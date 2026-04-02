//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Block-compressed quantum state vector using TurboQuant-variant for complex
// amplitudes. Based on TurboQuant (Zandieh et al., arXiv:2504.19874) and
// Apache 2.0 open-source implementation by TheTom (github.com/TheTom/turboquant_plus).
// Adapted for complex quantum state vectors by Dan Strano and (Anthropic) Claude.
//
// Each block of (1 << p) complex amplitudes is independently rotated by a
// random orthogonal matrix (applied separately to real and imaginary parts)
// and quantized per-coordinate at b bits.
//
// Key properties:
//   - get_probs(): decompresses block-by-block in parallel
//   - read()/write(): decompress one block, operate, recompress — O(block_size)
//   - write2() same block: decompress once — O(block_size)
//   - write2() cross block: decompress two blocks — O(2*block_size)
//   - shuffle(): block-swap when half-capacity is block-aligned
//   - Serialization: seed (8 bytes) + scales + packed data per block;
//     rotation matrices regenerated from seed on load — O(1) vs O(D²) storage
//
// Build configuration (set via CMake, overridable at runtime via constructor):
//   QRACK_TURBO_BITS — default bits per quantized coordinate (default 4)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "statevector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#ifndef QRACK_TURBO_BLOCK_SIZE
#define QRACK_TURBO_BLOCK_SIZE 64U
#endif

#ifndef QRACK_TURBO_BITS
#define QRACK_TURBO_BITS 4
#endif

namespace Qrack {

// ---------------------------------------------------------------------------
// TurboQuant helpers
// ---------------------------------------------------------------------------

// Build a random orthogonal d×d matrix from a fixed seed.
// Storing the seed rather than the matrix reduces serialized size from
// O(d²) to O(1) — critical for large block sizes.
// Column-major storage: column j starts at R[j*d].
static inline std::vector<real1> _tq_make_rotation(const size_t d, const uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<real1> normal(ZERO_R1, ONE_R1);
    std::vector<real1> R(d * d);
    for (auto& v : R) {
        v = normal(rng);
    }
    for (size_t j = 0U; j < d; ++j) {
        real1 nrm = ZERO_R1;
        for (size_t i = 0U; i < d; ++i) {
            nrm += R[j * d + i] * R[j * d + i];
        }
        nrm = std::sqrt(nrm);
        if (nrm < (real1)1e-8)
            nrm = (real1)1e-8;
        for (size_t i = 0U; i < d; ++i) {
            R[j * d + i] /= nrm;
        }
        for (size_t k = j + 1U; k < d; ++k) {
            real1 dot = ZERO_R1;
            for (size_t i = 0U; i < d; ++i) {
                dot += R[j * d + i] * R[k * d + i];
            }
            for (size_t i = 0U; i < d; ++i) {
                R[k * d + i] -= dot * R[j * d + i];
            }
        }
    }
    return R;
}

// Convenience overload: generate a random seed from hardware entropy,
// return both the rotation and the seed used (for later serialization).
static inline std::vector<real1> _tq_make_rotation(const size_t d, uint64_t* seed_out)
{
    std::random_device rd;
    *seed_out = ((uint64_t)rd() << 32U) | (uint64_t)rd();
    return _tq_make_rotation(d, *seed_out);
}

// Compute transpose of a d×d column-major matrix.
static inline std::vector<real1> _tq_transpose(const std::vector<real1>& R, const size_t d)
{
    std::vector<real1> T(d * d);
    for (size_t i = 0U; i < d; ++i)
        for (size_t j = 0U; j < d; ++j)
            T[i * d + j] = R[j * d + i];
    return T;
}

// Apply d×d column-major rotation R to vector v of length d.
// Result written into out (may alias v if caller provides scratch).
static inline void _tq_rotate(const real1* v, const std::vector<real1>& R, const size_t d, real1* out)
{
    for (size_t i = 0U; i < d; ++i) {
        real1 s = ZERO_R1;
        for (size_t j = 0U; j < d; ++j) {
            s += R[j * d + i] * v[j];
        }
        out[i] = s;
    }
}

// Quantize a single real value to bits-bit bucket index, given scale (std dev).
static inline int _tq_quant_bucket(const real1 val, const real1 scale, const int bits)
{
    const int levels = 1 << bits;
    const real1 lo = (real1)-3.0 * scale;
    const real1 hi = (real1)3.0 * scale;
    const real1 step = (hi - lo) / (real1)levels;
    if (step < (real1)1e-8)
        return 0;
    const real1 clamped = std::max(lo, std::min(hi - step, val));
    int bucket = (int)((clamped - lo) / step);
    if (bucket < 0)
        bucket = 0;
    if (bucket >= levels)
        bucket = levels - 1;
    return bucket;
}

// Dequantize a bucket index back to a real value.
static inline real1 _tq_dequant(const int bucket, const real1 scale, const int bits)
{
    const int levels = 1 << bits;
    const real1 lo = (real1)-3.0 * scale;
    const real1 hi = (real1)3.0 * scale;
    const real1 step = (hi - lo) / (real1)levels;
    return lo + ((real1)bucket + (real1)0.5) * step;
}

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------
static inline void _tq_write_size(std::ostream& os, const size_t x)
{
    os.write(reinterpret_cast<const char*>(&x), sizeof(size_t));
}
static inline void _tq_write_int(std::ostream& os, const int x)
{
    os.write(reinterpret_cast<const char*>(&x), sizeof(int));
}
static inline void _tq_write_bool(std::ostream& os, const bool x)
{
    os.write(reinterpret_cast<const char*>(&x), sizeof(bool));
}
static inline size_t _tq_read_size(std::istream& is)
{
    size_t x;
    is.read(reinterpret_cast<char*>(&x), sizeof(size_t));
    return x;
}
static inline int _tq_read_int(std::istream& is)
{
    int x;
    is.read(reinterpret_cast<char*>(&x), sizeof(int));
    return x;
}
static inline bool _tq_read_bool(std::istream& is)
{
    bool x;
    is.read(reinterpret_cast<char*>(&x), sizeof(bool));
    return x;
}

// ---------------------------------------------------------------------------
// TurboBlock
//
// Design rationale:
//   - Single rotation matrix R acts on the D-dimensional real vector formed
//     by interleaving re and im parts: [re_0, im_0, re_1, im_1, ...].
//     This respects the joint complex structure of Hilbert space — the
//     natural unit is the complex amplitude |ψ|² = re² + im², not re and im
//     independently. Under Haar measure, |ψ|² is uniformly distributed, and
//     the rotation concentrates the interleaved real vector into approximately
//     equal-variance Gaussian coordinates.
//   - Single scalar block_scale = RMS amplitude magnitude over the whole
//     block, estimated on first compression. Both re and im coordinates are
//     quantized against this same scale, which is theoretically correct since
//     Haar uniformity implies equal variance for re and im after rotation.
//   - One seed (8 bytes) replaces the full D×D rotation matrix, reducing
//     serialized size from O(D²) to O(1). The matrix is regenerated
//     deterministically from the seed on load.
//
// Serialized size per block:
//   header:      ~21 bytes
//   seed:         8 bytes
//   block_scale:  4 bytes (one real1)
//   packed:       (2*D*BITS + 63) / 64 * 8 bytes
//   ----------------------------------------
//   For D=64, BITS=4: 21 + 8 + 4 + 64 = ~97 bytes per block
//   For n=16 qubits, 1024 blocks: ~97 KB  (vs 512 KB uncompressed)
// ---------------------------------------------------------------------------
struct TurboBlock {
    size_t D; // block size = number of complex amplitudes
    int BITS; // bits per quantized coordinate
    int LEVELS; // 1 << BITS

    // Single seed for the one rotation matrix acting on the 2D-dimensional
    // interleaved [re_0, im_0, re_1, im_1, ...] real vector.
    uint64_t seed;

    // Rotation matrix R (2D×2D, col-major) and its transpose RT (= inverse).
    // Regenerated from seed — not serialized.
    std::vector<real1> R; // rotation: 2D×2D
    std::vector<real1> RT; // transpose of R

    // Single scalar scale: RMS amplitude magnitude over whole block.
    // Both re and im coordinates are quantized against this scale.
    real1 block_scale;

    // Packed bucket indices: 2D coordinates (D re + D im), BITS each.
    size_t NWORDS;
    std::unique_ptr<uint64_t[]> packed;
    bool initialized;

    TurboBlock(int p, int b)
        : D(1ULL << p)
        , BITS(b)
        , LEVELS(1 << b)
        , seed(0U)
        , R(_tq_make_rotation(2U * (1ULL << p), &seed))
        , RT(_tq_transpose(R, 2U * (1ULL << p)))
        , block_scale(ONE_R1)
        , NWORDS((2U * (1ULL << p) * b + 63U) / 64U)
        , packed(new uint64_t[(2U * (1ULL << p) * b + 63U) / 64U])
        , initialized(false)
    {
        std::fill(packed.get(), packed.get() + NWORDS, 0U);
    }

    TurboBlock(const TurboBlock& o)
        : D(o.D)
        , BITS(o.BITS)
        , LEVELS(o.LEVELS)
        , seed(o.seed)
        , R(o.R)
        , RT(o.RT)
        , block_scale(o.block_scale)
        , NWORDS(o.NWORDS)
        , packed(new uint64_t[o.NWORDS])
        , initialized(o.initialized)
    {
        std::copy(o.packed.get(), o.packed.get() + o.NWORDS, packed.get());
    }

    TurboBlock& operator=(const TurboBlock& o)
    {
        if (this == &o)
            return *this;
        D = o.D;
        BITS = o.BITS;
        LEVELS = o.LEVELS;
        seed = o.seed;
        R = o.R;
        RT = o.RT;
        block_scale = o.block_scale;
        NWORDS = o.NWORDS;
        packed.reset(new uint64_t[NWORDS]);
        initialized = o.initialized;
        std::copy(o.packed.get(), o.packed.get() + o.NWORDS, packed.get());
        return *this;
    }

    // Pack a bucket index into the packed array.
    // idx is the coordinate index.
    void pack_bucket(const size_t idx, const int bucket)
    {
        const size_t bit_offset = idx * (size_t)BITS;
        const size_t word = bit_offset / 64U;
        const size_t bit = bit_offset % 64U;
        const uint64_t mask = (uint64_t)(LEVELS - 1) << bit;
        packed[word] = (packed[word] & ~mask) | ((uint64_t)bucket << bit);
        if (bit + (size_t)BITS > 64U) {
            const size_t overflow = bit + (size_t)BITS - 64U;
            const uint64_t mask2 = ((uint64_t)1 << overflow) - 1U;
            packed[word + 1U] = (packed[word + 1U] & ~mask2) | ((uint64_t)bucket >> (BITS - (int)overflow));
        }
    }

    // Unpack a bucket index from the packed array.
    int unpack_bucket(const size_t idx) const
    {
        const size_t bit_offset = idx * (size_t)BITS;
        const size_t word = bit_offset / 64U;
        const size_t bit = bit_offset % 64U;
        int bucket = (int)((packed[word] >> bit) & (uint64_t)(LEVELS - 1));
        if (bit + (size_t)BITS > 64U) {
            const size_t overflow = bit + (size_t)BITS - 64U;
            const int high = (int)(packed[word + 1U] & (((uint64_t)1 << overflow) - 1U));
            bucket |= (high << (BITS - (int)overflow));
            bucket &= (LEVELS - 1);
        }
        return bucket;
    }

    // Compress an array of D complex amplitudes into this block.
    void compress(const complex* amps)
    {
        // Interleave re and im into a single 2D-dimensional real vector,
        // then apply the single joint rotation.
        const size_t D2 = 2U * D;
        std::vector<real1> v_in(D2), v_rot(D2);
        for (size_t i = 0U; i < D; ++i) {
            v_in[2U * i] = real(amps[i]);
            v_in[2U * i + 1U] = imag(amps[i]);
        }
        _tq_rotate(v_in.data(), R, D2, v_rot.data());

        // On first compression, estimate block_scale as RMS amplitude magnitude.
        // Under Haar uniformity both re and im contributions are equal, so a
        // single scalar captures the joint distribution correctly.
        if (!initialized) {
            real1 sum = ZERO_R1;
            for (size_t j = 0U; j < D2; ++j) {
                sum += v_rot[j] * v_rot[j];
            }
            block_scale = std::sqrt(sum / (real1)D2 + (real1)1e-8);
            initialized = true;
        }

        std::fill(packed.get(), packed.get() + NWORDS, 0U);
        for (size_t j = 0U; j < D2; ++j) {
            pack_bucket(j, _tq_quant_bucket(v_rot[j], block_scale, BITS));
        }
    }

    // Decompress this block into an array of D complex amplitudes.
    void decompress(complex* amps) const
    {
        const size_t D2 = 2U * D;
        std::vector<real1> v_rot(D2), v_out(D2);
        for (size_t j = 0U; j < D2; ++j) {
            v_rot[j] = _tq_dequant(unpack_bucket(j), block_scale, BITS);
        }
        _tq_rotate(v_rot.data(), RT, D2, v_out.data());
        for (size_t i = 0U; i < D; ++i) {
            amps[i] = complex(v_out[2U * i], v_out[2U * i + 1U]);
        }
    }

    // Total probability mass in this block — computable without full
    // decompression since ||Rv||² = ||v||² under orthogonal rotation.
    real1 get_total_prob() const
    {
        real1 total = ZERO_R1;
        const size_t D2 = 2U * D;
        for (size_t j = 0U; j < D2; ++j) {
            const real1 v = _tq_dequant(unpack_bucket(j), block_scale, BITS);
            total += v * v;
        }
        return total;
    }

    // --- Serialization ------------------------------------------------------
    //
    // Per-block binary format:
    //   size_t    D
    //   int       BITS
    //   bool      initialized
    //   uint64_t  seed          (rotation regenerated on load — O(1) storage)
    //   if initialized:
    //     real1   block_scale
    //   size_t    NWORDS
    //   uint64_t[NWORDS] packed

    void save(std::ostream& os) const
    {
        _tq_write_size(os, D);
        _tq_write_int(os, BITS);
        _tq_write_bool(os, initialized);
        os.write(reinterpret_cast<const char*>(&seed), sizeof(uint64_t));
        if (initialized) {
            os.write(reinterpret_cast<const char*>(&block_scale), sizeof(real1));
        }
        _tq_write_size(os, NWORDS);
        os.write(reinterpret_cast<const char*>(packed.get()), (std::streamsize)(NWORDS * sizeof(uint64_t)));
    }

    static TurboBlock load(std::istream& is)
    {
        const size_t D_in = _tq_read_size(is);
        const int BITS_in = _tq_read_int(is);
        const bool init = _tq_read_bool(is);

        // Compute p = log2(D_in)
        int p = 0;
        for (size_t tmp = D_in; tmp > 1U; tmp >>= 1U) {
            ++p;
        }

        uint64_t seed_in;
        is.read(reinterpret_cast<char*>(&seed_in), sizeof(uint64_t));

        // Reconstruct block, then overwrite rotation with seeded version
        TurboBlock blk(p, BITS_in);
        blk.seed = seed_in;
        blk.R = _tq_make_rotation(2U * D_in, seed_in);
        blk.RT = _tq_transpose(blk.R, 2U * D_in);
        blk.initialized = init;

        if (init) {
            is.read(reinterpret_cast<char*>(&blk.block_scale), sizeof(real1));
        }

        const size_t nwords = _tq_read_size(is);
        blk.NWORDS = nwords;
        blk.packed.reset(new uint64_t[nwords]);
        is.read(reinterpret_cast<char*>(blk.packed.get()), (std::streamsize)(nwords * sizeof(uint64_t)));

        return blk;
    }

    friend std::ostream& operator<<(std::ostream& os, const TurboBlock& b)
    {
        b.save(os);
        return os;
    }

    friend std::istream& operator>>(std::istream& is, TurboBlock& b)
    {
        b = TurboBlock::load(is);
        return is;
    }
};

// ---------------------------------------------------------------------------
// StateVectorTurboQuant
// ---------------------------------------------------------------------------
class StateVectorTurboQuant;
typedef std::shared_ptr<StateVectorTurboQuant> StateVectorTurboQuantPtr;

class StateVectorTurboQuant : public StateVector {
protected:
    size_t BLOCK;
    size_t num_blocks;
    std::vector<TurboBlock> blocks;
    std::vector<std::mutex> block_mutexes;

    size_t block_of(const bitCapIntOcl i) const { return (size_t)(i / BLOCK); }
    size_t offset_in(const bitCapIntOcl i) const { return (size_t)(i % BLOCK); }

    template <typename F> void with_block(const size_t b, F&& f)
    {
        std::lock_guard<std::mutex> lock(block_mutexes[b]);
        std::vector<complex> amps(BLOCK);
        blocks[b].decompress(amps.data());
        f(amps.data(), BLOCK);
        blocks[b].compress(amps.data());
    }

public:
    // Construct from raw amplitudes (nullptr = |0⟩)
    StateVectorTurboQuant(
        bitCapIntOcl cap, int p = (int)log2Ocl(QRACK_TURBO_BLOCK_SIZE), int b = 4, const complex* copyIn = nullptr)
        : StateVector(cap)
        , BLOCK(1ULL << p)
        , num_blocks((cap + (1ULL << p) - 1U) / (1ULL << p))
        , blocks(num_blocks, TurboBlock(p, b))
        , block_mutexes(num_blocks)
    {
        if (copyIn) {
            copy_in(copyIn);
        } else {
            // Initialize to |0⟩: amplitude 0 = 1, rest = 0
            std::vector<complex> init(BLOCK, ZERO_CMPLX);
            init[0] = complex(ONE_R1, ZERO_R1);
            blocks[0].compress(init.data());
            for (size_t b = 1U; b < num_blocks; ++b) {
                std::vector<complex> zeros(BLOCK, ZERO_CMPLX);
                blocks[b].compress(zeros.data());
            }
        }
    }

    bitCapIntOcl get_size() { return capacity; }

    // --- Serialization ------------------------------------------------------
    //
    // Stream format:
    //   size_t  capacity
    //   size_t  BLOCK
    //   size_t  num_blocks
    //   TurboBlock[num_blocks]

    void save(std::ostream& os) const
    {
        _tq_write_size(os, (size_t)capacity);
        _tq_write_size(os, BLOCK);
        _tq_write_size(os, num_blocks);
        for (size_t i = 0U; i < num_blocks; ++i) {
            blocks[i].save(os);
        }
    }

    static StateVectorTurboQuantPtr load(std::istream& is)
    {
        const bitCapIntOcl cap = (bitCapIntOcl)_tq_read_size(is);
        const size_t block_size = _tq_read_size(is);
        const size_t nblocks = _tq_read_size(is);

        int p = 0;
        for (size_t tmp = block_size; tmp > 1U; tmp >>= 1U) {
            ++p;
        }

        // Read all blocks
        std::vector<TurboBlock> loaded;
        loaded.reserve(nblocks);
        for (size_t i = 0U; i < nblocks; ++i) {
            loaded.push_back(TurboBlock::load(is));
        }

        const int bits = loaded.empty() ? QRACK_TURBO_BITS : loaded[0].BITS;

        // Construct shell with correct geometry, then overwrite blocks
        auto sv = std::make_shared<StateVectorTurboQuant>(cap, p, bits, nullptr);
        sv->blocks = std::move(loaded);

        return sv;
    }

    friend std::ostream& operator<<(std::ostream& os, const StateVectorTurboQuant& sv)
    {
        sv.save(os);
        return os;
    }

    friend std::istream& operator>>(std::istream& is, StateVectorTurboQuantPtr& sv)
    {
        sv = StateVectorTurboQuant::load(is);
        return is;
    }

    // --- StateVector interface ----------------------------------------------

    complex read(const bitCapInt& i) { return read((bitCapIntOcl)i); }
    complex read(const bitCapIntOcl& i)
    {
        std::vector<complex> amps(BLOCK);
        blocks[block_of(i)].decompress(amps.data());
        return amps[offset_in(i)];
    }

#if ENABLE_COMPLEX_X2
    complex2 read2(const bitCapInt& i1, const bitCapInt& i2) { return read2((bitCapIntOcl)i1, (bitCapIntOcl)i2); }
    complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2) { return complex2(read(i1), read(i2)); }
#endif

    void write(const bitCapInt& i, const complex& c) { write((bitCapIntOcl)i, c); }
    void write(const bitCapIntOcl& i, const complex& c)
    {
        with_block(block_of(i), [&](complex* amps, size_t) { amps[offset_in(i)] = c; });
    }

    void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2)
    {
        write2((bitCapIntOcl)i1, c1, (bitCapIntOcl)i2, c2);
    }

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        const size_t b1 = block_of(i1), b2 = block_of(i2);
        if (b1 == b2) {
            with_block(b1, [&](complex* amps, size_t) {
                amps[offset_in(i1)] = c1;
                amps[offset_in(i2)] = c2;
            });
        } else {
            const size_t blo = std::min(b1, b2), bhi = std::max(b1, b2);
            std::lock_guard<std::mutex> lo(block_mutexes[blo]);
            std::lock_guard<std::mutex> hi(block_mutexes[bhi]);
            std::vector<complex> a1(BLOCK), a2(BLOCK);
            blocks[b1].decompress(a1.data());
            blocks[b2].decompress(a2.data());
            a1[offset_in(i1)] = c1;
            a2[offset_in(i2)] = c2;
            blocks[b1].compress(a1.data());
            blocks[b2].compress(a2.data());
        }
    }

    void clear()
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> z(BLOCK, ZERO_CMPLX);
            blocks[b].compress(z.data());
        });
    }

    void copy_in(const complex* copyIn)
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> amps(BLOCK, ZERO_CMPLX);
            if (copyIn) {
                const size_t len = std::min(BLOCK, (size_t)(capacity - b * BLOCK));
                std::copy(copyIn + b * BLOCK, copyIn + b * BLOCK + len, amps.data());
            }
            blocks[b].compress(amps.data());
        });
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (!length)
            return;
        const size_t b0 = block_of(offset);
        const size_t b1 = block_of(offset + length - 1U);
        for (size_t b = b0; b <= b1; ++b) {
            with_block(b, [&](complex* amps, size_t) {
                const size_t base = b * BLOCK;
                for (size_t j = 0U; j < BLOCK; ++j) {
                    const size_t g = base + j;
                    if (g >= (size_t)offset && g < (size_t)(offset + length))
                        amps[j] = copyIn ? copyIn[g - offset] : ZERO_CMPLX;
                }
            });
        }
    }

    void copy_in(StateVectorPtr sv, const bitCapIntOcl src, const bitCapIntOcl dst, const bitCapIntOcl len)
    {
        std::vector<complex> tmp(len, ZERO_CMPLX);
        if (sv)
            for (bitCapIntOcl i = 0U; i < len; ++i)
                tmp[i] = sv->read(src + i);
        copy_in(sv ? tmp.data() : nullptr, dst, len);
    }

    void copy_out(complex* out)
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> amps(BLOCK);
            blocks[b].decompress(amps.data());
            const size_t len = std::min(BLOCK, (size_t)(capacity - b * BLOCK));
            std::copy(amps.data(), amps.data() + len, out + b * BLOCK);
        });
    }

    void copy_out(complex* out, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        for (bitCapIntOcl i = 0U; i < length; ++i)
            out[i] = read(offset + i);
    }

    void copy(StateVectorPtr toCopy)
    {
        auto src = std::dynamic_pointer_cast<StateVectorTurboQuant>(toCopy);
        if (src) {
            par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
                std::lock_guard<std::mutex> lk(block_mutexes[b]);
                blocks[b] = src->blocks[b];
            });
        } else {
            std::vector<complex> tmp(capacity);
            toCopy->copy_out(tmp.data());
            copy_in(tmp.data());
        }
    }

    void shuffle(StateVectorPtr svp)
    {
        // Swap upper and lower halves block-by-block.
        // For capacity that is a power of 2, the upper half starts at capacity/2
        auto other = std::dynamic_pointer_cast<StateVectorTurboQuant>(svp);
        const bitCapIntOcl half = capacity >> 1U;
        const size_t hb = (size_t)(half / BLOCK);
        if (other && (half % BLOCK == 0U)) {
            // Block-aligned shuffle: swap block pointers (swap the TurboBlock objects)
            par_for(
                0U, hb, [&](const bitCapIntOcl& b, const unsigned&) { std::swap(blocks[b + hb], other->blocks[b]); });
        } else {
            // Fallback: decompress, swap, recompress
            par_for(0U, half, [&](const bitCapIntOcl& i, const unsigned&) {
                complex amp = svp->read(i);
                svp->write(i, read(i + half));
                write(i + half, amp);
            });
        }
    }

    void get_probs(real1* outArray)
    {
        // Decompress block by block — norm preservation only holds for total block norm,
        // not per-amplitude, so we must decompress to get per-amplitude probs.
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> amps(BLOCK);
            blocks[b].decompress(amps.data());
            const size_t len = std::min(BLOCK, (size_t)(capacity - b * BLOCK));
            for (size_t j = 0U; j < len; ++j)
                outArray[b * BLOCK + j] = norm(amps[j]);
        });
    }

    bool is_sparse() { return false; }
};

typedef std::shared_ptr<StateVectorTurboQuant> StateVectorTurboQuantPtr;
} // namespace Qrack
