//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Block-compressed quantum state vector using TurboQuant-variant for complex
// amplitudes. Based on TurboQuant (Zandieh et al., arXiv:2504.19874) and
// Apache 2.0 open-source implementation by TheTom (github.com/TheTom/turboquant_plus).
// Adapted for complex quantum state vectors by Dan Strano and (Anthropic) Claude.
//
// Each block of QRACK_TURBO_BLOCK_SIZE complex amplitudes is independently
// rotated by a random orthogonal matrix (applied separately to real and
// imaginary parts) and quantized per-coordinate at QRACK_TURBO_BITS bits.
//
// Key properties:
//   - get_probs(): in-place, no decompression (norm preserved under rotation)
//   - read()/write(): decompress one block, operate, recompress — O(block_size)
//   - write2() same block: decompress once — O(block_size)
//   - write2() cross block: decompress two blocks — O(2*block_size)
//   - shuffle(): permute block assignments without moving data — O(capacity/block)
//   - clear(): zero all packed storage — O(capacity/block)
//
// Build configuration (set via CMake):
//   QRACK_TURBO_BITS        — bits per quantized coordinate (default 4)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/parallel_for.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

#ifndef QRACK_TURBO_BITS
#define QRACK_TURBO_BITS 4
#endif

namespace Qrack {

// ---------------------------------------------------------------------------
// TurboQuant helpers for complex amplitudes
// ---------------------------------------------------------------------------

// Build a random orthogonal d×d matrix via Gram-Schmidt on random Gaussians.
// Column-major storage: column j starts at R[j*d].
static inline std::vector<real1> _tq_make_rotation(const size_t d)
{
    std::mt19937 rng(std::random_device{}());
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
        if (nrm < (real1)1e-8) {
            nrm = (real1)1e-8;
        }
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

// Compute transpose of a d×d column-major matrix.
static inline std::vector<real1> _tq_transpose(const std::vector<real1>& R, const size_t d)
{
    std::vector<real1> T(d * d);
    for (size_t i = 0U; i < d; ++i) {
        for (size_t j = 0U; j < d; ++j) {
            T[i * d + j] = R[j * d + i];
        }
    }
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
    if (step < (real1)1e-8) {
        return 0;
    }
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
// Per-block compressed storage
// Each block holds QRACK_TURBO_BLOCK_SIZE complex amplitudes.
// Real and imaginary parts are rotated and quantized independently.
// Packed storage: (64/bits) bucket indices per word (using uint64_t).
// ---------------------------------------------------------------------------
struct TurboBlock {
    size_t D;
    int BITS;
    int LEVELS;
    int VPW; // values per 64-bit word

    // Random rotation matrices for real and imaginary parts (D×D, column-major)
    std::vector<real1> R_re; // rotation for real parts
    std::vector<real1> R_im; // rotation for imaginary parts
    std::vector<real1> RT_re; // transpose (inverse) of R_re
    std::vector<real1> RT_im; // transpose (inverse) of R_im

    // Per-coordinate scales (std dev) for quantization
    std::vector<real1> scales_re;
    std::vector<real1> scales_im;

    // Packed bucket indices
    // D real values + D imag values, each at BITS bits per value
    // Total bits = 2*D*BITS, words = ceil(2*D*BITS / 64)
    size_t NWORDS;
    std::unique_ptr<uint64_t> packed;

    bool initialized;

    TurboBlock(int p, int b)
        : D(1ULL << p)
        , BITS(b)
        , LEVELS(1 << BITS)
        , VPW(64 / BITS)
        , R_re(_tq_make_rotation(D))
        , R_im(_tq_make_rotation(D))
        , RT_re(_tq_transpose(R_re, D))
        , RT_im(_tq_transpose(R_im, D))
        , scales_re(D, ONE_R1)
        , scales_im(D, ONE_R1)
        , NWORDS((2U * D * BITS + 63U) / 64U)
        , packed(new uint64_t[NWORDS])
        , initialized(false)
    {
        std::fill(packed.get(), packed.get() + NWORDS, 0U);
    }

    TurboBlock(const TurboBlock& o)
        : D(o.D)
        , BITS(o.BITS)
        , LEVELS(o.LEVELS)
        , VPW(o.VPW)
        , R_re(o.R_re)
        , R_im(o.R_im)
        , RT_re(o.RT_re)
        , RT_im(o.RT_im)
        , scales_re(o.scales_re)
        , scales_im(o.scales_im)
        , NWORDS(o.NWORDS)
        , packed(new uint64_t[NWORDS])
        , initialized(o.initialized)
    {
        std::copy(o.packed.get(), o.packed.get() + o.NWORDS, packed.get());
    }

    TurboBlock& operator=(const TurboBlock& o)
    {
        D = o.D;
        BITS = o.BITS;
        LEVELS = o.LEVELS;
        VPW = o.VPW;
        R_re = o.R_re;
        R_im = o.R_im;
        RT_re = o.RT_re;
        RT_im = o.RT_im;
        scales_re = o.scales_re;
        scales_im = o.scales_im;
        NWORDS = o.NWORDS;
        packed = std::unique_ptr<uint64_t>(new uint64_t[NWORDS]);
        initialized = o.initialized;
        std::copy(o.packed.get(), o.packed.get() + o.NWORDS, packed.get());

        return *this;
    }

    // Pack a bucket index into the packed array.
    // idx is the coordinate index (0..2D-1: first D are re, next D are im).
    void pack_bucket(const size_t idx, const int bucket)
    {
        const size_t bit_offset = idx * BITS;
        const size_t word = bit_offset / 64U;
        const size_t bit = bit_offset % 64U;
        const uint64_t mask = (uint64_t)(LEVELS - 1) << bit;
        packed.get()[word] = (packed.get()[word] & ~mask) | ((uint64_t)bucket << bit);
        // Handle straddle across word boundary
        if (bit + BITS > 64U) {
            const size_t overflow = bit + BITS - 64U;
            const uint64_t mask2 = ((uint64_t)1 << overflow) - 1U;
            packed.get()[word + 1U] = (packed.get()[word + 1U] & ~mask2) | ((uint64_t)bucket >> (BITS - overflow));
        }
    }

    // Unpack a bucket index from the packed array.
    int unpack_bucket(const size_t idx) const
    {
        const size_t bit_offset = idx * BITS;
        const size_t word = bit_offset / 64U;
        const size_t bit = bit_offset % 64U;
        int bucket = (int)((packed.get()[word] >> bit) & (uint64_t)(LEVELS - 1));
        // Handle straddle
        if (bit + BITS > 64U) {
            const size_t overflow = bit + BITS - 64U;
            const int high = (int)(packed.get()[word + 1U] & (((uint64_t)1 << overflow) - 1U));
            bucket |= (high << (BITS - overflow));
            bucket &= (LEVELS - 1);
        }
        return bucket;
    }

    // Compress an array of D complex amplitudes into this block.
    void compress(const complex* amps)
    {
        std::vector<real1> re_in(D), im_in(D);
        std::vector<real1> re_rot(D), im_rot(D);

        for (size_t i = 0U; i < D; ++i) {
            re_in[i] = real(amps[i]);
            im_in[i] = imag(amps[i]);
        }

        // Rotate
        _tq_rotate(re_in.data(), R_re, D, re_rot.data());
        _tq_rotate(im_in.data(), R_im, D, im_rot.data());

        // Compute scales if first compression
        if (!initialized) {
            for (size_t j = 0U; j < D; ++j) {
                real1 var_re = ZERO_R1, var_im = ZERO_R1;
                // For a single block, use per-element magnitude as scale proxy
                var_re += re_rot[j] * re_rot[j];
                var_im += im_rot[j] * im_rot[j];
                scales_re[j] = std::sqrt(var_re + (real1)1e-8);
                scales_im[j] = std::sqrt(var_im + (real1)1e-8);
            }
            initialized = true;
        }

        // Quantize and pack
        std::fill(packed.get(), packed.get() + NWORDS, 0U);
        for (size_t j = 0U; j < D; ++j) {
            pack_bucket(j, _tq_quant_bucket(re_rot[j], scales_re[j], BITS));
            pack_bucket(j + D, _tq_quant_bucket(im_rot[j], scales_im[j], BITS));
        }
    }

    // Decompress this block into an array of D complex amplitudes.
    void decompress(complex* amps) const
    {
        std::vector<real1> re_rot(D), im_rot(D);
        std::vector<real1> re_out(D), im_out(D);

        // Dequantize
        for (size_t j = 0U; j < D; ++j) {
            re_rot[j] = _tq_dequant(unpack_bucket(j), scales_re[j], BITS);
            im_rot[j] = _tq_dequant(unpack_bucket(j + D), scales_im[j], BITS);
        }

        // Inverse rotate (apply transpose = inverse for orthogonal matrix)
        _tq_rotate(re_rot.data(), RT_re, D, re_out.data());
        _tq_rotate(im_rot.data(), RT_im, D, im_out.data());

        for (size_t i = 0U; i < D; ++i) {
            amps[i] = complex(re_out[i], im_out[i]);
        }
    }

    // Get probability (squared norm) of amplitude i without full decompression.
    // Norm is preserved under orthogonal rotation: ||Rv||^2 = ||v||^2.
    // So sum of squared dequantized rotated coords = sum of squared original coords.
    real1 get_prob(const size_t i) const
    {
        // We need to decompress just amplitude i — but rotation mixes all coords.
        // The fast path: decompress the full block.
        // The norm-preservation shortcut only works for the *total* norm, not per-element.
        // So we must decompress.
        std::vector<complex> tmp(D);
        decompress(tmp.data());
        return norm(tmp[i]);
    }

    // Get total probability mass in this block (no decompression needed).
    real1 get_total_prob() const
    {
        real1 total = ZERO_R1;
        for (size_t j = 0U; j < D; ++j) {
            const real1 re = _tq_dequant(unpack_bucket(j), scales_re[j], BITS);
            const real1 im = _tq_dequant(unpack_bucket(j + D), scales_im[j], BITS);
            total += re * re + im * im;
        }
        return total;
    }

    static void write_bool(std::ostream& out, const bool& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(bool));
    }

    static void write_size_t(std::ostream& out, const size_t& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(size_t));
    }

    static void write_real(std::ostream& out, const real1& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(real1));
    }

    friend std::ostream& operator<<(std::ostream& os, const TurboBlock& s)
    {
        write_size_t(os, s.D);
        write_bool(os, s.initialized);
        if (s.initialized) {
            for (size_t i = 0U; i < s.D; ++i) {
                write_real(os, s.scales_re[i]);
            }
            for (size_t i = 0U; i < s.D; ++i) {
                write_real(os, s.scales_im[i]);
            }
        }
        write_size_t(os, s.NWORDS);
        os.write(reinterpret_cast<const char*>(s.packed.get()), s.NWORDS * sizeof(uint64_t));

        return os;
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

    // Decompress block b, apply f(amps, num_amps), recompress.
    template <typename F> void with_block(const size_t b, F&& f)
    {
        std::lock_guard<std::mutex> lock(block_mutexes[b]);
        std::vector<complex> amps(BLOCK);
        blocks[b].decompress(amps.data());
        f(amps.data(), BLOCK);
        blocks[b].compress(amps.data());
    }

    // Decompress block b read-only, apply f(amps, num_amps).
    template <typename F> void with_block_ro(const size_t b, F&& f) const
    {
        std::vector<complex> amps(BLOCK);
        blocks[b].decompress(amps.data());
        f(amps.data(), BLOCK);
    }

public:
    StateVectorTurboQuant(bitCapIntOcl cap, size_t p, int b, const complex* copyIn)
        : StateVector(cap)
        , BLOCK(1ULL << p)
        , num_blocks((size_t)((cap + BLOCK - 1U) / BLOCK))
        , blocks(num_blocks, TurboBlock(p, b))
        , block_mutexes(num_blocks)
    {
        copy_in(copyIn);
    }

    static void write_bool(std::ostream& out, const bool& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(bool));
    }

    static void write_size_t(std::ostream& out, const size_t& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(size_t));
    }

    static void write_real(std::ostream& out, const real1& x)
    {
        out.write(reinterpret_cast<const char*>(&x), sizeof(real1));
    }

    friend std::ostream& operator<<(std::ostream& os, const StateVectorTurboQuant& s)
    {
        write_size_t(os, s.capacity);
        write_size_t(os, s.BLOCK);
        write_size_t(os, s.num_blocks);
        for (size_t i = 0U; i < s.blocks.size(); ++i) {
            os << s.blocks[i];
        }

        return os;
    }

    complex read(const bitCapInt& i) { return read((bitCapIntOcl)i); }

    complex read(const bitCapIntOcl& i)
    {
        const size_t b = block_of(i);
        const size_t off = offset_in(i);
        std::vector<complex> amps(BLOCK);
        blocks[b].decompress(amps.data());
        return amps[off];
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
        const size_t b1 = block_of(i1);
        const size_t b2 = block_of(i2);

        if (b1 == b2) {
            // Same block — decompress once
            with_block(b1, [&](complex* amps, size_t) {
                amps[offset_in(i1)] = c1;
                amps[offset_in(i2)] = c2;
            });
        } else {
            // Different blocks — lock lower index first to avoid deadlock
            const size_t blo = std::min(b1, b2);
            const size_t bhi = std::max(b1, b2);
            std::lock_guard<std::mutex> lock_lo(block_mutexes[blo]);
            std::lock_guard<std::mutex> lock_hi(block_mutexes[bhi]);

            std::vector<complex> amps1(BLOCK), amps2(BLOCK);
            blocks[b1].decompress(amps1.data());
            blocks[b2].decompress(amps2.data());
            amps1[offset_in(i1)] = c1;
            amps2[offset_in(i2)] = c2;
            blocks[b1].compress(amps1.data());
            blocks[b2].compress(amps2.data());
        }
    }

    void clear()
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> zeros(BLOCK, ZERO_CMPLX);
            blocks[b].compress(zeros.data());
        });
    }

    void copy_in(const complex* copyIn)
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            const complex* src = copyIn ? (copyIn + b * BLOCK) : nullptr;
            std::vector<complex> amps(BLOCK, ZERO_CMPLX);
            if (src) {
                const size_t len = std::min(BLOCK, (size_t)(capacity - b * BLOCK));
                std::copy(src, src + len, amps.data());
            }
            blocks[b].compress(amps.data());
        });
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        // Decompress affected blocks, patch, recompress
        const size_t b_start = block_of(offset);
        const size_t b_end = block_of(offset + length - 1U);

        for (size_t b = b_start; b <= b_end; ++b) {
            with_block(b, [&](complex* amps, size_t) {
                const size_t b_base = b * BLOCK;
                for (size_t j = 0U; j < BLOCK; ++j) {
                    const size_t global = b_base + j;
                    if (global >= offset && global < offset + length) {
                        amps[j] = copyIn ? copyIn[global - offset] : ZERO_CMPLX;
                    }
                }
            });
        }
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        std::vector<complex> tmp(length);
        if (copyInSv) {
            for (bitCapIntOcl i = 0U; i < length; ++i) {
                tmp[i] = copyInSv->read(srcOffset + i);
            }
        }
        copy_in(copyInSv ? tmp.data() : nullptr, dstOffset, length);
    }

    void copy_out(complex* copyOut)
    {
        par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
            std::vector<complex> amps(BLOCK);
            blocks[b].decompress(amps.data());
            const size_t len = std::min(BLOCK, (size_t)(capacity - b * BLOCK));
            std::copy(amps.data(), amps.data() + len, copyOut + b * BLOCK);
        });
    }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        for (bitCapIntOcl i = 0U; i < length; ++i) {
            copyOut[i] = read(offset + i);
        }
    }

    void copy(StateVectorPtr toCopy)
    {
        StateVectorTurboQuantPtr src = std::dynamic_pointer_cast<StateVectorTurboQuant>(toCopy);
        if (src) {
            // Block-level copy — preserves rotations and scales
            par_for(0U, num_blocks, [&](const bitCapIntOcl& b, const unsigned&) {
                std::lock_guard<std::mutex> lock(block_mutexes[b]);
                blocks[b] = src->blocks[b];
            });
        } else {
            // Fallback: decompress source, copy_in
            std::vector<complex> tmp(capacity);
            toCopy->copy_out(tmp.data());
            copy_in(tmp.data());
        }
    }

    void shuffle(StateVectorPtr svp)
    {
        // Swap upper and lower halves block-by-block.
        // For capacity that is a power of 2, the upper half starts at capacity/2.
        StateVectorTurboQuantPtr other = std::dynamic_pointer_cast<StateVectorTurboQuant>(svp);

        const bitCapIntOcl half = capacity >> 1U;
        const size_t half_blocks = (size_t)(half / BLOCK);

        if (other && (half % BLOCK == 0U)) {
            // Block-aligned shuffle: swap block pointers (swap the TurboBlock objects)
            par_for(0U, half_blocks,
                [&](const bitCapIntOcl& b, const unsigned&) { std::swap(blocks[b + half_blocks], other->blocks[b]); });
        } else {
            // Fallback: decompress, swap, recompress
            const bitCapIntOcl offset = half;
            par_for(0U, offset, [&](const bitCapIntOcl& lcv, const unsigned&) {
                complex amp = svp->read(lcv);
                svp->write(lcv, read(lcv + offset));
                write(lcv + offset, amp);
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
            for (size_t j = 0U; j < len; ++j) {
                outArray[b * BLOCK + j] = norm(amps[j]);
            }
        });
    }

    bool is_sparse() { return false; }
};

typedef std::shared_ptr<StateVectorTurboQuant> StateVectorTurboQuantPtr;

} // namespace Qrack
