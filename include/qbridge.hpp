//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// QBridge: bridged patching. A chain (optionally a ring) of state-vector patches, each one an ordinary
// Qrack QInterface holding its data qubits plus "bond" qubits shared with its nearest neighbours.
// The represented state is all patch kets contracted on every bond pair with the unnormalized virtual
// Bell projection sum_k <k|<k|, i.e. a matrix-product state whose sites are entire Qrack simulators.
// In-patch gates run natively on the patch engine (any engine stack: CPU, OpenCL, QUnit, ...).
// Cross-patch gates are teleported through fresh bond pairs (gates only); after each bridged gate (or
// once per deferred block) an MPS sweep over the touched patches recompresses the bonds, truncating
// to maxBonds per cut (greedy or Born-rule). Nothing larger than the patch kets and bond-sized
// matrices is ever formed: no density matrices, and no new library dependencies.
//
// By (Anthropic) Claude and Dan Strano. Port of PyQrack's QrackBridge.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qfactory.hpp"

#include <map>
#include <random>
#include <set>

namespace Qrack {

class QBridge;
typedef std::shared_ptr<QBridge> QBridgePtr;

class QBridge : public ParallelFor {
public:
    typedef std::complex<double> cmplx_d;

    /** A patch ket reshaped as A[l, d, r] (left bond, data, right bond), row-major. */
    struct Tensor {
        size_t l, d, r;
        std::vector<complex> a;
        Tensor()
            : l(1U)
            , d(1U)
            , r(1U)
        {
        }
        Tensor(size_t l_, size_t d_, size_t r_)
            : l(l_)
            , d(d_)
            , r(r_)
            , a(l_ * d_ * r_, ZERO_CMPLX)
        {
        }
    };

protected:
    struct Tag {
        char kind; // 'd' data, 'L' left bond, 'R' right bond, '0' scratch (always |0> at recompression)
        bitLenInt idx;
        bool operator==(const Tag& o) const { return (kind == o.kind) && (idx == o.idx); }
    };

    bitLenInt qubitCount;
    bitLenInt patchCount;
    bool isRing;
    bitLenInt maxBonds;
    bitLenInt lazyBonds;
    bitLenInt maxPatchQubits;
    bool isBorn;
    std::vector<QInterfaceEngine> engines;
    qrack_rand_gen_ptr rand_generator;
    std::mt19937_64 sampler;

    std::vector<bitLenInt> patchStart;
    std::vector<bitLenInt> patchSize;
    std::vector<std::vector<Tag>> layout;
    std::vector<bitLenInt> bond;
    std::vector<QInterfacePtr> sims;

    bitLenInt center;
    std::set<bitLenInt> touched;
    bool wrapTouched;
    int deferring;

    double fidelityEstimate;
    size_t truncations;
    size_t flushes;
    bitLenInt peakBonds;

    QInterfacePtr MakeEngine(bitLenInt n);
    void ParRange(size_t n, size_t grain, std::function<void(size_t, size_t, unsigned)> fn);
    void Locate(bitLenInt q, bitLenInt& p, bitLenInt& i) const;
    bool HasLeftCut(bitLenInt p) const { return (p > 0U) || isRing; }
    bool HasRightCut(bitLenInt p) const { return ((p + 1U) < patchCount) || isRing; }
    bitLenInt LeftCut(bitLenInt p) const { return p ? (p - 1U) : (patchCount - 1U); }
    bitLenInt RightCut(bitLenInt p) const { return p; }
    bitLenInt LeftBonds(bitLenInt p) const { return HasLeftCut(p) ? bond[LeftCut(p)] : 0U; }
    bitLenInt RightBonds(bitLenInt p) const { return HasRightCut(p) ? bond[RightCut(p)] : 0U; }
    std::vector<bitLenInt> Path(bitLenInt pa, bitLenInt pb) const;
    void NewBond(bitLenInt here, bitLenInt there, bitLenInt& idHere, bitLenInt& idThere);
    bitLenInt NewScratch(bitLenInt p);
    bitLenInt Ship(const std::vector<bitLenInt>& path, bitLenInt src);
    void Unship(const std::vector<bitLenInt>& path, bitLenInt z, bitLenInt dst);
    void Controlled(bitLenInt c, bitLenInt t, const complex* mtrx, bool anti);
    void PairGate(bitLenInt a, bitLenInt b, std::function<void(QInterfacePtr, bitLenInt, bitLenInt)> fn);

    Tensor ReadTensor(bitLenInt p);
    void StoreTensor(bitLenInt p, const Tensor& t);
    std::vector<size_t> Select(const std::vector<double>& sv, bool exactOnly);
    void Sweep(bitLenInt lo, bitLenInt hi, bool exactOnly);
    void CompressWrap(bool exactOnly);
    void Compress(bool exactOnly);
    bool OverBudget() const;
    void MaybeCompress();

    // Bond-sized linear algebra (Gram matrices + Hermitian Jacobi eigensolver), no external dependencies.
    void EighDesc(std::vector<cmplx_d>& g, size_t n, std::vector<double>& lam, std::vector<cmplx_d>& v);
    void LeftFactor(const std::vector<complex>& a, size_t rows, size_t cols, std::vector<complex>& q,
        std::vector<cmplx_d>& r, size_t& m);
    void GramSvd(
        const std::vector<complex>& m, size_t rows, size_t cols, std::vector<cmplx_d>& u, std::vector<double>& sv);

public:
    /**
     * @param eng           engine stack for every patch (e.g. { QINTERFACE_OPTIMAL } or { QINTERFACE_CPU })
     * @param qBitCount     logical qubits
     * @param mb            max bond qubits per cut (bond dimension 2^mb)
     * @param patches       patch count; patches are contiguous index blocks, equal up to remainder (the
     *                      last qBitCount % patches patches get one extra qubit)
     * @param wrap          close the chain into a ring (3+ patches only)
     * @param born          Born-rule group truncation instead of greedy
     * @param lazy          extra bond qubits a cut may buffer before a truncating sweep (0 = eager)
     * @param maxPatchQb    optional cap on any patch's total width that also forces a sweep (0 = none)
     * @param rgp           random generator (patch engines); seed seeds the sampler
     */
    QBridge(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitLenInt mb = 3U, bitLenInt patches = 2U,
        bool wrap = false, bool born = false, bitLenInt lazy = 0U, bitLenInt maxPatchQb = 0U,
        qrack_rand_gen_ptr rgp = nullptr, uint64_t seed = 0U);

    QBridgePtr Clone();

    bitLenInt GetQubitCount() const { return qubitCount; }
    bitLenInt GetPatchCount() const { return patchCount; }
    std::vector<bitLenInt> GetBonds() const { return bond; }
    double GetFidelityEstimate() const { return fidelityEstimate; }
    size_t GetTruncationCount() const { return truncations; }
    bitLenInt GetPeakBonds() const { return peakBonds; }

    // --- gates ---
    void Mtrx(const complex* mtrx, bitLenInt q);
    void U(bitLenInt q, real1_f theta, real1_f phi, real1_f lambda);
    void H(bitLenInt q);
    void S(bitLenInt q);
    void IS(bitLenInt q);
    void X(bitLenInt q);
    void Y(bitLenInt q);
    void Z(bitLenInt q);
    /** Single control only across patches; multi-control is supported when all qubits share a patch. */
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void CNOT(bitLenInt c, bitLenInt t);
    void CY(bitLenInt c, bitLenInt t);
    void CZ(bitLenInt c, bitLenInt t);
    void AntiCNOT(bitLenInt c, bitLenInt t);
    void AntiCY(bitLenInt c, bitLenInt t);
    void AntiCZ(bitLenInt c, bitLenInt t);
    /** Any two-qubit gate across patches: ship-and-return, 2 bonds per cut crossed, one sweep. */
    void Swap(bitLenInt a, bitLenInt b);
    void ISwap(bitLenInt a, bitLenInt b);
    void IISwap(bitLenInt a, bitLenInt b);
    void FSim(real1_f theta, real1_f phi, bitLenInt a, bitLenInt b);

    // --- truncation control ---
    /** Inside a deferred block, bonds are only recompressed exactly; one truncating sweep at the end. */
    void BeginDeferred() { ++deferring; }
    void EndDeferred();
    /** Truncate any buffered bonds back to maxBonds now (readout does this automatically). */
    void Flush() { Compress(false); }

    // --- readout ---
    /** Exact samples of the stored state (sequential conditional sampling, grouped by prefix). */
    std::vector<bitCapInt> MeasureShots(const std::vector<bitLenInt>& qubits, unsigned shots);
    /** Dense contracted state, little-endian; for testing (2^n memory). */
    void GetQuantumState(complex* outputState);
};

/** RAII guard: { QBridgeDeferred d(bridge); ...gates... } truncates once on scope exit. */
class QBridgeDeferred {
    QBridge& b;

public:
    explicit QBridgeDeferred(QBridge& br)
        : b(br)
    {
        b.BeginDeferred();
    }
    ~QBridgeDeferred() { b.EndDeferred(); }
};

} // namespace Qrack
