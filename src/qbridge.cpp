//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// QBridge: bridged patching (see qbridge.hpp). By (Anthropic) Claude and Dan Strano.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbridge.hpp"

#include <algorithm>
#include <array>
#include <future>
#include <numeric>

namespace Qrack {

// Relative singular-value floor treated as exactly zero (patch kets are single precision by default).
static const double QBRIDGE_TOL = 1e-6;

static bitLenInt bondBits(size_t dim)
{
    bitLenInt b = 0U;
    while (((size_t)1U << b) < dim) {
        ++b;
    }
    return b;
}

// --- vectorizable kernels on interleaved (re, im) arrays ----------------------------------------

// y[0..n) += c * b[0..n)
static inline void cAxpy(complex* y, const complex* b, size_t n, const std::complex<double>& c)
{
    const real1_f cr = (real1_f)c.real(), ci = (real1_f)c.imag();
    real1* Y = reinterpret_cast<real1*>(y);
    const real1* B = reinterpret_cast<const real1*>(b);
    for (size_t k = 0U; k < (n << 1U); k += 2U) {
        const real1_f br = (real1_f)B[k], bi = (real1_f)B[k + 1U];
        Y[k] = (real1)((real1_f)Y[k] + cr * br - ci * bi);
        Y[k + 1U] = (real1)((real1_f)Y[k + 1U] + cr * bi + ci * br);
    }
}

// sum_x a[x] conj(b[x]), accumulated in double
static inline std::complex<double> cDotc(const complex* a, const complex* b, size_t n)
{
    const real1* A = reinterpret_cast<const real1*>(a);
    const real1* B = reinterpret_cast<const real1*>(b);
    double sr = 0.0, si = 0.0;
    for (size_t k = 0U; k < (n << 1U); k += 2U) {
        const double ar = (double)A[k], ai = (double)A[k + 1U], br = (double)B[k], bi = (double)B[k + 1U];
        sr += ar * br + ai * bi;
        si += ai * br - ar * bi;
    }
    return std::complex<double>(sr, si);
}

QBridge::QBridge(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitLenInt mb, bitLenInt patches, bool wrap,
    bool born, bitLenInt lazy, bitLenInt maxPatchQb, qrack_rand_gen_ptr rgp, uint64_t seed)
    : qubitCount(qBitCount)
    , patchCount(patches)
    , isRing(wrap && (patches >= 3U))
    , maxBonds(mb)
    , lazyBonds(lazy)
    , maxPatchQubits(maxPatchQb)
    , isBorn(born)
    , engines(eng)
    , rand_generator(rgp)
    , sampler(seed)
    , center(0U)
    , wrapTouched(false)
    , deferring(0)
    , fidelityEstimate(1.0)
    , truncations(0U)
    , flushes(0U)
    , peakBonds(0U)
    , peakPatchQubits(0U)
{
    if (!patches || (patches > qBitCount)) {
        throw std::invalid_argument("QBridge needs 1 <= patches <= qubit count!");
    }
    const bitLenInt base = qBitCount / patches;
    const bitLenInt rem = qBitCount % patches;
    bitLenInt start = 0U;
    for (bitLenInt p = 0U; p < patches; ++p) {
        const bitLenInt sz = base + ((p >= (patches - rem)) ? 1U : 0U);
        patchStart.push_back(start);
        patchSize.push_back(sz);
        start += sz;
        std::vector<Tag> lay;
        for (bitLenInt i = 0U; i < sz; ++i) {
            lay.push_back(Tag{ 'd', i });
        }
        layout.push_back(lay);
        sims.push_back(MakeEngine(sz));
        peakPatchQubits = std::max(peakPatchQubits, sz);
    }
    bond.assign(isRing ? patches : (patches - 1U), 0U);
}

QInterfacePtr QBridge::MakeEngine(bitLenInt n)
{
    return CreateQuantumInterface(engines, n, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG, false, false);
}

void QBridge::ParRange(size_t n, size_t grain, std::function<void(size_t, size_t, unsigned)> fn)
{
    // Contiguous ranges, one per thread (Qrack's par_for is per-element and only splits above its stride).
    unsigned threads = GetConcurrencyLevel();
    const size_t chunks = (n + grain - 1U) / grain;
    if (threads > chunks) {
        threads = (unsigned)chunks;
    }
#if ENABLE_PTHREAD
    if (threads > 1U) {
        const size_t per = (n + threads - 1U) / threads;
        std::vector<std::future<void>> futures;
        for (unsigned t = 0U; t < threads; ++t) {
            const size_t b = t * per, e = std::min(n, b + per);
            if (b >= e) {
                break;
            }
            futures.emplace_back(std::async(std::launch::async, [b, e, t, &fn]() { fn(b, e, t); }));
        }
        for (auto& f : futures) {
            f.get();
        }
        return;
    }
#endif
    fn(0U, n, 0U);
}

QBridgePtr QBridge::Clone()
{
    QBridgePtr c = std::make_shared<QBridge>(*this);
    for (size_t p = 0U; p < sims.size(); ++p) {
        c->sims[p] = sims[p] ? sims[p]->Clone() : nullptr;
    }
    c->deferring = 0;
    return c;
}

void QBridge::Locate(bitLenInt q, bitLenInt& p, bitLenInt& i) const
{
    if (q >= qubitCount) {
        throw std::invalid_argument("QBridge qubit index out of range!");
    }
    p = 0U;
    while (((p + 1U) < patchCount) && (q >= patchStart[p + 1U])) {
        ++p;
    }
    i = q - patchStart[p];
}

std::vector<bitLenInt> QBridge::Path(bitLenInt pa, bitLenInt pb) const
{
    int step = (pb > pa) ? 1 : -1;
    if (isRing) {
        const int right = ((int)pb - (int)pa + (int)patchCount) % (int)patchCount;
        step = (right <= ((int)patchCount - right)) ? 1 : -1;
    }
    std::vector<bitLenInt> path{ pa };
    while (path.back() != pb) {
        path.push_back((bitLenInt)(((int)path.back() + step + (int)patchCount) % (int)patchCount));
    }
    return path;
}

QInterfacePtr QBridge::Eng(bitLenInt p)
{
    // An in-patch operation needs the patch's engine: write any host tensors back first (exactly, if a
    // deferred block or lazy buffer is still open; this is the only case an engine can exceed maxBonds).
    if (hostT.count(p)) {
        Compress((deferring > 0) || (lazyBonds > 0U), true);
    }
    return sims[p];
}

QBridge::Tensor QBridge::TakeTensor(bitLenInt p)
{
    auto it = hostT.find(p);
    if (it != hostT.end()) {
        Tensor t = std::move(it->second);
        hostT.erase(it);
        return t;
    }
    Tensor t = ReadTensor(p);
    sims[p] = nullptr; // release the engine while its tensor is on the host
    return t;
}

// Exact two-qubit unitary of a gate as Qrack applies it: G[(ya + 2 yb) * 4 + (xa + 2 xb)].
static void QBridgeGateMatrix(std::function<void(QInterfacePtr, bitLenInt, bitLenInt)> fn, std::complex<double>* G)
{
    for (int xin = 0; xin < 4; ++xin) {
        bitCapInt perm = ZERO_BCI;
        if (xin & 1) {
            bi_or_ip(&perm, pow2(0U));
        }
        if (xin & 2) {
            bi_or_ip(&perm, pow2(1U));
        }
        QInterfacePtr e = std::make_shared<QEngineCPU>(2U, perm, nullptr, CMPLX_DEFAULT_ARG, false, false);
        fn(e, 0U, 1U);
        complex out[4U];
        e->GetQuantumState(out);
        for (int y = 0; y < 4; ++y) {
            G[y * 4 + xin] = std::complex<double>(out[y]);
        }
    }
}

void QBridge::HostTwoQubit(bitLenInt a, bitLenInt b, const cmplx_d* G)
{
    // Bridged two-qubit gate without ancillae. Operator-Schmidt decomposition G = sum_k A_k (x) B_k
    // (K <= 4 terms; 2 for controlled gates), applied to the patch tensors on the host: patch a gets
    // its bond toward b extended by k with A_k on qubit a, patch b gets k with B_k on qubit b, and every
    // patch in between carries k unchanged. The usual sweep then recompresses before anything is
    // written back, so patch engines never hold more than data + bond qubits.
    bitLenInt pa, ia, pb, ib;
    Locate(a, pa, ia);
    Locate(b, pb, ib);
    std::vector<cmplx_d> M(16U), MM(16U, cmplx_d(0.0, 0.0)), U;
    for (int ya = 0; ya < 2; ++ya) {
        for (int xa = 0; xa < 2; ++xa) {
            for (int yb = 0; yb < 2; ++yb) {
                for (int xb = 0; xb < 2; ++xb) {
                    M[(ya * 2 + xa) * 4 + (yb * 2 + xb)] = G[(ya + 2 * yb) * 4 + (xa + 2 * xb)];
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int x = 0; x < 4; ++x) {
                MM[i * 4 + j] += M[i * 4 + x] * std::conj(M[j * 4 + x]);
            }
        }
    }
    std::vector<double> lam;
    EighDesc(MM, 4U, lam, U);
    size_t K = 0U;
    while ((K < 4U) && (lam[K] > (1e-12 * lam[0U]))) {
        ++K;
    }
    const size_t Kp = (K <= 1U) ? 1U : ((K <= 2U) ? 2U : 4U);
    std::vector<std::array<complex, 4U>> Ak(Kp), Bk(Kp);
    for (size_t k = 0U; k < Kp; ++k) {
        Ak[k].fill(ZERO_CMPLX);
        Bk[k].fill(ZERO_CMPLX);
        if (k >= K) {
            continue;
        }
        const double s = std::sqrt(lam[k]);
        const double rs = std::sqrt(s);
        for (int i = 0; i < 4; ++i) {
            const cmplx_d av = rs * U[i * 4 + k];
            Ak[k][i] = complex((real1)av.real(), (real1)av.imag());
            cmplx_d vh(0.0, 0.0);
            for (int j = 0; j < 4; ++j) {
                vh += std::conj(U[j * 4 + k]) * M[j * 4 + i];
            }
            vh *= rs / s;
            Bk[k][i] = complex((real1)vh.real(), (real1)vh.imag());
        }
    }

    const std::vector<bitLenInt> path = Path(pa, pb);
    std::vector<Tensor> T;
    for (const bitLenInt& p : path) {
        T.push_back(TakeTensor(p)); // read before any bond count changes
    }
    std::vector<bool> kL(path.size(), false), kR(path.size(), false);
    const bitLenInt kb = bondBits(Kp);
    for (size_t i = 0U; (i + 1U) < path.size(); ++i) {
        const bitLenInt u = path[i], v = path[i + 1U];
        const bool rightward = isRing ? (v == ((u + 1U) % patchCount)) : (v == (u + 1U));
        const bitLenInt c = rightward ? u : v;
        if (rightward) {
            kR[i] = true;
            kL[i + 1U] = true;
        } else {
            kL[i] = true;
            kR[i + 1U] = true;
        }
        bond[c] += kb;
        peakBonds = std::max(peakBonds, bond[c]);
        if (isRing && (c == (patchCount - 1U))) {
            wrapTouched = true;
        }
    }

    for (size_t i = 0U; i < path.size(); ++i) {
        const Tensor& t = T[i];
        const size_t L = t.l, D = t.d, R = t.r;
        const size_t nl = kL[i] ? (L * Kp) : L, nr = kR[i] ? (R * Kp) : R;
        Tensor n(nl, D, nr);
        const bool isA = (i == 0U), isB = ((i + 1U) == path.size());
        if (!isA && !isB) {
            // Relay: k passes through unchanged.
            ParRange(D, 1024U, [&](size_t b0, size_t e0, unsigned th) {
                for (size_t k = 0U; k < Kp; ++k) {
                    for (size_t l = 0U; l < L; ++l) {
                        for (size_t d = b0; d < e0; ++d) {
                            const complex* src = &t.a[(l * D + d) * R];
                            complex* dst = &n.a[((l * Kp + k) * D + d) * nr];
                            for (size_t r = 0U; r < R; ++r) {
                                dst[r * Kp + k] = src[r];
                            }
                        }
                    }
                }
            });
        } else {
            const std::vector<std::array<complex, 4U>>& op = isA ? Ak : Bk;
            const size_t bit = (size_t)1U << (isA ? ia : ib);
            const bool right = kR[i]; // an endpoint carries k on exactly one side
            ParRange(D, 1024U, [&](size_t b0, size_t e0, unsigned th) {
                for (size_t k = 0U; k < Kp; ++k) {
                    const complex o00 = op[k][0U], o01 = op[k][1U], o10 = op[k][2U], o11 = op[k][3U];
                    for (size_t l = 0U; l < L; ++l) {
                        for (size_t d = b0; d < e0; ++d) {
                            if (d & bit) {
                                continue;
                            }
                            const size_t d1 = d | bit;
                            const complex* s0 = &t.a[(l * D + d) * R];
                            const complex* s1 = &t.a[(l * D + d1) * R];
                            const size_t nlIdx = right ? l : (l * Kp + k);
                            complex* t0 = &n.a[(nlIdx * D + d) * nr];
                            complex* t1 = &n.a[(nlIdx * D + d1) * nr];
                            for (size_t r = 0U; r < R; ++r) {
                                const size_t nrIdx = right ? (r * Kp + k) : r;
                                t0[nrIdx] = o00 * s0[r] + o01 * s1[r];
                                t1[nrIdx] = o10 * s0[r] + o11 * s1[r];
                            }
                        }
                    }
                }
            });
        }
        hostT[path[i]] = std::move(n);
    }
    touched.insert(path.begin(), path.end());
    MaybeCompress();
}

void QBridge::Controlled(bitLenInt c, bitLenInt t, const complex* mtrx, bool anti)
{
    bitLenInt pc, ic, pt, it;
    Locate(c, pc, ic);
    Locate(t, pt, it);
    if (pc == pt) {
        if (anti) {
            Eng(pc)->MACMtrx({ ic }, mtrx, it);
        } else {
            Eng(pc)->MCMtrx({ ic }, mtrx, it);
        }
        return;
    }
    // G = |on><on| (x) U + |off><off| (x) I, qubit a = control, b = target.
    const int on = anti ? 0 : 1;
    cmplx_d G[16U];
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            const int ya = y & 1, yb = y >> 1, xa = x & 1, xb = x >> 1;
            cmplx_d v(0.0, 0.0);
            if (ya == xa) {
                if (xa == on) {
                    v = cmplx_d(mtrx[yb * 2 + xb]);
                } else if (yb == xb) {
                    v = cmplx_d(1.0, 0.0);
                }
            }
            G[y * 4 + x] = v;
        }
    }
    HostTwoQubit(c, t, G);
}

void QBridge::PairGate(bitLenInt a, bitLenInt b, std::function<void(QInterfacePtr, bitLenInt, bitLenInt)> fn)
{
    bitLenInt pa, ia, pb, ib;
    Locate(a, pa, ia);
    Locate(b, pb, ib);
    if (pa == pb) {
        fn(Eng(pa), ia, ib);
        return;
    }
    cmplx_d G[16U];
    QBridgeGateMatrix(fn, G);
    HostTwoQubit(a, b, G);
}

// --- gates -------------------------------------------------------------------

void QBridge::Mtrx(const complex* mtrx, bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->Mtrx(mtrx, i);
}
void QBridge::U(bitLenInt q, real1_f theta, real1_f phi, real1_f lambda)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->U(i, theta, phi, lambda);
}
void QBridge::H(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->H(i);
}
void QBridge::S(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->S(i);
}
void QBridge::IS(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->IS(i);
}
void QBridge::X(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->X(i);
}
void QBridge::Y(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->Y(i);
}
void QBridge::Z(bitLenInt q)
{
    bitLenInt p, i;
    Locate(q, p, i);
    Eng(p)->Z(i);
}

void QBridge::MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{
    if (controls.size() == 1U) {
        Controlled(controls[0U], target, mtrx, false);
        return;
    }
    bitLenInt pt, it;
    Locate(target, pt, it);
    std::vector<bitLenInt> lc;
    for (const bitLenInt& c : controls) {
        bitLenInt pc, ic;
        Locate(c, pc, ic);
        if (pc != pt) {
            throw std::domain_error("QBridge::MCMtrx: multi-control gates must lie within one patch!");
        }
        lc.push_back(ic);
    }
    Eng(pt)->MCMtrx(lc, mtrx, it);
}

void QBridge::MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{
    if (controls.size() == 1U) {
        Controlled(controls[0U], target, mtrx, true);
        return;
    }
    bitLenInt pt, it;
    Locate(target, pt, it);
    std::vector<bitLenInt> lc;
    for (const bitLenInt& c : controls) {
        bitLenInt pc, ic;
        Locate(c, pc, ic);
        if (pc != pt) {
            throw std::domain_error("QBridge::MACMtrx: multi-control gates must lie within one patch!");
        }
        lc.push_back(ic);
    }
    Eng(pt)->MACMtrx(lc, mtrx, it);
}

static const complex QB_X[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
static const complex QB_Y[4U]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
static const complex QB_Z[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };

void QBridge::CNOT(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_X, false); }
void QBridge::CY(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_Y, false); }
void QBridge::CZ(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_Z, false); }
void QBridge::AntiCNOT(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_X, true); }
void QBridge::AntiCY(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_Y, true); }
void QBridge::AntiCZ(bitLenInt c, bitLenInt t) { Controlled(c, t, QB_Z, true); }

void QBridge::Swap(bitLenInt a, bitLenInt b)
{
    PairGate(a, b, [](QInterfacePtr s, bitLenInt x, bitLenInt y) { s->Swap(x, y); });
}
void QBridge::ISwap(bitLenInt a, bitLenInt b)
{
    PairGate(a, b, [](QInterfacePtr s, bitLenInt x, bitLenInt y) { s->ISwap(x, y); });
}
void QBridge::IISwap(bitLenInt a, bitLenInt b)
{
    PairGate(a, b, [](QInterfacePtr s, bitLenInt x, bitLenInt y) { s->IISwap(x, y); });
}
void QBridge::FSim(real1_f theta, real1_f phi, bitLenInt a, bitLenInt b)
{
    PairGate(a, b, [theta, phi](QInterfacePtr s, bitLenInt x, bitLenInt y) { s->FSim(theta, phi, x, y); });
}

// --- bond-sized linear algebra -------------------------------------------------

void QBridge::EighDesc(std::vector<cmplx_d>& A, size_t n, std::vector<double>& lam, std::vector<cmplx_d>& V)
{
    // Hermitian eigensolver, as LAPACK does it, without LAPACK:
    //   1. Householder reduction A = Q T Q^dagger (T Hermitian tridiagonal), Q accumulated;
    //   2. a diagonal phase D makes T's off-diagonal real and nonnegative;
    //   3. implicit QL with Wilkinson shifts (tqli) on the real tridiagonal, accumulating Z;
    //   4. eigenvectors V = Q D Z, sorted by descending eigenvalue.
    V.assign(n * n, cmplx_d(0.0, 0.0));
    for (size_t i = 0U; i < n; ++i) {
        V[i * n + i] = cmplx_d(1.0, 0.0);
    }
    std::vector<cmplx_d> v(n), p(n), w(n);
    for (size_t k = 0U; (k + 2U) < n; ++k) {
        const size_t m = n - k - 1U; // length of the column below the diagonal
        double xn = 0.0;
        for (size_t i = 0U; i < m; ++i) {
            xn += std::norm(A[(k + 1U + i) * n + k]);
        }
        xn = std::sqrt(xn);
        if (xn <= 0.0) {
            continue;
        }
        const cmplx_d x0 = A[(k + 1U) * n + k];
        const double ax0 = std::abs(x0);
        const cmplx_d ph = (ax0 > 0.0) ? (x0 / ax0) : cmplx_d(1.0, 0.0);
        const cmplx_d alpha = -ph * xn;
        for (size_t i = 0U; i < m; ++i) {
            v[i] = A[(k + 1U + i) * n + k];
        }
        v[0U] -= alpha;
        double vn = 0.0;
        for (size_t i = 0U; i < m; ++i) {
            vn += std::norm(v[i]);
        }
        vn = std::sqrt(vn);
        if (vn <= 0.0) {
            continue;
        }
        for (size_t i = 0U; i < m; ++i) {
            v[i] /= vn;
        }
        // Trailing block B = A[k+1:, k+1:]:  B <- H B H = B - 2 v w^dag - 2 w v^dag,  w = Bv - (v^dag B v) v
        for (size_t i = 0U; i < m; ++i) {
            cmplx_d acc(0.0, 0.0);
            const cmplx_d* row = &A[(k + 1U + i) * n + (k + 1U)];
            for (size_t j = 0U; j < m; ++j) {
                acc += row[j] * v[j];
            }
            p[i] = acc;
        }
        cmplx_d sv(0.0, 0.0);
        for (size_t i = 0U; i < m; ++i) {
            sv += std::conj(v[i]) * p[i];
        }
        const double s = sv.real();
        for (size_t i = 0U; i < m; ++i) {
            w[i] = p[i] - s * v[i];
        }
        for (size_t i = 0U; i < m; ++i) {
            cmplx_d* row = &A[(k + 1U + i) * n + (k + 1U)];
            const cmplx_d vi = 2.0 * v[i], wi = 2.0 * w[i];
            for (size_t j = 0U; j < m; ++j) {
                row[j] -= vi * std::conj(w[j]) + wi * std::conj(v[j]);
            }
        }
        // Column k below the diagonal becomes (alpha, 0, ...); row k mirrors it.
        A[(k + 1U) * n + k] = alpha;
        A[k * n + (k + 1U)] = std::conj(alpha);
        for (size_t i = 1U; i < m; ++i) {
            A[(k + 1U + i) * n + k] = A[k * n + (k + 1U + i)] = cmplx_d(0.0, 0.0);
        }
        // Q <- Q H on columns k+1..n-1.
        for (size_t r = 0U; r < n; ++r) {
            cmplx_d* row = &V[r * n + (k + 1U)];
            cmplx_d acc(0.0, 0.0);
            for (size_t j = 0U; j < m; ++j) {
                acc += row[j] * v[j];
            }
            acc *= 2.0;
            for (size_t j = 0U; j < m; ++j) {
                row[j] -= acc * std::conj(v[j]);
            }
        }
    }
    // Real tridiagonal via phases: T[i+1][i] = |e_i| after D = diag(d_i), d_{i+1} = d_i e_i / |e_i|.
    std::vector<double> d(n), e(n, 0.0);
    std::vector<cmplx_d> ph(n, cmplx_d(1.0, 0.0));
    for (size_t i = 0U; i < n; ++i) {
        d[i] = A[i * n + i].real();
    }
    for (size_t i = 0U; (i + 1U) < n; ++i) {
        const cmplx_d ei = A[(i + 1U) * n + i];
        const double a = std::abs(ei);
        e[i] = a;
        ph[i + 1U] = (a > 0.0) ? (ph[i] * ei / a) : ph[i];
    }
    // tqli (Numerical Recipes), e[i] couples i and i+1, e[n-1] = 0; Z accumulates rotations.
    std::vector<double> Z(n * n, 0.0);
    for (size_t i = 0U; i < n; ++i) {
        Z[i * n + i] = 1.0;
    }
    const int N = (int)n;
    for (int l = 0; l < N; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < (N - 1); ++m) {
                const double dd = std::abs(d[m]) + std::abs(d[m + 1]);
                if ((std::abs(e[m]) + dd) == dd) {
                    break;
                }
            }
            if (m != l) {
                if (++iter > 200) {
                    break;
                }
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + ((g >= 0.0) ? r : -r));
                double s = 1.0, c = 1.0, p2 = 0.0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    double f = s * e[i];
                    const double b = c * e[i];
                    e[i + 1] = (r = std::hypot(f, g));
                    if (r == 0.0) {
                        d[i + 1] -= p2;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p2;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p2 = s * r);
                    g = c * r - b;
                    for (int k = 0; k < N; ++k) {
                        f = Z[k * n + i + 1];
                        Z[k * n + i + 1] = s * Z[k * n + i] + c * f;
                        Z[k * n + i] = c * Z[k * n + i] - s * f;
                    }
                }
                if ((r == 0.0) && (i >= l)) {
                    continue;
                }
                d[l] -= p2;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return d[x] > d[y]; });
    lam.resize(n);
    std::vector<cmplx_d> W(n * n, cmplx_d(0.0, 0.0));
    // V_final = Q D Z: (Q D)[r][j] = Q[r][j] ph[j]
    for (size_t r = 0U; r < n; ++r) {
        for (size_t j = 0U; j < n; ++j) {
            const cmplx_d qd = V[r * n + j] * ph[j];
            if (qd == cmplx_d(0.0, 0.0)) {
                continue;
            }
            const double* zrow = &Z[j * n];
            cmplx_d* wrow = &W[r * n];
            for (size_t c = 0U; c < n; ++c) {
                wrow[c] += qd * zrow[order[c]];
            }
        }
    }
    for (size_t c = 0U; c < n; ++c) {
        lam[c] = d[order[c]];
    }
    V.swap(W);
}

void QBridge::LeftFactor(const std::vector<complex>& a, size_t rows, size_t cols, std::vector<complex>& q,
    std::vector<cmplx_d>& r, size_t& m)
{
    // a = q r, q with orthonormal columns, through the cols x cols Gram matrix (cols is bond-sized);
    // zero modes are dropped exactly.
    const unsigned nc = GetConcurrencyLevel();
    std::vector<std::vector<double>> part(nc, std::vector<double>(2U * cols * cols, 0.0));
    ParRange(rows, 64U, [&](size_t b, size_t e, unsigned t) {
        std::vector<double>& g = part[t];
        std::vector<double> row(2U * cols);
        for (size_t x = b; x < e; ++x) {
            const real1* A = reinterpret_cast<const real1*>(&a[x * cols]);
            for (size_t k = 0U; k < (cols << 1U); ++k) {
                row[k] = (double)A[k];
            }
            for (size_t i = 0U; i < cols; ++i) {
                const double ar = row[2U * i], ai = -row[2U * i + 1U]; // conj(a[x][i])
                if ((ar == 0.0) && (ai == 0.0)) {
                    continue;
                }
                double* gi = &g[2U * i * cols];
                for (size_t j = i; j < cols; ++j) {
                    const double br = row[2U * j], bi = row[2U * j + 1U];
                    gi[2U * j] += ar * br - ai * bi;
                    gi[2U * j + 1U] += ar * bi + ai * br;
                }
            }
        }
    });
    std::vector<cmplx_d> G(cols * cols, cmplx_d(0.0, 0.0));
    for (const auto& g : part) {
        for (size_t i = 0U; i < cols; ++i) {
            for (size_t j = i; j < cols; ++j) {
                G[i * cols + j] += cmplx_d(g[2U * (i * cols + j)], g[2U * (i * cols + j) + 1U]);
            }
        }
    }
    for (size_t i = 0U; i < cols; ++i) {
        for (size_t j = 0U; j < i; ++j) {
            G[i * cols + j] = std::conj(G[j * cols + i]);
        }
    }
    std::vector<double> lam;
    std::vector<cmplx_d> V;
    EighDesc(G, cols, lam, V);
    m = 1U;
    while ((m < cols) && (lam[m] > (QBRIDGE_TOL * QBRIDGE_TOL * lam[0U]))) {
        ++m;
    }
    std::vector<double> s(m);
    for (size_t k = 0U; k < m; ++k) {
        s[k] = (lam[k] > 0.0) ? std::sqrt(lam[k]) : 1.0;
    }
    // Vs[j][k] = V[j][k] / s[k], as interleaved doubles
    std::vector<double> Vs(2U * cols * m);
    for (size_t j = 0U; j < cols; ++j) {
        for (size_t k = 0U; k < m; ++k) {
            const cmplx_d z = V[j * cols + k] / s[k];
            Vs[2U * (j * m + k)] = z.real();
            Vs[2U * (j * m + k) + 1U] = z.imag();
        }
    }
    q.assign(rows * m, ZERO_CMPLX);
    ParRange(rows, 64U, [&](size_t b, size_t e, unsigned t) {
        std::vector<double> acc(2U * m);
        for (size_t x = b; x < e; ++x) {
            std::fill(acc.begin(), acc.end(), 0.0);
            const real1* A = reinterpret_cast<const real1*>(&a[x * cols]);
            for (size_t j = 0U; j < cols; ++j) {
                const double ar = (double)A[2U * j], ai = (double)A[2U * j + 1U];
                if ((ar == 0.0) && (ai == 0.0)) {
                    continue;
                }
                const double* vj = &Vs[2U * j * m];
                for (size_t k = 0U; k < m; ++k) {
                    acc[2U * k] += ar * vj[2U * k] - ai * vj[2U * k + 1U];
                    acc[2U * k + 1U] += ar * vj[2U * k + 1U] + ai * vj[2U * k];
                }
            }
            for (size_t k = 0U; k < m; ++k) {
                q[x * m + k] = complex((real1)acc[2U * k], (real1)acc[2U * k + 1U]);
            }
        }
    });
    r.assign(m * cols, cmplx_d(0.0, 0.0));
    for (size_t k = 0U; k < m; ++k) {
        for (size_t j = 0U; j < cols; ++j) {
            r[k * cols + j] = s[k] * std::conj(V[j * cols + k]);
        }
    }
}

void QBridge::GramSvd(
    const std::vector<complex>& M, size_t rows, size_t cols, std::vector<cmplx_d>& u, std::vector<double>& sv)
{
    // Left singular vectors and values of a short-wide M (rows bond-sized) through its rows x rows
    // Gram matrix: contiguous row dot products, double accumulation.
    const unsigned nc = GetConcurrencyLevel();
    std::vector<std::vector<cmplx_d>> part(nc, std::vector<cmplx_d>(rows * rows, cmplx_d(0.0, 0.0)));
    ParRange(cols, 4096U, [&](size_t b, size_t e, unsigned t) {
        std::vector<cmplx_d>& g = part[t];
        for (size_t i = 0U; i < rows; ++i) {
            for (size_t j = i; j < rows; ++j) {
                g[i * rows + j] += cDotc(&M[i * cols + b], &M[j * cols + b], e - b);
            }
        }
    });
    std::vector<cmplx_d> G(rows * rows, cmplx_d(0.0, 0.0));
    for (const auto& g : part) {
        for (size_t i = 0U; i < rows; ++i) {
            for (size_t j = i; j < rows; ++j) {
                G[i * rows + j] += g[i * rows + j];
            }
        }
    }
    for (size_t i = 0U; i < rows; ++i) {
        for (size_t j = 0U; j < i; ++j) {
            G[i * rows + j] = std::conj(G[j * rows + i]);
        }
    }
    std::vector<double> lam;
    EighDesc(G, rows, lam, u);
    sv.resize(rows);
    for (size_t k = 0U; k < rows; ++k) {
        sv[k] = (lam[k] > 0.0) ? std::sqrt(lam[k]) : 0.0;
    }
}

// --- patch tensors through the kets ------------------------------------------

QBridge::Tensor QBridge::ReadTensor(bitLenInt p, QInterfacePtr eng)
{
    const std::vector<Tag>& lay = layout[p];
    const size_t k = lay.size();
    const bitLenInt bl = LeftBonds(p), br = RightBonds(p);
    const bitLenInt nd = patchSize[p];
    std::unique_ptr<complex[]> v(new complex[(size_t)1U << k]);
    (eng ? eng : sims[p])->GetQuantumState(v.get());
    auto pos = [&](char kind, bitLenInt idx) -> size_t {
        for (size_t x = 0U; x < k; ++x) {
            if ((lay[x].kind == kind) && (lay[x].idx == idx)) {
                return x;
            }
        }
        throw std::runtime_error("QBridge internal error: missing qubit tag!");
    };
    auto table = [&](char kind, bitLenInt bits) {
        std::vector<size_t> t((size_t)1U << bits, 0U);
        std::vector<size_t> ps(bits);
        for (bitLenInt b = 0U; b < bits; ++b) {
            ps[b] = pos(kind, b);
        }
        for (size_t i = 0U; i < t.size(); ++i) {
            size_t o = 0U;
            for (bitLenInt b = 0U; b < bits; ++b) {
                if ((i >> b) & 1U) {
                    o |= (size_t)1U << ps[b];
                }
            }
            t[i] = o;
        }
        return t;
    };
    const std::vector<size_t> Lt = table('L', bl), Dt = table('d', nd), Rt = table('R', br);
    Tensor T(Lt.size(), Dt.size(), Rt.size());
    ParRange(T.d, 4096U, [&](size_t b, size_t e, unsigned t) {
        for (size_t i = 0U; i < T.l; ++i) {
            for (size_t x = b; x < e; ++x) {
                complex* out = &T.a[(i * T.d + x) * T.r];
                const size_t o = Lt[i] | Dt[x];
                for (size_t j = 0U; j < T.r; ++j) {
                    out[j] = v[o | Rt[j]];
                }
            }
        }
    });
    return T;
}

void QBridge::StoreTensor(bitLenInt p, const Tensor& t)
{
    const bitLenInt bl = LeftBonds(p), br = RightBonds(p);
    const bitLenInt nd = patchSize[p];
    const size_t BL = (size_t)1U << bl, D = (size_t)1U << nd, BR = (size_t)1U << br;
    if ((t.l > BL) || (t.r > BR) || (t.d != D)) {
        throw std::runtime_error("QBridge internal error: tensor does not fit its bonds!");
    }
    const bitLenInt k = nd + bl + br;
    std::vector<complex> v((size_t)1U << k, ZERO_CMPLX);
    std::vector<double> nrms(GetConcurrencyLevel(), 0.0);
    ParRange(D, 4096U, [&](size_t b, size_t e, unsigned th) {
        double nrm = 0.0;
        for (size_t j = 0U; j < t.r; ++j) {
            for (size_t i = 0U; i < t.l; ++i) {
                complex* out = &v[D * (i + BL * j)];
                for (size_t x = b; x < e; ++x) {
                    const complex a = t.a[(i * D + x) * t.r + j];
                    out[x] = a;
                    nrm += (double)norm(a);
                }
            }
        }
        nrms[th] += nrm;
    });
    double nrm = 0.0;
    for (const double& x : nrms) {
        nrm += x;
    }
    const real1 scale = (real1)(1.0 / std::sqrt(nrm));
    ParRange(v.size(), 4096U, [&](size_t b, size_t e, unsigned th) {
        for (size_t x = b; x < e; ++x) {
            v[x] *= scale;
        }
    });
    std::vector<Tag> lay;
    for (bitLenInt i = 0U; i < nd; ++i) {
        lay.push_back(Tag{ 'd', i });
    }
    for (bitLenInt i = 0U; i < bl; ++i) {
        lay.push_back(Tag{ 'L', i });
    }
    for (bitLenInt i = 0U; i < br; ++i) {
        lay.push_back(Tag{ 'R', i });
    }
    layout[p] = lay;
    peakPatchQubits = std::max(peakPatchQubits, k);
    sims[p] = MakeEngine(k);
    sims[p]->SetQuantumState(v.data());
}

// --- recompression -------------------------------------------------------------

std::vector<size_t> QBridge::Select(const std::vector<double>& sv, bool exactOnly)
{
    std::vector<size_t> idx;
    if (sv.empty() || (sv[0U] <= 0.0)) {
        idx.push_back(0U);
        return idx;
    }
    size_t live = 0U;
    while ((live < sv.size()) && (sv[live] > (QBRIDGE_TOL * sv[0U]))) {
        ++live;
    }
    const bitLenInt bExact = bondBits(live);
    const bitLenInt bNew = exactOnly ? bExact : std::min(bExact, maxBonds);
    const size_t keep = std::min(live, (size_t)1U << bNew);
    if (keep >= live) {
        for (size_t k = 0U; k < keep; ++k) {
            idx.push_back(k);
        }
        return idx;
    }
    ++truncations;
    double total = 0.0;
    for (size_t k = 0U; k < live; ++k) {
        total += sv[k] * sv[k];
    }
    if (!isBorn) {
        double kept = 0.0;
        for (size_t k = 0U; k < keep; ++k) {
            kept += sv[k] * sv[k];
            idx.push_back(k);
        }
        fidelityEstimate *= kept / total;
        return idx;
    }
    std::vector<double> pg;
    for (size_t g = 0U; g < live; g += keep) {
        double w = 0.0;
        for (size_t k = g; k < std::min(g + keep, live); ++k) {
            w += sv[k] * sv[k];
        }
        pg.push_back(w / total);
    }
    double p2 = 0.0;
    for (const double& w : pg) {
        p2 += w * w;
    }
    fidelityEstimate *= p2;
    std::discrete_distribution<size_t> pick(pg.begin(), pg.end());
    const size_t g = pick(sampler) * keep;
    for (size_t k = g; k < std::min(g + keep, live); ++k) {
        idx.push_back(k);
    }
    return idx;
}

void QBridge::Sweep(bitLenInt lo, bitLenInt hi, bool exactOnly, bool toEngine)
{
    const bitLenInt start = std::min(lo, center), end = std::max(hi, center);
    if (start == end) {
        center = lo;
        if (toEngine && hostT.count(lo)) {
            StoreTensor(lo, TakeTensor(lo));
        }
        return;
    }
    std::vector<Tensor> T;
    for (bitLenInt k = start; k <= end; ++k) {
        T.push_back(TakeTensor(k));
    }
    // Left-isometric up to the centre at `end`.
    for (bitLenInt k = start; k < end; ++k) {
        Tensor& A = T[k - start];
        std::vector<complex> q;
        std::vector<cmplx_d> rr;
        size_t m;
        LeftFactor(A.a, A.l * A.d, A.r, q, rr, m);
        const size_t oldR = A.r;
        A.r = m;
        A.a.swap(q);
        Tensor& B = T[k + 1U - start];
        const size_t cols = B.d * B.r;
        std::vector<complex> nb(m * cols, ZERO_CMPLX);
        ParRange(cols, 4096U, [&](size_t b, size_t e, unsigned th) {
            for (size_t i = 0U; i < m; ++i) {
                for (size_t j = 0U; j < oldR; ++j) {
                    cAxpy(&nb[i * cols + b], &B.a[j * cols + b], e - b, rr[i * oldR + j]);
                }
            }
        });
        B.l = m;
        B.a.swap(nb);
        bond[k] = bondBits(m);
    }
    // Right-isometric back down to `lo`, truncating each cut.
    for (bitLenInt k = end; k > lo; --k) {
        Tensor& M = T[k - start];
        const size_t cols = M.d * M.r;
        std::vector<cmplx_d> u;
        std::vector<double> sv;
        GramSvd(M.a, M.l, cols, u, sv);
        const std::vector<size_t> idx = Select(sv, exactOnly);
        const size_t kk = idx.size();
        std::vector<complex> nm(kk * cols, ZERO_CMPLX);
        ParRange(cols, 4096U, [&](size_t b, size_t e, unsigned th) {
            for (size_t c = 0U; c < kk; ++c) {
                for (size_t i = 0U; i < M.l; ++i) {
                    cAxpy(&nm[c * cols + b], &M.a[i * cols + b], e - b, std::conj(u[i * M.l + idx[c]]) / sv[idx[c]]);
                }
            }
        });
        std::vector<cmplx_d> W(M.l * kk);
        for (size_t i = 0U; i < M.l; ++i) {
            for (size_t c = 0U; c < kk; ++c) {
                W[i * kk + c] = u[i * M.l + idx[c]] * sv[idx[c]];
            }
        }
        const size_t inner = M.l;
        M.l = kk;
        M.a.swap(nm);
        Tensor& P = T[k - 1U - start];
        const size_t prow = P.l * P.d;
        std::vector<complex> np(prow * kk, ZERO_CMPLX);
        ParRange(prow, 64U, [&](size_t b, size_t e, unsigned th) {
            for (size_t x = b; x < e; ++x) {
                complex* out = &np[x * kk];
                for (size_t j = 0U; j < inner; ++j) {
                    const complex pj = P.a[x * inner + j];
                    if (pj == ZERO_CMPLX) {
                        continue;
                    }
                    const cmplx_d pjd(pj);
                    // out[c] += pj * W[j][c]
                    real1* O = reinterpret_cast<real1*>(out);
                    const cmplx_d* wj = &W[j * kk];
                    for (size_t c = 0U; c < kk; ++c) {
                        O[2U * c] = (real1)((real1_f)O[2U * c] +
                            (real1_f)(pjd.real() * wj[c].real() - pjd.imag() * wj[c].imag()));
                        O[2U * c + 1U] = (real1)((real1_f)O[2U * c + 1U] +
                            (real1_f)(pjd.real() * wj[c].imag() + pjd.imag() * wj[c].real()));
                    }
                }
            }
        });
        P.r = kk;
        P.a.swap(np);
        bond[k - 1U] = bondBits(kk);
    }
    for (bitLenInt k = start; k <= end; ++k) {
        if (toEngine) {
            StoreTensor(k, T[k - start]);
        } else {
            hostT[k] = std::move(T[k - start]);
        }
    }
    center = lo;
}

void QBridge::CompressWrap(bool exactOnly)
{
    // Ring only: recompress the wrap cut (P-1 | 0) locally, from both sides' orthonormal factors.
    const bitLenInt last = patchCount - 1U;
    Tensor A = TakeTensor(last), B = TakeTensor(0U);
    const size_t w = A.r, bcols = B.d * B.r;
    std::vector<complex> qa, qb;
    std::vector<cmplx_d> ra, rb;
    size_t m1, m2;
    LeftFactor(A.a, A.l * A.d, w, qa, ra, m1);
    std::vector<complex> bt(bcols * w);
    for (size_t i = 0U; i < w; ++i) {
        for (size_t x = 0U; x < bcols; ++x) {
            bt[x * w + i] = B.a[i * bcols + x];
        }
    }
    LeftFactor(bt, bcols, w, qb, rb, m2);
    std::vector<cmplx_d> core(m1 * m2, cmplx_d(0.0, 0.0));
    for (size_t i = 0U; i < m1; ++i) {
        for (size_t j = 0U; j < m2; ++j) {
            for (size_t c = 0U; c < w; ++c) {
                core[i * m2 + j] += ra[i * w + c] * rb[j * w + c];
            }
        }
    }
    std::vector<complex> corec(m1 * m2);
    for (size_t i = 0U; i < core.size(); ++i) {
        corec[i] = complex((real1)core[i].real(), (real1)core[i].imag());
    }
    std::vector<cmplx_d> u;
    std::vector<double> sv;
    GramSvd(corec, m1, m2, u, sv);
    const std::vector<size_t> idx = Select(sv, exactOnly);
    const size_t kk = idx.size();
    std::vector<cmplx_d> vh(kk * m2, cmplx_d(0.0, 0.0));
    for (size_t c = 0U; c < kk; ++c) {
        for (size_t j = 0U; j < m2; ++j) {
            for (size_t i = 0U; i < m1; ++i) {
                vh[c * m2 + j] += std::conj(u[i * m1 + idx[c]]) * core[i * m2 + j];
            }
            vh[c * m2 + j] /= sv[idx[c]];
        }
    }
    Tensor nA(A.l, A.d, kk);
    const size_t arows = A.l * A.d;
    par_for(0U, arows, [&](const size_t& x, const unsigned& cpu) {
        for (size_t c = 0U; c < kk; ++c) {
            cmplx_d acc(0.0, 0.0);
            for (size_t i = 0U; i < m1; ++i) {
                acc += cmplx_d(qa[x * m1 + i]) * u[i * m1 + idx[c]];
            }
            acc *= sv[idx[c]];
            nA.a[x * kk + c] = complex((real1)acc.real(), (real1)acc.imag());
        }
    });
    Tensor nB(kk, B.d, B.r);
    par_for(0U, bcols, [&](const size_t& x, const unsigned& cpu) {
        for (size_t c = 0U; c < kk; ++c) {
            cmplx_d acc(0.0, 0.0);
            for (size_t j = 0U; j < m2; ++j) {
                acc += vh[c * m2 + j] * cmplx_d(qb[x * m2 + j]);
            }
            nB.a[c * bcols + x] = complex((real1)acc.real(), (real1)acc.imag());
        }
    });
    bond[last] = bondBits(kk);
    hostT[last] = std::move(nA); // the sweep that follows writes these back (or keeps them on the host)
    hostT[0U] = std::move(nB);
}

void QBridge::Compress(bool exactOnly, bool toEngine)
{
    if (touched.empty()) {
        return;
    }
    if (wrapTouched) {
        CompressWrap(exactOnly);
        touched.insert(0U);
        touched.insert(patchCount - 1U);
    }
    Sweep(*touched.begin(), *touched.rbegin(), exactOnly, toEngine);
    if (!exactOnly) {
        ++flushes;
        touched.clear();
        wrapTouched = false;
    }
}

bool QBridge::OverBudget() const
{
    for (const bitLenInt& b : bond) {
        if (b > (maxBonds + lazyBonds)) {
            return true;
        }
    }
    if (maxPatchQubits) {
        for (bitLenInt p = 0U; p < patchCount; ++p) {
            if ((patchSize[p] + LeftBonds(p) + RightBonds(p)) > maxPatchQubits) {
                return true;
            }
        }
    }
    return false;
}

void QBridge::MaybeCompress()
{
    if (deferring) {
        // Inside a deferred block: drop exact zeros on the host, never truncate, never widen an engine.
        if (!lazyBonds || OverBudget()) {
            Compress(true, false);
        }
        return;
    }
    // Eager: truncate and write back after every bridged gate. Lazy: let bonds grow on the host until
    // a cut exceeds maxBonds + lazyBonds (or a patch would exceed maxPatchQubits).
    if (!lazyBonds || OverBudget()) {
        Compress(false, true);
    }
}

void QBridge::EndDeferred()
{
    if (deferring > 0) {
        --deferring;
    }
    if (!deferring) {
        MaybeCompress();
    }
}

// --- readout -----------------------------------------------------------------

std::vector<bitCapInt> QBridge::MeasureShots(const std::vector<bitLenInt>& qubits, unsigned shots)
{
    Compress(false);
    Sweep(0U, 0U, true); // centre at patch 0: patches 1.. right-isometric
    std::vector<Tensor> T;
    for (bitLenInt p = 0U; p < patchCount; ++p) {
        T.push_back(ReadTensor(p));
    }
    const size_t cw = isRing ? T[0U].l : 1U;
    // Ring: right environments R[k][l, w, l', w'] with the wrap index open.
    std::vector<std::vector<cmplx_d>> R(patchCount + 1U);
    if (isRing) {
        R[patchCount].assign(cw * cw * cw * cw, cmplx_d(0.0, 0.0));
        for (size_t r = 0U; r < cw; ++r) {
            for (size_t s = 0U; s < cw; ++s) {
                R[patchCount][((r * cw + r) * cw + s) * cw + s] = cmplx_d(1.0, 0.0);
            }
        }
        for (bitLenInt k = patchCount - 1U; k > 0U; --k) {
            const Tensor& A = T[k];
            const std::vector<cmplx_d>& Rn = R[k + 1U];
            const size_t L = A.l, D = A.d, Rr = A.r;
            const unsigned nc = GetConcurrencyLevel();
            std::vector<std::vector<cmplx_d>> part(nc, std::vector<cmplx_d>(L * cw * L * cw, cmplx_d(0.0, 0.0)));
            par_for(0U, D, [&](const size_t& d, const unsigned& cpu) {
                std::vector<cmplx_d> tmp(L * cw * Rr * cw, cmplx_d(0.0, 0.0));
                for (size_t l = 0U; l < L; ++l) {
                    for (size_t r = 0U; r < Rr; ++r) {
                        const cmplx_d a(A.a[(l * D + d) * Rr + r]);
                        if (a == cmplx_d(0.0, 0.0)) {
                            continue;
                        }
                        for (size_t w = 0U; w < cw; ++w) {
                            for (size_t s = 0U; s < Rr; ++s) {
                                for (size_t v = 0U; v < cw; ++v) {
                                    tmp[((l * cw + w) * Rr + s) * cw + v] += a * Rn[((r * cw + w) * Rr + s) * cw + v];
                                }
                            }
                        }
                    }
                }
                std::vector<cmplx_d>& out = part[cpu];
                for (size_t lp = 0U; lp < L; ++lp) {
                    for (size_t s = 0U; s < Rr; ++s) {
                        const cmplx_d b = std::conj(cmplx_d(A.a[(lp * D + d) * Rr + s]));
                        if (b == cmplx_d(0.0, 0.0)) {
                            continue;
                        }
                        for (size_t l = 0U; l < L; ++l) {
                            for (size_t w = 0U; w < cw; ++w) {
                                for (size_t v = 0U; v < cw; ++v) {
                                    out[((l * cw + w) * L + lp) * cw + v] += tmp[((l * cw + w) * Rr + s) * cw + v] * b;
                                }
                            }
                        }
                    }
                }
            });
            R[k].assign(L * cw * L * cw, cmplx_d(0.0, 0.0));
            for (const auto& pt : part) {
                for (size_t i = 0U; i < pt.size(); ++i) {
                    R[k][i] += pt[i];
                }
            }
        }
    }

    struct Group {
        std::vector<cmplx_d> X; // w x l
        size_t rows, cols;
        unsigned count;
        std::vector<bool> bits;
    };
    std::vector<Group> groups(1U);
    groups[0U].rows = cw;
    groups[0U].cols = cw;
    groups[0U].X.assign(cw * cw, cmplx_d(0.0, 0.0));
    for (size_t w = 0U; w < cw; ++w) {
        groups[0U].X[w * cw + w] = cmplx_d(1.0, 0.0);
    }
    groups[0U].count = shots;
    groups[0U].bits.assign(qubitCount, false);

    for (bitLenInt k = 0U; k < patchCount; ++k) {
        const Tensor& A = T[k];
        const size_t D = A.d, Rr = A.r;
        std::vector<Group> next;
        for (const Group& g : groups) {
            const size_t W = g.rows;
            std::vector<cmplx_d> M(W * D * Rr, cmplx_d(0.0, 0.0));
            par_for(0U, D, [&](const size_t& d, const unsigned& cpu) {
                for (size_t w = 0U; w < W; ++w) {
                    for (size_t l = 0U; l < g.cols; ++l) {
                        const cmplx_d x = g.X[w * g.cols + l];
                        if (x == cmplx_d(0.0, 0.0)) {
                            continue;
                        }
                        for (size_t r = 0U; r < Rr; ++r) {
                            M[(w * D + d) * Rr + r] += x * cmplx_d(A.a[(l * D + d) * Rr + r]);
                        }
                    }
                }
            });
            std::vector<double> prob(D, 0.0);
            par_for(0U, D, [&](const size_t& d, const unsigned& cpu) {
                double pr = 0.0;
                if (!isRing) {
                    for (size_t w = 0U; w < W; ++w) {
                        for (size_t r = 0U; r < Rr; ++r) {
                            pr += std::norm(M[(w * D + d) * Rr + r]);
                        }
                    }
                } else {
                    const std::vector<cmplx_d>& Rn = R[k + 1U];
                    cmplx_d acc(0.0, 0.0);
                    for (size_t w = 0U; w < W; ++w) {
                        for (size_t r = 0U; r < Rr; ++r) {
                            const cmplx_d m = M[(w * D + d) * Rr + r];
                            if (m == cmplx_d(0.0, 0.0)) {
                                continue;
                            }
                            for (size_t s = 0U; s < Rr; ++s) {
                                for (size_t v = 0U; v < W; ++v) {
                                    acc +=
                                        m * Rn[((r * cw + w) * Rr + s) * cw + v] * std::conj(M[(v * D + d) * Rr + s]);
                                }
                            }
                        }
                    }
                    pr = std::max(acc.real(), 0.0);
                }
                prob[d] = pr;
            });
            std::discrete_distribution<size_t> pick(prob.begin(), prob.end());
            std::map<size_t, unsigned> counts;
            for (unsigned s = 0U; s < g.count; ++s) {
                ++counts[pick(sampler)];
            }
            for (const auto& dc : counts) {
                const size_t d = dc.first;
                Group ng;
                ng.rows = W;
                ng.cols = Rr;
                ng.count = dc.second;
                ng.X.resize(W * Rr);
                double nrm = 0.0;
                for (size_t w = 0U; w < W; ++w) {
                    for (size_t r = 0U; r < Rr; ++r) {
                        ng.X[w * Rr + r] = M[(w * D + d) * Rr + r];
                        nrm += std::norm(ng.X[w * Rr + r]);
                    }
                }
                nrm = std::sqrt(nrm);
                for (cmplx_d& x : ng.X) {
                    x /= nrm;
                }
                ng.bits = g.bits;
                for (bitLenInt i = 0U; i < patchSize[k]; ++i) {
                    ng.bits[patchStart[k] + i] = (d >> i) & 1U;
                }
                next.push_back(ng);
            }
        }
        groups.swap(next);
    }

    std::vector<bitCapInt> out;
    out.reserve(shots);
    for (const Group& g : groups) {
        bitCapInt v = ZERO_BCI;
        for (size_t j = 0U; j < qubits.size(); ++j) {
            if (g.bits[qubits[j]]) {
                bi_or_ip(&v, pow2((bitLenInt)j));
            }
        }
        for (unsigned c = 0U; c < g.count; ++c) {
            out.push_back(v);
        }
    }
    std::shuffle(out.begin(), out.end(), sampler);
    return out;
}

void QBridge::GetQuantumState(complex* outputState)
{
    Compress(false);
    std::vector<Tensor> T;
    for (bitLenInt p = 0U; p < patchCount; ++p) {
        T.push_back(ReadTensor(p));
    }
    const size_t cw = T[0U].l;
    // C[w, x, r]: wrap index open on the left, accumulated data x, right bond r.
    size_t Dacc = T[0U].d, Rr = T[0U].r;
    std::vector<cmplx_d> C(cw * Dacc * Rr);
    for (size_t i = 0U; i < C.size(); ++i) {
        C[i] = cmplx_d(T[0U].a[i]);
    }
    for (bitLenInt k = 1U; k < patchCount; ++k) {
        const Tensor& A = T[k];
        std::vector<cmplx_d> N(cw * Dacc * A.d * A.r, cmplx_d(0.0, 0.0));
        for (size_t w = 0U; w < cw; ++w) {
            for (size_t x = 0U; x < Dacc; ++x) {
                for (size_t j = 0U; j < Rr; ++j) {
                    const cmplx_d c = C[(w * Dacc + x) * Rr + j];
                    if (c == cmplx_d(0.0, 0.0)) {
                        continue;
                    }
                    for (size_t xp = 0U; xp < A.d; ++xp) {
                        for (size_t q = 0U; q < A.r; ++q) {
                            N[(w * Dacc * A.d + (xp * Dacc + x)) * A.r + q] +=
                                c * cmplx_d(A.a[(j * A.d + xp) * A.r + q]);
                        }
                    }
                }
            }
        }
        Dacc *= A.d;
        Rr = A.r;
        C.swap(N);
    }
    double nrm = 0.0;
    std::vector<cmplx_d> psi(Dacc, cmplx_d(0.0, 0.0));
    for (size_t x = 0U; x < Dacc; ++x) {
        for (size_t w = 0U; w < cw; ++w) {
            psi[x] += C[(w * Dacc + x) * Rr + (isRing ? w : 0U)];
        }
        nrm += std::norm(psi[x]);
    }
    nrm = std::sqrt(nrm);
    for (size_t x = 0U; x < Dacc; ++x) {
        outputState[x] = complex((real1)(psi[x].real() / nrm), (real1)(psi[x].imag() / nrm));
    }
}

void QBridge::TransferStep(std::vector<cmplx_d>& E, size_t W, const Tensor& A, const Tensor& B)
{
    // E'[w][ra][rb] = sum_{d, la, lb} E[w][la][lb] conj(A[la, d, ra]) B[lb, d, rb], for W open-index pairs w.
    const size_t LA = A.l, LB = B.l, RA = A.r, RB = B.r, D = A.d;
    const unsigned nc = GetConcurrencyLevel();
    std::vector<std::vector<cmplx_d>> part(nc, std::vector<cmplx_d>(W * RA * RB, cmplx_d(0.0, 0.0)));
    ParRange(D, 64U, [&](size_t b, size_t e, unsigned th) {
        std::vector<cmplx_d>& out = part[th];
        std::vector<cmplx_d> tmp(LA * RB);
        for (size_t d = b; d < e; ++d) {
            for (size_t w = 0U; w < W; ++w) {
                const cmplx_d* Ew = &E[w * LA * LB];
                // tmp[la][rb] = sum_lb E[w][la][lb] B[lb, d, rb]
                std::fill(tmp.begin(), tmp.end(), cmplx_d(0.0, 0.0));
                for (size_t lb = 0U; lb < LB; ++lb) {
                    const complex* brow = &B.a[(lb * D + d) * RB];
                    for (size_t la = 0U; la < LA; ++la) {
                        const cmplx_d ev = Ew[la * LB + lb];
                        if (ev == cmplx_d(0.0, 0.0)) {
                            continue;
                        }
                        cmplx_d* t = &tmp[la * RB];
                        for (size_t rb = 0U; rb < RB; ++rb) {
                            t[rb] += ev * cmplx_d(brow[rb]);
                        }
                    }
                }
                // out[w][ra][rb] += sum_la conj(A[la, d, ra]) tmp[la][rb]
                cmplx_d* ow = &out[w * RA * RB];
                for (size_t la = 0U; la < LA; ++la) {
                    const complex* arow = &A.a[(la * D + d) * RA];
                    const cmplx_d* t = &tmp[la * RB];
                    for (size_t ra = 0U; ra < RA; ++ra) {
                        const cmplx_d av = std::conj(cmplx_d(arow[ra]));
                        if (av == cmplx_d(0.0, 0.0)) {
                            continue;
                        }
                        cmplx_d* o = &ow[ra * RB];
                        for (size_t rb = 0U; rb < RB; ++rb) {
                            o[rb] += av * t[rb];
                        }
                    }
                }
            }
        }
    });
    E.assign(W * RA * RB, cmplx_d(0.0, 0.0));
    for (const auto& pt : part) {
        for (size_t i = 0U; i < E.size(); ++i) {
            E[i] += pt[i];
        }
    }
}

real1_f QBridge::ExpectationPauliAll(const std::vector<bitLenInt>& bits, const std::vector<Pauli>& paulis)
{
    if (bits.size() != paulis.size()) {
        throw std::invalid_argument("QBridge::ExpectationPauliAll: qubit and Pauli lists differ in length!");
    }
    Compress(false);
    std::vector<std::vector<std::pair<bitLenInt, Pauli>>> per(patchCount);
    for (size_t k = 0U; k < bits.size(); ++k) {
        if (paulis[k] == PauliI) {
            continue;
        }
        bitLenInt p, i;
        Locate(bits[k], p, i);
        per[p].push_back(std::make_pair(i, paulis[k]));
    }
    // Patch tensors |psi> (A) and P|psi> (B, where P acts), with P applied natively to an engine clone.
    std::vector<Tensor> A, B;
    std::vector<bool> hasB(patchCount, false);
    for (bitLenInt p = 0U; p < patchCount; ++p) {
        A.push_back(ReadTensor(p));
        if (per[p].empty()) {
            B.push_back(Tensor());
            continue;
        }
        QInterfacePtr c = sims[p]->Clone();
        for (const auto& ip : per[p]) {
            bitLenInt pos = 0U;
            while (!((layout[p][pos].kind == 'd') && (layout[p][pos].idx == ip.first))) {
                ++pos;
            }
            if (ip.second == PauliX) {
                c->X(pos);
            } else if (ip.second == PauliY) {
                c->Y(pos);
            } else {
                c->Z(pos);
            }
        }
        B.push_back(ReadTensor(p, c));
        hasB[p] = true;
    }
    // Transfer matrices from the left. On a ring, the bra-side wrap index wa is fixed one value at a
    // time and the ket-side wrap index wb is carried (W = cw open pairs), so memory stays O(cw chi^2).
    const size_t cw = A[0U].l;
    cmplx_d e(0.0, 0.0), n(0.0, 0.0);
    for (size_t wa = 0U; wa < cw; ++wa) {
        std::vector<cmplx_d> E(cw * cw * cw, cmplx_d(0.0, 0.0)), N;
        for (size_t wb = 0U; wb < cw; ++wb) {
            E[(wb * cw + wa) * cw + wb] = cmplx_d(1.0, 0.0);
        }
        N = E;
        for (bitLenInt p = 0U; p < patchCount; ++p) {
            TransferStep(E, cw, A[p], hasB[p] ? B[p] : A[p]);
            TransferStep(N, cw, A[p], A[p]);
        }
        // Close the ring: the last right indices must equal the wrap indices (wa on the bra, wb on the ket).
        const size_t R = A[patchCount - 1U].r;
        for (size_t wb = 0U; wb < cw; ++wb) {
            e += E[(wb * R + wa) * R + wb];
            n += N[(wb * R + wa) * R + wb];
        }
    }
    return (real1_f)(e.real() / n.real());
}

} // namespace Qrack
