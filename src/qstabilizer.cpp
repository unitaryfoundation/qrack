//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Adapted from:
//
// CHP: CNOT-Hadamard-Phase
// Stabilizer Quantum Computer Simulator
// by Scott Aaronson
// Last modified June 30, 2004
//
// Thanks to Simon Anders and Andrew Cross for bugfixes
//
// https://www.scottaaronson.com/chp/
//
// Daniel Strano and the Qrack contributers appreciate Scott Aaronson's open sharing of the CHP code, and we hope that
// vm6502q/qrack is one satisfactory framework by which CHP could be adapted to enter the C++ STL. Our project
// philosophy aims to raise the floor of decentralized quantum computing technology access across all modern platforms,
// for all people, not commercialization.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qstabilizer.hpp"

#include <algorithm>
#include <chrono>

#if SEED_DEVRAND
#include <sys/random.h>
#endif

#define IS_0_R1(r) (abs(r) <= REAL1_EPSILON)
#define IS_1_R1(r) (abs(r) <= REAL1_EPSILON)

namespace Qrack {

QStabilizer::QStabilizer(bitLenInt n, const bitCapInt& perm, qrack_rand_gen_ptr rgp, const complex& phaseFac,
    bool doNorm, bool randomGlobalPhase, bool ignored2, int64_t ignored3, bool useHardwareRNG, bool ignored4,
    real1_f ignored5, std::vector<int64_t> ignored6, bitLenInt ignored7, real1_f ignored8)
    : QInterface(n, rgp, doNorm, useHardwareRNG, randomGlobalPhase, REAL1_EPSILON)
    , rawRandBools(0U)
    , rawRandBoolsRemaining(0U)
    , phaseOffset(ZERO_R1)
    , isGaussianCached(false)
    , gaussianCached(0U)
#if BOOST_AVAILABLE
    , isTransposed(false)
#endif
    , r(2U, BoolVector((n << 1U) + 1U))
    , x((n << 1U) + 1U, BoolVector(n))
    , z((n << 1U) + 1U, BoolVector(n))
    , bBuffer(n, ZERO_R1_F)
    , pBuffer(n, ZERO_R1_F)
    , bPhase(n, false)
    , pPhase(n, false)
{
    maxStateMapCacheQubitCount = getenv("QRACK_MAX_CPU_QB")
        ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_CPU_QB")))
        : 28U - ((QBCAPPOW < FPPOW) ? 1U : (1U + QBCAPPOW - FPPOW));

    SetPermutation(perm, phaseFac);
}

void QStabilizer::ParFor(StabilizerParallelFunc fn, std::vector<bitLenInt> qubits)
{
    for (const bitLenInt& qubit : qubits) {
        ValidateQubitIndex(qubit);
    }

    Dispatch([this, fn] {
        const bitLenInt maxLcv = qubitCount << 1U;
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            fn(i);
        }
    });
}

QInterfacePtr QStabilizer::Clone()
{
    Finish();

    QStabilizerPtr clone = std::make_shared<QStabilizer>(qubitCount, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG, false,
        randGlobalPhase, false, -1, !!hardware_rand_generator);
    clone->Finish();

    clone->x = x;
    clone->z = z;
    clone->r = r;
    clone->phaseOffset = phaseOffset;
    clone->isGaussianCached = isGaussianCached;
    clone->gaussianCached = gaussianCached;
#if BOOST_AVAILABLE
    clone->isTransposed = isTransposed;
#endif

    return clone;
}

void QStabilizer::SetPermutation(const bitCapInt& perm, const complex& phaseFac)
{
    Dump();

#if BOOST_AVAILABLE
    const bitLenInt rowCount = qubitCount;
    const bitLenInt rc = (bitLenInt)x.size();
    for (bitLenInt i = 0U; i < rc; ++i) {
        x[i].reset();
        z[i].reset();
    }
    SetTransposeState(true);
#else
    const bitLenInt rowCount = (qubitCount << 1U);
#endif

    isGaussianCached = false;

    if (phaseFac != CMPLX_DEFAULT_ARG) {
        phaseOffset = std::arg(phaseFac);
    } else if (randGlobalPhase) {
        phaseOffset = (real1)(2 * PI_R1 * Rand() - PI_R1);
    } else {
        phaseOffset = ZERO_R1;
    }

#if BOOST_AVAILABLE
    r[0U].reset();
    r[1U].reset();
#else
    std::fill(r[0U].begin(), r[0U].end(), false);
    std::fill(r[1U].begin(), r[1U].end(), false);
#endif

    std::fill(bBuffer.begin(), bBuffer.end(), ZERO_R1_F);
    std::fill(pBuffer.begin(), pBuffer.end(), ZERO_R1_F);
    std::fill(bPhase.begin(), bPhase.end(), false);
    std::fill(pPhase.begin(), pPhase.end(), false);

    for (bitLenInt i = 0; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
#if BOOST_AVAILABLE
        xi.set(i);
        zi.set(i + qubitCount);
#else
        std::fill(xi.begin(), xi.end(), false);
        std::fill(zi.begin(), zi.end(), false);

        if (i < qubitCount) {
            xi[i] = true;
        } else {
            zi[i - qubitCount] = true;
        }
#endif
    }

    if (bi_compare_0(perm) == 0) {
        return;
    }

    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        if (bi_and_1(perm >> j)) {
            X(j);
        }
    }
}

/// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
uint8_t QStabilizer::clifford(const bitLenInt& i, const bitLenInt& k)
{
    const BoolVector& xi = x[i];
    const BoolVector& zi = z[i];
    const BoolVector& xk = x[k];
    const BoolVector& zk = z[k];

    // Power to which i is raised
    bitLenInt e = 0U;

#if BOOST_AVAILABLE
    // Build the phase-bit-0 and phase-bit-1 masks directly
    BoolVector e0, e1;

    // X & ~Z case
    {
        BoolVector cond = xk & ~zk;
        BoolVector exp = cond & ~xi & zi;
        // XY = iZ  → phase +1
        e0 = cond & xi & zi;
        // XZ = -iY → phase +2
        e1 = ~e0 & exp;
        e0 ^= exp;
    }

    // X & Z case (Y)
    {
        BoolVector cond = xk & zk;
        BoolVector exp = cond & ~xi & zi;
        e1 ^= e0 & exp; // YZ = iX
        e0 ^= exp;

        exp = cond & xi & ~zi;
        e1 ^= (~e0) & exp; // YX = -iZ
        e0 ^= exp;
    }

    // ~X & Z case
    {
        BoolVector cond = ~xk & zk;
        BoolVector exp = cond & xi & ~zi;
        e1 ^= e0 & exp; // ZX = iY
        e0 ^= exp;

        exp = cond & xi & zi;
        e1 ^= (~e0) & exp; // ZY = -iX
        e0 ^= exp;
    }

    e = e0.count() + (e1.count() << 1U);
#else
    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        // X
        if (xk[j] && !zk[j]) {
            // XY=iZ
            e += xi[j] && zi[j];
            // XZ=-iY
            e -= !xi[j] && zi[j];
        }
        // Y
        if (xk[j] && zk[j]) {
            // YZ=iX
            e += !xi[j] && zi[j];
            // YX=-iZ
            e -= xi[j] && !zi[j];
        }
        // Z
        if (!xk[j] && zk[j]) {
            // ZX=iY
            e += xi[j] && !zi[j];
            // ZY=-iX
            e -= xi[j] && zi[j];
        }
    }
#endif

    e = (e + GetR(i) + GetR(k)) & 0x3U;

    return e;
}

/**
 * Do Gaussian elimination to put the stabilizer generators in the following form:
 * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
 * (Return value = number of such generators = log_2 of number of nonzero basis states)
 * At the bottom, generators containing Z's only in quasi-upper-triangular form.
 */
bitLenInt QStabilizer::gaussian(bool s)
{
#if BOOST_AVAILABLE
    SetTransposeState(false);
#endif

    if (isGaussianCached) {
        if (s) {
            seed(gaussianCached);
        }

        return gaussianCached;
    }

    // For brevity:
    const bitLenInt& n = qubitCount;
    const bitLenInt maxLcv = n << 1U;
    bitLenInt i = n;
    bitLenInt k;

    for (bitLenInt j = 0U; j < n; ++j) {

        // Find a generator containing X in jth column
        for (k = i; k < maxLcv; ++k) {
            if (!x[k][j]) {
                continue;
            }

            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; ++k2) {
                if (x[k2][j]) {
                    // Gaussian elimination step:
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            ++i;

            break;
        }
    }

    const bitLenInt g = i - n;

    for (bitLenInt j = 0U; j < n; ++j) {

        // Find a generator containing Z in jth column
        for (k = i; k < maxLcv; ++k) {
            if (!z[k][j]) {
                continue;
            }

            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; ++k2) {
                if (z[k2][j]) {
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            ++i;

            break;
        }
    }

    if (s) {
        seed(g);
    }

    isGaussianCached = true;
    gaussianCached = g;

    return g;
}

/**
 * Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
 * writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
 * performed on q.  g is the return value from gaussian(q).
 */
void QStabilizer::seed(const bitLenInt& g)
{
    const bitLenInt elemCount = qubitCount << 1U;
    int min = 0;

    // Wipe the scratch space clean
    ResetR(elemCount);

    BoolVector& xec = x[elemCount];
    BoolVector& zec = z[elemCount];
#if BOOST_AVAILABLE
    xec.reset();
    zec.reset();
#else
    std::fill(xec.begin(), xec.end(), false);
    std::fill(zec.begin(), zec.end(), false);
#endif

    const int qcg = (int)(qubitCount + g);
    for (int i = elemCount - 1; i >= qcg; i--) {
        int f = GetR(i);
        for (int j = qubitCount - 1; j >= 0; j--) {
            if (z[i][j]) {
                min = j;
                if (xec[j]) {
                    f = (f + 2) & 0x3;
                }
            }
        }

        if (f == 2) {
            const int j = min;
            // Make the seed consistent with the ith equation
            xec[j] = !xec[j];
        }
    }
}

/// Helper for setBasisState() and setBasisProb()
AmplitudeEntry QStabilizer::getBasisAmp(const real1_f& nrm)
{
    const bitLenInt elemCount = qubitCount << 1U;
    uint8_t e = GetR(elemCount);
    const BoolVector& xRow = x[elemCount];
    const BoolVector& zRow = z[elemCount];

    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        // Pauli operator is "Y"
        if (xRow[j] && zRow[j]) {
            e = (e + 1U) & 0x3U;
        }
    }

    complex amp((real1)nrm, ZERO_R1);
    if (e & 1U) {
        amp *= I_CMPLX;
    }
    if (e & 2U) {
        amp *= -ONE_CMPLX;
    }
    amp *= std::polar(ONE_R1, phaseOffset);

    bitCapInt perm = ZERO_BCI;
    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        if (xRow[j]) {
            bi_or_ip(&perm, pow2(j));
        }
    }

    return AmplitudeEntry(perm, amp);
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1_f& nrm, complex* stateVec)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    stateVec[(bitCapIntOcl)entry.permutation] = entry.amplitude;
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1_f& nrm, QInterfacePtr eng)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    eng->SetAmplitude(entry.permutation, entry.amplitude);
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1_f& nrm, std::map<bitCapInt, complex>& stateMap)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    stateMap[entry.permutation] = entry.amplitude;
}

/// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisProb(const real1_f& nrm, real1* outputProbs)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    outputProbs[(bitCapIntOcl)entry.permutation] = norm(entry.amplitude);
}

real1_f QStabilizer::getExpectation(const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
    const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    bitCapInt retIndex = ZERO_BCI;
    for (size_t b = 0U; b < bitPowers.size(); ++b) {
        bi_add_ip(
            &retIndex, (bi_compare_0(entry.permutation & bitPowers[b]) != 0) ? perms[(b << 1U) | 1U] : perms[b << 1U]);
    }
    return (real1_f)(bi_to_double(offset + retIndex) * norm(entry.amplitude));
}

real1_f QStabilizer::getExpectation(
    const real1_f& nrm, const std::vector<bitCapInt>& bitPowers, const std::vector<real1_f>& weights)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    real1_f weight = ONE_R1_F;
    for (size_t b = 0U; b < bitPowers.size(); ++b) {
        weight *= (bi_compare_0(entry.permutation & bitPowers[b]) != 0) ? weights[(b << 1U) | 1U] : weights[b << 1U];
    }
    return weight * norm(entry.amplitude);
}

real1_f QStabilizer::getVariance(const real1_f& mean, const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
    const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    bitCapInt retIndex = ZERO_BCI;
    for (size_t b = 0U; b < bitPowers.size(); ++b) {
        bi_add_ip(
            &retIndex, (bi_compare_0(entry.permutation & bitPowers[b]) != 0) ? perms[(b << 1U) | 1U] : perms[b << 1U]);
    }
    const real1_f diff = ((real1_f)bi_to_double(offset + retIndex)) - mean;
    return diff * diff * norm(entry.amplitude);
}

real1_f QStabilizer::getVariance(const real1_f& mean, const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
    const std::vector<real1_f>& weights)
{
    const AmplitudeEntry entry = getBasisAmp(nrm);
    real1_f weight = ONE_R1_F;
    for (size_t b = 0U; b < bitPowers.size(); ++b) {
        weight *= (bi_compare_0(entry.permutation & bitPowers[b]) != 0) ? weights[(b << 1U) | 1U] : weights[b << 1U];
    }
    const real1_f diff = weight - mean;
    return diff * diff * norm(entry.amplitude);
}

/// Get all probabilities corresponding to ket notation
bitCapInt QStabilizer::HighestProbAll()
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));
    const AmplitudeEntry entry = getBasisAmp(nrm);

    return entry.permutation;
}

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(complex* stateVec)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    // (1 / permCount) ^ (1/2)
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    // init stateVec as all 0 values
    par_for(0, pow2Ocl(qubitCount), [&](const bitCapIntOcl& lcv, const unsigned& cpu) { stateVec[lcv] = ZERO_CMPLX; });

    setBasisState(nrm, stateVec);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, stateVec);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(QInterfacePtr eng)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    // init stateVec as all 0 values
    eng->SetPermutation(ZERO_BCI);
    eng->SetAmplitude(ZERO_BCI, ZERO_CMPLX);

    setBasisState(nrm, eng);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, eng);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
std::map<bitCapInt, complex> QStabilizer::GetQuantumState()
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    std::map<bitCapInt, complex> stateMap;

    setBasisState(nrm, stateMap);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, stateMap);
    }

    return stateMap;
}

/// Get all probabilities corresponding to ket notation
void QStabilizer::GetProbs(real1* outputProbs)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    // init stateVec as all 0 values
    par_for(0, pow2Ocl(qubitCount), [&](const bitCapIntOcl& lcv, const unsigned& cpu) { outputProbs[lcv] = ZERO_R1; });

    setBasisProb(nrm, outputProbs);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisProb(nrm, outputProbs);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
complex QStabilizer::GetAmplitude(const bitCapInt& perm)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry entry = getBasisAmp(nrm);
    if (bi_compare(entry.permutation, perm) == 0) {
        return entry.amplitude;
    }
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry entry = getBasisAmp(nrm);
        if (bi_compare(entry.permutation, perm) == 0) {
            return entry.amplitude;
        }
    }

    return ZERO_CMPLX;
}

/// Convert the state to ket notation (warning: could be huge!)
std::vector<complex> QStabilizer::GetAmplitudes(std::vector<bitCapInt> perms)
{
    std::set<bitCapInt> prms{ perms.begin(), perms.end() };
    std::map<bitCapInt, complex> amps;

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry entry = getBasisAmp(nrm);
    if (prms.find(entry.permutation) != prms.end()) {
        amps[entry.permutation] = entry.amplitude;
    }
    if (amps.size() < perms.size()) {
        for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
            const bitCapInt t2 = t ^ (t + ONE_BCI);
            for (bitLenInt i = 0U; i < g; ++i) {
                if (bi_and_1(t2 >> i)) {
                    rowmult(elemCount, qubitCount + i);
                }
            }
            const AmplitudeEntry entry = getBasisAmp(nrm);
            if (prms.find(entry.permutation) != prms.end()) {
                amps[entry.permutation] = entry.amplitude;
                if (amps.size() >= perms.size()) {
                    break;
                }
            }
        }
    }

    std::vector<complex> toRet(perms.size());
    for (size_t i = 0U; i < perms.size(); ++i) {
        toRet[i] = amps[perms[i]];
    }

    return toRet;
}

AmplitudeEntry QStabilizer::GetAnyAmplitude()
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const real1_f nrm = sqrt(ONE_R1_F / pow2Ocl(g));

    return getBasisAmp(nrm);
}

AmplitudeEntry QStabilizer::GetQubitAmplitude(bitLenInt t, bool m)
{
    const bitCapInt tPow = pow2(t);
    const bitCapInt mPow = m ? tPow : ZERO_BCI;

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry entry = getBasisAmp(nrm);
    if (bi_compare(entry.permutation & tPow, mPow) == 0) {
        return entry;
    }
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry entry = getBasisAmp(nrm);
        if (bi_compare(entry.permutation & tPow, mPow) == 0) {
            return entry;
        }
    }

    return AmplitudeEntry(ZERO_BCI, ZERO_CMPLX);
}

real1_f QStabilizer::ExpectationBitsFactorized(
    const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    if (perms.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QStabilizer::ExpectationBitsFactorized must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QStabilizer::ExpectationBitsAllRdm parameter qubits vector values must be within allocated qubit bounds!");

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    real1 expectation = (real1)getExpectation(nrm, bitPowers, perms, offset);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        expectation += (real1)getExpectation(nrm, bitPowers, perms, offset);
    }

    return (real1_f)expectation;
}

real1_f QStabilizer::ExpectationFloatsFactorized(
    const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
{
    if (weights.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QStabilizer::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QStabilizer::ExpectationFloatsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    real1_f expectation = getExpectation(nrm, bitPowers, weights);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        expectation += getExpectation(nrm, bitPowers, weights);
    }

    return expectation;
}

real1_f QStabilizer::VarianceBitsFactorized(
    const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    if (perms.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QStabilizer::ExpectationBitsFactorized must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QStabilizer::ExpectationBitsAllRdm parameter qubits vector values must be within allocated qubit bounds!");

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const real1_f mean = ExpectationBitsFactorized(bits, perms, offset);
    real1 expectation = (real1)getVariance(mean, nrm, bitPowers, perms, offset);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        expectation += (real1)getVariance(mean, nrm, bitPowers, perms, offset);
    }

    return (real1_f)expectation;
}

real1_f QStabilizer::VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
{
    if (weights.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QStabilizer::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QStabilizer::ExpectationFloatsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const real1_f mean = ExpectationFloatsFactorized(bits, weights);
    real1_f expectation = getVariance(mean, nrm, bitPowers, weights);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        expectation += getVariance(mean, nrm, bitPowers, weights);
    }

    return expectation;
}

real1_f QStabilizer::ProbPermRdm(const bitCapInt& _perm, bitLenInt ancillaeStart)
{
    if (ancillaeStart > qubitCount) {
        throw std::invalid_argument("QStabilizer::ProbPermRDM ancillaeStart is out-of-bounds!");
    }

    if (ancillaeStart == qubitCount) {
        return ProbAll(_perm);
    }

    bitCapInt qubitMask = pow2(ancillaeStart);
    bi_decrement(&qubitMask, 1U);
    bitCapInt perm = _perm & qubitMask;

    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry firstAmp = getBasisAmp(nrm);
    real1 prob = (bi_compare(firstAmp.permutation & qubitMask, perm) == 0) ? norm(firstAmp.amplitude) : ZERO_R1;
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry amp = getBasisAmp(nrm);
        if (bi_compare(perm, amp.permutation & qubitMask) == 0) {
            prob += norm(amp.amplitude);
        }
    }

    return prob;
}

real1_f QStabilizer::ProbMask(const bitCapInt& mask, const bitCapInt& perm)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry firstAmp = getBasisAmp(nrm);
    real1 prob = (bi_compare(firstAmp.permutation & mask, perm) == 0) ? norm(firstAmp.amplitude) : ZERO_R1;
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry amp = getBasisAmp(nrm);
        if (bi_compare(perm, amp.permutation & mask) == 0) {
            prob += norm(amp.amplitude);
        }
    }

    return prob;
}

/// Apply a CNOT gate with control and target
void QStabilizer::CNOT(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase || isNearClifford(c) || isNearClifford(t)) {
        H(t);
        CZ(c, t);
        return H(t);
    }

    isGaussianCached = false;

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    xt ^= xc;
    zc ^= zt;
    r[1U] ^= zt & xc & ~(xt ^ zc);
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            xi[t] = xi[t] ^ xi[c];

            if (zi[t]) {
                zi[c] = !zi[c];

                if (xi[c] && (xi[t] == zi[c])) {
                    r[1U][i].flip();
                }
            }
        },
        { c, t });
#endif
}

/// Apply an (anti-)CNOT gate with control and target
void QStabilizer::AntiCNOT(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase || isNearClifford(c) || isNearClifford(t)) {
        H(t);
        AntiCZ(c, t);
        return H(t);
    }

    isGaussianCached = false;

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    xt ^= xc;
    zc ^= zt;
    r[1U] ^= zt & (~xc | (xt ^ zc));
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            xi[t] = xi[t] ^ xi[c];

            if (zi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] || (xi[t] != zi[c])) {
                    r[1U][i].flip();
                }
            }
        },
        { c, t });
#endif
}

/// Apply a CY gate with control and target
void QStabilizer::CY(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase || isNearClifford(c) || isNearClifford(t)) {
        IS(t);
        CNOT(c, t);
        return S(t);
    }

    isGaussianCached = false;

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;
    xt ^= xc;
    r[1U] ^= zt & (xc & ~(xt ^ zc));
    zc ^= zt;
    zt ^= xt;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            zi[t] = zi[t] ^ xi[t];
            xi[t] = xi[t] ^ xi[c];

            if (zi[t]) {
                if (xi[c] && (xi[t] == zi[c])) {
                    r[1U][i].flip();
                }

                zi[c] = !zi[c];
            }

            zi[t] = zi[t] ^ xi[t];
        },
        { c, t });
#endif
}

/// Apply an (anti-)CY gate with control and target
void QStabilizer::AntiCY(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase || isNearClifford(c) || isNearClifford(t)) {
        IS(t);
        AntiCNOT(c, t);
        return S(t);
    }

    isGaussianCached = false;

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;
    xt ^= xc;
    r[1U] ^= zt & (~xc | (xt ^ zc));
    zc ^= zt;
    zt ^= xt;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            zi[t] = zi[t] ^ xi[t];
            xi[t] = xi[t] ^ xi[c];

            if (zi[t]) {
                if (!xi[c] || (xi[t] != zi[c])) {
                    r[1U][i].flip();
                }

                zi[c] = !zi[c];
            }

            zi[t] = zi[t] ^ xi[t];
        },
        { c, t });
#endif
}

/// Apply a CZ gate with control and target
void QStabilizer::CZ(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(c)) {
        if (M(c)) {
            Z(t);
        }

        return;
    }

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(c, false);

    isGaussianCached = false;

    CZNearClifford(c, t);

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xt;
    r[1U] ^= xt & xc & ~(zt ^ zc);
    zt ^= xc;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            if (xi[t]) {
                zi[c] = !zi[c];

                if (xi[c] && (zi[t] == zi[c])) {
                    r[1U][i].flip();
                }
            }

            zi[t] = zi[t] ^ xi[c];
        },
        { c, t });
#endif

    if (!randGlobalPhase) {
        SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
    }
}

/// Apply an (anti-)CZ gate with control and target
void QStabilizer::AntiCZ(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(c)) {
        if (!M(c)) {
            Z(t);
        }

        return;
    }

    const AmplitudeEntry ampEntry = randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(c, true);

    isGaussianCached = false;

    pBuffer[c] *= -ONE_R1;
    CZNearClifford(c, t);
    pBuffer[c] *= -ONE_R1;

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xt;
    r[1U] ^= xt & (~xc | (zt ^ zc));
    zt ^= xc;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            if (xi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] || (zi[t] != zi[c])) {
                    r[1U][i].flip();
                }
            }

            zi[t] = zi[t] ^ xi[c];
        },
        { c, t });
#endif

    if (!randGlobalPhase) {
        SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
    }
}

void QStabilizer::Swap(bitLenInt c, bitLenInt t)
{
    if (c == t) {
        return;
    }

    if (!randGlobalPhase) {
        return QInterface::Swap(c, t);
    }

    isGaussianCached = false;

    SwapNearClifford(c, t);

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);
    std::swap(x[c], x[t]);
    std::swap(z[c], z[t]);
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            BoolVector::swap(xi[c], xi[t]);
            BoolVector::swap(zi[c], zi[t]);
        },
        { c, t });
#endif
}

void QStabilizer::ISwap(bitLenInt c, bitLenInt t)
{
    if (c == t) {
        return;
    }

    if (!randGlobalPhase) {
        return QInterface::ISwap(c, t);
    }

    isGaussianCached = false;

    SwapNearClifford(c, t);
    CZNearClifford(c, t);

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    std::swap(xc, xt);
    std::swap(zc, zt);

    zc ^= xt;
    r[1U] ^= xt & ~xc & zt;
    zt ^= xc;
    r[1U] ^= xc & zc & ~xt;
    zc ^= xc;
    zt ^= xt;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            BoolVector::swap(xi[c], xi[t]);
            BoolVector::swap(zi[c], zi[t]);

            if (xi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] && zi[t]) {
                    r[1U][i].flip();
                }
            }

            if (xi[c]) {
                zi[t] = !zi[t];

                if (zi[c] && !xi[t]) {
                    r[1U][i].flip();
                }
            }

            zi[c] = zi[c] ^ xi[c];
            zi[t] = zi[t] ^ xi[t];
        },
        { c, t });
#endif
}

void QStabilizer::IISwap(bitLenInt c, bitLenInt t)
{
    if (c == t) {
        return;
    }

    if (!randGlobalPhase) {
        return QInterface::IISwap(c, t);
    }

    isGaussianCached = false;

    CZNearClifford(c, t);
    SwapNearClifford(c, t);

#if BOOST_AVAILABLE
    ValidateQubitIndex(c);
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xc;
    zt ^= xt;
    zt ^= xc;
    r[1U] ^= xc & zc & ~xt;
    zc ^= xt;
    r[1U] ^= xt & ~xc & zt;
    std::swap(xc, xt);
    std::swap(zc, zt);
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            zi[c] = zi[c] ^ xi[c];
            zi[t] = zi[t] ^ xi[t];

            if (xi[c]) {
                zi[t] = !zi[t];

                if (zi[c] && !xi[t]) {
                    r[1U][i].flip();
                }
            }

            if (xi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] && zi[t]) {
                    r[1U][i].flip();
                }
            }

            BoolVector::swap(xi[c], xi[t]);
            BoolVector::swap(zi[c], zi[t]);
        },
        { c, t });
#endif
}

/// Apply a Hadamard gate to target
void QStabilizer::H(bitLenInt t)
{
    const QStabilizerPtr clone = randGlobalPhase ? nullptr : std::dynamic_pointer_cast<QStabilizer>(Clone());

    isGaussianCached = false;

    std::swap(bBuffer[t], pBuffer[t]);
    std::vector<bool>::swap(bPhase[t], pPhase[t]);

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    std::swap(xt, zt);

    r[1U] ^= xt & zt;
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            BoolVector::swap(xi[t], zi[t]);
            if (xi[t] && zi[t]) {
                r[1U][i].flip();
            }
        },
        { t });
#endif

    if (randGlobalPhase) {
        return;
    }

    const bool oIsSepZ = clone->IsSeparableZ(t);
    const bool nIsSepZ = IsSeparableZ(t);

    const bitCapInt tPow = pow2(t);
    const bitLenInt g = gaussian();
    const bitCapInt permCountMinus1 = pow2Mask(g);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

    const AmplitudeEntry entry = getBasisAmp(nrm);
    if (nIsSepZ || (bi_compare_0(entry.permutation & tPow) == 0)) {
        const complex oAmp = clone->GetAmplitude(oIsSepZ ? entry.permutation : (entry.permutation & ~tPow));
        if (norm(oAmp) > FP_NORM_EPSILON) {
            return SetPhaseOffset(phaseOffset + std::arg(oAmp) - std::arg(entry.amplitude));
        }
    }
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry entry = getBasisAmp(nrm);
        if (nIsSepZ || (bi_compare_0(entry.permutation & tPow) == 0)) {
            const complex oAmp = clone->GetAmplitude(oIsSepZ ? entry.permutation : (entry.permutation & ~tPow));
            if (norm(oAmp) > FP_NORM_EPSILON) {
                return SetPhaseOffset(phaseOffset + std::arg(oAmp) - std::arg(entry.amplitude));
            }
        }
    }
}

/// Apply an X (or NOT) gate to target
void QStabilizer::X(bitLenInt t)
{
    if (!randGlobalPhase) {
        H(t);
        Z(t);
        return H(t);
    }

    isGaussianCached = false;

    pBuffer[t] *= -ONE_R1;

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);
    SetTransposeState(true);
    r[1U] ^= z[t];
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (z[i][t]) {
                r[1U][i] = !r[1U][i];
            }
        },
        { t });
#endif
}

/// Apply a Pauli Y gate to target
void QStabilizer::Y(bitLenInt t)
{
    // Y is composed as IS, X, S, with overall -i phase
    if (!randGlobalPhase && IsSeparableZ(t)) {
        IS(t);
        X(t);
        return S(t);
    }

    isGaussianCached = false;

    pBuffer[t] *= -ONE_R1;
    bBuffer[t] *= -ONE_R1;

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);
    SetTransposeState(true);
    r[1U] ^= z[t] ^ x[t];
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (z[i][t] ^ x[i][t]) {
                r[1U][i] = !r[1U][i];
            }
        },
        { t });
#endif
}

/// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
void QStabilizer::Z(bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(t)) {
        if (M(t)) {
            SetPhaseOffset(phaseOffset + PI_R1);
        }
        return;
    }

    isGaussianCached = false;

    bBuffer[t] *= -ONE_R1;

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);
    SetTransposeState(true);
    r[1U] ^= x[t];
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (x[i][t]) {
                r[1U][i] = !r[1U][i];
            }
        },
        { t });
#endif

    if (randGlobalPhase) {
        return;
    }

    SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::S(bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(t)) {
        if (M(t)) {
            SetPhaseOffset(phaseOffset + PI_R1 / 2);
        }
        return;
    }

    SBase(t);

    bBuffer[t] *= I_CMPLX;
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::SBase(bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(t)) {
        if (M(t)) {
            SetPhaseOffset(phaseOffset + PI_R1 / 2);
        }
        return;
    }

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

    isGaussianCached = false;

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    r[1U] ^= xt & zt;
    zt ^= xt;
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            if (xi[t] && zi[t]) {
                r[1U][i] = !r[1U][i];
            }
            zi[t] = zi[t] ^ xi[t];
        },
        { t });
#endif

    if (randGlobalPhase) {
        return;
    }

    SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::IS(bitLenInt t)
{
    if (!randGlobalPhase && IsSeparableZ(t)) {
        if (M(t)) {
            SetPhaseOffset(phaseOffset - PI_R1 / 2);
        }
        return;
    }

    ISBase(t);

    bBuffer[t] *= -I_CMPLX;
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::ISBase(bitLenInt t)
{
    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

    isGaussianCached = false;

#if BOOST_AVAILABLE
    ValidateQubitIndex(t);

    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;
    r[1U] ^= xt & zt;
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            zi[t] = zi[t] ^ xi[t];
            if (xi[t] && zi[t]) {
                r[1U][i] = !r[1U][i];
            }
        },
        { t });
#endif

    if (randGlobalPhase) {
        return;
    }

    SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
}

/// Apply half a phase gate
void QStabilizer::RZ(real1_f angle, bitLenInt t)
{
    angle = FixAnglePeriod(angle);
    while (angle >= HALF_PI_R1) {
        S(t);
        angle = FixAnglePeriod(angle - HALF_PI_R1);
    }
    while (angle <= -HALF_PI_R1) {
        IS(t);
        angle = FixAnglePeriod(angle + HALF_PI_R1);
    }

    angle = FixAnglePeriod((pPhase[t] ? -std::real(pBuffer[t]) : std::real(pBuffer[t])) + angle);

    if (angle >= HALF_PI_R1) {
        S(t);
        angle = FixAnglePeriod(angle - HALF_PI_R1);
        while (angle >= HALF_PI_R1) {
            // S(t);
            angle = FixAnglePeriod(angle - HALF_PI_R1);
        }
        if (pPhase[t]) {
            angle = -angle;
            pPhase[t] = false;
            pBuffer[t].imag(-imag(pBuffer[t]));
        }
    } else if (angle <= -HALF_PI_R1) {
        IS(t);
        angle = FixAnglePeriod(angle + HALF_PI_R1);
        while (angle <= -HALF_PI_R1) {
            // IS(t);
            angle = FixAnglePeriod(angle + HALF_PI_R1);
        }
        if (pPhase[t]) {
            angle = -angle;
            pPhase[t] = false;
            pBuffer[t].imag(-imag(pBuffer[t]));
        }
    } else if ((RandFloat() * HALF_PI_R1) < std::abs(angle)) {
        if (angle > 0) {
            S(t);
            angle = FixAnglePeriod(angle - HALF_PI_R1);
        } else {
            IS(t);
            angle = FixAnglePeriod(angle + HALF_PI_R1);
        }
    }
    pBuffer[t].real(pPhase[t] ? -angle : angle);
}

/**
 * Returns all qubits entangled with "target" (including itself)
 */
std::vector<bitLenInt> QStabilizer::EntangledQubits(const bitLenInt& target, const bool& g)
{
    // for brevity
    const bitLenInt n = qubitCount;
    if (g) {
        // This is technically necessary, but inefficient.
        // Skipping it might return a larger set.
        gaussian(false);
    }
    BoolVector bits(qubitCount);
    bits[target] = true;
    BoolVector origBits;

    do {
        origBits = bits;

        std::vector<bitLenInt> toCheck;
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
#if BOOST_AVAILABLE
            if (!bits.test(i)) {
#else
            if (!bits[i]) {
#endif
                toCheck.push_back(i);
            }
        }

        for (bitLenInt b = 0U; b < qubitCount; ++b) {
#if BOOST_AVAILABLE
            if (!origBits.test(b)) {
#else
            if (!origBits[b]) {
#endif
                continue;
            }

            const bitLenInt bpn = b + n;
#if BOOST_AVAILABLE
            if (isTransposed) {
                for (const bitLenInt& i : toCheck) {
                    bits[i] |= x[i][b] || z[i][b] || x[i][bpn] || z[i][bpn] || x[b][i] || z[b][i] || x[b][i + n] ||
                        z[b][i + n];
                }
            } else {
                bits |= x[b] | z[b] | x[bpn] | z[bpn];
                for (const bitLenInt& i : toCheck) {
                    bits[i] |= x[i][b] || z[i][b] || x[i + n][b] || z[i + n][b];
                }
            }
#else
            for (const bitLenInt& i : toCheck) {
                bits[i] = bits[i] || x[b][i] || z[b][i] || x[bpn][i] || z[bpn][i] || x[i][b] || z[i][b] ||
                    x[i + n][b] || z[i + n][b];
            }
#endif
        }
    } while (origBits != bits);

    std::vector<bitLenInt> toReturn;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
#if BOOST_AVAILABLE
        if (!bits.test(i)) {
#else
        if (!bits[i]) {
#endif
            toReturn.push_back(i);
        }
    }

    return toReturn;
}

/**
 * Returns "true" if target qubit is a Z basis eigenstate
 */
bool QStabilizer::IsSeparableZ(const bitLenInt& t)
{
    if (t >= qubitCount) {
        throw std::invalid_argument("QStabilizer::IsSeparableZ qubit index is out-of-bounds!");
    }

    Finish();

    // for brevity
    const bitLenInt& n = qubitCount;
#if BOOST_AVAILABLE
    if (isTransposed) {
        return x[t].find_next(n - 1U) == BoolVector::npos;
    }
#endif
    const bitLenInt& nt2 = qubitCount << 1U;
    // loop over stabilizer generators
    for (bitLenInt p = n; p < nt2; ++p) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p][t]) {
            return false;
        }
    }

    return true;
}

/**
 * Returns "true" if target qubit is an X basis eigenstate
 */
bool QStabilizer::IsSeparableX(const bitLenInt& t)
{
    H(t);
    const bool isSeparable = IsSeparableZ(t);
    H(t);

    return isSeparable;
}

/**
 * Returns "true" if target qubit is a Y basis eigenstate
 */
bool QStabilizer::IsSeparableY(const bitLenInt& t)
{
    IS(t);
    const bool isSeparable = IsSeparableX(t);
    S(t);

    return isSeparable;
}

/**
 * Returns:
 * 0 if target qubit is not separable
 * 1 if target qubit is a Z basis eigenstate
 * 2 if target qubit is an X basis eigenstate
 * 3 if target qubit is a Y basis eigenstate
 */
uint8_t QStabilizer::IsSeparable(const bitLenInt& t)
{
    if (IsSeparableZ(t)) {
        return 1U;
    }

    if (IsSeparableX(t)) {
        return 2U;
    }

    if (IsSeparableY(t)) {
        return 3U;
    }

    return 0U;
}

/// Measure qubit t
bool QStabilizer::ForceM(bitLenInt t, bool result, bool doForce, bool doApply)
{
    if (t >= qubitCount) {
        throw std::invalid_argument("QStabilizer::ForceM qubit index is out-of-bounds!");
    }

    if (doForce && !doApply) {
        return result;
    }

    Finish();

    const bitLenInt elemCount = qubitCount << 1U;
    // for brevity
    const bitLenInt& n = qubitCount;

    // pivot row in stabilizer
    bitLenInt p;
    // loop over stabilizer generators
#if BOOST_AVAILABLE
    if (isTransposed) {
        const auto pos = x[t].find_next(n - 1U);
        p = (pos == BoolVector::npos) ? n : (bitLenInt)(pos - n);
    } else {
        for (p = n; p < elemCount; ++p) {
            // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
            if (x[p][t]) {
                // The outcome is random
                p -= n;
                break;
            }
        }
    }
#else
    for (p = n; p < elemCount; ++p) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p][t]) {
            // The outcome is random
            p -= n;
            break;
        }
    }
#endif

    // If outcome is indeterminate
    if (p < n) {
        // moment of quantum randomness
        if (!doForce) {
            result = Rand();
        }

        if (!doApply) {
            return result;
        }

#if BOOST_AVAILABLE
        SetTransposeState(false);
#endif

        const QStabilizerPtr clone = randGlobalPhase ? nullptr : std::dynamic_pointer_cast<QStabilizer>(Clone());

        isGaussianCached = false;

        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, t + n);

        // Set the new stabilizer result phase
        r[1U][p + n] = result;
        r[0U][p + n] = false;

        // Now update the Xbar's and Zbar's that don't commute with Z_b
        for (bitLenInt i = 0U; i < p; ++i) {
            if (x[i][t]) {
                rowmult(i, p);
            }
        }
        // (Skip "p" row)
        for (bitLenInt i = p + 1U; i < elemCount; ++i) {
            if (x[i][t]) {
                rowmult(i, p);
            }
        }

        if (randGlobalPhase) {
            return result;
        }

        const bitLenInt g = gaussian();
        const bitCapInt permCountMinus1 = pow2Mask(g);
        const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

        const AmplitudeEntry entry = getBasisAmp(nrm);
        const complex oAmp = clone->GetAmplitude(entry.permutation);
        if (norm(oAmp) > FP_NORM_EPSILON) {
            SetPhaseOffset(phaseOffset + std::arg(oAmp) - std::arg(entry.amplitude));
            return result;
        }
        for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
            const bitCapInt t2 = t ^ (t + ONE_BCI);
            for (bitLenInt i = 0U; i < g; ++i) {
                if (bi_and_1(t2 >> i)) {
                    rowmult(elemCount, qubitCount + i);
                }
            }
            const AmplitudeEntry entry = getBasisAmp(nrm);
            const complex oAmp = GetAmplitude(entry.permutation);
            if (norm(oAmp) > FP_NORM_EPSILON) {
                SetPhaseOffset(phaseOffset + std::arg(oAmp) - std::arg(entry.amplitude));
                return result;
            }
        }

        return result;
    }

    // If outcome is determinate

    // Before, we were checking if stabilizer generators commute with Z_b; now, we're checking destabilizer
    // generators

    // pivot row in destabilizer
#if BOOST_AVAILABLE
    if (isTransposed) {
        const auto pos = x[t].find_first();
        p = ((pos == BoolVector::npos) || (pos >= n)) ? n : (bitLenInt)pos;
    } else {
        for (p = 0U; p < n; ++p) {
            if (x[p][t]) {
                break;
            }
        }
    }
#else
    for (p = 0U; p < n; ++p) {
        if (x[p][t]) {
            break;
        }
    }
#endif

    if (p == n) {
        // For example, diagonal permutation state is |0>.
        return false;
    }

#if BOOST_AVAILABLE
    SetTransposeState(false);
#endif

    rowcopy(elemCount, p + n);
    for (bitLenInt i = p + 1U; i < n; ++i) {
        if (x[i][t]) {
            rowmult(elemCount, i + n);
        }
    }

    if (doForce && (result != (bool)GetR(elemCount))) {
        throw std::invalid_argument("QStabilizer::ForceM() forced a measurement with 0 probability!");
    }

    return GetR(elemCount);
}

bitLenInt QStabilizer::Compose(QStabilizerPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QStabilizer::Compose start index is out-of-bounds!");
    }

    // We simply insert the (elsewhere initialized and valid) "toCopy" stabilizers and destabilizers in corresponding
    // position, and we set the new padding to 0. This is immediately a valid state, if the two original QStablizer
    // instances are valid.

    toCopy->Finish();
    Finish();

    isGaussianCached = false;

    SetPhaseOffset(phaseOffset + toCopy->phaseOffset);

    const bitLenInt length = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + length;
    const bitLenInt endLength = qubitCount - start;

#if BOOST_AVAILABLE
    const bitLenInt end = start + length;

    QStabilizerPtr nQubits = std::make_shared<QStabilizer>(nQubitCount, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG,
        false, randGlobalPhase, false, -1, !!hardware_rand_generator);

    nQubits->isGaussianCached = false;

    nQubits->r[0U].reset();
    nQubits->r[1U].reset();
    for (size_t i = 0U; i < nQubits->x.size(); ++i) {
        nQubits->x[i].reset();
        nQubits->z[i].reset();
    }

    SetTransposeState(true);
    toCopy->SetTransposeState(true);
    nQubits->SetTransposeState(true);

    for (bitLenInt i = 0U; i < start; ++i) {
        const bitLenInt ia = i + nQubitCount;
        const bitLenInt ib = i + qubitCount;
        nQubits->pBuffer[i] = pBuffer[i];
        nQubits->bBuffer[i] = bBuffer[i];
        nQubits->pPhase[i] = pPhase[i];
        nQubits->bPhase[i] = bPhase[i];
        nQubits->r[0U][i] = r[0U][i];
        nQubits->r[1U][i] = r[1U][i];
        nQubits->r[0U][ia] = r[0U][ib];
        nQubits->r[1U][ia] = r[1U][ib];
        for (bitLenInt j = 0U; j < start; ++j) {
            nQubits->x[j][i] = x[j][i];
            nQubits->z[j][i] = z[j][i];

            nQubits->x[j][ia] = x[j][ib];
            nQubits->z[j][ia] = z[j][ib];
        }
        for (bitLenInt j = 0U; j < endLength; ++j) {
            const bitLenInt ja = j + end;
            const bitLenInt jb = j + start;
            nQubits->x[ja][i] = x[jb][i];
            nQubits->z[ja][i] = z[jb][i];

            nQubits->x[ja][ia] = x[jb][ib];
            nQubits->z[ja][ia] = z[jb][ib];
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt ia = i + start;
        const bitLenInt ib = ia + nQubitCount;
        const bitLenInt ic = i + length;
        nQubits->pBuffer[ia] = toCopy->pBuffer[i];
        nQubits->bBuffer[ia] = toCopy->bBuffer[i];
        nQubits->pPhase[ia] = toCopy->pPhase[i];
        nQubits->bPhase[ia] = toCopy->bPhase[i];
        nQubits->r[0U][ia] = toCopy->r[0U][i];
        nQubits->r[1U][ia] = toCopy->r[1U][i];
        nQubits->r[0U][ib] = toCopy->r[0U][ic];
        nQubits->r[1U][ib] = toCopy->r[1U][ic];
        for (bitLenInt j = 0U; j < length; ++j) {
            const bitLenInt ja = j + start;
            nQubits->x[ja][ia] = toCopy->x[j][i];
            nQubits->z[ja][ia] = toCopy->z[j][i];

            nQubits->x[ja][ib] = toCopy->x[j][ic];
            nQubits->z[ja][ib] = toCopy->z[j][ic];
        }
    }

    for (bitLenInt i = 0; i < endLength; ++i) {
        const bitLenInt ia = i + end;
        const bitLenInt ib = i + start;
        const bitLenInt ic = ia + nQubitCount;
        const bitLenInt id = ib + qubitCount;
        nQubits->pBuffer[ia] = pBuffer[ib];
        nQubits->bBuffer[ia] = bBuffer[ib];
        nQubits->pPhase[ia] = pPhase[ib];
        nQubits->bPhase[ia] = bPhase[ib];
        nQubits->r[0U][ia] = r[0U][ib];
        nQubits->r[1U][ia] = r[1U][ib];
        nQubits->r[0U][ic] = r[0U][id];
        nQubits->r[1U][ic] = r[1U][id];
        for (bitLenInt j = 0; j < start; ++j) {
            nQubits->x[j][ia] = x[j][ib];
            nQubits->z[j][ia] = z[j][ib];

            nQubits->x[j][ic] = x[j][id];
            nQubits->z[j][ic] = z[j][id];
        }
        for (bitLenInt j = 0; j < endLength; ++j) {
            const bitLenInt ja = j + end;
            const bitLenInt jb = j + start;
            nQubits->x[ja][ia] = x[jb][ib];
            nQubits->z[ja][ia] = z[jb][ib];

            nQubits->x[ja][ic] = x[jb][id];
            nQubits->z[ja][ic] = z[jb][id];
        }
    }

    Copy(nQubits);
#else
    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    const bitLenInt dLen = length << 1U;
    const bitLenInt secondStart = qubitCount + start;

    for (bitLenInt i = 0U; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
        xi.insert(xi.begin() + start, length, false);
        zi.insert(zi.begin() + start, length, false);
    }

    x.insert(x.begin() + secondStart, toCopy->x.begin() + length, toCopy->x.begin() + dLen);
    z.insert(z.begin() + secondStart, toCopy->z.begin() + length, toCopy->z.begin() + dLen);
    r[0U].insert(r[0U].begin() + secondStart, toCopy->r[0U].begin() + length, toCopy->r[0U].begin() + dLen);
    r[1U].insert(r[1U].begin() + secondStart, toCopy->r[1U].begin() + length, toCopy->r[1U].begin() + dLen);

    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt offset = secondStart + i;
        BoolVector& xo = x[offset];
        BoolVector& zo = z[offset];
        xo.insert(xo.begin(), start, false);
        xo.insert(xo.end(), endLength, false);
        zo.insert(zo.begin(), start, false);
        zo.insert(zo.end(), endLength, false);
    }

    pBuffer.insert(pBuffer.begin() + start, toCopy->pBuffer.begin(), toCopy->pBuffer.begin() + length);
    bBuffer.insert(bBuffer.begin() + start, toCopy->bBuffer.begin(), toCopy->bBuffer.begin() + length);
    pPhase.insert(pPhase.begin() + start, toCopy->pPhase.begin(), toCopy->pPhase.begin() + length);
    bPhase.insert(bPhase.begin() + start, toCopy->bPhase.begin(), toCopy->bPhase.begin() + length);
    x.insert(x.begin() + start, toCopy->x.begin(), toCopy->x.begin() + length);
    z.insert(z.begin() + start, toCopy->z.begin(), toCopy->z.begin() + length);
    r[0U].insert(r[0U].begin() + start, toCopy->r[0U].begin(), toCopy->r[0U].begin() + length);
    r[1U].insert(r[1U].begin() + start, toCopy->r[1U].begin(), toCopy->r[1U].begin() + length);

    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt offset = start + i;
        BoolVector& xo = x[offset];
        BoolVector& zo = z[offset];
        xo.insert(xo.begin(), start, false);
        xo.insert(xo.end(), endLength, false);
        zo.insert(zo.begin(), start, false);
        zo.insert(zo.end(), endLength, false);
    }

    SetQubitCount(nQubitCount);
#endif

    return start;
}

QInterfacePtr QStabilizer::Decompose(bitLenInt start, bitLenInt length)
{
    QStabilizerPtr dest = std::make_shared<QStabilizer>(length, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG, false,
        randGlobalPhase, false, -1, !!hardware_rand_generator);
    Decompose(start, dest);

    return dest;
}

bool QStabilizer::CanDecomposeDispose(const bitLenInt start, const bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QStabilizer::CanDecomposeDispose range is out-of-bounds!");
    }

    if (qubitCount == 1U) {
        return true;
    }

    Finish();

    // We want to have the maximum number of 0 cross terms possible.
    gaussian(false);

    const bitLenInt end = start + length;

    for (bitLenInt i = 0U; i < start; ++i) {
        const bitLenInt i2 = i + qubitCount;
        const BoolVector& xi = x[i];
        const BoolVector& zi = z[i];
        const BoolVector& xi2 = x[i2];
        const BoolVector& zi2 = z[i2];
        for (bitLenInt j = start; j < end; ++j) {
            if (xi[j] || zi[j] || xi2[j] || zi2[j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = end; i < qubitCount; ++i) {
        const bitLenInt i2 = i + qubitCount;
        const BoolVector& xi = x[i];
        const BoolVector& zi = z[i];
        const BoolVector& xi2 = x[i2];
        const BoolVector& zi2 = z[i2];
        for (bitLenInt j = start; j < end; ++j) {
            if (xi[j] || zi[j] || xi2[j] || zi2[j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = start; i < end; ++i) {
        const bitLenInt i2 = i + qubitCount;
        const BoolVector& xi = x[i];
        const BoolVector& zi = z[i];
        const BoolVector& xi2 = x[i2];
        const BoolVector& zi2 = z[i2];
        for (bitLenInt j = 0U; j < start; ++j) {
            if (xi[j] || zi[j] || xi2[j] || zi2[j]) {
                return false;
            }
        }
        for (bitLenInt j = end; j < qubitCount; ++j) {
            if (xi[j] || zi[j] || xi2[j] || zi2[j]) {
                return false;
            }
        }
    }

    return true;
}

void QStabilizer::DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr dest)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QStabilizer::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    if (dest) {
        dest->Dump();
    }
    Finish();

    const AmplitudeEntry ampEntry = (randGlobalPhase || dest) ? AmplitudeEntry(ZERO_BCI, ONE_CMPLX) : GetAnyAmplitude();

    // We want to have the maximum number of 0 cross terms possible.
    gaussian(false);

    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "dest" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    const bitCapInt oMaxQMask = pow2Mask(qubitCount);
    const bitLenInt end = start + length;
    const bitLenInt nQubitCount = qubitCount - length;

#if BOOST_AVAILABLE
    const bitLenInt endLength = qubitCount - end;

    QStabilizerPtr nQubits = std::make_shared<QStabilizer>(nQubitCount, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG,
        false, randGlobalPhase, false, -1, !!hardware_rand_generator);

    nQubits->isGaussianCached = false;

    nQubits->r[0U].reset();
    nQubits->r[1U].reset();
    for (size_t i = 0U; i < nQubits->x.size(); ++i) {
        nQubits->x[i].reset();
        nQubits->z[i].reset();
    }

    if (dest) {
        dest->r[0U].reset();
        dest->r[1U].reset();
        for (size_t i = 0U; i < dest->x.size(); ++i) {
            dest->x[i].reset();
            dest->z[i].reset();
        }
        dest->SetTransposeState(true);
    }

    SetTransposeState(true);
    nQubits->SetTransposeState(true);

    for (bitLenInt i = 0U; i < start; ++i) {
        const bitLenInt ia = i + nQubitCount;
        const bitLenInt ib = i + qubitCount;
        nQubits->pBuffer[i] = pBuffer[i];
        nQubits->bBuffer[i] = bBuffer[i];
        nQubits->pPhase[i] = pPhase[i];
        nQubits->bPhase[i] = bPhase[i];
        nQubits->r[0U][i] = r[0U][i];
        nQubits->r[1U][i] = r[1U][i];
        nQubits->r[0U][ia] = r[0U][ib];
        nQubits->r[1U][ia] = r[1U][ib];
        for (bitLenInt j = 0U; j < start; ++j) {
            nQubits->x[j][i] = x[j][i];
            nQubits->z[j][i] = z[j][i];

            nQubits->x[j][ia] = x[j][ib];
            nQubits->z[j][ia] = z[j][ib];
        }
        for (bitLenInt j = 0U; j < endLength; ++j) {
            const bitLenInt ja = j + start;
            const bitLenInt jb = j + end;
            nQubits->x[ja][i] = x[jb][i];
            nQubits->z[ja][i] = z[jb][i];

            nQubits->x[ja][ia] = x[jb][ib];
            nQubits->z[ja][ia] = z[jb][ib];
        }
    }

    if (dest) {
        for (bitLenInt i = 0U; i < length; ++i) {
            const bitLenInt ia = i + start;
            const bitLenInt ib = i + length;
            const bitLenInt ic = ia + qubitCount;
            dest->pBuffer[i] = pBuffer[ia];
            dest->bBuffer[i] = bBuffer[ia];
            dest->pPhase[i] = pPhase[ia];
            dest->bPhase[i] = bPhase[ia];
            dest->r[0U][i] = r[0U][ia];
            dest->r[1U][i] = r[1U][ia];
            dest->r[0U][ib] = r[0U][ic];
            dest->r[1U][ib] = r[1U][ic];
            for (bitLenInt j = 0U; j < length; ++j) {
                const bitLenInt ja = j + start;
                dest->x[j][i] = x[ja][ia];
                dest->z[j][i] = z[ja][ia];

                dest->x[j][ib] = x[ja][ic];
                dest->z[j][ib] = z[ja][ic];
            }
        }
    }

    for (bitLenInt i = 0; i < endLength; ++i) {
        const bitLenInt ia = i + start;
        const bitLenInt ib = i + end;
        const bitLenInt ic = ia + nQubitCount;
        const bitLenInt id = ib + qubitCount;
        nQubits->pBuffer[ia] = pBuffer[ib];
        nQubits->bBuffer[ia] = bBuffer[ib];
        nQubits->pPhase[ia] = pPhase[ib];
        nQubits->bPhase[ia] = bPhase[ib];
        nQubits->r[0U][ia] = r[0U][ib];
        nQubits->r[1U][ia] = r[1U][ib];
        nQubits->r[0U][ic] = r[0U][id];
        nQubits->r[1U][ic] = r[1U][id];
        for (bitLenInt j = 0; j < start; ++j) {
            nQubits->x[j][ia] = x[j][ib];
            nQubits->z[j][ia] = z[j][ib];

            nQubits->x[j][ic] = x[j][id];
            nQubits->z[j][ic] = z[j][id];
        }
        for (bitLenInt j = 0; j < endLength; ++j) {
            const bitLenInt ja = j + start;
            const bitLenInt jb = j + end;
            nQubits->x[ja][ia] = x[jb][ib];
            nQubits->z[ja][ia] = z[jb][ib];

            nQubits->x[ja][ic] = x[jb][id];
            nQubits->z[ja][ic] = z[jb][id];
        }
    }

    Copy(nQubits);
#else
    const bitLenInt secondStart = qubitCount + start;
    const bitLenInt secondEnd = qubitCount + end;

    if (dest) {
        for (bitLenInt i = 0U; i < length; ++i) {
            bitLenInt j = start + i;
            const BoolVector& xj = x[j];
            const BoolVector& zj = z[j];
            std::copy(xj.begin() + start, xj.begin() + end, dest->x[i].begin());
            std::copy(zj.begin() + start, zj.begin() + end, dest->z[i].begin());

            j = qubitCount + start + i;
            const bitLenInt i2 = i + length;
            const BoolVector& xj2 = x[j];
            const BoolVector& zj2 = z[j];
            std::copy(xj2.begin() + start, xj2.begin() + end, dest->x[i2].begin());
            std::copy(zj2.begin() + start, zj2.begin() + end, dest->z[i2].begin());
        }
        bitLenInt j = start;
        std::copy(pBuffer.begin() + j, pBuffer.begin() + j + length, dest->pBuffer.begin());
        std::copy(bBuffer.begin() + j, bBuffer.begin() + j + length, dest->bBuffer.begin());
        std::copy(pPhase.begin() + j, pPhase.begin() + j + length, dest->pPhase.begin());
        std::copy(bPhase.begin() + j, bPhase.begin() + j + length, dest->bPhase.begin());
        std::copy(r[0U].begin() + j, r[0U].begin() + j + length, dest->r[0U].begin());
        std::copy(r[1U].begin() + j, r[1U].begin() + j + length, dest->r[1U].begin());
        j = qubitCount + start;
        std::copy(r[0U].begin() + j, r[0U].begin() + j + length, dest->r[0U].begin() + length);
        std::copy(r[1U].begin() + j, r[1U].begin() + j + length, dest->r[1U].begin() + length);
    }

    x.erase(x.begin() + secondStart, x.begin() + secondEnd);
    z.erase(z.begin() + secondStart, z.begin() + secondEnd);
    r[0U].erase(r[0U].begin() + secondStart, r[0U].begin() + secondEnd);
    r[1U].erase(r[1U].begin() + secondStart, r[1U].begin() + secondEnd);
    x.erase(x.begin() + start, x.begin() + end);
    z.erase(z.begin() + start, z.begin() + end);
    r[0U].erase(r[0U].begin() + start, r[0U].begin() + end);
    r[1U].erase(r[1U].begin() + start, r[1U].begin() + end);
    pBuffer.erase(pBuffer.begin() + start, pBuffer.begin() + end);
    bBuffer.erase(bBuffer.begin() + start, bBuffer.begin() + end);
    pPhase.erase(pPhase.begin() + start, pPhase.begin() + end);
    bPhase.erase(bPhase.begin() + start, bPhase.begin() + end);

    SetQubitCount(nQubitCount);

    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    for (bitLenInt i = 0U; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
        xi.erase(xi.begin() + start, xi.begin() + end);
        zi.erase(zi.begin() + start, zi.begin() + end);
    }
#endif

    isGaussianCached = false;

    if (randGlobalPhase || dest) {
        return;
    }

    const bitCapInt startMask = pow2Mask(start);
    const bitCapInt endMask = oMaxQMask ^ pow2Mask(start + length);
    const bitCapInt nPerm = (ampEntry.permutation & startMask) | ((ampEntry.permutation & endMask) >> length);

    SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(nPerm)));
}

real1_f QStabilizer::ApproxCompareHelper(QStabilizerPtr toCompare, real1_f error_tol, bool isDiscrete)
{
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    if (!qubitCount) {
        // Both instances have 0 qubits, hence they are equal.
        return ZERO_R1_F;
    }

    toCompare->Finish();
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCount = pow2(g);
    bitCapInt permCountMinus1 = permCount;
    bi_decrement(&permCountMinus1, 1U);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f pNrm = ONE_R1_F / (real1_f)bi_to_double(permCount);
    const real1_f nrm = sqrt(pNrm);

    if (isDiscrete) {
        const AmplitudeEntry entry = getBasisAmp(nrm);
        real1_f potential = pNrm;
        complex proj = conj(entry.amplitude) * toCompare->GetAmplitude(entry.permutation);
        if ((potential - abs(proj)) > error_tol) {
            return ONE_R1_F;
        }
        for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
            const bitCapInt t2 = t ^ (t + ONE_BCI);
            for (bitLenInt i = 0U; i < g; ++i) {
                if (bi_and_1(t2 >> i)) {
                    rowmult(elemCount, qubitCount + i);
                }
            }
            const AmplitudeEntry entry = getBasisAmp(nrm);
            potential += pNrm;
            proj += conj(entry.amplitude) * toCompare->GetAmplitude(entry.permutation);
            if ((potential - abs(proj)) > error_tol) {
                return ONE_R1_F;
            }
        }

        return ONE_R1_F - clampProb((real1_f)norm(proj));
    }

    if (toCompare->PermCount() < pow2(maxStateMapCacheQubitCount)) {
        const std::map<bitCapInt, complex> stateMapCache = toCompare->GetQuantumState();

        complex proj = ZERO_CMPLX;
        const AmplitudeEntry entry = getBasisAmp(nrm);
        const auto it = stateMapCache.find(entry.permutation);
        if (it != stateMapCache.end()) {
            proj += conj(entry.amplitude) * it->second;
        }
        for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
            const bitCapInt t2 = t ^ (t + ONE_BCI);
            for (bitLenInt i = 0U; i < g; ++i) {
                if (bi_and_1(t2 >> i)) {
                    rowmult(elemCount, qubitCount + i);
                }
            }
            const AmplitudeEntry entry = getBasisAmp(nrm);
            const auto it = stateMapCache.find(entry.permutation);
            if (it != stateMapCache.end()) {
                proj += conj(entry.amplitude) * it->second;
            }
        }

        return ONE_R1_F - clampProb((real1_f)norm(proj));
    }

    const AmplitudeEntry entry = getBasisAmp(nrm);
    complex proj = conj(entry.amplitude) * toCompare->GetAmplitude(entry.permutation);
    for (bitCapInt t = ZERO_BCI; bi_compare(t, permCountMinus1) < 0; bi_increment(&t, 1U)) {
        const bitCapInt t2 = t ^ (t + ONE_BCI);
        for (bitLenInt i = 0U; i < g; ++i) {
            if (bi_and_1(t2 >> i)) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry entry = getBasisAmp(nrm);
        proj += conj(entry.amplitude) * toCompare->GetAmplitude(entry.permutation);
    }

    return ONE_R1_F - clampProb((real1_f)norm(proj));
}

void QStabilizer::SetQuantumState(const complex* inputState)
{
    if (qubitCount > 1U) {
        throw std::domain_error("QStabilizer::SetQuantumState() not generally implemented!");
    }

    SetPermutation(ZERO_BCI);

    const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
    const real1 sqrtProb = sqrt(prob);
    const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
    const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
    const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
    const complex mtrx[4U]{ sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
    Mtrx(mtrx, 0U);
}

real1_f QStabilizer::Prob(bitLenInt qubit)
{
    if (IsSeparableZ(qubit)) {
        return M(qubit) ? ONE_R1_F : ZERO_R1_F;
    }

    // Otherwise, state appears locally maximally mixed.
    return HALF_R1_F;
}

void QStabilizer::Mtrx(const complex* mtrx, bitLenInt target)
{
    const complex& mtrx0 = mtrx[0U];
    const complex& mtrx1 = mtrx[1U];
    const complex& mtrx2 = mtrx[2U];
    const complex& mtrx3 = mtrx[3U];

    if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
        return Phase(mtrx0, mtrx3, target);
    }

    if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
        return Invert(mtrx1, mtrx2, target);
    }

    if (IS_SAME(mtrx0, mtrx1) && IS_SAME(mtrx0, mtrx2) && IS_SAME(mtrx0, -mtrx3)) {
        H(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, mtrx1) && IS_SAME(mtrx0, -mtrx2) && IS_SAME(mtrx0, mtrx3)) {
        X(target);
        H(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -mtrx1) && IS_SAME(mtrx0, mtrx2) && IS_SAME(mtrx0, mtrx3)) {
        H(target);
        X(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -mtrx1) && IS_SAME(mtrx0, -mtrx2) && IS_SAME(mtrx0, -mtrx3)) {
        X(target);
        H(target);
        X(target);
        // Reverses sign
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0) + PI_R1);
    }

    if (IS_SAME(mtrx0, mtrx1) && IS_SAME(mtrx0, -I_CMPLX * mtrx2) && IS_SAME(mtrx0, I_CMPLX * mtrx3)) {
        H(target);
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, mtrx1) && IS_SAME(mtrx0, I_CMPLX * mtrx2) && IS_SAME(mtrx0, -I_CMPLX * mtrx3)) {
        H(target);
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -mtrx1) && IS_SAME(mtrx0, I_CMPLX * mtrx2) && IS_SAME(mtrx0, I_CMPLX * mtrx3)) {
        H(target);
        X(target);
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -mtrx1) && IS_SAME(mtrx0, -I_CMPLX * mtrx2) && IS_SAME(mtrx0, -I_CMPLX * mtrx3)) {
        H(target);
        X(target);
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, I_CMPLX * mtrx1) && IS_SAME(mtrx0, mtrx2) && IS_SAME(mtrx0, -I_CMPLX * mtrx3)) {
        IS(target);
        H(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -I_CMPLX * mtrx1) && IS_SAME(mtrx0, mtrx2) && IS_SAME(mtrx0, I_CMPLX * mtrx3)) {
        S(target);
        H(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -I_CMPLX * mtrx1) && IS_SAME(mtrx0, -mtrx2) && IS_SAME(mtrx0, -I_CMPLX * mtrx3)) {
        IS(target);
        H(target);
        X(target);
        Z(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, I_CMPLX * mtrx1) && IS_SAME(mtrx0, -mtrx2) && IS_SAME(mtrx0, I_CMPLX * mtrx3)) {
        S(target);
        H(target);
        X(target);
        Z(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, I_CMPLX * mtrx1) && IS_SAME(mtrx0, I_CMPLX * mtrx2) && IS_SAME(mtrx0, mtrx3)) {
        IS(target);
        H(target);
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -I_CMPLX * mtrx1) && IS_SAME(mtrx0, -I_CMPLX * mtrx2) && IS_SAME(mtrx0, mtrx3)) {
        S(target);
        H(target);
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, I_CMPLX * mtrx1) && IS_SAME(mtrx0, -I_CMPLX * mtrx2) && IS_SAME(mtrx0, -mtrx3)) {
        IS(target);
        H(target);
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    if (IS_SAME(mtrx0, -I_CMPLX * mtrx1) && IS_SAME(mtrx0, I_CMPLX * mtrx2) && IS_SAME(mtrx0, -mtrx3)) {
        S(target);
        H(target);
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(mtrx0));
    }

    throw std::domain_error("QStabilizer::Mtrx() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::Phase(const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (IS_SAME(topLeft, bottomRight)) {
        return SetPhaseOffset(phaseOffset + std::arg(topLeft));
    }

    if (IS_SAME(topLeft, -bottomRight)) {
        Z(target);
        return SetPhaseOffset(phaseOffset + std::arg(topLeft));
    }

    if (IS_SAME(topLeft, -I_CMPLX * bottomRight)) {
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(topLeft));
    }

    if (IS_SAME(topLeft, I_CMPLX * bottomRight)) {
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(topLeft));
    }

    if (IsSeparableZ(target)) {
        // This gate has no effect.
        if (M(target)) {
            Phase(bottomRight, bottomRight, target);
        } else {
            Phase(topLeft, topLeft, target);
        }
        return;
    }

    throw std::domain_error("QStabilizer::Phase() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::Invert(const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (IS_SAME(topRight, bottomLeft)) {
        X(target);
        return SetPhaseOffset(phaseOffset + std::arg(topRight));
    }

    if (IS_SAME(topRight, -bottomLeft)) {
        Y(target);
        // Y is composed as IS, X, S, with overall -i phase
        return SetPhaseOffset(phaseOffset + std::arg(topRight) + PI_R1 / 2);
    }

    if (IS_SAME(topRight, -I_CMPLX * bottomLeft)) {
        X(target);
        S(target);
        return SetPhaseOffset(phaseOffset + std::arg(topRight));
    }

    if (IS_SAME(topRight, I_CMPLX * bottomLeft)) {
        X(target);
        IS(target);
        return SetPhaseOffset(phaseOffset + std::arg(topRight));
    }

    if (IsSeparableZ(target)) {
        if (M(target)) {
            Invert(topRight, topRight, target);
        } else {
            Invert(bottomLeft, bottomLeft, target);
        }
        return;
    }

    throw std::domain_error("QStabilizer::Invert() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::MCPhase(
    const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    if (controls.empty()) {
        return Phase(topLeft, bottomRight, target);
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0U];

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            return;
        }
        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            return CZ(control, target);
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            CNOT(control, target);
            CZ(control, target);
            return CNOT(control, target);
        }
        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            CZ(control, target);
            CNOT(control, target);
            CZ(control, target);
            return CNOT(control, target);
        }
    } else if (IS_SAME(topLeft, I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            CZ(control, target);
            CY(control, target);
            return CNOT(control, target);
        }
        if (IS_SAME(bottomRight, -I_CMPLX)) {
            CY(control, target);
            return CNOT(control, target);
        }
    } else if (IS_SAME(topLeft, -I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            CNOT(control, target);
            return CY(control, target);
        }
        if (IS_SAME(bottomRight, -I_CMPLX)) {
            CY(control, target);
            CZ(control, target);
            return CNOT(control, target);
        }
    }

    throw std::domain_error(
        "QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::MACPhase(
    const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    if (controls.empty()) {
        return Phase(topLeft, bottomRight, target);
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MACPhase() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0U];

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            return;
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            return AntiCZ(control, target);
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            AntiCNOT(control, target);
            AntiCZ(control, target);

            return AntiCNOT(control, target);
        }

        if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            AntiCZ(control, target);
            AntiCNOT(control, target);
            AntiCZ(control, target);

            return AntiCNOT(control, target);
        }
    } else if (IS_SAME(topLeft, I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            AntiCZ(control, target);
            AntiCY(control, target);

            return AntiCNOT(control, target);
        }

        if (IS_SAME(bottomRight, -I_CMPLX)) {
            AntiCY(control, target);

            return AntiCNOT(control, target);
        }
    } else if (IS_SAME(topLeft, -I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            AntiCNOT(control, target);

            return AntiCY(control, target);
        }

        if (IS_SAME(bottomRight, -I_CMPLX)) {
            AntiCY(control, target);
            AntiCZ(control, target);

            return AntiCNOT(control, target);
        }
    }

    throw std::domain_error(
        "QStabilizer::MACPhase() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::MCInvert(
    const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (controls.empty()) {
        return Invert(topRight, bottomLeft, target);
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0U];

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            return CNOT(control, target);
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            CNOT(control, target);

            return CZ(control, target);
        }
    } else if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            CZ(control, target);

            return CNOT(control, target);
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            CZ(control, target);
            CNOT(control, target);

            return CZ(control, target);
        }
    } else if (IS_SAME(topRight, I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            CZ(control, target);

            return CY(control, target);
        }

        if (IS_SAME(bottomLeft, -I_CMPLX)) {
            CZ(control, target);
            CY(control, target);

            return CZ(control, target);
        }
    } else if (IS_SAME(topRight, -I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            return CY(control, target);
        }

        if (IS_SAME(bottomLeft, -I_CMPLX)) {
            CY(control, target);

            return CZ(control, target);
        }
    }

    throw std::domain_error(
        "QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::MACInvert(
    const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (controls.empty()) {
        return Invert(topRight, bottomLeft, target);
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MACInvert() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0U];

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            return AntiCNOT(control, target);
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            AntiCNOT(control, target);

            return AntiCZ(control, target);
        }
    } else if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            AntiCZ(control, target);

            return AntiCNOT(control, target);
        }

        if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            AntiCZ(control, target);
            AntiCNOT(control, target);

            return AntiCZ(control, target);
        }
    } else if (IS_SAME(topRight, I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            AntiCZ(control, target);

            return AntiCY(control, target);
        }

        if (IS_SAME(bottomLeft, -I_CMPLX)) {
            AntiCZ(control, target);
            AntiCY(control, target);

            return AntiCZ(control, target);
        }
    } else if (IS_SAME(topRight, -I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            return AntiCY(control, target);
        }

        if (IS_SAME(bottomLeft, -I_CMPLX)) {
            AntiCY(control, target);

            return AntiCZ(control, target);
        }
    }

    throw std::domain_error(
        "QStabilizer::MACInvert() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    const std::vector<bitLenInt> controls{ qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if (IS_0_R1(sinTheta)) {
        return MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
    }

    if (IS_1_R1(-sinTheta)) {
        ISwap(qubit1, qubit2);
        return MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
    }

    throw std::domain_error("QStabilizer::FSim() not implemented for non-Clifford/Pauli cases!");
}

bool QStabilizer::TrySeparate(const std::vector<bitLenInt>& qubits, real1_f ignored)
{
    std::vector<bitLenInt> lQubits(qubits);
    std::sort(lQubits.begin(), lQubits.end());

    for (size_t i = 0U; i < lQubits.size(); ++i) {
        Swap(lQubits[i], i);
    }

    const bool toRet = CanDecomposeDispose(0U, lQubits.size());

    const bitLenInt last = lQubits.size() - 1U;
    for (size_t i = 0U; i < lQubits.size(); ++i) {
        Swap(lQubits[last - i], last - i);
    }

    return toRet;
}

std::ostream& operator<<(std::ostream& os, const QStabilizerPtr s)
{
    s->gaussian(false);
    const size_t qubitCount = (size_t)s->GetQubitCount();
    os << qubitCount << std::endl;

    const size_t rows = qubitCount << 1U;
    for (size_t row = 0U; row < rows; ++row) {
#if BOOST_AVAILABLE
        const boost::dynamic_bitset<>& xRow = s->x[row];
#else
        const std::vector<bool>& xRow = s->x[row];
#endif
        for (size_t i = 0U; i < xRow.size(); ++i) {
            os << xRow[i] << " ";
        }

#if BOOST_AVAILABLE
        const boost::dynamic_bitset<>& zRow = s->z[row];
#else
        const std::vector<bool>& zRow = s->z[row];
#endif
        for (size_t i = 0U; i < zRow.size(); ++i) {
            os << zRow[i] << " ";
        }

        os << (int)s->GetR(row) << std::endl;
    }

    return os;
}
std::istream& operator>>(std::istream& is, const QStabilizerPtr s)
{
    size_t n;
    is >> n;
    s->SetQubitCount(n);
#if BOOST_AVAILABLE
    s->isTransposed = false;
#endif

    const size_t rows = n << 1U;
#if BOOST_AVAILABLE
    s->r = std::vector<boost::dynamic_bitset<>>(2U, boost::dynamic_bitset<>(rows + 1U));
    s->x = std::vector<boost::dynamic_bitset<>>(rows + 1U, boost::dynamic_bitset<>(n));
    s->z = std::vector<boost::dynamic_bitset<>>(rows + 1U, boost::dynamic_bitset<>(n));
#else
    s->r = std::vector<std::vector<bool>>(2U, std::vector<bool>(rows + 1U));
    s->x = std::vector<std::vector<bool>>(rows + 1U, std::vector<bool>(n));
    s->z = std::vector<std::vector<bool>>(rows + 1U, std::vector<bool>(n));
#endif

    for (size_t row = 0U; row < rows; ++row) {
#if BOOST_AVAILABLE
        boost::dynamic_bitset<>& xRow = s->x[row];
#else
        std::vector<bool>& xRow = s->x[row];
#endif
        for (size_t i = 0U; i < n; ++i) {
            bool x;
            is >> x;
            xRow[i] = x;
        }

#if BOOST_AVAILABLE
        boost::dynamic_bitset<>& zRow = s->z[row];
#else
        std::vector<bool>& zRow = s->z[row];
#endif
        for (size_t i = 0U; i < n; ++i) {
            bool y;
            is >> y;
            zRow[i] = y;
        }

        size_t _r;
        is >> _r;
        s->r[0U][row] = (bool)(_r & 1U);
        s->r[1U][row] = (bool)(_r & 2U);
    }

    return is;
}
} // namespace Qrack
