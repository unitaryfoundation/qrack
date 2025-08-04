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
#if BOOST_AVAILABLE
    , isTransposed(false)
#endif
    , r((n << 1U) + 1U)
    , x((n << 1U) + 1U, BoolVector(n))
    , z((n << 1U) + 1U, BoolVector(n))
{
    maxStateMapCacheQubitCount = getenv("QRACK_MAX_CPU_QB")
        ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_CPU_QB")))
        : 28U - ((QBCAPPOW < FPPOW) ? 1U : (1U + QBCAPPOW - FPPOW));

    SetPermutation(perm, phaseFac);
}

void QStabilizer::ParFor(StabilizerParallelFunc fn, std::vector<bitLenInt> qubits)
{
    for (const bitLenInt& qubit : qubits) {
        if (qubit >= qubitCount) {
            throw std::domain_error("QStabilizer gate qubit indices are out-of-bounds!");
        }
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
    clone->isTransposed = isTransposed;

    return clone;
}

void QStabilizer::SetPermutation(const bitCapInt& perm, const complex& phaseFac)
{
    Dump();
    isTransposed = false;

    if (phaseFac != CMPLX_DEFAULT_ARG) {
        phaseOffset = std::arg(phaseFac);
    } else if (randGlobalPhase) {
        phaseOffset = (real1)(2 * PI_R1 * Rand() - PI_R1);
    } else {
        phaseOffset = ZERO_R1;
    }

    const bitLenInt rowCount = (qubitCount << 1U);

    std::fill(r.begin(), r.end(), 0U);

    for (bitLenInt i = 0; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
#if BOOST_AVAILABLE
        xi.reset();
        zi.reset();

        if (i < qubitCount) {
            xi.set(i);
        } else {
            zi.set(i - qubitCount);
        }
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
#if BOOST_AVAILABLE
    SetTransposeState(false);
#endif

    const BoolVector& xi = x[i];
    const BoolVector& zi = z[i];
    const BoolVector& xk = x[k];
    const BoolVector& zk = z[k];

    // Power to which i is raised
    bitLenInt e = 0U;

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

    e = (e + r[i] + r[k]) & 0x3U;

    return e;
}

/**
 * Do Gaussian elimination to put the stabilizer generators in the following form:
 * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
 * (Return value = number of such generators = log_2 of number of nonzero basis states)
 * At the bottom, generators containing Z's only in quasi-upper-triangular form.
 */
bitLenInt QStabilizer::gaussian()
{
#if BOOST_AVAILABLE
    SetTransposeState(false);
#endif

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

    return g;
}

/**
 * Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
 * writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
 * performed on q.  g is the return value from gaussian(q).
 */
void QStabilizer::seed(const bitLenInt& g)
{
#if BOOST_AVAILABLE
    SetTransposeState(false);
#endif

    const bitLenInt elemCount = qubitCount << 1U;
    int min = 0;

    // Wipe the scratch space clean
    r[elemCount] = 0U;

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
        int f = r[i];
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
    uint8_t e = r[elemCount];
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

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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

    seed(g);

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
    if (!randGlobalPhase) {
        H(t);
        CZ(c, t);
        return H(t);
    }

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    xt ^= xc;
    zc ^= zt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!zt[i]) {
            continue;
        }

        if (xc[i] && (xt[i] == zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            xi[t] = xi[t] ^ xi[c];

            if (zi[t]) {
                zi[c] = !zi[c];

                if (xi[c] && (xi[t] == zi[c])) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
                }
            }
        },
        { c, t });
#endif
}

/// Apply an (anti-)CNOT gate with control and target
void QStabilizer::AntiCNOT(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase) {
        H(t);
        AntiCZ(c, t);
        return H(t);
    }

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    xt ^= xc;
    zc ^= zt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!zt[i]) {
            continue;
        }

        if (xc[i] && (xt[i] == zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            if (xi[c]) {
                xi[t] = !xi[t];
            }

            if (zi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] || (xi[t] != zi[c])) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
                }
            }
        },
        { c, t });
#endif
}

/// Apply a CY gate with control and target
void QStabilizer::CY(bitLenInt c, bitLenInt t)
{
    if (!randGlobalPhase) {
        IS(t);
        CNOT(c, t);
        return S(t);
    }

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;
    xt ^= xc;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!zt[i]) {
            continue;
        }

        if (xc[i] && (xt[i] == zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

    zc ^= zt;
    zt ^= xt;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            zi[t] = zi[t] ^ xi[t];

            if (xi[c]) {
                xi[t] = !xi[t];
            }

            if (zi[t]) {
                if (xi[c] && (xi[t] == zi[c])) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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
    if (!randGlobalPhase) {
        IS(t);
        AntiCNOT(c, t);
        return S(t);
    }

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;
    xt ^= xc;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!zt[i]) {
            continue;
        }

        if (!xc[i] && (xt[i] != zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

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
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xt[i]) {
            continue;
        }

        if (xc[i] && (xt[i] == zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

    zt ^= xc;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            if (xi[t]) {
                zi[c] = !zi[c];

                if (xi[c] && (zi[t] == zi[c])) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xt[i]) {
            continue;
        }

        if (!xc[i] && (xt[i] != zc[i])) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

    zt ^= xc;
#else
    ParFor(
        [this, c, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];

            if (xi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] || (zi[t] != zi[c])) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
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

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    std::swap(xc, xt);
    std::swap(zc, zt);

    zc ^= xt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xt[i]) {
            continue;
        }

        if (!xc[i] && zt[i]) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

    zt ^= xc;

    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xc[i]) {
            continue;
        }

        if (zc[i] && !xt[i]) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

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
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
                }
            }

            if (xi[c]) {
                zi[t] = !zi[t];

                if (zi[c] && !xi[t]) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xc = x[c];
    BoolVector& zc = z[c];
    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zc ^= xc;
    zt ^= xt;

    zt ^= xc;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xc[i]) {
            continue;
        }

        if (zc[i] && !xt[i]) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

    zc ^= xt;

    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xt[i]) {
            continue;
        }

        if (!xc[i] && zt[i]) {
            uint8_t& ri = r[i];
            ri = (ri + 2U) & 0x3U;
        }
    }

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
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
                }
            }

            if (xi[t]) {
                zi[c] = !zi[c];

                if (!xi[c] && zi[t]) {
                    uint8_t& ri = r[i];
                    ri = (ri + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    std::swap(xt, zt);

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (!xt[i] || !zt[i]) {
            continue;
        }
        uint8_t& ri = r[i];
        ri = (ri + 2U) & 0x3U;
    }
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            BoolVector::swap(xi[t], zi[t]);
            if (xi[t] && zi[t]) {
                uint8_t& ri = r[i];
                ri = (ri + 2U) & 0x3U;
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

    seed(g);

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

#if BOOST_AVAILABLE
    const bitLenInt maxLcv = qubitCount << 1U;
    if (isTransposed) {
        BoolVector& zt = z[t];
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (zt[i]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    } else {
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (z[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    }
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (z[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
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

#if BOOST_AVAILABLE
    const bitLenInt maxLcv = qubitCount << 1U;
    if (isTransposed) {
        BoolVector& zt = z[t];
        BoolVector& xt = x[t];
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (zt[i] ^ xt[i]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    } else {
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (z[i][t] ^ x[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    }
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (z[i][t] ^ x[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
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

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

#if BOOST_AVAILABLE
    const bitLenInt maxLcv = qubitCount << 1U;
    if (isTransposed) {
        BoolVector& xt = x[t];
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (xt[i]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    } else {
        for (bitLenInt i = 0; i < maxLcv; ++i) {
            if (x[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        }
    }
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            if (x[i][t]) {
                r[i] = (r[i] + 2U) & 0x3U;
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

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (xt[i] && zt[i]) {
            r[i] = (r[i] + 2U) & 0x3U;
        }
    }

    zt ^= xt;
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            if (xi[t] && zi[t]) {
                r[i] = (r[i] + 2U) & 0x3U;
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

    const AmplitudeEntry ampEntry =
        randGlobalPhase ? AmplitudeEntry(ZERO_BCI, ZERO_CMPLX) : GetQubitAmplitude(t, false);

#if BOOST_AVAILABLE
    SetTransposeState(true);

    BoolVector& xt = x[t];
    BoolVector& zt = z[t];

    zt ^= xt;

    const bitLenInt maxLcv = qubitCount << 1U;
    for (bitLenInt i = 0; i < maxLcv; ++i) {
        if (xt[i] && zt[i]) {
            r[i] = (r[i] + 2U) & 0x3U;
        }
    }
#else
    ParFor(
        [this, t](const bitLenInt& i) {
            const BoolVector& xi = x[i];
            BoolVector& zi = z[i];
            zi[t] = zi[t] ^ xi[t];
            if (xi[t] && zi[t]) {
                r[i] = (r[i] + 2U) & 0x3U;
            }
        },
        { t });
#endif

    if (randGlobalPhase) {
        return;
    }

    SetPhaseOffset(phaseOffset + std::arg(ampEntry.amplitude) - std::arg(GetAmplitude(ampEntry.permutation)));
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
    SetTransposeState(true);
    return x[t].find_next(n - 1U) == BoolVector::npos;
#else
    const bitLenInt& nt2 = n << 1U;
    // loop over stabilizer generators
    for (bitLenInt p = n; p < nt2; ++p) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p][t]) {
            return false;
        }
    }
#endif

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
        p = (bitLenInt)x[t].find_next(n - 1U) - n;
    } else {
        for (p = 0U; p < n; ++p) {
            // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
            if (x[p + n][t]) {
                // The outcome is random
                break;
            }
        }
    }
#else
    for (p = 0U; p < n; ++p) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t]) {
            // The outcome is random
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

        SetTransposeState(false);

        const QStabilizerPtr clone = randGlobalPhase ? nullptr : std::dynamic_pointer_cast<QStabilizer>(Clone());

        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, t + n);

        // Set the new stabilizer result phase
        r[p + n] = result ? 2U : 0U;

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
        const bitLenInt elemCount = qubitCount << 1U;
        const real1_f nrm = sqrt(ONE_R1_F / (real1_f)bi_to_double(pow2(g)));

        seed(g);

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
    bitLenInt m;
#if BOOST_AVAILABLE
    if (isTransposed) {
        m = (bitLenInt)x[t].find_first();
    } else {
        for (m = 0U; m < n; ++m) {
            if (x[m][t]) {
                break;
            }
        }
    }
#else
    for (m = 0U; m < n; ++m) {
        if (x[m][t]) {
            break;
        }
    }
#endif

    if (m >= n) {
        // For example, diagonal permutation state is |0>.
        return false;
    }

    rowcopy(elemCount, m + n);
    for (bitLenInt i = m + 1U; i < n; ++i) {
        if (x[i][t]) {
            rowmult(elemCount, i + n);
        }
    }

    if (doForce && (result != (bool)r[elemCount])) {
        throw std::invalid_argument("QStabilizer::ForceM() forced a measurement with 0 probability!");
    }

    return r[elemCount];
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

    SetPhaseOffset(phaseOffset + toCopy->phaseOffset);

    const bitLenInt length = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + length;
    const bitLenInt endLength = qubitCount - start;
    const bitLenInt secondStart = qubitCount + start;

#if BOOST_AVAILABLE
    QStabilizerPtr nQubits = std::make_shared<QStabilizer>(nQubitCount, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG,
        false, randGlobalPhase, false, -1, !!hardware_rand_generator);

    SetTransposeState(false);
    toCopy->SetTransposeState(false);

    const bitLenInt oRowLength = (nQubitCount << 1U) + 1U;
    for (bitLenInt i = 0U; i < oRowLength; ++i) {
        nQubits->x[i].reset();
        nQubits->z[i].reset();
    }

    for (bitLenInt i = 0U; i < start; ++i) {
        for (bitLenInt j = 0U; j < start; ++j) {
            nQubits->r[i] = r[i];
            nQubits->x[i][j] = x[i][j];
            nQubits->z[i][j] = z[i][j];
            nQubits->r[i + nQubitCount] = r[i + qubitCount];
            nQubits->x[i + nQubitCount][j] = x[i + qubitCount][j];
            nQubits->z[i + nQubitCount][j] = z[i + qubitCount][j];
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        for (bitLenInt j = 0U; j < length; ++j) {
            nQubits->r[i + start] = toCopy->r[i];
            nQubits->x[i + start][j] = toCopy->x[i][j];
            nQubits->z[i + start][j] = toCopy->z[i][j];
            nQubits->r[i + nQubitCount + start] = r[i + length];
            nQubits->x[i + nQubitCount + start][j] = toCopy->x[i + length][j];
            nQubits->z[i + nQubitCount + start][j] = toCopy->z[i + length][j];
        }
    }

    const bitLenInt end = start + length;
    for (bitLenInt i = 0; i < endLength; ++i) {
        for (bitLenInt j = 0; j < endLength; ++j) {
            nQubits->r[i + end] = r[i + start];
            nQubits->x[i + end][j] = x[i + start][j];
            nQubits->z[i + end][j] = z[i + start][j];
            nQubits->r[i + nQubitCount + end] = r[i + secondStart];
            nQubits->x[i + nQubitCount + end][j] = x[i + secondStart][j];
            nQubits->z[i + nQubitCount + end][j] = z[i + secondStart][j];
        }
    }

    Copy(nQubits);
#else
    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    const bitLenInt dLen = length << 1U;

    for (bitLenInt i = 0U; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
        xi.insert(xi.begin() + start, length, false);
        zi.insert(zi.begin() + start, length, false);
    }

    x.insert(x.begin() + secondStart, toCopy->x.begin() + length, toCopy->x.begin() + dLen);
    z.insert(z.begin() + secondStart, toCopy->z.begin() + length, toCopy->z.begin() + dLen);
    r.insert(r.begin() + secondStart, toCopy->r.begin() + length, toCopy->r.begin() + dLen);

    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt offset = secondStart + i;
        BoolVector& xo = x[offset];
        BoolVector& zo = z[offset];
        xo.insert(xo.begin(), start, false);
        xo.insert(xo.end(), endLength, false);
        zo.insert(zo.begin(), start, false);
        zo.insert(zo.end(), endLength, false);
    }

    x.insert(x.begin() + start, toCopy->x.begin(), toCopy->x.begin() + length);
    z.insert(z.begin() + start, toCopy->z.begin(), toCopy->z.begin() + length);
    r.insert(r.begin() + start, toCopy->r.begin(), toCopy->r.begin() + length);
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
    gaussian();

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
    gaussian();

    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "dest" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    const bitCapInt oMaxQMask = pow2Mask(qubitCount);
    const bitLenInt end = start + length;
    const bitLenInt nQubitCount = qubitCount - length;
    const bitLenInt secondStart = qubitCount + start;
    const bitLenInt secondEnd = qubitCount + end;

    if (dest) {
        for (bitLenInt i = 0U; i < length; ++i) {
            bitLenInt j = start + i;
            const BoolVector& xj = x[j];
            const BoolVector& zj = z[j];
            for (bitLenInt k = 0U; k < length; ++k) {
                dest->x[i][k] = xj[start + k];
                dest->z[i][k] = zj[start + k];
            }

            j = qubitCount + start + i;
            const bitLenInt i2 = i + length;
            const BoolVector& xj2 = x[j];
            const BoolVector& zj2 = z[j];
            for (bitLenInt k = 0U; k < length; ++k) {
                dest->x[i2][k] = xj2[start + k];
                dest->z[i2][k] = zj2[start + k];
            }
        }
        bitLenInt j = start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin());
        j = qubitCount + start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin() + length);
    }

    x.erase(x.begin() + secondStart, x.begin() + secondEnd);
    z.erase(z.begin() + secondStart, z.begin() + secondEnd);
    r.erase(r.begin() + secondStart, r.begin() + secondEnd);
    x.erase(x.begin() + start, x.begin() + end);
    z.erase(z.begin() + start, z.begin() + end);
    r.erase(r.begin() + start, r.begin() + end);

    SetQubitCount(nQubitCount);

#if BOOST_AVAILABLE
    SetTransposeState(true);
    x.erase(x.begin() + start, x.begin() + end);
    z.erase(z.begin() + start, z.begin() + end);
#else
    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    for (bitLenInt i = 0U; i < rowCount; ++i) {
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
        xi.erase(xi.begin() + start, xi.begin() + end);
        zi.erase(zi.begin() + start, zi.begin() + end);
    }
#endif

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

    toCompare->SetTransposeState(false);

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapInt permCount = pow2(g);
    bitCapInt permCountMinus1 = permCount;
    bi_decrement(&permCountMinus1, 1U);
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f pNrm = ONE_R1_F / (real1_f)bi_to_double(permCount);
    const real1_f nrm = sqrt(pNrm);

    seed(g);

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
    s->gaussian();
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

        os << (int)s->r[row] << std::endl;
    }

    return os;
}
std::istream& operator>>(std::istream& is, const QStabilizerPtr s)
{
    size_t n;
    is >> n;
    s->SetQubitCount(n);
    s->isTransposed = false;

    const size_t rows = n << 1U;
    s->r = std::vector<uint8_t>(rows + 1U);
#if BOOST_AVAILABLE
    s->x = std::vector<boost::dynamic_bitset<>>(rows + 1U, boost::dynamic_bitset<>(n));
    s->z = std::vector<boost::dynamic_bitset<>>(rows + 1U, boost::dynamic_bitset<>(n));
#else
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
        s->r[row] = (uint8_t)_r;
    }

    return is;
}
} // namespace Qrack
