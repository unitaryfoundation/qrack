//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
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

#include <chrono>

#if SEED_DEVRAND
#include <sys/random.h>
#endif

#define IS_0_R1(r) (abs(r) <= REAL1_EPSILON)
#define IS_1_R1(r) (abs(r) <= REAL1_EPSILON)

namespace Qrack {

QStabilizer::QStabilizer(bitLenInt n, bitCapInt perm, qrack_rand_gen_ptr rgp, complex ignored, bool doNorm,
    bool randomGlobalPhase, bool ignored2, int ignored3, bool useHardwareRNG, bool ignored4, real1_f ignored5,
    std::vector<int> ignored6, bitLenInt ignored7, real1_f ignored8)
    : QInterface(n, rgp, doNorm, useHardwareRNG, randomGlobalPhase, REAL1_EPSILON)
    , x((n << 1U) + 1U, BoolVector(n, false))
    , z((n << 1U) + 1U, BoolVector(n, false))
    , r((n << 1U) + 1U)
    , phaseOffset(ONE_CMPLX)
    , rawRandBools(0)
    , rawRandBoolsRemaining(0)
{
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#if ENABLE_ENV_VARS
    dispatchThreshold =
        (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
    dispatchThreshold = PSTRIDEPOW;
#endif
#endif

    SetPermutation(perm);
}

void QStabilizer::SetPermutation(const bitCapInt& perm)
{
    Dump();

    const bitLenInt rowCount = (qubitCount << 1U);

    std::fill(r.begin(), r.end(), 0);

    for (bitLenInt i = 0; i < rowCount; i++) {
        // Dealloc, first
        x[i] = BoolVector();
        z[i] = BoolVector();

        x[i] = BoolVector(qubitCount, false);
        z[i] = BoolVector(qubitCount, false);

        if (i < qubitCount) {
            x[i][i] = true;
        } else {
            bitLenInt j = i - qubitCount;
            z[i][j] = true;
        }
    }

    if (!perm) {
        return;
    }

    for (bitLenInt j = 0; j < qubitCount; j++) {
        if ((perm >> j) & 1) {
            X(j);
        }
    }
}

/// Sets row i equal to row k
void QStabilizer::rowcopy(const bitLenInt& i, const bitLenInt& k)
{
    if (i == k) {
        return;
    }

    x[i] = x[k];
    z[i] = z[k];
    r[i] = r[k];
}

/// Swaps row i and row k
void QStabilizer::rowswap(const bitLenInt& i, const bitLenInt& k)
{
    if (i == k) {
        return;
    }

    std::swap(x[k], x[i]);
    std::swap(z[k], z[i]);
    std::swap(r[k], r[i]);
}

/// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)
void QStabilizer::rowset(const bitLenInt& i, bitLenInt b)
{
    r[i] = 0;

    // Dealloc, first
    x[i] = BoolVector();
    z[i] = BoolVector();

    x[i] = BoolVector(qubitCount, false);
    z[i] = BoolVector(qubitCount, false);

    if (b < qubitCount) {
        z[i][b] = true;
    } else {
        b -= qubitCount;
        x[i][b] = true;
    }
}

/// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
uint8_t QStabilizer::clifford(const bitLenInt& i, const bitLenInt& k)
{
    // Power to which i is raised
    bitLenInt e = 0U;

    for (bitLenInt j = 0; j < qubitCount; j++) {
        // X
        if (x[k][j] && !z[k][j]) {
            // XY=iZ
            e += x[i][j] && z[i][j];
            // XZ=-iY
            e -= !x[i][j] && z[i][j];
        }
        // Y
        if (x[k][j] && z[k][j]) {
            // YZ=iX
            e += !x[i][j] && z[i][j];
            // YX=-iZ
            e -= x[i][j] && !z[i][j];
        }
        // Z
        if (!x[k][j] && z[k][j]) {
            // ZX=iY
            e += x[i][j] && !z[i][j];
            // ZY=-iX
            e -= x[i][j] && z[i][j];
        }
    }

    e = (e + r[i] + r[k]) & 0x3U;

    return e;
}

/// Left-multiply row i by row k
void QStabilizer::rowmult(const bitLenInt& i, const bitLenInt& k)
{
    r[i] = clifford(i, k);
    for (bitLenInt j = 0; j < qubitCount; j++) {
        x[i][j] = x[i][j] ^ x[k][j];
        z[i][j] = z[i][j] ^ z[k][j];
    }
}

/**
 * Do Gaussian elimination to put the stabilizer generators in the following form:
 * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
 * (Return value = number of such generators = log_2 of number of nonzero basis states)
 * At the bottom, generators containing Z's only in quasi-upper-triangular form.
 */
bitLenInt QStabilizer::gaussian()
{
    // For brevity:
    const bitLenInt n = qubitCount;
    const bitLenInt maxLcv = n << 1U;
    bitLenInt i = n;
    bitLenInt k;

    for (bitLenInt j = 0; j < n; j++) {

        // Find a generator containing X in jth column
        for (k = i; k < maxLcv; k++) {
            if (x[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; k2++) {
                if (x[k2][j]) {
                    // Gaussian elimination step:
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            i++;
        }
    }

    const bitLenInt g = i - n;

    for (bitLenInt j = 0; j < n; j++) {

        // Find a generator containing Z in jth column
        for (k = i; k < maxLcv; k++) {
            if (z[k][j]) {
                break;
            }
        }

        if (k < maxLcv) {
            rowswap(i, k);
            rowswap(i - n, k - n);
            for (bitLenInt k2 = i + 1U; k2 < maxLcv; k2++) {
                if (z[k2][j]) {
                    rowmult(k2, i);
                    rowmult(i - n, k2 - n);
                }
            }
            i++;
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
    const bitLenInt elemCount = qubitCount << 1U;
    int min = 0;

    // Wipe the scratch space clean
    r[elemCount] = 0;

    // Dealloc, first
    x[elemCount] = BoolVector();
    z[elemCount] = BoolVector();

    x[elemCount] = BoolVector(qubitCount, false);
    z[elemCount] = BoolVector(qubitCount, false);

    for (int i = elemCount - 1; i >= (int)(qubitCount + g); i--) {
        int f = r[i];
        for (int j = qubitCount - 1; j >= 0; j--) {
            if (z[i][j]) {
                min = j;
                if (x[elemCount][j]) {
                    f = (f + 2) & 0x3;
                }
            }
        }

        if (f == 2) {
            const int j = min;
            // Make the seed consistent with the ith equation
            x[elemCount][j] = !x[elemCount][j];
        }
    }
}

/// Helper for setBasisState() and setBasisProb()
AmplitudeEntry QStabilizer::getBasisAmp(const real1_f& nrm)
{
    const bitLenInt elemCount = qubitCount << 1U;
    uint8_t e = r[elemCount];

    for (bitLenInt j = 0; j < qubitCount; j++) {
        // Pauli operator is "Y"
        if (x[elemCount][j] && z[elemCount][j]) {
            e = (e + 1) & 0x3U;
        }
    }

    complex amp((real1)nrm, ZERO_R1);
    if (e & 1) {
        amp *= I_CMPLX;
    }
    if (e & 2) {
        amp *= -ONE_CMPLX;
    }
    if (!randGlobalPhase) {
        amp *= phaseOffset;
    }

    bitCapIntOcl perm = 0;
    for (bitLenInt j = 0; j < qubitCount; j++) {
        if (x[elemCount][j]) {
            perm |= pow2Ocl(j);
        }
    }

    return AmplitudeEntry(perm, amp);
}

/// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisState(const real1_f& nrm, complex* stateVec, QInterfacePtr eng)
{
    AmplitudeEntry entry = getBasisAmp(nrm);
    if (entry.amplitude == ZERO_CMPLX) {
        return;
    }

    if (stateVec) {
        stateVec[entry.permutation] = entry.amplitude;
    }

    if (eng) {
        eng->SetAmplitude(entry.permutation, entry.amplitude);
    }
}

/// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
void QStabilizer::setBasisProb(const real1_f& nrm, real1* outputProbs)
{
    AmplitudeEntry entry = getBasisAmp(nrm);
    outputProbs[entry.permutation] = norm(entry.amplitude);
}

bool QStabilizer::isEqual(QStabilizerPtr toCompare)
{
    if (!toCompare) {
        return false;
    }

    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    if (!qubitCount) {
        return true;
    }

    if ((!randGlobalPhase || !toCompare->randGlobalPhase) && !IS_NORM_0(phaseOffset - toCompare->phaseOffset)) {
        return false;
    }

    gaussian();
    toCompare->gaussian();

    const bitLenInt elemCount = qubitCount << 1U;

    for (bitLenInt i = 0; i < elemCount; i++) {
        if (r[i] != toCompare->r[i]) {
            return false;
        }
        for (bitLenInt j = 0; j < qubitCount; j++) {
            if (x[i][j] != toCompare->x[i][j]) {
                return false;
            }
            if (z[i][j] != toCompare->z[i][j]) {
                return false;
            }
        }
    }

    return true;
}

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)

real1_f QStabilizer::FirstNonzeroPhase()
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    const AmplitudeEntry entry0 = getBasisAmp(nrm);
    if (entry0.amplitude != ZERO_CMPLX) {
        return (real1_f)std::arg(entry0.amplitude);
    }
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        const AmplitudeEntry entry = getBasisAmp(nrm);
        if (entry.amplitude != ZERO_CMPLX) {
            return (real1_f)std::arg(entry.amplitude);
        }
    }

    return ZERO_R1;
}

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(complex* stateVec)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    std::fill(stateVec, stateVec + pow2Ocl(qubitCount), ZERO_CMPLX);

    setBasisState(nrm, stateVec, NULL);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, stateVec, NULL);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
void QStabilizer::GetQuantumState(QInterfacePtr eng)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    eng->SetPermutation(0);
    eng->SetAmplitude(0, ZERO_CMPLX);

    setBasisState(nrm, NULL, eng);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1U);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisState(nrm, NULL, eng);
    }
}

/// Get all probabilities corresponding to ket notation
void QStabilizer::GetProbs(real1* outputProbs)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    // init stateVec as all 0 values
    std::fill(outputProbs, outputProbs + pow2Ocl(qubitCount), ZERO_R1);

    setBasisProb(nrm, outputProbs);
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        setBasisProb(nrm, outputProbs);
    }
}

/// Convert the state to ket notation (warning: could be huge!)
complex QStabilizer::GetAmplitude(bitCapInt perm)
{
    Finish();

    // log_2 of number of nonzero basis states
    const bitLenInt g = gaussian();
    const bitCapIntOcl permCount = pow2Ocl(g);
    const bitCapIntOcl permCountMin1 = permCount - ONE_BCI;
    const bitLenInt elemCount = qubitCount << 1U;
    const real1_f nrm = sqrt(ONE_R1 / permCount);

    seed(g);

    AmplitudeEntry entry = getBasisAmp(nrm);
    if (entry.permutation == perm) {
        return entry.amplitude;
    }
    for (bitCapIntOcl t = 0; t < permCountMin1; t++) {
        bitCapIntOcl t2 = t ^ (t + 1);
        for (bitLenInt i = 0; i < g; i++) {
            if ((t2 >> i) & 1U) {
                rowmult(elemCount, qubitCount + i);
            }
        }
        AmplitudeEntry entry = getBasisAmp(nrm);
        if (entry.permutation == perm) {
            return entry.amplitude;
        }
    }

    return ZERO_R1;
}

/// Apply a CNOT gate with control and target
void QStabilizer::CNOT(bitLenInt c, bitLenInt t)
{
    ParFor([this, c, t](const bitLenInt& i) {
        if (x[i][c]) {
            x[i][t] = !x[i][t];
        }

        if (z[i][t]) {
            z[i][c] = !z[i][c];

            if (x[i][c] && (x[i][t] == z[i][c])) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }
    });
}

/// Apply a CY gate with control and target
void QStabilizer::CY(bitLenInt c, bitLenInt t)
{
    ParFor([this, c, t](const bitLenInt& i) {
        z[i][t] = z[i][t] ^ x[i][t];

        if (x[i][c]) {
            x[i][t] = !x[i][t];
        }

        if (z[i][t]) {
            if (x[i][c] && (x[i][t] == z[i][c])) {
                r[i] = (r[i] + 2) & 0x3U;
            }

            z[i][c] = !z[i][c];
        }

        z[i][t] = z[i][t] ^ x[i][t];
    });
}

/// Apply a CZ gate with control and target
void QStabilizer::CZ(bitLenInt c, bitLenInt t)
{
    ParFor([this, c, t](const bitLenInt& i) {
        if (x[i][t]) {
            z[i][c] = !z[i][c];

            if (x[i][c] && (z[i][t] == z[i][c])) {
                r[i] = (r[i] + 2) & 0x3U;
            }
        }

        if (x[i][c]) {
            z[i][t] = !z[i][t];
        }
    });
}

void QStabilizer::Swap(bitLenInt c, bitLenInt t)
{
    if (c == t) {
        return;
    }

    ParFor([this, c, t](const bitLenInt& i) {
        BoolVector::swap(x[i][c], x[i][t]);
        BoolVector::swap(z[i][c], z[i][t]);
    });
}

/// Apply a Hadamard gate to target
void QStabilizer::H(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        BoolVector::swap(x[i][t], z[i][t]);
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::S(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
        z[i][t] = z[i][t] ^ x[i][t];
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::IS(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        z[i][t] = z[i][t] ^ x[i][t];
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
void QStabilizer::Z(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (x[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply an X (or NOT) gate to target
void QStabilizer::X(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply a Pauli Y gate to target
void QStabilizer::Y(bitLenInt t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (z[i][t] ^ x[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply square root of X gate
void QStabilizer::StabilizerSqrtX(const bitLenInt& t)
{
    ParFor([this, t](const bitLenInt& i) {
        x[i][t] = x[i][t] ^ z[i][t];
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply inverse square root of X gate
void QStabilizer::StabilizerISqrtX(const bitLenInt& t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
        x[i][t] = x[i][t] ^ z[i][t];
    });
}

/// Apply square root of Y gate
void QStabilizer::StabilizerSqrtY(const bitLenInt& t)
{
    ParFor([this, t](const bitLenInt& i) {
        BoolVector::swap(x[i][t], z[i][t]);
        if (!x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
    });
}

/// Apply inverse square root of Y gate
void QStabilizer::StabilizerISqrtY(const bitLenInt& t)
{
    ParFor([this, t](const bitLenInt& i) {
        if (!x[i][t] && z[i][t]) {
            r[i] = (r[i] + 2) & 0x3U;
        }
        BoolVector::swap(x[i][t], z[i][t]);
    });
}

/**
 * Returns "true" if target qubit is a Z basis eigenstate
 */
bool QStabilizer::IsSeparableZ(const bitLenInt& t)
{
    Finish();

    // for brevity
    const bitLenInt n = qubitCount;

    // loop over stabilizer generators
    for (bitLenInt p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t]) {
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
    H(t);
    S(t);
    const bool isSeparable = IsSeparableZ(t);
    IS(t);
    H(t);

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
        return 1;
    }

    H(t);

    if (IsSeparableZ(t)) {
        H(t);
        return 2;
    }

    S(t);

    if (IsSeparableZ(t)) {
        IS(t);
        H(t);
        return 3;
    }

    IS(t);
    H(t);

    return 0;
}

/**
 * Measure qubit b
 */
bool QStabilizer::ForceM(bitLenInt t, bool result, bool doForce, bool doApply)
{
    if (doForce && !doApply) {
        return result;
    }

    Finish();

    const bitLenInt elemCount = qubitCount << 1U;
    // for brevity
    const bitLenInt n = qubitCount;

    // pivot row in stabilizer
    bitLenInt p;
    // pivot row in destabilizer
    bitLenInt m;

    // loop over stabilizer generators
    for (p = 0; p < n; p++) {
        // if a Zbar does NOT commute with Z_b (the operator being measured), then outcome is random
        if (x[p + n][t]) {
            // The outcome is random
            break;
        }
    }

    // If outcome is indeterminate
    if (p < n) {
        // moment of quantum randomness
        if (!doForce) {
            result = Rand();
        }

        if (!doApply) {
            return result;
        }

        // Set Xbar_p := Zbar_p
        rowcopy(p, p + n);
        // Set Zbar_p := Z_b
        rowset(p + n, t + n);

        r[p + n] = result ? 2U : 0U;
        // Now update the Xbar's and Zbar's that don't commute with Z_b
        for (bitLenInt i = 0; i < elemCount; i++) {
            if ((i != p) && x[i][t]) {
                rowmult(i, p);
            }
        }

        return result;
    }

    // If outcome is determinate

    // Before, we were checking if stabilizer generators commute with Z_b; now, we're checking destabilizer
    // generators
    for (m = 0; m < n; m++) {
        if (x[m][t]) {
            break;
        }
    }

    if (m >= n) {
        return r[elemCount];
    }

    rowcopy(elemCount, m + n);
    for (bitLenInt i = m + 1U; i < n; i++) {
        if (x[i][t]) {
            rowmult(elemCount, i + n);
        }
    }

    return r[elemCount];
}

bitLenInt QStabilizer::Compose(QStabilizerPtr toCopy, bitLenInt start)
{
    // We simply insert the (elsewhere initialized and valid) "toCopy" stabilizers and destabilizers in corresponding
    // position, and we set the new padding to 0. This is immediately a valid state, if the two original QStablizer
    // instances are valid.

    toCopy->Finish();
    Finish();

    const bitLenInt rowCount = (qubitCount << 1U) + 1U;
    const bitLenInt length = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + length;
    const bitLenInt secondStart = nQubitCount + start;
    const BoolVector row(length, 0);

    for (bitLenInt i = 0; i < rowCount; i++) {
        x[i].insert(x[i].begin() + start, row.begin(), row.end());
        z[i].insert(z[i].begin() + start, row.begin(), row.end());
    }

    std::vector<BoolVector> xGroup(length, BoolVector(nQubitCount, false));
    std::vector<BoolVector> zGroup(length, BoolVector(nQubitCount, false));
    for (bitLenInt i = 0; i < length; i++) {
        std::copy(toCopy->x[i].begin(), toCopy->x[i].end(), xGroup[i].begin() + start);
        std::copy(toCopy->z[i].begin(), toCopy->z[i].end(), zGroup[i].begin() + start);
    }
    x.insert(x.begin() + start, xGroup.begin(), xGroup.end());
    z.insert(z.begin() + start, zGroup.begin(), zGroup.end());
    r.insert(r.begin() + start, toCopy->r.begin(), toCopy->r.begin() + length);

    std::vector<BoolVector> xGroup2(length, BoolVector(nQubitCount, false));
    std::vector<BoolVector> zGroup2(length, BoolVector(nQubitCount, false));
    for (bitLenInt i = 0; i < length; i++) {
        bitLenInt j = length + i;
        std::copy(toCopy->x[j].begin(), toCopy->x[j].end(), xGroup2[i].begin() + start);
        std::copy(toCopy->z[j].begin(), toCopy->z[j].end(), zGroup2[i].begin() + start);
    }
    x.insert(x.begin() + secondStart, xGroup2.begin(), xGroup2.end());
    z.insert(z.begin() + secondStart, zGroup2.begin(), zGroup2.end());
    r.insert(r.begin() + secondStart, toCopy->r.begin() + length, toCopy->r.begin() + (length << 1U));

    qubitCount = nQubitCount;

    return start;
}
QInterfacePtr QStabilizer::Decompose(bitLenInt start, bitLenInt length)
{
    QStabilizerPtr dest = std::make_shared<QStabilizer>(qubitCount, 0, rand_generator, CMPLX_DEFAULT_ARG, false,
        randGlobalPhase, false, -1, hardware_rand_generator != NULL);
    Decompose(start, dest);

    return dest;
}

bool QStabilizer::CanDecomposeDispose(const bitLenInt start, const bitLenInt length)
{
    if (qubitCount == 1U) {
        return true;
    }

    Finish();

    // We want to have the maximum number of 0 cross terms possible.
    gaussian();

    const bitLenInt end = start + length;

    for (bitLenInt i = 0; i < start; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = start; j < end; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = end; i < qubitCount; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = start; j < end; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    for (bitLenInt i = start; i < end; i++) {
        bitLenInt i2 = i + qubitCount;
        for (bitLenInt j = 0; j < start; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
        for (bitLenInt j = end; j < qubitCount; j++) {
            if (x[i][j] || z[i][j] || x[i2][j] || z[i2][j]) {
                return false;
            }
        }
    }

    return true;
}

void QStabilizer::DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr dest)
{
    if (length == 0) {
        return;
    }

    if (dest) {
        dest->Dump();
    }
    Finish();

    // We assume that the bits to "decompose" the representation of already have 0 cross-terms in their generators
    // outside inter- "dest" cross terms. (Usually, we're "decomposing" the representation of a just-measured single
    // qubit.)

    const bitLenInt end = start + length;
    const bitLenInt nQubitCount = qubitCount - length;
    const bitLenInt secondStart = nQubitCount + start;
    const bitLenInt secondEnd = nQubitCount + end;

    if (dest) {
        for (bitLenInt i = 0; i < length; i++) {
            bitLenInt j = start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[i].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[i].begin());

            j = qubitCount + start + i;
            std::copy(x[j].begin() + start, x[j].begin() + end, dest->x[(i + length)].begin());
            std::copy(z[j].begin() + start, z[j].begin() + end, dest->z[(i + length)].begin());
        }
        bitLenInt j = start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin());
        j = qubitCount + start;
        std::copy(r.begin() + j, r.begin() + j + length, dest->r.begin() + length);
    }

    x.erase(x.begin() + start, x.begin() + end);
    z.erase(z.begin() + start, z.begin() + end);
    r.erase(r.begin() + start, r.begin() + end);
    x.erase(x.begin() + secondStart, x.begin() + secondEnd);
    z.erase(z.begin() + secondStart, z.begin() + secondEnd);
    r.erase(r.begin() + secondStart, r.begin() + secondEnd);

    qubitCount = nQubitCount;

    const bitLenInt rowCount = (qubitCount << 1U) + 1U;

    for (bitLenInt i = 0; i < rowCount; i++) {
        x[i].erase(x[i].begin() + start, x[i].begin() + end);
        z[i].erase(z[i].begin() + start, z[i].begin() + end);
    }
}

bool QStabilizer::ApproxCompare(QStabilizerPtr o)
{
    if (qubitCount != o->qubitCount) {
        return false;
    }

    o->Finish();
    Finish();

    o->gaussian();
    gaussian();

    const bitLenInt rowCount = (qubitCount << 1U);

    for (bitLenInt i = 0; i < rowCount; i++) {
        if (r[i] != o->r[i]) {
            return false;
        }

        for (bitLenInt j = 0; j < qubitCount; j++) {
            if (x[i][j] != o->x[i][j]) {
                return false;
            }
            if (z[i][j] != o->z[i][j]) {
                return false;
            }
        }
    }

    return true;
}

real1_f QStabilizer::Prob(bitLenInt qubit)
{
    if (IsSeparableZ(qubit)) {
        return M(qubit) ? ONE_R1 : ZERO_R1;
    }

    // Otherwise, state appears locally maximally mixed.
    return ONE_R1 / 2;
}

void QStabilizer::Mtrx(const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        Phase(mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        Invert(mtrx[1], mtrx[2], target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        H(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        // Equivalent to X before H
        StabilizerISqrtY(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        // Equivalent to H before X
        StabilizerSqrtY(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        // Equivalent to X-H-X
        X(target);
        StabilizerSqrtY(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        H(target);
        S(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        StabilizerISqrtY(target);
        S(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        H(target);
        X(target);
        IS(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        H(target);
        X(target);
        S(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        IS(target);
        H(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        S(target);
        H(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        IS(target);
        H(target);
        X(target);
        Z(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        S(target);
        H(target);
        X(target);
        Z(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        StabilizerSqrtX(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        StabilizerISqrtX(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        StabilizerSqrtX(target);
        Z(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    if (IS_SAME(mtrx[0], -I_CMPLX * mtrx[1]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        Z(target);
        StabilizerSqrtX(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= mtrx[0] / std::abs(mtrx[0]);
            } else {
                phaseOffset *= mtrx[1] / std::abs(mtrx[1]);
            }
        }
        return;
    }

    throw std::domain_error("QStabilizer::Mtrx() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::Phase(complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_SAME(topLeft, bottomRight)) {
        if (!randGlobalPhase) {
            phaseOffset *= topLeft;
        }
        return;
    }

    if (IS_SAME(topLeft, -bottomRight)) {
        Z(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topLeft;
            } else {
                phaseOffset *= bottomRight;
            }
        }
        return;
    }

    if (IS_SAME(topLeft, -I_CMPLX * bottomRight)) {
        S(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topLeft;
            } else {
                phaseOffset *= bottomRight;
            }
        }
        return;
    }

    if (IS_SAME(topLeft, I_CMPLX * bottomRight)) {
        IS(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topLeft;
            } else {
                phaseOffset *= bottomRight;
            }
        }
        return;
    }

    if (randGlobalPhase && IsSeparableZ(target)) {
        // This gate has no effect.
        return;
    }

    throw std::domain_error("QStabilizer::Phase() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::Invert(complex topRight, complex bottomLeft, bitLenInt target)
{
    if (IS_SAME(topRight, bottomLeft)) {
        X(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topRight;
            } else {
                phaseOffset *= bottomLeft;
            }
        }
        return;
    }

    if (IS_SAME(topRight, -bottomLeft)) {
        X(target);
        Z(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topRight;
            } else {
                phaseOffset *= bottomLeft;
            }
        }
        return;
    }

    if (IS_SAME(topRight, -I_CMPLX * bottomLeft)) {
        X(target);
        S(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topRight;
            } else {
                phaseOffset *= bottomLeft;
            }
        }
        return;
    }

    if (IS_SAME(topRight, I_CMPLX * bottomLeft)) {
        X(target);
        IS(target);
        if (!randGlobalPhase) {
            if (!IsSeparableZ(target) || !M(target)) {
                phaseOffset *= topRight;
            } else {
                phaseOffset *= bottomLeft;
            }
        }
        return;
    }

    if (randGlobalPhase && IsSeparableZ(target)) {
        // This gate has no meaningful effect on phase.
        X(target);
        return;
    }

    throw std::domain_error("QStabilizer::Invert() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::MCMtrx(const bitLenInt* lControls, bitLenInt lControlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MCPhase(lControls, lControlLen, mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MCInvert(lControls, lControlLen, mtrx[1], mtrx[2], target);
        return;
    }

    throw std::domain_error("QStabilizer::MCMtrx() not implemented for non-Clifford/Pauli cases!");
}

void QStabilizer::MCPhase(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (IS_NORM_0(topLeft - ONE_CMPLX) || IS_NORM_0(bottomRight - ONE_CMPLX)) {
        real1_f prob = Prob(target);
        if (IS_NORM_0(topLeft - ONE_CMPLX) && (prob == ZERO_R1)) {
            return;
        }
        if (IS_NORM_0(bottomRight - ONE_CMPLX) && (prob == ONE_R1)) {
            return;
        }
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0];

    if (IS_SAME(topLeft, ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            return;
        } else if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            CZ(control, target);
            return;
        }
    } else if (IS_SAME(topLeft, -ONE_CMPLX)) {
        if (IS_SAME(bottomRight, ONE_CMPLX)) {
            CNOT(control, target);
            CZ(control, target);
            CNOT(control, target);
            return;
        } else if (IS_SAME(bottomRight, -ONE_CMPLX)) {
            CZ(control, target);
            CNOT(control, target);
            CZ(control, target);
            CNOT(control, target);
            return;
        }
    } else if (IS_SAME(topLeft, I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            CZ(control, target);
            CY(control, target);
            CNOT(control, target);
            return;
        } else if (IS_SAME(bottomRight, -I_CMPLX)) {
            CY(control, target);
            CNOT(control, target);
            return;
        }
    } else if (IS_SAME(topLeft, -I_CMPLX)) {
        if (IS_SAME(bottomRight, I_CMPLX)) {
            CNOT(control, target);
            CY(control, target);
            return;
        } else if (IS_SAME(bottomRight, -I_CMPLX)) {
            CY(control, target);
            CZ(control, target);
            CNOT(control, target);
            return;
        }
    }

    throw std::domain_error(
        "QStabilizer::MCPhase() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::MCInvert(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (controls.size() > 1U) {
        throw std::domain_error(
            "QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Too many controls)");
    }

    const bitLenInt control = controls[0];

    if (IS_SAME(topRight, ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            CNOT(control, target);
            return;
        } else if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            CNOT(control, target);
            CZ(control, target);
            return;
        }
    } else if (IS_SAME(topRight, -ONE_CMPLX)) {
        if (IS_SAME(bottomLeft, ONE_CMPLX)) {
            CZ(control, target);
            CNOT(control, target);
            return;
        } else if (IS_SAME(bottomLeft, -ONE_CMPLX)) {
            CZ(control, target);
            CNOT(control, target);
            CZ(control, target);
            return;
        }
    } else if (IS_SAME(topRight, I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            CZ(control, target);
            CY(control, target);
            return;
        } else if (IS_SAME(bottomLeft, -I_CMPLX)) {
            CZ(control, target);
            CY(control, target);
            CZ(control, target);
            return;
        }
    } else if (IS_SAME(topRight, -I_CMPLX)) {
        if (IS_SAME(bottomLeft, I_CMPLX)) {
            CY(control, target);
            return;
        } else if (IS_SAME(bottomLeft, -I_CMPLX)) {
            CY(control, target);
            CZ(control, target);
            return;
        }
    }

    throw std::domain_error(
        "QStabilizer::MCInvert() not implemented for non-Clifford/Pauli cases! (Non-Clifford/Pauli target payload)");
}

void QStabilizer::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    bitLenInt controls[1] = { qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if (IS_0_R1(sinTheta)) {
        MCPhase(controls, 1, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    if (IS_1_R1(-sinTheta)) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, 1, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    throw std::domain_error("QStabilizer::FSim() not implemented for non-Clifford/Pauli cases!");
}
} // namespace Qrack
