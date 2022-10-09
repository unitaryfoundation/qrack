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

#pragma once

#include "qinterface.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif

#include <cstdint>

namespace Qrack {

struct AmplitudeEntry {
    bitCapIntOcl permutation;
    complex amplitude;

    AmplitudeEntry(const bitCapInt& p, const complex& a)
        : permutation(p)
        , amplitude(a)
    {
    }
};

class QStabilizer;
typedef std::shared_ptr<QStabilizer> QStabilizerPtr;

class QStabilizer : public QInterface {
protected:
    unsigned rawRandBools;
    unsigned rawRandBoolsRemaining;
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
    bitLenInt dispatchThreshold;
    DispatchQueue dispatchQueue;
#endif
    complex phaseOffset;

    // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    std::vector<uint8_t> r;
    // Typedef for special type std::vector<bool> compatibility
    typedef std::vector<bool> BoolVector;
    // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at the bottom)
    std::vector<BoolVector> x;
    // (2n+1)*n matrix for z bits
    std::vector<BoolVector> z;

    typedef std::function<void(const bitLenInt&)> StabilizerParallelFunc;
    typedef std::function<void(void)> DispatchFn;
    void Dispatch(DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        if (qubitCount >= dispatchThreshold) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    void ParFor(StabilizerParallelFunc fn, std::vector<bitLenInt> qubits)
    {
        for (size_t i = 0U; i < qubits.size(); ++i) {
            if (qubits[i] >= qubitCount) {
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

    bool TrimControls(bitLenInt const* lControls, bitLenInt lControlLen, bool isAnti, std::vector<bitLenInt>& output);

public:
    QStabilizer(bitLenInt n, bitCapInt perm = 0U, qrack_rand_gen_ptr rgp = nullptr, complex ignored = CMPLX_DEFAULT_ARG,
        bool doNorm = false, bool randomGlobalPhase = true, bool ignored2 = false, int64_t ignored3 = -1,
        bool useHardwareRNG = true, bool ignored4 = false, real1_f ignored5 = REAL1_EPSILON,
        std::vector<int64_t> ignored6 = {}, bitLenInt ignored7 = 0U, real1_f ignored8 = FP_NORM_EPSILON_F);

    QInterfacePtr Clone()
    {
        Finish();

        QStabilizerPtr clone = std::make_shared<QStabilizer>(qubitCount, 0U, rand_generator, CMPLX_DEFAULT_ARG, false,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL);
        clone->Finish();

        clone->x = x;
        clone->z = z;
        clone->r = r;
        clone->phaseOffset = phaseOffset;
        clone->randomSeed = randomSeed;

        return clone;
    }

    ~QStabilizer() { Dump(); }

    bool isClifford() { return true; };
    bool isClifford(bitLenInt qubit) { return true; };

    void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.finish();
#endif
    };

    bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        return dispatchQueue.isFinished();
#else
        return true;
#endif
    }

    void Dump()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dump();
#endif
    }

    bitLenInt GetQubitCount() { return qubitCount; }

    bitCapInt GetMaxQPower() { return pow2(qubitCount); }

    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    void SetRandomSeed(uint32_t seed)
    {
        if (rand_generator != NULL) {
            rand_generator->seed(seed);
        }
    }

    void SetDevice(int64_t dID) {}

    bool Rand()
    {
        if (hardware_rand_generator != NULL) {
            if (!rawRandBoolsRemaining) {
                rawRandBools = hardware_rand_generator->NextRaw();
                rawRandBoolsRemaining = sizeof(unsigned) * bitsInByte;
            }
            --rawRandBoolsRemaining;

            return (bool)((rawRandBools >> rawRandBoolsRemaining) & 1U);
        } else {
            return (bool)rand_distribution(*rand_generator);
        }
    }

protected:
    /// Sets row i equal to row k
    void rowcopy(const bitLenInt& i, const bitLenInt& k)
    {
        if (i == k) {
            return;
        }

        x[i] = x[k];
        z[i] = z[k];
        r[i] = r[k];
    }
    /// Swaps row i and row k - does not change the logical state
    void rowswap(const bitLenInt& i, const bitLenInt& k)
    {
        if (i == k) {
            return;
        }

        std::swap(x[k], x[i]);
        std::swap(z[k], z[i]);
        std::swap(r[k], r[i]);
    }
    /// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)
    void rowset(const bitLenInt& i, bitLenInt b)
    {
        // Dealloc, first
        x[i] = BoolVector();
        z[i] = BoolVector();

        x[i] = BoolVector(qubitCount, false);
        z[i] = BoolVector(qubitCount, false);
        r[i] = 0;

        if (b < qubitCount) {
            x[i][b] = true;
        } else {
            b -= qubitCount;
            z[i][b] = true;
        }
    }
    /// Left-multiply row i by row k - does not change the logical state
    void rowmult(const bitLenInt& i, const bitLenInt& k)
    {
        r[i] = clifford(i, k);
        for (bitLenInt j = 0U; j < qubitCount; ++j) {
            x[i][j] = x[i][j] ^ x[k][j];
            z[i][j] = z[i][j] ^ z[k][j];
        }
    }
    /// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
    uint8_t clifford(const bitLenInt& i, const bitLenInt& k);

    /**
     * Do Gaussian elimination to put the stabilizer generators in the following form:
     * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
     * (Return value = number of such generators = log_2 of number of nonzero basis states)
     * At the bottom, generators containing Z's only in quasi-upper-triangular form.
     */
    bitLenInt gaussian();

    /**
     * Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
     * writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
     * performed on q.  g is the return value from gaussian(q).
     */
    void seed(const bitLenInt& g);

    /// Helper for setBasisState() and setBasisProb()
    AmplitudeEntry getBasisAmp(const real1_f& nrm);

    /// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisState(const real1_f& nrm, complex* stateVec);

    /// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisState(const real1_f& nrm, QInterfacePtr eng);

    /// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisProb(const real1_f& nrm, real1* outputProbs);

    void DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr toCopy);

    real1_f ApproxCompareHelper(
        QStabilizerPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);

public:
    void SetQuantumState(complex const* inputState)
    {
        if (qubitCount > 1U) {
            throw std::domain_error("QStabilizer::SetQuantumState() not generally implemented!");
        }

        SetPermutation(0U);

        const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
        const real1 sqrtProb = sqrt(prob);
        const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
        const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
        const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
        const complex mtrx[4U] = { sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1,
            -sqrt1MinProb * phase1 };
        Mtrx(mtrx, 0U);
    }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        throw std::domain_error("QStabilizer::SetAmplitude() not implemented!");
    }

    /// Apply a CNOT gate with control and target
    void CNOT(bitLenInt control, bitLenInt target);
    /// Apply a CY gate with control and target
    void CY(bitLenInt control, bitLenInt target);
    /// Apply a CZ gate with control and target
    void CZ(bitLenInt control, bitLenInt target);
    /// Apply an (anti-)CNOT gate with control and target
    void AntiCNOT(bitLenInt control, bitLenInt target);
    /// Apply an (anti-)CY gate with control and target
    void AntiCY(bitLenInt control, bitLenInt target);
    /// Apply an (anti-)CZ gate with control and target
    void AntiCZ(bitLenInt control, bitLenInt target);
    /// Apply a Hadamard gate to target
    using QInterface::H;
    void H(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(bitLenInt qubitIndex);
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    void IS(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    void Z(bitLenInt qubitIndex);
    /// Apply an X (or NOT) gate to target
    using QInterface::X;
    void X(bitLenInt qubitIndex);
    /// Apply a Pauli Y gate to target
    void Y(bitLenInt qubitIndex);
    // Swap two bits
    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    // Swap two bits and apply a phase factor of i if they are different
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    // Swap two bits and apply a phase factor of -i if they are different
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /// Measure qubit t
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Measure all qubits
    bitCapInt MAll()
    {
        bitCapInt toRet = QInterface::MAll();
        SetPermutation(toRet);
        return toRet;
    }

    /// Get the phase radians of the lowest permutation nonzero amplitude
    real1_f FirstNonzeroPhase();

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    void GetQuantumState(QInterfacePtr eng);

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    complex GetAmplitude(bitCapInt perm);

    /**
     * Returns "true" if target qubit is a Z basis eigenstate
     */
    bool IsSeparableZ(const bitLenInt& target);
    /**
     * Returns "true" if target qubit is an X basis eigenstate
     */
    bool IsSeparableX(const bitLenInt& target);
    /**
     * Returns "true" if target qubit is a Y basis eigenstate
     */
    bool IsSeparableY(const bitLenInt& target);
    /**
     * Returns:
     * 0 if target qubit is not separable
     * 1 if target qubit is a Z basis eigenstate
     * 2 if target qubit is an X basis eigenstate
     * 3 if target qubit is a Y basis eigenstate
     */
    uint8_t IsSeparable(const bitLenInt& target);

    using QInterface::Compose;
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizer>(toCopy)); }
    bitLenInt Compose(QStabilizerPtr toCopy) { return Compose(toCopy, qubitCount); }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizer>(toCopy), start);
    }
    bitLenInt Compose(QStabilizerPtr toCopy, bitLenInt start);
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QStabilizer>(dest));
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QStabilizerPtr)NULL); }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt ignored)
    {
        DecomposeDispose(start, length, (QStabilizerPtr)NULL);
    }
    bool CanDecomposeDispose(const bitLenInt start, const bitLenInt length);
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        QStabilizerPtr nQubits = std::make_shared<QStabilizer>(length, 0U, rand_generator, CMPLX_DEFAULT_ARG, false,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL);
        return Compose(nQubits, start);
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        if (!randGlobalPhase) {
            phaseOffset *= std::polar(ONE_R1, (real1)phaseArg);
        }
    }
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizer>(toCompare), false);
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >= ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizer>(toCompare), true, error_tol);
    }

    real1_f Prob(bitLenInt qubit);

    void Mtrx(complex const* mtrx, bitLenInt target);
    void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    void Invert(complex topRight, complex bottomLeft, bitLenInt target);
    void MCPhase(
        bitLenInt const* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MACPhase(
        bitLenInt const* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(
        bitLenInt const* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    void MACInvert(
        bitLenInt const* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    void MCMtrx(bitLenInt const* controls, bitLenInt controlLen, complex const* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            MCPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
            return;
        }

        if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
            MCInvert(controls, controlLen, mtrx[1U], mtrx[2U], target);
            return;
        }

        throw std::domain_error("QStabilizer::MCMtrx() not implemented for non-Clifford/Pauli cases!");
    }
    void MACMtrx(bitLenInt const* controls, bitLenInt controlLen, complex const* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            MACPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
            return;
        }

        if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
            MACInvert(controls, controlLen, mtrx[1U], mtrx[2U], target);
            return;
        }

        throw std::domain_error("QStabilizer::MACMtrx() not implemented for non-Clifford/Pauli cases!");
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);

    bool TrySeparate(bitLenInt const* qubits, bitLenInt length, real1_f ignored);
    bool TrySeparate(bitLenInt qubit) { return CanDecomposeDispose(qubit, 1U); }
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        Swap(qubit1, 0U);
        Swap(qubit2, 1U);

        const bool toRet = CanDecomposeDispose(0U, 2U);

        Swap(qubit1, 0U);
        Swap(qubit2, 1U);

        return toRet;
    }
};
} // namespace Qrack
