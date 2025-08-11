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

#pragma once

#include "qinterface.hpp"

#if BOOST_AVAILABLE
#include <boost/dynamic_bitset.hpp>
#endif

namespace Qrack {

struct AmplitudeEntry {
    bitCapInt permutation;
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
    real1 phaseOffset;
    bitLenInt maxStateMapCacheQubitCount;
    bool isGaussianCached;
    bitLenInt gaussianCached;
#if BOOST_AVAILABLE
    bool isTransposed;
#endif

    // Typedef for special type std::vector<bool> compatibility
#if BOOST_AVAILABLE
    typedef boost::dynamic_bitset<> BoolVector;
#else
    typedef std::vector<bool> BoolVector;
#endif
    // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    std::vector<BoolVector> r;
    // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at the bottom)
    std::vector<BoolVector> x;
    // (2n+1)*n matrix for z bits
    std::vector<BoolVector> z;

    typedef std::function<void(const bitLenInt&)> StabilizerParallelFunc;
    typedef std::function<void(void)> DispatchFn;
    void Dispatch(DispatchFn fn) { fn(); }

    void ParFor(StabilizerParallelFunc fn, std::vector<bitLenInt> qubits);

    void SetPhaseOffset(real1_f phaseArg)
    {
        phaseOffset = (real1)phaseArg;
        const bool isNeg = phaseOffset < 0;
        if (isNeg) {
            phaseOffset = -phaseOffset;
        }
        phaseOffset -= (real1)(((size_t)(phaseOffset / (2 * PI_R1))) * (2 * PI_R1));
        if (phaseOffset > PI_R1) {
            phaseOffset -= 2 * PI_R1;
        }
        if (isNeg) {
            phaseOffset = -phaseOffset;
        }
    }

    uint8_t GetR(size_t i) { return (r[0U][i] ? 1U : 0U) | (r[1U][i] ? 2U : 0U); }
    void ResetR(size_t i)
    {
#if BOOST_AVAILABLE
        r[0U].reset(i);
        r[1U].reset(i);
#else
        r[0U][i] = false;
        r[1U][i] = false;
#endif
    }

    using QInterface::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QStabilizer>(orig)); }
    void Copy(QStabilizerPtr orig)
    {
        QInterface::Copy(std::dynamic_pointer_cast<QInterface>(orig));
        rawRandBools = orig->rawRandBools;
        rawRandBoolsRemaining = orig->rawRandBoolsRemaining;
        phaseOffset = orig->phaseOffset;
        isGaussianCached = orig->isGaussianCached;
        gaussianCached = orig->gaussianCached;
#if BOOST_AVAILABLE
        isTransposed = orig->isTransposed;
#endif
        maxStateMapCacheQubitCount = orig->maxStateMapCacheQubitCount;
        r = orig->r;
        x = orig->x;
        z = orig->z;
    }

#if BOOST_AVAILABLE
    // By Elara (OpenAI custom GPT)
    std::vector<boost::dynamic_bitset<>> FastTranspose(const std::vector<boost::dynamic_bitset<>>& matrix)
    {
        const size_t num_rows = matrix.size();
        const size_t num_cols = matrix[0].size();

        std::vector<BoolVector> transposed(num_cols, BoolVector(num_rows));

        for (size_t row = 0; row < num_rows; ++row) {
            const BoolVector& bit_row = matrix[row];
            for (size_t col = bit_row.find_first(); col != BoolVector::npos; col = bit_row.find_next(col)) {
                transposed[col].set(row);
            }
        }

        return transposed;
    }

    void SetTransposeState(bool isTrans)
    {
        if (isTrans == isTransposed) {
            return;
        }

        if (!isTransposed) {
            x.pop_back();
            z.pop_back();
            r[0U].pop_back();
            r[1U].pop_back();
        }

        x = FastTranspose(x);
        z = FastTranspose(z);
        isTransposed = isTrans;

        if (!isTransposed) {
            x.emplace_back(qubitCount);
            z.emplace_back(qubitCount);
            r[0U].push_back(0U);
            r[1U].push_back(0U);
        }
    }
#endif

    void ValidateQubitIndex(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::domain_error("QStabilizer gate qubit indices are out-of-bounds!");
        }
    }

public:
    QStabilizer(bitLenInt n, const bitCapInt& perm = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool ignored2 = false, int64_t ignored3 = -1, bool useHardwareRNG = true, bool ignored4 = false,
        real1_f ignored5 = REAL1_EPSILON, std::vector<int64_t> ignored6 = {}, bitLenInt ignored7 = 0U,
        real1_f ignored8 = _qrack_qunit_sep_thresh);

    ~QStabilizer() { Dump(); }

    QInterfacePtr Clone();

    bool isClifford() { return true; };
    bool isClifford(bitLenInt qubit) { return true; };

    bitLenInt GetQubitCount() { return qubitCount; }

    bitCapInt GetMaxQPower() { return pow2(qubitCount); }

    void ResetPhaseOffset() { phaseOffset = ZERO_R1; }
    complex GetPhaseOffset() { return std::polar(ONE_R1, phaseOffset); }

    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    void SetRandomSeed(uint32_t seed)
    {
        if (!!rand_generator) {
            rand_generator->seed(seed);
        }
    }

    bool Rand()
    {
        if (!!hardware_rand_generator) {
            if (!rawRandBoolsRemaining) {
                rawRandBools = hardware_rand_generator->NextRaw();
                rawRandBoolsRemaining = sizeof(unsigned) * bitsInByte;
            }
            --rawRandBoolsRemaining;

            return (bool)((rawRandBools >> rawRandBoolsRemaining) & 1U);
        } else {
            return (rand_distribution(*rand_generator) >= HALF_R1_F);
        }
    }

    void Clear()
    {
        x.clear();
        z.clear();
#if BOOST_AVAILABLE
        r[0U].resize(0U);
        r[1U].resize(0U);
        isTransposed = false;
#else
        r[0U].clear();
        r[1U].clear();
#endif
        phaseOffset = ZERO_R1;
        qubitCount = 0U;
        maxQPower = ONE_BCI;
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
        r[0U][i] = r[0U][k];
        r[1U][i] = r[1U][k];
    }
    /// Swaps row i and row k - does not change the logical state
    void rowswap(const bitLenInt& i, const bitLenInt& k)
    {
        if (i == k) {
            return;
        }

        std::swap(x[k], x[i]);
        std::swap(z[k], z[i]);
#if BOOST_AVAILABLE
        bool _r = r[0U][k];
        r[0U][k] = r[0U][i];
        r[0U][i] = _r;
        _r = r[1U][k];
        r[1U][k] = r[1U][i];
        r[1U][i] = _r;
#else
        BoolVector::swap(r[0U][k], r[0U][i]);
        BoolVector::swap(r[1U][k], r[1U][i]);
#endif
    }
    /// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)
    void rowset(const bitLenInt& i, bitLenInt b)
    {
        ResetR(i);

        BoolVector& xi = x[i];
        BoolVector& zi = z[i];
#if BOOST_AVAILABLE
        xi.reset();
        zi.reset();

        if (b < qubitCount) {
            xi.set(b);
        } else {
            b -= qubitCount;
            zi.set(b);
        }
#else
        std::fill(xi.begin(), xi.end(), false);
        std::fill(zi.begin(), zi.end(), false);

        if (b < qubitCount) {
            xi[b] = true;
        } else {
            b -= qubitCount;
            zi[b] = true;
        }
#endif
    }
    /// Left-multiply row i by row k - does not change the logical state
    void rowmult(const bitLenInt& i, const bitLenInt& k)
    {
        uint8_t phase = clifford(i, k);
        r[0U][i] = (bool)(phase & 1U);
        r[1U][i] = (bool)(phase & 2U);
        BoolVector& xi = x[i];
        BoolVector& zi = z[i];

#if BOOST_AVAILABLE
        xi ^= x[k];
        zi ^= z[k];
#else
        for (bitLenInt j = 0U; j < qubitCount; ++j) {
            xi[j] = xi[j] ^ x[k][j];
            zi[j] = zi[j] ^ z[k][j];
        }
#endif
    }
    /// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
    uint8_t clifford(const bitLenInt& i, const bitLenInt& k);

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

    /// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisState(const real1_f& nrm, std::map<bitCapInt, complex>& stateMap);

    /// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisProb(const real1_f& nrm, real1* outputProbs);

    /// Returns the (partial) expectation value from a state vector amplitude.
    real1_f getExpectation(const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset);

    /// Returns the (partial) expectation value from a state vector amplitude.
    real1_f getExpectation(
        const real1_f& nrm, const std::vector<bitCapInt>& bitPowers, const std::vector<real1_f>& weights);

    /// Returns the (partial) variance from a state vector amplitude.
    real1_f getVariance(const real1_f& mean, const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset);

    /// Returns the (partial) variance a state vector amplitude.
    real1_f getVariance(const real1_f& mean, const real1_f& nrm, const std::vector<bitCapInt>& bitPowers,
        const std::vector<real1_f>& weights);

    void DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr toCopy);

    real1_f ApproxCompareHelper(
        QStabilizerPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON, bool isDiscrete = false);

public:
    /**
     * Do Gaussian elimination to put the stabilizer generators in the following form:
     * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
     * (Return value = number of such generators = log_2 of number of nonzero basis states)
     * At the bottom, generators containing Z's only in quasi-upper-triangular form.
     */
    bitLenInt gaussian(bool s = true);

    bitCapInt PermCount() { return pow2(gaussian(false)); }

    void SetQuantumState(const complex* inputState);
    void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        throw std::domain_error("QStabilizer::SetAmplitude() not implemented!");
    }

    void SetRandGlobalPhase(bool isRand) { randGlobalPhase = isRand; }

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
    /// Apply an X (or NOT) gate to target
    using QInterface::X;
    void X(bitLenInt qubitIndex);
    /// Apply a Pauli Y gate to target
    void Y(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    void Z(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(bitLenInt qubitIndex);
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    void IS(bitLenInt qubitIndex);
    // Swap two bits
    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    // Swap two bits and apply a phase factor of i if they are different
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    // Swap two bits and apply a phase factor of -i if they are different
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /// Measure qubit t
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    void GetQuantumState(QInterfacePtr eng);

    /// Convert the state to sparse ket notation
    std::map<bitCapInt, complex> GetQuantumState();

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    complex GetAmplitude(const bitCapInt& perm);

    /// Get a single basis state amplitude
    std::vector<complex> GetAmplitudes(std::vector<bitCapInt> perms);

    /// Get any single basis state amplitude
    AmplitudeEntry GetAnyAmplitude();

    /// Get any single basis state amplitude where qubit "t" has value "m"
    AmplitudeEntry GetQubitAmplitude(bitLenInt t, bool m);

    /// Get expectation of qubits, interpreting each permutation as an unsigned integer.
    real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI);
    real1_f ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);
    /// Get variance of qubits, interpreting each permutation as an unsigned integer.
    real1_f VarianceBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI);
    real1_f VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);

    /// Under assumption of a QStabilizerHybrid ancillary buffer, trace out the permutation probability
    /// of the reduced density matrx without ancillae.
    real1_f ProbPermRdm(const bitCapInt& perm, bitLenInt ancillaeStart);

    /// Direct measure of masked permutation probability
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation);

    /// Returns all qubits entangled with "target" (including itself)
    std::vector<bitLenInt> EntangledQubits(const bitLenInt& target, const bool& g = true);
    /// Returns "true" if target qubit is a Z basis eigenstate
    bool IsSeparableZ(const bitLenInt& target);
    /// Returns "true" if target qubit is an X basis eigenstate
    bool IsSeparableX(const bitLenInt& target);
    /// Returns "true" if target qubit is a Y basis eigenstate
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
    void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QStabilizerPtr) nullptr); }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& ignored)
    {
        DecomposeDispose(start, length, (QStabilizerPtr) nullptr);
    }
    bool CanDecomposeDispose(const bitLenInt start, const bitLenInt length);
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        if (start > qubitCount) {
            throw std::out_of_range("QStabilizer::Allocate() cannot start past end of register!");
        }

        if (!qubitCount) {
            SetQubitCount(length);
            SetPermutation(ZERO_BCI);
            return 0U;
        }

        QStabilizerPtr nQubits = std::make_shared<QStabilizer>(length, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG,
            false, randGlobalPhase, false, -1, !!hardware_rand_generator);
        return Compose(nQubits, start);
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        if (!randGlobalPhase) {
            SetPhaseOffset(phaseOffset + (real1)phaseArg);
        }
    }
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizer>(toCompare));
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QStabilizer>(toCompare), error_tol);
    }
    bool ApproxCompare(QStabilizerPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >= ApproxCompareHelper(toCompare, error_tol, true);
    }
    bool GlobalPhaseCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return GlobalPhaseCompare(std::dynamic_pointer_cast<QStabilizer>(toCompare), error_tol);
    }
    bool GlobalPhaseCompare(QStabilizerPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        const AmplitudeEntry thisAmpEntry = GetAnyAmplitude();
        real1 argDiff = (real1)abs(
            (std::arg(thisAmpEntry.amplitude) - std::arg(toCompare->GetAmplitude(thisAmpEntry.permutation))) /
            (2 * PI_R1));
        argDiff -= (real1)(size_t)argDiff;
        if (argDiff > HALF_R1) {
            argDiff -= ONE_R1;
        }
        if (FP_NORM_EPSILON >= abs(argDiff)) {
            return false;
        }
        return error_tol >= ApproxCompareHelper(toCompare, error_tol, true);
    }

    real1_f Prob(bitLenInt qubit);

    void Mtrx(const complex* mtrx, bitLenInt target);
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt target);
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt target);
    void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target);
    void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target);
    void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target);
    void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target);
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            return MCPhase(controls, mtrx[0U], mtrx[3U], target);
        }

        if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
            return MCInvert(controls, mtrx[1U], mtrx[2U], target);
        }

        throw std::domain_error("QStabilizer::MCMtrx() not implemented for non-Clifford/Pauli cases!");
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            return MACPhase(controls, mtrx[0U], mtrx[3U], target);
        }

        if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
            return MACInvert(controls, mtrx[1U], mtrx[2U], target);
        }

        throw std::domain_error("QStabilizer::MACMtrx() not implemented for non-Clifford/Pauli cases!");
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);

    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f ignored);
    bool TrySeparate(bitLenInt qubit) { return CanDecomposeDispose(qubit, 1U); }
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit2 < qubit1) {
            std::swap(qubit1, qubit2);
        }

        Swap(qubit1, 0U);
        Swap(qubit2, 1U);

        const bool toRet = CanDecomposeDispose(0U, 2U);

        Swap(qubit2, 1U);
        Swap(qubit1, 0U);

        return toRet;
    }

    friend std::ostream& operator<<(std::ostream& os, const QStabilizerPtr s);
    friend std::istream& operator>>(std::istream& is, const QStabilizerPtr s);
};
} // namespace Qrack
