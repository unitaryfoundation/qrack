//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qinterface.hpp"

#include <algorithm>
#include <random>
#include <thread>

#if SEED_DEVRAND
#include <sys/random.h>
#endif

namespace Qrack {

QInterface::QInterface(
    bitLenInt n, qrack_rand_gen_ptr rgp, bool doNorm, bool useHardwareRNG, bool randomGlobalPhase, real1_f norm_thresh)
    : doNormalize(doNorm)
    , randGlobalPhase(randomGlobalPhase)
    , useRDRAND(useHardwareRNG)
    , qubitCount(n)
    , amplitudeFloor(norm_thresh)
    , maxQPower(pow2(qubitCount))
    , rand_distribution(ZERO_R1_F, ONE_R1_F)
    , hardware_rand_generator(NULL)
{
#if !ENABLE_RDRAND && !ENABLE_RNDFILE && !ENABLE_DEVRAND
    useHardwareRNG = false;
#endif

    if (useHardwareRNG) {
        hardware_rand_generator = std::make_shared<RdRandom>();
#if !ENABLE_RNDFILE && !ENABLE_DEVRAND
        useRDRAND = hardware_rand_generator->SupportsRDRAND();
        if (!useRDRAND) {
            hardware_rand_generator = NULL;
        }
#endif
    }

    if ((rgp == NULL) && (hardware_rand_generator == NULL)) {
        rand_generator = std::make_shared<qrack_rand_gen>();
#if SEED_DEVRAND
        // The original author of this code block (Daniel Strano) is NOT a cryptography expert. However, here's the
        // author's justification for preferring /dev/random used to seed Mersenne twister, in this case. We state
        // firstly, our use case is probably more dependent on good statistical randomness than CSPRNG security.
        // Casually, we can list a few reasons our design:
        //
        // * (As a total guess, if clock manipulation isn't a completely obvious problem,) either of /dev/random or
        // /dev/urandom is probably both statistically and cryptographically preferable to the system clock, as a
        // one-time seed.
        //
        // * We need VERY LITTLE entropy for this seeding, even though its repeated a few times depending on the
        // simulation method stack. Tests of 30+ qubits don't run out of random numbers, this way, and there's no
        // detectable slow-down in Qrack.
        //
        // * The blocking behavior of /dev/random (specifically on startup) is GOOD for us, here. We WANT Qrack to block
        // until the entropy pool is ready on virtual machine and container images that start a Qrack-based application
        // on boot. (We're not crypotgraphers; we're quantum computer simulator developers and users.)
        //
        // * (I have a very basic appreciation for the REFUTATION to historical confusion over the quantity of "entropy"
        // in the device pools, but...) If our purpose is PHYSICAL REALISM of quantum computer simulation, rather than
        // cryptography, then we probably should have a tiny preference for higher "true" entropy. Although, even as a
        // developer in the quantum computing field, I must say that there might be no provable empirical difference
        // between "true quantum randomness" and "perfect statistical (whether pseudo-)randomness" as ontological
        // categories, now might there?

        const int max_rdrand_tries = 10;
        int i;
        for (i = 0; i < max_rdrand_tries; ++i) {
            if (sizeof(randomSeed) == getrandom(reinterpret_cast<char*>(&randomSeed), sizeof(randomSeed), GRND_RANDOM))
                break;
        }
        if (i == max_rdrand_tries) {
            throw std::runtime_error("Failed to seed RNG!");
        }
#else
        randomSeed = (uint32_t)std::time(0);
#endif
        SetRandomSeed(randomSeed);
    } else {
        rand_generator = rgp;
    }

    SetConcurrencyLevel(std::thread::hardware_concurrency());
}

/// Set to a specific permutation of all qubits
void QInterface::SetPermutation(bitCapInt perm, complex ignored)
{
    const bitCapInt measured = MAll();
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (((perm ^ measured) >> i) & ONE_BCI) {
            X(i);
        }
    }
}

/// Quantum Fourier Transform - Optimized for going from |0>/|1> to |+>/|-> basis
void QInterface::QFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (!length) {
        return;
    }

    const bitLenInt end = start + (length - 1U);
    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt hBit = end - i;
        for (bitLenInt j = 0U; j < i; ++j) {
            bitLenInt c = hBit;
            bitLenInt t = hBit + 1U + j;
            CPhaseRootN(j + 2U, c, t);
            if (trySeparate) {
                TrySeparate(c, t);
            }
        }
        H(hBit);
    }
}

/// Inverse Quantum Fourier Transform - Quantum Fourier transform optimized for going from |+>/|-> to |0>/|1> basis
void QInterface::IQFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (!length) {
        return;
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        for (bitLenInt j = 0U; j < i; ++j) {
            const bitLenInt c = (start + i) - (j + 1U);
            const bitLenInt t = start + i;
            CIPhaseRootN(j + 2U, c, t);
            if (trySeparate) {
                TrySeparate(c, t);
            }
        }
        H(start + i);
    }
}

/// Quantum Fourier Transform - Optimized for going from |0>/|1> to |+>/|-> basis
void QInterface::QFTR(const std::vector<bitLenInt>& qubits, bool trySeparate)
{
    if (!qubits.size()) {
        return;
    }

    const bitLenInt end = (qubits.size() - 1U);
    for (size_t i = 0U; i < qubits.size(); ++i) {
        H(qubits[end - i]);
        for (size_t j = 0U; j < ((qubits.size() - 1U) - i); ++j) {
            CPhaseRootN(j + 2U, qubits[(end - i) - (j + 1U)], qubits[end - i]);
        }

        if (trySeparate) {
            TrySeparate(qubits[end - i]);
        }
    }
}

/// Inverse Quantum Fourier Transform - Quantum Fourier transform optimized for going from |+>/|-> to |0>/|1> basis
void QInterface::IQFTR(const std::vector<bitLenInt>& qubits, bool trySeparate)
{
    if (!qubits.size()) {
        return;
    }

    for (size_t i = 0U; i < qubits.size(); ++i) {
        for (size_t j = 0U; j < i; ++j) {
            CIPhaseRootN(j + 2U, qubits[i - (j + 1U)], qubits[i]);
        }
        H(qubits[i]);

        if (trySeparate) {
            TrySeparate(qubits[i]);
        }
    }
}

/// Set register bits to given permutation
void QInterface::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (bool)(value & 1));
        return;
    }

    if (!start && (length == qubitCount)) {
        SetPermutation(value);
        return;
    }

    const bitCapInt regVal = MReg(start, length);
    for (bitLenInt i = 0U; i < length; ++i) {
        const bool bitVal = (bitCapIntOcl)bitSlice(i, regVal);
        if (bitVal == !bitSlice(i, value)) {
            X(start + i);
        }
    }
}

/// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    bitCapInt res = 0U;
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        const bitCapInt power = pow2(bit);
        res |= ForceM(start + bit, (bool)(power & result), doForce, doApply) ? power : 0U;
    }
    return res;
}

/// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceM(const std::vector<bitLenInt>& bits, const std::vector<bool>& values, bool doApply)
{
    if (values.size() && (bits.size() != values.size())) {
        throw std::invalid_argument(
            "QInterface::ForceM() boolean values vector length does not match bit vector length!");
    }

    bitCapInt result = 0U;

    if (values.size()) {
        for (size_t bit = 0U; bit < bits.size(); ++bit) {
            result |= ForceM(bits[bit], values[bit], true, doApply) ? pow2(bits[bit]) : 0U;
        }
        return result;
    }

    if (doApply) {
        for (size_t bit = 0U; bit < bits.size(); ++bit) {
            result |= M(bits[bit]) ? pow2(bits[bit]) : 0U;
        }
        return result;
    }

    std::vector<bitCapInt> qPowers(bits.size());
    std::transform(bits.begin(), bits.end(), qPowers.begin(), pow2);
    result = MultiShotMeasureMask(qPowers, 1).begin()->first;

    return result;
}

/// Returns probability of permutation of the register
real1_f QInterface::ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
{
    const bitCapIntOcl startMask = pow2Ocl(start) - ONE_BCI;
    const bitCapIntOcl maxLcv = ((bitCapIntOcl)maxQPower) >> length;
    const bitCapIntOcl p = (bitCapIntOcl)permutation;
    real1 prob = ZERO_R1;
    for (bitCapIntOcl lcv = 0U; lcv < maxLcv; ++lcv) {
        bitCapIntOcl i = lcv & startMask;
        i |= ((lcv ^ i) | p) << length;
        prob += ProbAll(i);
    }

    return (real1_f)prob;
}

/// Returns probability of permutation of the mask
real1_f QInterface::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    real1 prob = ZERO_R1;
    for (bitCapInt lcv = 0U; lcv < maxQPower; ++lcv) {
        if ((lcv & mask) == permutation) {
            prob += ProbAll(lcv);
        }
    }

    return (real1_f)prob;
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QInterface::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    shift %= length;
    if (!shift) {
        return;
    }

    const bitLenInt end = start + length;
    Reverse(start, end);
    Reverse(start, start + shift);
    Reverse(start + shift, end);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QInterface::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) { ROL(length - shift, start, length); }

bitLenInt QInterface::Compose(QInterfacePtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    const bitLenInt origSize = qubitCount;
    ROL(origSize - start, 0, qubitCount);
    const bitLenInt result = Compose(toCopy);
    ROR(origSize - start, 0, qubitCount);

    return result;
}

std::map<QInterfacePtr, bitLenInt> QInterface::Compose(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto&& q : toCopy) {
        ret[q] = Compose(q);
    }

    return ret;
}

void QInterface::ProbMaskAll(bitCapInt mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    std::vector<bitCapInt> bitPowers;
    for (bitLenInt length = 0U; v; ++length) {
        bitCapInt oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        bitPowers.push_back((v ^ oldV) & oldV);
    }

    std::fill(probsArray, probsArray + pow2Ocl(bitPowers.size()), ZERO_R1);

    for (bitCapInt lcv = 0U; lcv < maxQPower; ++lcv) {
        bitCapIntOcl retIndex = 0U;
        for (size_t p = 0U; p < bitPowers.size(); ++p) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2Ocl(p);
            }
        }
        probsArray[retIndex] += ProbAll(lcv);
    }
}

void QInterface::ProbBitsAll(const std::vector<bitLenInt>& bits, real1* probsArray)
{
    if (bits.size() == qubitCount) {
        bool isOrdered = true;
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            if (bits[i] != i) {
                isOrdered = false;
                break;
            }
        }

        if (isOrdered) {
            GetProbs(probsArray);
            return;
        }
    }

    std::fill(probsArray, probsArray + pow2Ocl(bits.size()), ZERO_R1);

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    for (bitCapInt lcv = 0U; lcv < maxQPower; ++lcv) {
        bitCapIntOcl retIndex = 0U;
        for (size_t p = 0U; p < bits.size(); ++p) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2Ocl(p);
            }
        }
        probsArray[retIndex] += ProbAll(lcv);
    }
}

real1_f QInterface::ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset)
{
    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QInterface::ExpectationBitsAll parameter controls array values must be within allocated qubit bounds!");

    if (bits.size() == 1U) {
        return Prob(bits[0]);
    }

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    real1_f expectation = 0;
    for (bitCapInt lcv = 0U; lcv < maxQPower; ++lcv) {
        bitCapInt retIndex = 0U;
        for (size_t p = 0U; p < bits.size(); ++p) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2(p);
            }
        }
        expectation += (bitCapIntOcl)(offset + retIndex) * ProbAll(lcv);
    }

    return expectation;
}

std::map<bitCapInt, int> QInterface::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    std::vector<bitLenInt> bitMap(qPowers.size());
    std::transform(qPowers.begin(), qPowers.end(), bitMap.begin(), log2);

    ThrowIfQbIdArrayIsBad(bitMap, qubitCount,
        "QInterface::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");

    const bitCapIntOcl maskMaxQPower = pow2Ocl(qPowers.size());
    std::vector<real1> maskProbsVec((bitCapIntOcl)maskMaxQPower);
    ProbBitsAll(bitMap, &(maskProbsVec[0]));
    std::discrete_distribution<bitCapIntOcl> dist(maskProbsVec.begin(), maskProbsVec.end());

    std::random_device rd;
    std::mt19937 gen(rd());

    std::map<bitCapInt, int> results;
    for (unsigned int shot = 0U; shot < shots; ++shot) {
        ++(results[dist(gen)]);
    }

    return results;
}

void QInterface::MultiShotMeasureMask(
    const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    std::vector<bitLenInt> bitMap(qPowers.size());
    std::transform(qPowers.begin(), qPowers.end(), bitMap.begin(), log2);

    ThrowIfQbIdArrayIsBad(bitMap, qubitCount,
        "QInterface::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");

    const bitCapIntOcl maskMaxQPower = pow2Ocl(qPowers.size());
    std::vector<real1> maskProbsVec((bitCapIntOcl)maskMaxQPower);
    ProbBitsAll(bitMap, &(maskProbsVec[0]));
    std::discrete_distribution<bitCapIntOcl> dist(maskProbsVec.begin(), maskProbsVec.end());

    std::random_device rd;
    std::mt19937 gen(rd());

    par_for(0, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) { shotsArray[shot] = (unsigned)dist(gen); });
}

bool QInterface::TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol)
{
    Finish();

    const bool tempDoNorm = doNormalize;
    doNormalize = false;
    QInterfacePtr unitCopy = Clone();
    doNormalize = tempDoNorm;

    unitCopy->Decompose(start, dest);
    unitCopy->Compose(dest, start);

    const bool didSeparate = ApproxCompare(unitCopy, error_tol);

    if (didSeparate) {
        // The subsystem is separable.
        Dispose(start, dest->GetQubitCount());
    }

    return didSeparate;
}

#define REG_GATE_1(gate)                                                                                               \
    void QInterface::gate(bitLenInt start, bitLenInt length)                                                           \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(start + bit);                                                                                         \
        }                                                                                                              \
    }

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
REG_GATE_1(H);

#if ENABLE_REG_GATES

#define REG_GATE_2(gate)                                                                                               \
    void QInterface::gate(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)                                        \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(qubit1 + bit, qubit2 + bit);                                                                          \
        }                                                                                                              \
    }

#define REG_GATE_3(gate)                                                                                               \
    void QInterface::gate(bitLenInt qubit1, bitLenInt qubit2, bitLenInt qubit3, bitLenInt length)                      \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(qubit1 + bit, qubit2 + bit, qubit3 + bit);                                                            \
        }                                                                                                              \
    }

#define REG_GATE_3B(gate)                                                                                              \
    void QInterface::gate(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)    \
    {                                                                                                                  \
        for (bitLenInt i = 0U; i < length; ++i) {                                                                      \
            gate(qInputStart + i, (bitCapIntOcl)bitSlice(i, classicalInput), outputStart + i);                         \
        }                                                                                                              \
    }

#define REG_GATE_1R(gate)                                                                                              \
    void QInterface::gate(real1_f radians, bitLenInt start, bitLenInt length)                                          \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(radians, start + bit);                                                                                \
        }                                                                                                              \
    }

#define REG_GATE_1D(gate)                                                                                              \
    void QInterface::gate(int numerator, int denominator, bitLenInt start, bitLenInt length)                           \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(numerator, denominator, start + bit);                                                                 \
        }                                                                                                              \
    }

#define REG_GATE_C1_1(gate)                                                                                            \
    void QInterface::gate(bitLenInt control, bitLenInt target, bitLenInt length)                                       \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(control + bit, target + bit);                                                                         \
        }                                                                                                              \
    }

#define REG_GATE_C2_1(gate)                                                                                            \
    void QInterface::gate(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)                  \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(control1 + bit, control2 + bit, target + bit);                                                        \
        }                                                                                                              \
    }

#define REG_GATE_C1_1R(gate)                                                                                           \
    void QInterface::gate(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length)                      \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(radians, control + bit, target + bit);                                                                \
        }                                                                                                              \
    }

#define REG_GATE_C1_1D(gate)                                                                                           \
    void QInterface::gate(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)       \
    {                                                                                                                  \
        for (bitLenInt bit = 0U; bit < length; ++bit) {                                                                \
            gate(numerator, denominator, control + bit, target + bit);                                                 \
        }                                                                                                              \
    }

/// Bit-wise apply swap to two registers
REG_GATE_2(Swap);

/// Bit-wise apply iswap to two registers
REG_GATE_2(ISwap);

/// Bit-wise apply inverse iswap to two registers
REG_GATE_2(IISwap);

/// Bit-wise apply square root of swap to two registers
REG_GATE_2(SqrtSwap);

/// Bit-wise apply inverse square root of swap to two registers
REG_GATE_2(ISqrtSwap);

void QInterface::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        FSim(theta, phi, qubit1 + bit, qubit2 + bit);
    }
}

/// Bit-wise apply "anti-"controlled-z to two control registers and one target register
REG_GATE_C2_1(AntiCCZ);

/// Bit-wise apply doubly-controlled-z to two control registers and one target register
REG_GATE_C2_1(CCZ);

/// Apply "Anti-"CZ gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCZ);

/// Apply controlled Pauli Z matrix to each bit
REG_GATE_C1_1(CZ);

/// Bit-wise apply "anti-"controlled-not to two control registers and one target register
REG_GATE_C2_1(AntiCCNOT);

/// Bit-wise apply controlled-not to two control registers and one target register
REG_GATE_C2_1(CCNOT);

/// Apply "Anti-"CNOT gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCNOT);

/// Apply CNOT gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(CNOT);

/// Bit-wise apply "anti-"controlled-y to two control registers and one target register
REG_GATE_C2_1(AntiCCY);

/// Bit-wise apply doubly-controlled-y to two control registers and one target register
REG_GATE_C2_1(CCY);

// Apply "Anti-"CY gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCY);

/// Apply controlled Pauli Y matrix to each bit
REG_GATE_C1_1(CY);

/// Apply S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
REG_GATE_1(S);

/// Apply inverse S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
REG_GATE_1(IS);

/// Apply T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
REG_GATE_1(T);

/// Apply inverse T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
REG_GATE_1(IT);

/// Apply square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SqrtX);

/// Apply inverse square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(ISqrtX);

/// Apply Y-basis transformation gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SH);

/// Apply inverse Y-basis transformation gate to each bit in "length," starting from bit index "start"
REG_GATE_1(HIS);

/// Apply square root of Hadamard gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SqrtH);

/// Apply Pauli Y matrix to each bit
REG_GATE_1(Y);

/// Apply square root of Pauli Y matrix to each bit
REG_GATE_1(SqrtY);

/// Apply square root of Pauli Y matrix to each bit
REG_GATE_1(ISqrtY);

/// Apply Pauli Z matrix to each bit
REG_GATE_1(Z);

/// Apply controlled H gate to each bit
REG_GATE_C1_1(CH);

/// Apply controlled S gate to each bit
REG_GATE_C1_1(CS);

/// Apply controlled IS gate to each bit
REG_GATE_C1_1(CIS);

/// Apply controlled T gate to each bit
REG_GATE_C1_1(CT);

/// Apply controlled IT gate to each bit
REG_GATE_C1_1(CIT);

/// "AND" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(AND);

/// "OR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(OR);

/// "XOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(XOR);

/// "NAND" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(NAND);

/// "NOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(NOR);

/// "XNOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(XNOR);

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLAND);

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLOR);

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLXOR);

/// "NAND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLNAND);

/// "NOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLNOR);

/// "XNOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLXNOR);

/// Apply "PhaseRootN" gate (1/(2^N) phase rotation) to each bit in "length", starting from bit index "start"
void QInterface::PhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        PhaseRootN(n, start + bit);
    }
}

/// Apply inverse "PhaseRootN" gate (1/(2^N) phase rotation) to each bit in "length", starting from bit index "start"
void QInterface::IPhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        IPhaseRootN(n, start + bit);
    }
}

/// Apply controlled "PhaseRootN" gate to each bit
void QInterface::CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length)
{
    if (!n) {
        return;
    }
    if (n == 1) {
        CZ(control, target, length);
        return;
    }

    for (bitLenInt bit = 0U; bit < length; ++bit) {
        CPhaseRootN(n, control + bit, target + bit);
    }
}

/// Apply controlled IT gate to each bit
void QInterface::CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length)
{
    if (!n) {
        return;
    }
    if (n == 1) {
        CZ(control, target, length);
        return;
    }

    for (bitLenInt bit = 0U; bit < length; ++bit) {
        CIPhaseRootN(n, control + bit, target + bit);
    }
}
/// Apply general unitary gate to each bit in "length," starting from bit index "start"
void QInterface::U(bitLenInt start, bitLenInt length, real1_f theta, real1_f phi, real1_f lambda)
{
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        U(start + bit, theta, phi, lambda);
    }
}

/// Apply 2-parameter unitary gate to each bit in "length," starting from bit index "start"
void QInterface::U2(bitLenInt start, bitLenInt length, real1_f phi, real1_f lambda)
{
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        U2(start + bit, phi, lambda);
    }
}
#endif

#if ENABLE_ROT_API
inline real1_f dyadAngle(int numerator, int denomPower) { return (-M_PI * numerator * 2) / pow(2, denomPower); };

/// Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around |1> state.
void QInterface::RTDyad(int numerator, int denomPower, bitLenInt qubit) { RT(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction (identity) exponentiation gate - Applies exponentiation of the identity operator
void QInterface::ExpDyad(int numerator, int denomPower, bitLenInt qubit)
{
    Exp(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
void QInterface::ExpXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpX(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
void QInterface::ExpYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpY(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
void QInterface::ExpZDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpZ(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::RXDyad(int numerator, int denomPower, bitLenInt qubit) { RX(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RYDyad(int numerator, int denomPower, bitLenInt qubit) { RY(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RZDyad(int numerator, int denomPower, bitLenInt qubit) { RZ(dyadAngle(numerator, denomPower), qubit); }

/// Controlled dyadic "phase shift gate" - if control bit is true, rotates target bit as e^(i*(M_PI * numerator) /
/// 2^denomPower) around |1> state
void QInterface::CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRT(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRX(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRY(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction z axis rotation gate - Rotates around Pauli z axis.
void QInterface::CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRZ(dyadAngle(numerator, denomPower), control, target);
}

#if ENABLE_REG_GATES
///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
REG_GATE_1R(RT);

/// Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
REG_GATE_1D(RTDyad);

/// Bitwise (identity) exponentiation gate - Applies exponentiation of the identity operator
REG_GATE_1R(Exp);

/// Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
REG_GATE_1D(ExpDyad);

/// Bitwise Pauli X exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
REG_GATE_1R(ExpX);

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
REG_GATE_1D(ExpXDyad);

/// Bitwise Pauli Y exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
REG_GATE_1R(ExpY);

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
REG_GATE_1D(ExpYDyad);

/// Bitwise Pauli Z exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
REG_GATE_1R(ExpZ);

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
REG_GATE_1D(ExpZDyad);

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
REG_GATE_1R(RX);

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x
REG_GATE_1D(RXDyad);

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
REG_GATE_1R(RY);

/// Dyadic fraction y axis rotation gate - Rotates each bit around Pauli y axis.
REG_GATE_1D(RYDyad);

/// z axis rotation gate - Rotates each bit around Pauli z axis
REG_GATE_1R(RZ);

/// Dyadic fraction z axis rotation gate - Rotates each bit around Pauli y axis.
REG_GATE_1D(RZDyad)

/// Controlled "phase shift gate"
REG_GATE_C1_1R(CRT);

/// Controlled dyadic fraction "phase shift gate"
REG_GATE_C1_1D(CRTDyad);

/// Controlled x axis rotation
REG_GATE_C1_1R(CRX);

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
REG_GATE_C1_1D(CRXDyad);

/// Controlled y axis rotation
REG_GATE_C1_1R(CRY);

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
REG_GATE_C1_1D(CRYDyad);

/// Controlled z axis rotation
REG_GATE_C1_1R(CRZ);

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
REG_GATE_C1_1D(CRZDyad);
#endif

#endif
} // namespace Qrack
