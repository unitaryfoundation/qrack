//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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
#include <mutex>
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
    , rand_generator(rgp)
    , rand_distribution(ZERO_R1_F, ONE_R1_F)
    , hardware_rand_generator{ nullptr }
{
    if (qubitCount > QRACK_MAX_QUBITS) {
        throw std::invalid_argument(
            "Requested QInterface qubit count is larger than the limits of the current Qrack build options!");
    }

#if ENABLE_RDRAND || ENABLE_RNDFILE || ENABLE_DEVRAND
    if (useHardwareRNG) {
        hardware_rand_generator = std::make_shared<RdRandom>();
#if !ENABLE_RNDFILE && !ENABLE_DEVRAND
        useRDRAND = hardware_rand_generator->SupportsRDRAND();
        if (!useRDRAND) {
            hardware_rand_generator = nullptr;
        }
#endif
    } else if (!rand_generator) {
#endif
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

        constexpr int max_rdrand_tries = 10;
        int i;
        for (i = 0; i < max_rdrand_tries; ++i) {
            if (sizeof(randomSeed) == getrandom(reinterpret_cast<char*>(&randomSeed), sizeof(randomSeed), GRND_RANDOM))
                break;
        }
        if (i == max_rdrand_tries) {
            throw std::runtime_error("Failed to seed RNG!");
        }
#else
    std::random_device rd;
    std::uniform_int_distribution<uint32_t> dist(0U, (unsigned int)(pow2Ocl(32U) - 1U));
    randomSeed = dist(rd);
#endif
        SetRandomSeed(randomSeed);
#if ENABLE_RDRAND || ENABLE_RNDFILE || ENABLE_DEVRAND
    }
#endif
}

/// Set to a specific permutation of all qubits
void QInterface::SetPermutation(const bitCapInt& perm, const complex& ignored)
{
    const bitCapInt measured = MAll();
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (bi_and_1((perm ^ measured) >> i)) {
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
            const bitLenInt c = hBit;
            const bitLenInt t = hBit + 1U + j;
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
void QInterface::SetReg(bitLenInt start, bitLenInt length, const bitCapInt& value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        return SetBit(start, (bool)(bi_and_1(value)));
    }

    if (!start && (length == qubitCount)) {
        return SetPermutation(value);
    }

    const bitCapInt regVal = MReg(start, length);
    for (bitLenInt i = 0U; i < length; ++i) {
        if ((bi_compare_0(bitSlice(i, regVal)) == 0) != (bi_compare_0(bitSlice(i, value)) == 0)) {
            X(start + i);
        }
    }
}

/// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceMReg(bitLenInt start, bitLenInt length, const bitCapInt& result, bool doForce, bool doApply)
{
    bitCapInt res = ZERO_BCI;
    for (bitLenInt bit = 0U; bit < length; ++bit) {
        const bitCapInt power = pow2(bit);
        if (ForceM(start + bit, bi_compare_0(power & result) != 0, doForce, doApply)) {
            bi_or_ip(&res, power);
        }
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

    if (values.size()) {
        bitCapInt result = ZERO_BCI;
        for (size_t bit = 0U; bit < bits.size(); ++bit) {
            if (ForceM(bits[bit], values[bit], true, doApply)) {
                bi_or_ip(&result, pow2(bits[bit]));
            }
        }

        return result;
    }

    if (doApply) {
        bitCapInt result = ZERO_BCI;
        for (size_t bit = 0U; bit < bits.size(); ++bit) {
            if (M(bits[bit])) {
                bi_or_ip(&result, pow2(bits[bit]));
            }
        }

        return result;
    }

    std::vector<bitCapInt> qPowers(bits.size());
    std::transform(bits.begin(), bits.end(), qPowers.begin(), pow2);

    return MultiShotMeasureMask(qPowers, 1).begin()->first;
}

/// Returns probability of permutation of the register
real1_f QInterface::ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation)
{
    const bitCapIntOcl startMask = pow2Ocl(start) - 1U;
    const bitCapIntOcl maxLcv = (bitCapIntOcl)(maxQPower >> length);
    const bitCapIntOcl p = (bitCapIntOcl)permutation;
    real1 prob = ZERO_R1;
    for (bitCapIntOcl lcv = 0U; lcv < maxLcv; ++lcv) {
        bitCapIntOcl i = lcv & startMask;
        i |= ((lcv ^ i) | p) << length;
        prob += ProbAll(i);
    }

    return (real1_f)clampProb(prob);
}

/// Returns probability of permutation of the mask
real1_f QInterface::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    if (bi_compare(maxQPower - ONE_BCI, mask) == 0) {
        return ProbAll(permutation);
    }

    real1 prob = ZERO_R1;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        if (bi_compare(lcv & mask, permutation) == 0) {
            prob += ProbAll(lcv);
        }
    }

    return (real1_f)clampProb(prob);
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QInterface::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (length < 2U) {
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
void QInterface::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (length < 2U) {
        return;
    }

    shift %= length;

    if (!shift) {
        return;
    }

    const bitLenInt end = start + length;
    Reverse(start + shift, end);
    Reverse(start, start + shift);
    Reverse(start, end);
}

/// Arithmetic shift left, with last 2 bits as sign and carry
void QInterface::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (!length || !shift) {
        return;
    }

    if (shift >= length) {
        SetReg(start, length, ZERO_BCI);
    } else {
        const bitLenInt end = start + length;
        Swap(end - 1U, end - 2U);
        ROL(shift, start, length);
        SetReg(start, shift, ZERO_BCI);
        Swap(end - 1U, end - 2U);
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void QInterface::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (!length || !shift) {
        return;
    }

    if (shift >= length) {
        SetReg(start, length, ZERO_BCI);
    } else {
        const bitLenInt end = start + length;
        Swap(end - 1U, end - 2U);
        ROR(shift, start, length);
        SetReg(end - shift - 1U, shift, ZERO_BCI);
        Swap(end - 1U, end - 2U);
    }
}

/// Logical shift left, filling the extra bits with |0>
void QInterface::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (!length || !shift) {
        return;
    }

    if (shift >= length) {
        SetReg(start, length, ZERO_BCI);
    } else {
        ROL(shift, start, length);
        SetReg(start, shift, ZERO_BCI);
    }
}

/// Logical shift right, filling the extra bits with |0>
void QInterface::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (!length || !shift) {
        return;
    }

    if (shift >= length) {
        SetReg(start, length, ZERO_BCI);
    } else {
        SetReg(start, shift, ZERO_BCI);
        ROR(shift, start, length);
    }
}

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

void QInterface::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    std::vector<bitCapInt> bitPowers;
    while (bi_compare_0(v) != 0) {
        bitCapInt oldV = v;
        bi_and_ip(&v, v - ONE_BCI); // clear the least significant bit set
        bitPowers.push_back((v ^ oldV) & oldV);
    }

    std::fill(probsArray, probsArray + pow2Ocl(bitPowers.size()), ZERO_R1);

    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        bitCapIntOcl retIndex = 0U;
        for (size_t p = 0U; p < bitPowers.size(); ++p) {
            if (bi_compare_0(lcv & bitPowers[p]) != 0) {
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
            return GetProbs(probsArray);
        }
    }

    std::fill(probsArray, probsArray + pow2Ocl(bits.size()), ZERO_R1);

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        bitCapIntOcl retIndex = 0U;
        for (size_t p = 0U; p < bits.size(); ++p) {
            if (bi_compare_0(lcv & bitPowers[p]) != 0) {
                retIndex |= pow2Ocl(p);
            }
        }
        probsArray[retIndex] += ProbAll(lcv);
    }
}

real1_f QInterface::ExpVarUnitaryAll(bool isExp, const std::vector<bitLenInt>& bits,
    const std::vector<std::shared_ptr<complex>>& basisOps, std::vector<real1_f> eigenVals)
{
    if (bits.empty()) {
        return ONE_R1;
    }

    if (eigenVals.empty()) {
        eigenVals.reserve(bits.size() << 1U);
        for (size_t i = 0U; i < bits.size(); ++i) {
            eigenVals.push_back(ONE_R1);
            eigenVals.push_back(-ONE_R1);
        }
    }

    std::unique_ptr<complex[]> inv(new complex[4U]);
    for (size_t i = 0U; i < bits.size(); ++i) {
        const std::shared_ptr<complex>& mtrx = basisOps[i];
        inv2x2(mtrx.get(), inv.get());
        Mtrx(inv.get(), bits[i]);
    }

    const real1_f toRet =
        isExp ? ExpectationFloatsFactorized(bits, eigenVals) : VarianceFloatsFactorized(bits, eigenVals);

    for (size_t i = 0U; i < bits.size(); ++i) {
        const std::shared_ptr<complex>& mtrx = basisOps[i];
        Mtrx(mtrx.get(), bits[i]);
    }

    return toRet;
}

real1_f QInterface::ExpVarUnitaryAll(bool isExp, const std::vector<bitLenInt>& bits,
    const std::vector<real1_f>& basisOps, std::vector<real1_f> eigenVals)
{
    if (bits.empty()) {
        return ONE_R1;
    }

    if (eigenVals.empty()) {
        eigenVals.reserve(bits.size() << 1U);
        for (size_t i = 0U; i < bits.size(); ++i) {
            eigenVals.push_back(ONE_R1);
            eigenVals.push_back(-ONE_R1);
        }
    }

    for (size_t i = 0U; i < bits.size(); ++i) {
        const size_t i3 = 3U * i;
        U(bits[i], -basisOps[i3], -basisOps[i3 + 1U], -basisOps[i3 + 2U]);
    }

    const real1_f toRet =
        isExp ? ExpectationFloatsFactorized(bits, eigenVals) : VarianceFloatsFactorized(bits, eigenVals);

    for (size_t i = 0U; i < bits.size(); ++i) {
        const size_t i3 = 3U * i;
        U(bits[i], basisOps[i3], basisOps[i3 + 1U], basisOps[i3 + 2U]);
    }

    return toRet;
}

real1_f QInterface::ExpectationBitsFactorized(
    const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    if (perms.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QInterface::ExpectationBitsFactorized() must supply at least twice as many 'perms' as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QInterface::ExpectationBitsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    if (bits.empty()) {
        return ONE_R1;
    }

    if (bits.size() == 1U) {
        const real1_f prob = Prob(bits[0]);
        return (
            real1_f)(bi_to_double(perms[0U] + offset) * (ONE_R1_F - prob) + bi_to_double(perms[1U] + offset) * prob);
    }

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    real1_f expectation = ZERO_R1;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        bitCapInt retIndex = offset;
        for (size_t p = 0U; p < bits.size(); ++p) {
            bi_add_ip(&retIndex, (bi_compare_0(lcv & bitPowers[p]) != 0) ? perms[(p << 1U) | 1U] : perms[p << 1U]);
        }
        expectation += (real1_f)(bi_to_double(retIndex) * ProbAll(lcv));
    }

    return expectation;
}

real1_f QInterface::VarianceBitsFactorized(
    const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    if (perms.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QInterface::VarianceBitsFactorized() must supply at least twice as many 'perms' as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QInterface::VarianceBitsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    if (bits.empty()) {
        return ONE_R1;
    }

    const real1_f mean = ExpectationBitsFactorized(bits, perms, offset);
    if (bits.size() == 1U) {
        const real1_f prob = Prob(bits[0]);
        const real1_f diff0 = bi_to_double(perms[0U] + offset) - mean;
        const real1_f diff1 = bi_to_double(perms[1U] + offset) - mean;

        return diff0 * diff0 * (ONE_R1_F - prob) + diff1 * diff1 * prob;
    }

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    real1_f expectation = ZERO_R1;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        bitCapInt retIndex = offset;
        for (size_t p = 0U; p < bits.size(); ++p) {
            bi_add_ip(&retIndex, (bi_compare_0(lcv & bitPowers[p]) != 0) ? perms[(p << 1U) | 1U] : perms[p << 1U]);
        }
        const real1_f diff = (real1_f)bi_to_double(retIndex) - mean;
        expectation += diff * diff * ProbAll(lcv);
    }

    return expectation;
}

real1_f QInterface::VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
{
    if (weights.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QInterface::VarianceFloatsFactorized() must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QInterface::VarianceFloatsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    if (bits.empty()) {
        return ONE_R1;
    }

    const real1_f mean = ExpectationFloatsFactorized(bits, weights);
    if (bits.size() == 1U) {
        const real1_f prob = Prob(bits[0]);
        const real1_f var0 = weights[0U] - mean;
        const real1_f var1 = weights[1U] - mean;

        return var0 * var0 * (ONE_R1_F - prob) + var1 * var1 * prob;
    }

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    real1_f expectation = ZERO_R1_F;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        real1_f weight = ONE_R1_F;
        for (size_t p = 0U; p < bits.size(); ++p) {
            weight *= (bi_compare_0(lcv & bitPowers[p]) == 0) ? weights[p << 1U] : weights[(p << 1U) | 1U];
        }
        expectation += (weight - mean) * ProbAll(lcv);
    }

    return expectation;
}

real1_f QInterface::VariancePauliAll(std::vector<bitLenInt> bits, std::vector<Pauli> paulis)
{
    for (size_t i = 0U; i < bits.size(); ++i) {
        const size_t j = bits.size() - (i + 1U);
        if (paulis[j] == PauliI) {
            bits.erase(bits.begin() + j);
            paulis.erase(paulis.begin() + j);
        }
    }

    if (bits.empty()) {
        return ONE_R1;
    }

    std::vector<real1_f> eigenVals;
    eigenVals.reserve(bits.size() << 1U);
    for (size_t i = 0U; i < bits.size(); ++i) {
        eigenVals.push_back(ONE_R1);
        eigenVals.push_back(-ONE_R1);

        switch (paulis[i]) {
        case PauliX:
            H(bits[i]);
            break;
        case PauliY:
            IS(bits[i]);
            H(bits[i]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }

    const real1_f toRet = VarianceFloatsFactorized(bits, eigenVals);

    for (size_t i = 0U; i < bits.size(); ++i) {
        switch (paulis[i]) {
        case PauliX:
            H(bits[i]);
            break;
        case PauliY:
            H(bits[i]);
            S(bits[i]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }

    return toRet;
}

real1_f QInterface::ExpectationPauliAll(std::vector<bitLenInt> bits, std::vector<Pauli> paulis)
{
    for (size_t i = 0U; i < bits.size(); ++i) {
        const size_t j = bits.size() - (i + 1U);
        if (paulis[j] == PauliI) {
            bits.erase(bits.begin() + j);
            paulis.erase(paulis.begin() + j);
        }
    }

    if (bits.empty()) {
        return ONE_R1;
    }

    std::vector<real1_f> eigenVals;
    eigenVals.reserve(bits.size() << 1U);
    for (size_t i = 0U; i < bits.size(); ++i) {
        eigenVals.push_back(ONE_R1);
        eigenVals.push_back(-ONE_R1);

        switch (paulis[i]) {
        case PauliX:
            H(bits[i]);
            break;
        case PauliY:
            IS(bits[i]);
            H(bits[i]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }

    const real1_f toRet = ExpectationFloatsFactorized(bits, eigenVals);

    for (size_t i = 0U; i < bits.size(); ++i) {
        switch (paulis[i]) {
        case PauliX:
            H(bits[i]);
            break;
        case PauliY:
            H(bits[i]);
            S(bits[i]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }

    return toRet;
}

real1_f QInterface::ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
{
    if (weights.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QInterface::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QInterface::ExpectationFloatsFactorized() parameter qubits vector values must be within allocated qubit "
        "bounds!");

    if (bits.empty()) {
        return ONE_R1;
    }

    if (bits.size() == 1U) {
        const real1_f prob = Prob(bits[0]);

        return weights[0U] * (ONE_R1_F - prob) + weights[1U] * prob;
    }

    std::vector<bitCapInt> bitPowers(bits.size());
    std::transform(bits.begin(), bits.end(), bitPowers.begin(), pow2);

    real1_f expectation = ZERO_R1_F;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxQPower) < 0; bi_increment(&lcv, 1U)) {
        real1_f weight = ONE_R1_F;
        for (size_t p = 0U; p < bits.size(); ++p) {
            weight *= (bi_compare_0(lcv & bitPowers[p]) != 0) ? weights[(p << 1U) | 1U] : weights[p << 1U];
        }
        expectation += weight * ProbAll(lcv);
    }

    return expectation;
}

std::map<bitCapInt, int> QInterface::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    std::map<bitCapInt, int> results;
    std::mutex resultsMutex;
    par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
        const bitCapInt sample = SampleClone(qPowers);
        std::lock_guard<std::mutex> lock(resultsMutex);
        ++(results[sample]);
    });

    return results;
}

void QInterface::MultiShotMeasureMask(
    const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
        shotsArray[shot] = (unsigned)(bitCapIntOcl)SampleClone(qPowers);
    });
}

bool QInterface::TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol)
{
    Finish();

    QInterfacePtr orig = Copy();
    orig->Decompose(start, dest);
    QInterfacePtr output = orig->Copy();
    orig->Compose(dest, start);

    const bool didSeparate = ApproxCompare(orig, error_tol);

    if (didSeparate) {
        // The subsystem is separable.
        Copy(output);
    }

    return didSeparate;
}

void QInterface::GetReducedDensityMatrix(const std::vector<bitLenInt>& qubits, complex* outputState)
{
    const bitLenInt nQubits = GetQubitCount();
    const bitLenInt keptQubits = (bitLenInt)qubits.size();
    const bitCapIntOcl dimKept = pow2Ocl(keptQubits);
    const bitLenInt envQubits = nQubits - keptQubits;
    const bitCapInt dimEnv = pow2(envQubits);

    std::fill(outputState, outputState + dimKept * dimKept, complex{ ZERO_R1, ZERO_R1 });

    // Precompute bit positions for kept qubits and env qubits
    std::vector<bitLenInt> keptBitPos = qubits;
    std::vector<bool> isKept(nQubits, false);
    for (auto q : keptBitPos) {
        isKept[q] = true;
    }

    std::vector<bitLenInt> envBitPos;
    envBitPos.reserve(envQubits);
    for (bitLenInt q = 0; q < nQubits; ++q) {
        if (!isKept[q]) {
            envBitPos.push_back(q);
        }
    }

    // Precompute full-index masks for each kept basis state given an env pattern
    // We build a base index for each env state where kept qubits are all zero
    for (bitCapInt envState = ZERO_BCI; envState < dimEnv; ++envState) {
        bitCapInt envBaseIndex = ZERO_BCI;
        for (bitLenInt e = 0U; e < envQubits; ++e) {
            if (bi_and_1(envState >> e)) {
                envBaseIndex = envBaseIndex | pow2(envBitPos[e]);
            }
        }

        // For each pair of kept basis states, accumulate contributions
        for (bitCapIntOcl kept_i = 0U; kept_i < dimKept; ++kept_i) {
            bitCapInt fullIndex_i = envBaseIndex;
            for (bitLenInt k = 0U; k < keptQubits; ++k) {
                if (bi_and_1(kept_i >> k)) {
                    fullIndex_i = fullIndex_i | pow2(keptBitPos[k]);
                }
            }

            const complex amp_i = GetAmplitude(fullIndex_i);

            for (bitCapIntOcl kept_j = 0U; kept_j < dimKept; ++kept_j) {
                bitCapInt fullIndex_j = envBaseIndex;
                for (bitLenInt k = 0U; k < keptQubits; ++k) {
                    if (bi_and_1(kept_j >> k)) {
                        fullIndex_j = fullIndex_j | pow2(keptBitPos[k]);
                    }
                }

                outputState[kept_i * dimKept + kept_j] += amp_i * std::conj(GetAmplitude(fullIndex_j));
            }
        }
    }
}

struct _PermProb {
    bitCapInt perm;
    real1_f prob;

    _PermProb()
        : perm(ZERO_BCI)
        , prob(ZERO_R1_F)
    {
    }

    _PermProb(const bitCapInt& prm, const real1_f& prb)
        : perm(prm)
        , prob(prb)
    {
    }
};
std::vector<bitCapInt> QInterface::HighestProbAll(size_t n)
{
    if (!n) {
        return std::vector<bitCapInt>();
    }

    if (n == 1U) {
        return std::vector<bitCapInt>{ HighestProbAll() };
    }

    if (n > maxQPower) {
        throw std::invalid_argument("QInterface::HighestProbAll(n) requested more !");
    }

    real1_f totProb = ZERO_R1_F;
    std::vector<_PermProb> highestProbs(n);
    for (bitCapInt p = ZERO_BCI; p < maxQPower; ++p) {
        real1_f prob = ProbAll(p);
        totProb += prob;
        for (size_t t = 0U; t < n; ++t) {
            if (prob <= highestProbs[t].prob) {
                continue;
            }
            _PermProb orig = highestProbs[t];
            for (size_t i = t + 1U; i < n; ++i) {
                std::swap(orig, highestProbs[i]);
            }
            highestProbs[t] = _PermProb(p, prob);
            break;
        }
        if (highestProbs.back().prob > (ONE_R1_F - totProb)) {
            break;
        }
    }

    std::vector<bitCapInt> bestPerms(n);
    for (size_t t = 0U; t < n; ++t) {
        bestPerms[t] = highestProbs[t].perm;
    }

    return bestPerms;
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
    void QInterface::gate(                                                                                             \
        bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length)               \
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
        return CZ(control, target, length);
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
        return CZ(control, target, length);
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
