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

#include "qengine.hpp"

#include <algorithm>

namespace Qrack {

void QEngine::Mtrx(const complex* mtrx, bitLenInt qubit)
{
    if (IsIdentity(mtrx, false)) {
        return;
    }

    const bitCapIntOcl qPowers[1U]{ pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0U], mtrx, 1U, qPowers, doNormalize && !(IsPhase(mtrx) || IsInvert(mtrx)));
}

void QEngine::EitherMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, bool isAnti)
{
    if (controls.empty()) {
        return Mtrx(mtrx, target);
    }

    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (isAnti) {
        ApplyAntiControlled2x2(controls, target, mtrx);
    } else {
        ApplyControlled2x2(controls, target, mtrx);
    }

    if (doNormalize && !(IsPhase(mtrx) || IsInvert(mtrx))) {
        UpdateRunningNorm();
    }
}

void QEngine::UCMtrx(
    const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, const bitCapInt& controlPerm)
{
    if (controls.empty()) {
        return Mtrx(mtrx, target);
    }

    if (IsIdentity(mtrx, true)) {
        return;
    }

    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 1U]);
    const bitCapIntOcl targetMask = pow2Ocl(target);
    bitCapIntOcl fullMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
        if (bi_and_1(controlPerm >> i)) {
            fullMask |= qPowersSorted[i];
        }
    }
    const bitCapIntOcl controlMask = fullMask;
    qPowersSorted[controls.size()] = targetMask;
    fullMask |= targetMask;
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 1U);
    Apply2x2(controlMask, fullMask, mtrx, controls.size() + 1U, qPowersSorted.get(), false);
}

/// PSEUDO-QUANTUM - Acts like a measurement gate, except with a specified forced result.
bool QEngine::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QEngine::ForceM qubit index parameter must be within allocated qubit bounds!");
    }

    const real1_f oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            result = (Rand() <= oneChance);
        }
    }

    const real1_f nrmlzr = result ? oneChance : (ONE_R1 - oneChance);
    if (nrmlzr <= ZERO_R1) {
        throw std::invalid_argument("QEngine::ForceM() forced a measurement result with 0 probability!");
    }

    if (doApply && ((ONE_R1 - nrmlzr) > REAL1_EPSILON)) {
        const bitCapInt qPower = pow2(qubit);
        ApplyM(qPower, result, GetNonunitaryPhase() / (real1)(std::sqrt((real1_s)nrmlzr)));
    }

    return result;
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceM(const std::vector<bitLenInt>& bits, const std::vector<bool>& values, bool doApply)
{
    if (values.size() && (bits.size() != values.size())) {
        throw std::invalid_argument(
            "QInterface::ForceM() boolean values vector length does not match bit vector length!");
    }

    for (const bitLenInt& bit : bits) {
        if (bit >= qubitCount) {
            throw std::invalid_argument(
                "QEngine::ForceM qubit index parameter array values must be within allocated qubit bounds!");
        }
    }

    // Single bit operations are better optimized for this special case:
    if (bits.size() == 1U) {
        if (ForceM(bits[0U], values.empty() ? false : values[0U], false, doApply)) {
            return pow2(bits[0U]);
        }

        return ZERO_BCI;
    }

    std::unique_ptr<bitCapInt[]> qPowers(new bitCapInt[bits.size()]);
    bitCapInt regMask = ZERO_BCI;
    for (bitCapIntOcl i = 0U; i < bits.size(); ++i) {
        qPowers[i] = pow2(bits[i]);
        bi_or_ip(&regMask, qPowers[i]);
    }
    std::sort(qPowers.get(), qPowers.get() + bits.size());

    const complex phase = GetNonunitaryPhase();
    if (values.size()) {
        bitCapIntOcl result = 0U;
        for (size_t j = 0U; j < values.size(); ++j) {
            if (values[j]) {
                result |= pow2Ocl(bits[j]);
            }
        }
        real1_f nrmlzr = ProbMask(regMask, result);
        complex nrm = phase / (real1)(std::sqrt((real1_s)nrmlzr));
        if ((ONE_R1_F - nrmlzr) > FP_NORM_EPSILON) {
            ApplyM(regMask, (bitCapInt)result, nrm);
        }

        // No need to check against probabilities:
        return result;
    }

    if (doNormalize) {
        NormalizeState();
    }

    const bitCapIntOcl lengthPower = pow2Ocl(bits.size());
    real1_f prob = Rand();
    std::unique_ptr<real1[]> probArray(new real1[lengthPower]);

    ProbMaskAll(regMask, probArray.get());

    bitCapIntOcl lcv = 0U;
    real1 lowerProb = probArray[0U];
    bitCapIntOcl result = lengthPower - 1U;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    real1 nrmlzr = ONE_R1;
    while ((lowerProb < prob) && (lcv < lengthPower)) {
        ++lcv;
        lowerProb += probArray[lcv];
        if (probArray[lcv] > ZERO_R1) {
            nrmlzr = probArray[lcv];
            result = lcv;
        }
    }
    if (lcv < lengthPower) {
        nrmlzr = probArray[lcv];
        result = lcv;
    }

    probArray.reset();

    bitCapIntOcl i = 0U;
    for (size_t p = 0U; p < bits.size(); ++p) {
        if (result & pow2Ocl(p)) {
            i |= (bitCapIntOcl)qPowers[p];
        }
    }
    result = i;

    qPowers.reset();

    const complex nrm = phase / (real1)(std::sqrt((real1_s)nrmlzr));

    if (doApply && ((ONE_R1 - nrmlzr) > REAL1_EPSILON)) {
        ApplyM(regMask, (bitCapInt)result, nrm);
    }

    return result;
}

void QEngine::CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return Swap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    bitCapIntOcl skipMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(skipMask | pow2Ocl(qubit1), skipMask | pow2Ocl(qubit2), pauliX, controls.size() + 2U, qPowersSorted.get(),
        false);
}

void QEngine::AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return Swap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(pow2Ocl(qubit1), pow2Ocl(qubit2), pauliX, controls.size() + 2U, qPowersSorted.get(), false);
}

void QEngine::CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return SqrtSwap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    QRACK_CONST complex sqrtX[4]{ HALF_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX };
    bitCapIntOcl skipMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(skipMask | pow2Ocl(qubit1), skipMask | pow2Ocl(qubit2), sqrtX, controls.size() + 2U, qPowersSorted.get(),
        false);
}

void QEngine::AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return SqrtSwap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    QRACK_CONST complex sqrtX[4]{ HALF_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX };
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(pow2Ocl(qubit1), pow2Ocl(qubit2), sqrtX, controls.size() + 2U, qPowersSorted.get(), false);
}

void QEngine::CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return ISqrtSwap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    QRACK_CONST complex iSqrtX[4]{ HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX, HALF_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX };
    bitCapIntOcl skipMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(skipMask | pow2Ocl(qubit1), skipMask | pow2Ocl(qubit2), iSqrtX, controls.size() + 2U, qPowersSorted.get(),
        false);
}

void QEngine::AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controls.empty()) {
        return ISqrtSwap(qubit1, qubit2);
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    QRACK_CONST complex iSqrtX[4U]{ HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX, HALF_I_HALF_CMPLX,
        HALF_NEG_I_HALF_CMPLX };
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(pow2Ocl(qubit1), pow2Ocl(qubit2), iSqrtX, controls.size() + 2U, qPowersSorted.get(), false);
}

void QEngine::ApplyControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx)
{
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 1U]);
    const bitCapIntOcl targetMask = pow2Ocl(target);
    bitCapIntOcl fullMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
        fullMask |= qPowersSorted[i];
    }
    const bitCapIntOcl controlMask = fullMask;
    qPowersSorted[controls.size()] = targetMask;
    fullMask |= targetMask;
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 1U);
    Apply2x2(controlMask, fullMask, mtrx, controls.size() + 1U, qPowersSorted.get(), false);
}

void QEngine::ApplyAntiControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx)
{
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 1U]);
    const bitCapIntOcl targetMask = pow2Ocl(target);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
    }
    qPowersSorted[controls.size()] = targetMask;
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 1U);
    Apply2x2(0U, targetMask, mtrx, controls.size() + 1U, qPowersSorted.get(), false);
}

#define _QRACK_QENGINE_SWAP_PREAMBLE()                                                                                 \
    if (qubit1 == qubit2) {                                                                                            \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    if (qubit2 < qubit1) {                                                                                             \
        std::swap(qubit1, qubit2);                                                                                     \
    }
void QEngine::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    const complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], pauliX, 2U, qPowersSorted, false);
}
void QEngine::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    const complex pauliX[4U]{ ZERO_CMPLX, I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], pauliX, 2U, qPowersSorted, false);
}
void QEngine::IISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    const complex pauliX[4U]{ ZERO_CMPLX, -I_CMPLX, -I_CMPLX, ZERO_CMPLX };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], pauliX, 2U, qPowersSorted, false);
}
void QEngine::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    QRACK_CONST complex sqrtX[4U]{ HALF_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], sqrtX, 2U, qPowersSorted, false);
}
void QEngine::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    QRACK_CONST complex iSqrtX[4U]{ HALF_NEG_I_HALF_CMPLX, HALF_I_HALF_CMPLX, HALF_I_HALF_CMPLX,
        HALF_NEG_I_HALF_CMPLX };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], iSqrtX, 2U, qPowersSorted, false);
}
void QEngine::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const real1 sinTheta = (real1)sin(theta);
    if ((sinTheta * sinTheta) > FP_NORM_EPSILON) {
        const real1 cosTheta = (real1)cos(theta);
        const complex fSimSwap[4U]{ complex(cosTheta, ZERO_R1), complex(ZERO_R1, -sinTheta),
            complex(ZERO_R1, -sinTheta), complex(cosTheta, ZERO_R1) };
        const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
        Apply2x2(qPowersSorted[0U], qPowersSorted[1U], fSimSwap, 2U, qPowersSorted, false);
    }

    const std::vector<bitLenInt> controls{ qubit1 };
    MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, -(real1)phi)), qubit2);
}

real1_f QEngine::CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target)
{
    if (controlState) {
        AntiCNOT(control, target);
    } else {
        CNOT(control, target);
    }
    const real1_f prob = Prob(target);
    if (controlState) {
        AntiCNOT(control, target);
    } else {
        CNOT(control, target);
    }
    return prob;
}

void QEngine::ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray)
{
    const bitCapIntOcl lengthMask = pow2Ocl(length) - 1U;
    std::fill(probsArray, probsArray + lengthMask + 1U, ZERO_R1);
    for (bitCapIntOcl i = 0U; i < maxQPowerOcl; ++i) {
        bitCapIntOcl reg = (i >> start) & lengthMask;
        probsArray[reg] += ProbAll(i);
    }
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceMReg(bitLenInt start, bitLenInt length, const bitCapInt& _result, bool doForce, bool doApply)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngine::ForceMReg range is out-of-bounds!");
    }

    // Single bit operations are better optimized for this special case:
    if (length == 1U) {
        return ForceM(start, bi_and_1(_result), doForce, doApply) ? ONE_BCI : ZERO_BCI;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl regMask = (lengthPower - 1U) << (bitCapIntOcl)start;
    real1 nrmlzr = ONE_R1;

    bitCapInt result = _result;
    if (doForce) {
        nrmlzr = ProbMask(regMask, result << start);
    } else {
        bitCapIntOcl lcv = 0;
        std::unique_ptr<real1[]> probArray(new real1[lengthPower]);
        ProbRegAll(start, length, probArray.get());

        const real1_f prob = Rand();
        real1_f lowerProb = ZERO_R1_F;
        result = lengthPower - 1U;

        /*
         * The value of 'lcv' should not exceed lengthPower unless the stateVec is
         * in a bug-induced topology - some value in stateVec must always be a
         * vector.
         */
        while ((lowerProb < prob) && (lcv < lengthPower)) {
            lowerProb += (real1_f)probArray[lcv];
            if (probArray[lcv] > ZERO_R1) {
                nrmlzr = probArray[lcv];
                result = lcv;
            }
            ++lcv;
        }

        probArray.reset();
    }

    if (doApply) {
        const bitCapInt resultPtr = result << start;
        const complex nrm = GetNonunitaryPhase() / (real1)(std::sqrt((real1_s)nrmlzr));
        ApplyM(regMask, resultPtr, nrm);
    }

    return result;
}

std::map<bitCapInt, int> QEngine::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
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
    std::vector<std::mt19937> genVec;
    const unsigned numThreads = GetConcurrencyLevel();
    genVec.reserve(numThreads);
    for (unsigned i = 0U; i < numThreads; ++i) {
        genVec.emplace_back(rd());
    }
    std::vector<std::map<bitCapInt, int>> resultsVec(numThreads);

    par_for(0, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) { ++(resultsVec[cpu][dist(genVec[cpu])]); });

    std::map<bitCapInt, int> results(resultsVec[0U]);
    for (size_t i = 1U; i < resultsVec.size(); ++i) {
        const std::map<bitCapInt, int>& toAdd = resultsVec[i];
        for (auto const& kv : toAdd) {
            results[kv.first] += kv.second;
        }
    }

    return results;
}

void QEngine::MultiShotMeasureMask(
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
    std::vector<std::mt19937> genVec;
    const unsigned numThreads = GetConcurrencyLevel();
    genVec.reserve(numThreads);
    for (unsigned i = 0U; i < numThreads; ++i) {
        genVec.emplace_back(rd());
    }

    par_for(0, shots,
        [&](const bitCapIntOcl& shot, const unsigned& cpu) { shotsArray[shot] = (unsigned)dist(genVec[cpu]); });
}

} // namespace Qrack
