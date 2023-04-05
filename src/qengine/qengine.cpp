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

#include "qengine.hpp"

#include <algorithm>

namespace Qrack {

void QEngine::Mtrx(complex const* mtrx, bitLenInt qubit)
{
    if (IsIdentity(mtrx, false)) {
        return;
    }

    const bitCapIntOcl qPowers[1U]{ pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0U], mtrx, 1U, qPowers, doNormalize && !(IsPhase(mtrx) || IsInvert(mtrx)));
}

void QEngine::EitherMtrx(const std::vector<bitLenInt>& controls, complex const* mtrx, bitLenInt target, bool isAnti)
{
    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
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

    for (size_t i = 0U; i < bits.size(); ++i) {
        if (bits[i] >= qubitCount) {
            throw std::invalid_argument(
                "QEngine::ForceM qubit index parameter array values must be within allocated qubit bounds!");
        }
    }

    // Single bit operations are better optimized for this special case:
    if (bits.size() == 1U) {
        if (!values.size()) {
            if (M(bits[0U])) {
                return pow2(bits[0U]);
            } else {
                return 0U;
            }
        } else {
            if (ForceM(bits[0U], values[0U], true, doApply)) {
                return pow2(bits[0U]);
            } else {
                return 0U;
            }
        }
    }

    std::unique_ptr<bitCapInt[]> qPowers(new bitCapInt[bits.size()]);
    bitCapInt regMask = 0U;
    for (bitCapIntOcl i = 0U; i < bits.size(); ++i) {
        qPowers[i] = pow2(bits[i]);
        regMask |= qPowers[i];
    }
    std::sort(qPowers.get(), qPowers.get() + bits.size());

    const complex phase = GetNonunitaryPhase();
    real1 nrmlzr = ONE_R1;
    bitCapInt result;
    complex nrm;
    if (values.size()) {
        bitCapInt result = 0U;
        for (size_t j = 0U; j < bits.size(); ++j) {
            result |= values[j] ? pow2(bits[j]) : 0U;
        }
        nrmlzr = ProbMask(regMask, result);
        nrm = phase / (real1)(std::sqrt((real1_s)nrmlzr));
        if (nrmlzr != ONE_R1) {
            ApplyM(regMask, result, nrm);
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
    result = lengthPower - ONE_BCI;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
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
        if (pow2(p) & result) {
            i |= (bitCapIntOcl)qPowers[p];
        }
    }
    result = i;

    qPowers.reset();

    nrm = phase / (real1)(std::sqrt((real1_s)nrmlzr));

    if (doApply && ((ONE_R1 - nrmlzr) > REAL1_EPSILON)) {
        ApplyM(regMask, result, nrm);
    }

    return result;
}

void QEngine::CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (!controls.size()) {
        Swap(qubit1, qubit2);
        return;
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
    if (!controls.size()) {
        Swap(qubit1, qubit2);
        return;
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
    if (!controls.size()) {
        SqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex sqrtX[4]{ complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f,
        complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f };
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
    if (!controls.size()) {
        SqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex sqrtX[4]{ complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f,
        complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f };
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
    if (!controls.size()) {
        ISqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex iSqrtX[4]{ complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f,
        complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f };
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
    if (!controls.size()) {
        ISqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const complex iSqrtX[4U]{ complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f,
        complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f };
    std::unique_ptr<bitCapIntOcl[]> qPowersSorted(new bitCapIntOcl[controls.size() + 2U]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowersSorted[i] = pow2Ocl(controls[i]);
    }
    qPowersSorted[controls.size()] = pow2Ocl(qubit1);
    qPowersSorted[controls.size() + 1U] = pow2Ocl(qubit2);
    std::sort(qPowersSorted.get(), qPowersSorted.get() + controls.size() + 2U);
    Apply2x2(pow2Ocl(qubit1), pow2Ocl(qubit2), iSqrtX, controls.size() + 2U, qPowersSorted.get(), false);
}

void QEngine::ApplyControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, complex const* mtrx)
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

void QEngine::ApplyAntiControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, complex const* mtrx)
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
    const complex sqrtX[4U]{ complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f,
        complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f };
    const bitCapIntOcl qPowersSorted[2U]{ pow2Ocl(qubit1), pow2Ocl(qubit2) };
    Apply2x2(qPowersSorted[0U], qPowersSorted[1U], sqrtX, 2U, qPowersSorted, false);
}
void QEngine::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    _QRACK_QENGINE_SWAP_PREAMBLE()
    const complex iSqrtX[4U]{ complex(ONE_R1, -ONE_R1) / (real1)2.0f, complex(ONE_R1, ONE_R1) / (real1)2.0f,
        complex(ONE_R1, ONE_R1) / (real1)2.0f, complex(ONE_R1, -ONE_R1) / (real1)2.0f };
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
    MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
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
    const bitCapIntOcl lengthMask = pow2Ocl(length) - ONE_BCI;
    std::fill(probsArray, probsArray + lengthMask + ONE_BCI, ZERO_R1);
    for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
        bitCapIntOcl reg = (i >> start) & lengthMask;
        probsArray[reg] += ProbAll(i);
    }
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngine::ForceMReg range is out-of-bounds!");
    }

    // Single bit operations are better optimized for this special case:
    if (length == 1U) {
        if (ForceM(start, ((bitCapIntOcl)result) & ONE_BCI, doForce, doApply)) {
            return ONE_BCI;
        } else {
            return 0U;
        }
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl regMask = (lengthPower - ONE_BCI) << (bitCapIntOcl)start;
    real1 nrmlzr = ONE_R1;

    if (doForce) {
        nrmlzr = ProbMask(regMask, result << (bitCapIntOcl)start);
    } else {
        bitCapIntOcl lcv = 0;
        std::unique_ptr<real1[]> probArray(new real1[lengthPower]);
        ProbRegAll(start, length, probArray.get());

        const real1_f prob = Rand();
        real1_f lowerProb = ZERO_R1_F;
        result = lengthPower - ONE_BCI;

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
        const bitCapInt resultPtr = result << (bitCapIntOcl)start;
        const complex nrm = GetNonunitaryPhase() / (real1)(std::sqrt((real1_s)nrmlzr));
        ApplyM(regMask, resultPtr, nrm);
    }

    return result;
}

} // namespace Qrack
