//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qengine.hpp"

namespace Qrack {

class QHybrid;
typedef std::shared_ptr<QHybrid> QHybridPtr;

/**
 * General purpose QHybrid implementation
 */
class QHybrid : public QInterface {
protected:
    const bitLenInt MIN_OCL_QUBIT_COUNT = 6U;
    QInterfacePtr qEngine;
    QInterfaceEngine qEngineType;
    int deviceID;
    bool useRDRAND;
    bool isSparse;
    bool useHostRam;

    QInterfacePtr ConvertEngineType(
        QInterfaceEngine oQEngineType, QInterfaceEngine nQEngineType, QInterfacePtr oQEngine);

public:
    /**
     * \defgroup HybridInterface Special implementations for QHybrid.
     *
     * @{
     */

    QHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int devID = -1, bool useHardwareRNG = true, bool ignored = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored2 = {});

    QHybrid(QInterfacePtr qEngineCopy, QInterfaceEngine qEngineTypeCopy, qrack_rand_gen_ptr rgp, bool doNorm,
        bool randomGlobalPhase, bool useHostMem, int devID, bool useHardwareRNG, bool isSparseStateVec)
        : QInterface(qEngineCopy->GetQubitCount(), rgp, doNorm, useHardwareRNG, randomGlobalPhase, isSparseStateVec)
        , qEngineType(qEngineTypeCopy)
        , deviceID(devID)
        , useRDRAND(useHardwareRNG)
        , isSparse(isSparseStateVec)
        , useHostRam(useHostMem)
    {
        qEngine = qEngineCopy->Clone();
    }

    bitLenInt GetQubitCount() { return qEngine->GetQubitCount(); }

    bitCapInt GetMaxQPower() { return qEngine->GetMaxQPower(); }

    using QInterface::Compose;
    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        QHybridPtr toCopyH = std::dynamic_pointer_cast<QHybrid>(toCopy);

        QInterfaceEngine composeType = QINTERFACE_CPU;
        if (qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if (toCopyH->qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if ((qEngine->GetQubitCount() + toCopyH->GetQubitCount()) >= MIN_OCL_QUBIT_COUNT) {
            composeType = QINTERFACE_OPENCL;
        }

        qEngine = ConvertEngineType(qEngineType, composeType, qEngine);
        QInterfacePtr nCopyQEngine = ConvertEngineType(toCopyH->qEngineType, composeType, toCopyH->qEngine);

        bitLenInt toRet = qEngine->Compose(nCopyQEngine);

        qEngineType = composeType;

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        QHybridPtr toCopyH = std::dynamic_pointer_cast<QHybrid>(toCopy);

        QInterfaceEngine composeType = QINTERFACE_CPU;
        if (qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if (toCopyH->qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if ((qEngine->GetQubitCount() + toCopyH->GetQubitCount()) >= MIN_OCL_QUBIT_COUNT) {
            composeType = QINTERFACE_OPENCL;
        }

        qEngine = ConvertEngineType(qEngineType, composeType, qEngine);
        QInterfacePtr nCopyQEngine = ConvertEngineType(toCopyH->qEngineType, composeType, toCopyH->qEngine);

        bitLenInt toRet = qEngine->Compose(nCopyQEngine, start);

        qEngineType = composeType;

        return toRet;
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        QHybridPtr destH = std::dynamic_pointer_cast<QHybrid>(destH);

        QInterfaceEngine decomposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            decomposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            decomposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, destH->qEngineType, qEngine);
        qEngine->Decompose(start, length, destH->qEngine);

        if (decomposeType != destH->qEngineType) {
            qEngine = ConvertEngineType(qEngineType, decomposeType, qEngine);
        }

        qEngineType = decomposeType;
    }
    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        QInterfaceEngine disposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            disposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            disposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, disposeType, qEngine);
        qEngine->Dispose(start, length);

        qEngineType = disposeType;
    }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        QInterfaceEngine disposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            disposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            disposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, disposeType, qEngine);
        qEngine->Dispose(start, length, disposedPerm);

        qEngineType = disposeType;
    }

    virtual void SetQuantumState(const complex* inputState) { qEngine->SetQuantumState(inputState); }

    virtual void GetQuantumState(complex* outputState) { qEngine->GetQuantumState(outputState); }
    virtual void GetProbs(real1* outputProbs) { qEngine->GetProbs(outputProbs); }
    virtual complex GetAmplitude(bitCapInt perm) { return qEngine->GetAmplitude(perm); }
    virtual void SetAmplitude(bitCapInt perm, complex amp) { qEngine->SetAmplitude(perm, amp); }

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QHybrid>(toCompare));
    }
    virtual bool ApproxCompare(QHybridPtr toCompare) { return qEngine->ApproxCompare(toCompare->qEngine); }
    virtual QInterfacePtr Clone()
    {
        return std::make_shared<QHybrid>(qEngine, qEngineType, rand_generator, doNormalize, randGlobalPhase, useHostRam,
            deviceID, useRDRAND, isSparse);
    }

    /** @} */

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex)
    {
        qEngine->ApplySingleBit(mtrx, qubitIndex);
    }
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        qEngine->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        qEngine->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qEngine->Swap(qubitIndex1, qubitIndex2); }
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qEngine->ISwap(qubitIndex1, qubitIndex2); }
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qEngine->SqrtSwap(qubitIndex1, qubitIndex2); }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        qEngine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        qEngine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->AntiCSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        qEngine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return qEngine->ForceM(qubit, result, doForce, doApply);
    }

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { qEngine->ROL(shift, start, length); }
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { qEngine->INC(toAdd, start, length); }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->INCC(toAdd, start, length, carryIndex);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        qEngine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        qEngine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->INCSC(toAdd, start, length, carryIndex);
    }
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { qEngine->INCBCD(toAdd, start, length); }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->INCBCDC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->DECC(toSub, start, length, carryIndex);
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        qEngine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->DECSC(toSub, start, length, carryIndex);
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qEngine->DECBCDC(toSub, start, length, carryIndex);
    }
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qEngine->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qEngine->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        qEngine->FullAdd(inputBit1, inputBit2, carryInSumOut, carryOut);
    }
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        qEngine->IFullAdd(inputBit1, inputBit2, carryInSumOut, carryOut);
    }

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) { qEngine->ZeroPhaseFlip(start, length); }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        qEngine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        qEngine->PhaseFlipIfLess(greaterPerm, start, length);
    }
    virtual void PhaseFlip() { qEngine->PhaseFlip(); }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        qEngine->SetPermutation(perm, phaseFac);
    }
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        return qEngine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return qEngine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return qEngine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        qEngine->Hash(start, length, values);
    }
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        qEngine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1 Prob(bitLenInt qubitIndex) { return qEngine->Prob(qubitIndex); }
    virtual real1 ProbAll(bitCapInt fullRegister) { return qEngine->ProbAll(fullRegister); }
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
    {
        return qEngine->ProbReg(start, length, permutation);
    }
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        return qEngine->ProbMask(mask, permutation);
    }
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        qEngine->NormalizeState(nrm, norm_thresh);
    }
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG) { qEngine->UpdateRunningNorm(norm_thresh); }

    /** @} */
};
} // namespace Qrack
