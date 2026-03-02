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
#pragma once

#include "qengine.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error OpenCL or CUDA has not been enabled
#endif

#if ENABLE_OPENCL
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#else
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

namespace Qrack {

class QHybrid;
typedef std::shared_ptr<QHybrid> QHybridPtr;

/**
 * A "Qrack::QHybrid" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QHybrid : public QEngine {
protected:
    bool isGpu;
    bool isPager;
    bool useRDRAND;
    bool isSparse;
    bitLenInt gpuThresholdQubits;
    bitLenInt pagerThresholdQubits;
    real1_f separabilityThreshold;
    int64_t devID;
    QEnginePtr engine;
    complex phaseFactor;
    std::vector<int64_t> deviceIDs;

    using QEngine::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QHybrid>(orig)); }
    void Copy(QHybridPtr orig)
    {
        QEngine::Copy(std::dynamic_pointer_cast<QEngine>(orig));
        isGpu = orig->isGpu;
        isPager = orig->isPager;
        useRDRAND = orig->useRDRAND;
        isSparse = orig->isSparse;
        gpuThresholdQubits = orig->gpuThresholdQubits;
        pagerThresholdQubits = orig->pagerThresholdQubits;
        separabilityThreshold = orig->separabilityThreshold;
        devID = orig->devID;
        engine = orig->engine;
        phaseFactor = orig->phaseFactor;
        deviceIDs = orig->deviceIDs;
    }

public:
    QHybrid(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f ignored2 = _qrack_qunit_sep_thresh);

    void SetQubitCount(bitLenInt qb)
    {
        const bool isHigher = qb > qubitCount;
        if (isHigher) {
            SwitchModes(qb >= gpuThresholdQubits, qb > pagerThresholdQubits);
        }
        QEngine::SetQubitCount(qb);
        if (!isHigher) {
            SwitchModes(qb >= gpuThresholdQubits, qb > pagerThresholdQubits);
        }

        if (engine->IsZeroAmplitude()) {
            engine->SetQubitCount(qb);
        }
    }

    QEnginePtr MakeEngine(bool isOpenCL);

    bool isOpenCL() { return isGpu; }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        engine->SetConcurrency(GetConcurrencyLevel());
    }

    /**
     * Switches between CPU and GPU modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchGpuMode(bool useGpu)
    {
        QEnginePtr nEngine{ nullptr };
        if (!isGpu && useGpu) {
            nEngine = MakeEngine(true);
        } else if (isGpu && !useGpu) {
            nEngine = MakeEngine(false);
        }

        if (nEngine) {
            nEngine->CopyStateVec(engine);
            engine = nEngine;
        }

        isGpu = useGpu;
    }

    /**
     * Switches between paged and non-paged modes. (This will not incur a performance penalty, if the chosen mode
     * matches the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and
     * Decompose() might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchPagerMode(bool usePager)
    {
        if (!isPager && usePager) {
            std::vector<QInterfaceEngine> engines{ isGpu ? QRACK_GPU_ENGINE : QINTERFACE_CPU };
            engine = std::make_shared<QPager>(engine, engines, qubitCount, ZERO_BCI, rand_generator, phaseFactor,
                doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
                deviceIDs, 0U, separabilityThreshold);
        } else if (isPager && !usePager) {
            engine = std::dynamic_pointer_cast<QPager>(engine)->ReleaseEngine();
        }

        isPager = usePager;
    }

    void SwitchModes(bool useGpu, bool usePager)
    {
        if (!usePager) {
            SwitchPagerMode(false);
        }
        SwitchGpuMode(useGpu);
        if (usePager) {
            SwitchPagerMode(true);
        }
    }

    real1_f GetRunningNorm() { return engine->GetRunningNorm(); }

    void ZeroAmplitudes() { engine->ZeroAmplitudes(); }

    bool IsZeroAmplitude() { return engine->IsZeroAmplitude(); }

    real1_f FirstNonzeroPhase() { return engine->FirstNonzeroPhase(); }

    void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QHybrid>(src)); }
    void CopyStateVec(QHybridPtr src)
    {
        SwitchModes(src->isGpu, src->isPager);
        engine->CopyStateVec(src->engine);
    }

    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        engine->GetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        engine->SetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(QHybridPtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        pageEnginePtr->SwitchModes(isGpu, isPager);
        engine->SetAmplitudePage(pageEnginePtr->engine, srcOffset, dstOffset, length);
    }
    void SetAmplitudePage(QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QHybrid>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    void ShuffleBuffers(QEnginePtr oEngine) { ShuffleBuffers(std::dynamic_pointer_cast<QHybrid>(oEngine)); }
    void ShuffleBuffers(QHybridPtr oEngine)
    {
        oEngine->SwitchModes(isGpu, isPager);
        engine->ShuffleBuffers(oEngine->engine);
    }
    void QueueSetDoNormalize(bool doNorm) { engine->QueueSetDoNormalize(doNorm); }
    void QueueSetRunningNorm(real1_f runningNrm) { engine->QueueSetRunningNorm(runningNrm); }

    using QEngine::ApplyM;
    void ApplyM(const bitCapInt& regMask, const bitCapInt& result, const complex& nrm)
    {
        engine->ApplyM(regMask, result, nrm);
    }
    real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation)
    {
        return engine->ProbReg(start, length, permutation);
    }
    void ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray)
    {
        engine->ProbRegAll(start, length, probsArray);
    }

    using QEngine::Compose;
    bitLenInt Compose(QHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchModes(isGpu, isPager);
        return engine->Compose(toCopy->engine);
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QHybrid>(toCopy)); }
    bitLenInt Compose(QHybridPtr toCopy, bitLenInt start)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchModes(isGpu, isPager);
        return engine->Compose(toCopy->engine, start);
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchModes(isGpu, isPager);
        return engine->ComposeNoClone(toCopy->engine);
    }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QHybrid>(toCopy));
    }
    using QEngine::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest) { Decompose(start, std::dynamic_pointer_cast<QHybrid>(dest)); }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QHybrid>(dest), error_tol);
    }
    void Decompose(bitLenInt start, QHybridPtr dest)
    {
        dest->SwitchModes(isGpu, isPager);
        engine->Decompose(start, dest->engine);
        SetQubitCount(qubitCount - dest->GetQubitCount());
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        engine->Dispose(start, length);
        SetQubitCount(qubitCount - length);
    }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        engine->Dispose(start, length, disposedPerm);
        SetQubitCount(qubitCount - length);
    }

    using QEngine::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        QHybridPtr nQubits = std::make_shared<QHybrid>(length, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            gpuThresholdQubits, separabilityThreshold);
        nQubits->SetConcurrency(GetConcurrencyLevel());

        return Compose(nQubits, start);
    }

    bool TryDecompose(bitLenInt start, QHybridPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        const bitLenInt nQubitCount = qubitCount - dest->GetQubitCount();
        SwitchModes(nQubitCount >= gpuThresholdQubits, nQubitCount > pagerThresholdQubits);
        dest->SwitchModes(isGpu, isPager);
        const bool result = engine->TryDecompose(start, dest->engine, error_tol);
        if (result) {
            SetQubitCount(nQubitCount);
        } else {
            SwitchModes(qubitCount >= gpuThresholdQubits, qubitCount > pagerThresholdQubits);
        }
        return result;
    }

    void SetQuantumState(const complex* inputState) { engine->SetQuantumState(inputState); }
    void GetQuantumState(complex* outputState) { engine->GetQuantumState(outputState); }
    void GetProbs(real1* outputProbs) { engine->GetProbs(outputProbs); }
    complex GetAmplitude(const bitCapInt& perm) { return engine->GetAmplitude(perm); }
    void SetAmplitude(const bitCapInt& perm, const complex& amp) { engine->SetAmplitude(perm, amp); }
    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        engine->SetPermutation(perm, phaseFac);
    }

    using QEngine::X;
    void X(bitLenInt target) { engine->X(target); }
    using QEngine::Z;
    void Z(bitLenInt target) { engine->Z(target); }
    using QEngine::Invert;
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubitIndex)
    {
        engine->Invert(topRight, bottomLeft, qubitIndex);
    }
    using QEngine::Phase;
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubitIndex)
    {
        engine->Phase(topLeft, bottomRight, qubitIndex);
    }
    void XMask(const bitCapInt& mask) { engine->XMask(mask); }
    void PhaseParity(real1_f radians, const bitCapInt& mask) { engine->PhaseParity(radians, mask); }
    void PhaseRootNMask(bitLenInt n, const bitCapInt& mask) { engine->PhaseRootNMask(n, mask); }
    void Mtrx(const complex* mtrx, bitLenInt qubitIndex) { engine->Mtrx(mtrx, qubitIndex); }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MCMtrx(controls, mtrx, target);
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MACMtrx(controls, mtrx, target);
    }

    using QEngine::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt> mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask)
    {
        engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
    }

    real1_f CProb(bitLenInt control, bitLenInt target) { return engine->CProb(control, target); }
    real1_f ACProb(bitLenInt control, bitLenInt target) { return engine->ACProb(control, target); }

    void UniformParityRZ(const bitCapInt& mask, real1_f angle) { engine->UniformParityRZ(mask, angle); }
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle)
    {
        engine->CUniformParityRZ(controls, mask, angle);
    }

    void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSwap(controls, qubit1, qubit2);
    }
    void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSwap(controls, qubit1, qubit2);
    }
    void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSqrtSwap(controls, qubit1, qubit2);
    }
    void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CISqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCISqrtSwap(controls, qubit1, qubit2);
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return engine->ForceM(qubit, result, doForce, doApply);
    }
    bitCapInt MAll() { return engine->MAll(); }

    void ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { engine->ROL(shift, start, length); }

#if ENABLE_ALU
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length) { engine->INC(toAdd, start, length); }
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        engine->CINC(toAdd, inOutStart, length, controls);
    }
    void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCC(toAdd, start, length, carryIndex);
    }
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECC(toSub, start, length, carryIndex);
    }
    void DECSC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    void DECSC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length) { engine->INCBCD(toAdd, start, length); }
    void INCBCDC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    void DECBCDC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values) { engine->Hash(start, length, values); }

    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length)
    {
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->Swap(qubitIndex1, qubitIndex2); }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->ISwap(qubitIndex1, qubitIndex2); }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->IISwap(qubitIndex1, qubitIndex2); }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->SqrtSwap(qubitIndex1, qubitIndex2); }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->ISqrtSwap(qubitIndex1, qubitIndex2); }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f Prob(bitLenInt qubitIndex) { return engine->Prob(qubitIndex); }
    real1_f CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target)
    {
        return engine->CtrlOrAntiProb(controlState, control, target);
    }
    real1_f ProbAll(const bitCapInt& fullRegister) { return engine->ProbAll(fullRegister); }
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        return engine->ProbMask(mask, permutation);
    }
    real1_f ProbParity(const bitCapInt& mask) { return engine->ProbParity(mask); }
    bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
    {
        return engine->ForceMParity(mask, result, doForce);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QHybrid>(toCompare)); }
    real1_f SumSqrDiff(QHybridPtr toCompare)
    {
        toCompare->SwitchModes(isGpu, isPager);
        return engine->SumSqrDiff(toCompare->engine);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return engine->ExpectationBitsAll(bits, offset);
    }

    void Finish() { engine->Finish(); }

    bool isFinished() { return engine->isFinished(); }

    void Dump() { engine->Dump(); }

    QInterfacePtr Clone()
    {
        QHybridPtr c = std::make_shared<QHybrid>(qubitCount, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            gpuThresholdQubits, separabilityThreshold);
        c->runningNorm = runningNorm;
        c->SetConcurrency(GetConcurrencyLevel());
        c->engine->CopyStateVec(engine);
        return c;
    }
    QEnginePtr CloneEmpty()
    {
        QHybridPtr c = std::make_shared<QHybrid>(0U, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            gpuThresholdQubits, separabilityThreshold);
        c->SetQubitCount(qubitCount);
        return c;
    }
    QInterfacePtr Copy()
    {
        QHybridPtr c = std::dynamic_pointer_cast<QHybrid>(CloneEmpty());
        c->engine = std::dynamic_pointer_cast<QEngine>(engine->Copy());
        return c;
    }

    void SetDevice(int64_t dID)
    {
        devID = dID;
        engine->SetDevice(dID);
    }

    void SetDeviceList(std::vector<int64_t> dIDs)
    {
        deviceIDs = dIDs;
        engine->SetDeviceList(dIDs);
    }

    int64_t GetDevice() { return devID; }
    std::vector<int64_t> GetDeviceList() { return deviceIDs; }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };

protected:
    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        return engine->GetExpectation(valueStart, valueLength);
    }

    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        engine->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    void ApplyControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx)
    {
        engine->ApplyControlled2x2(controls, target, mtrx);
    }
    void ApplyAntiControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx)
    {
        engine->ApplyAntiControlled2x2(controls, target, mtrx);
    }

#if ENABLE_ALU
    void INCDECC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCDECC(toMod, inOutStart, length, carryIndex);
    }
    void INCDECSC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCDECSC(toMod, inOutStart, length, carryIndex);
    }
    void INCDECSC(
        const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex);
    }
#if ENABLE_BCD
    void INCDECBCDC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCDECBCDC(toMod, inOutStart, length, carryIndex);
    }
#endif
#endif
};
} // namespace Qrack
