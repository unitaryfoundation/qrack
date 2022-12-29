//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qstabilizer.hpp"

#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QStabilizer and Qrack::QEngine to maximize
 * performance.
 */
#if ENABLE_ALU
class QStabilizerHybrid : public QAlu, public QParity, public QInterface {
#else
class QStabilizerHybrid : public QParity, public QInterface {
#endif
protected:
    bool useHostRam;
    bool doNormalize;
    bool isSparse;
    bool useTGadget;
    bitLenInt thresholdQubits;
    bitLenInt ancillaCount;
    bitLenInt maxQubitPlusAncillaCount;
    real1_f separabilityThreshold;
    int64_t devID;
    complex phaseFactor;
    QInterfacePtr engine;
    QStabilizerPtr stabilizer;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engineTypes;
    std::vector<QInterfaceEngine> cloneEngineTypes;
    std::vector<MpsShardPtr> shards;

    QStabilizerPtr MakeStabilizer(bitCapInt perm = 0U);
    QInterfacePtr MakeEngine(bitCapInt perm = 0U);
    QInterfacePtr MakeEngine(bitCapInt perm, bitLenInt qbCount);

    void InvertBuffer(bitLenInt qubit);
    void FlushH(bitLenInt qubit);
    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false);
    bool CollapseSeparableShard(bitLenInt qubit);
    bool TrimControls(const std::vector<bitLenInt>& lControls, std::vector<bitLenInt>& output, bool anti = false);
    void CacheEigenstate(bitLenInt target);
    void FlushBuffers();
    void DumpBuffers()
    {
        for (size_t i = 0; i < shards.size(); ++i) {
            shards[i] = NULL;
        }
    }
    bool IsBuffered()
    {
        for (size_t i = 0U; i < shards.size(); ++i) {
            if (shards[i]) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsProbBuffered()
    {
        for (size_t i = 0U; i < shards.size(); ++i) {
            MpsShardPtr shard = shards[i];
            if (shard && !((norm(shard->gate[1]) <= FP_NORM_EPSILON) && (norm(shard->gate[2]) <= FP_NORM_EPSILON))) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }

    real1_f ApproxCompareHelper(
        QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);
    void ISwapHelper(bitLenInt qubit1, bitLenInt qubit2, bool inverse);

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QStabilizerHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    void SetTInjection(bool useGadget) { useTGadget = useGadget; }
    bool GetTInjection() { return useTGadget; }

    void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    void Dump()
    {
        if (stabilizer) {
            stabilizer->Dump();
        } else {
            engine->Dump();
        }
    }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (engine) {
            SetConcurrency(GetConcurrencyLevel());
        }
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchToEngine();

    bool isClifford() { return !engine; }

    bool isClifford(bitLenInt qubit) { return !engine && !shards[qubit]; };

    bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QInterface::Compose;
    bitLenInt Compose(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, false); };
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy)); }
    bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, true); };
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    bitLenInt ComposeEither(QStabilizerHybridPtr toCopy, bool willDestroy);
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(bitCapInt perm);
    void SetQuantumState(complex const* inputState);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();

        engine = NULL;

        if (stabilizer && !ancillaCount) {
            stabilizer->SetPermutation(perm);
        } else {
            ancillaCount = 0U;
            stabilizer = MakeStabilizer(perm);
        }
    }

    void Swap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        std::swap(shards[qubit1], shards[qubit2]);

        if (stabilizer) {
            stabilizer->Swap(qubit1, qubit2);
        } else {
            engine->Swap(qubit1, qubit2);
        }
    }

    void ISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, false); }
    void IISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, true); }

    real1_f Prob(bitLenInt qubit);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll();

    void Mtrx(complex const* mtrx, bitLenInt target);
    void MCMtrx(const std::vector<bitLenInt>& controls, complex const* mtrx, bitLenInt target);
    void MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target);
    void MACMtrx(const std::vector<bitLenInt>& controls, complex const* mtrx, bitLenInt target);
    void MACPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target);
    void MACInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target);

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(
        const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, complex const* mtrxs)
    {
        if (stabilizer) {
            QInterface::UniformlyControlledSingleBit(controls, qubitIndex, mtrxs);
            return;
        }

        engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs);
    }

    void CSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, controls, false)) {
                return;
            }
            if (!controls.size()) {
                stabilizer->Swap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CSwap(lControls, qubit1, qubit2);
    }
    void CSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, controls, false)) {
                return;
            }
            if (!controls.size()) {
                QInterface::SqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CSqrtSwap(lControls, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, controls, true)) {
                return;
            }
            if (!controls.size()) {
                QInterface::SqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->AntiCSqrtSwap(lControls, qubit1, qubit2);
    }
    void CISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, controls, false)) {
                return;
            }
            if (!controls.size()) {
                QInterface::ISqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CISqrtSwap(lControls, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, controls, true)) {
                return;
            }
            if (!controls.size()) {
                QInterface::ISqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->AntiCISqrtSwap(lControls, qubit1, qubit2);
    }

    void XMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->XMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            X(log2(mask ^ v));
            mask = v;
        }
    }

    void YMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->YMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            Y(log2(mask ^ v));
            mask = v;
        }
    }

    void ZMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->ZMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            Z(log2(mask ^ v));
            mask = v;
        }
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);
    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    real1_f ProbParity(bitCapInt mask)
    {
        if (!mask) {
            return ZERO_R1_F;
        }

        if (!(mask & (mask - ONE_BCI))) {
            return Prob(log2(mask));
        }

        SwitchToEngine();
        return QINTERFACE_TO_QPARITY(engine)->ProbParity(mask);
    }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        // If no bits in mask:
        if (!mask) {
            return false;
        }

        // If only one bit in mask:
        if (!(mask & (mask - ONE_BCI))) {
            return ForceM(log2(mask), result, doForce);
        }

        SwitchToEngine();
        return QINTERFACE_TO_QPARITY(engine)->ForceMParity(mask, result, doForce);
    }
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->CUniformParityRZ(controls, mask, angle);
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }
    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->PhaseFlipIfLess(greaterPerm, start, length);
    }

    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::INC(toAdd, start, length);
            return;
        }

        engine->INC(toAdd, start, length);
    }
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::DEC(toSub, start, length);
            return;
        }

        engine->DEC(toSub, start, length);
    }
    void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::DECS(toSub, start, length, overflowIndex);
            return;
        }

        engine->DECS(toSub, start, length, overflowIndex);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (stabilizer) {
            QInterface::CINC(toAdd, inOutStart, length, controls);
            return;
        }

        engine->CINC(toAdd, inOutStart, length, controls);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::INCS(toAdd, start, length, overflowIndex);
            return;
        }

        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (stabilizer) {
            QInterface::INCDECC(toAdd, start, length, carryIndex);
            return;
        }

        engine->INCDECC(toAdd, start, length, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCBCD(toAdd, start, length);
    }
    void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECBCDC(toAdd, start, length, carryIndex);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedLDA(
            indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedADC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedSBC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->Hash(start, length, values);
    }
#endif

    void PhaseFlip()
    {
        if (stabilizer) {
            stabilizer->PhaseFlip();
        } else {
            engine->PhaseFlip();
        }
    }
    void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }

    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::SqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::ISqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f ProbMask(bitCapInt mask, bitCapInt permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), false);
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >=
            ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), true, error_tol);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset = 0)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, offset);
        }

        return engine->ExpectationBitsAll(bits, offset);
    }

    bool TrySeparate(bitLenInt qubit);
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);
    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol);

    QInterfacePtr Clone();

    void SetDevice(int64_t dID)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID);
        }
    }

    int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }
};
} // namespace Qrack
