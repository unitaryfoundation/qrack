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

#include "common/qrack_types.hpp"

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qparity.hpp"
#include "qstabilizer.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
#if ENABLE_ALU
class QStabilizerHybrid : public QAlu, public QParity, public QInterface {
#else
class QStabilizerHybrid : public QParity, public QInterface {
#endif
protected:
    std::vector<QInterfaceEngine> engineTypes;
    QInterfacePtr engine;
    QStabilizerPtr stabilizer;
    std::vector<MpsShardPtr> shards;
    int devID;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool isSparse;
    bool isDefaultPaging;
    real1_f separabilityThreshold;
    bitLenInt thresholdQubits;
    bitLenInt maxPageQubits;
    std::vector<int> deviceIDs;

    QStabilizerPtr MakeStabilizer(bitCapInt perm = 0);
    QInterfacePtr MakeEngine(bitCapInt perm = 0);

    void InvertBuffer(bitLenInt qubit)
    {
        complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
        pauliShard->Compose(shards[qubit]->gate);
        shards[qubit] = pauliShard->IsIdentity() ? NULL : pauliShard;
        stabilizer->X(qubit);
    }

    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false)
    {
        if (engine) {
            return;
        }

        if (shards[target] && shards[target]->IsInvert()) {
            InvertBuffer(target);
        }

        if (shards[control] && shards[control]->IsInvert()) {
            InvertBuffer(control);
        }

        bool isBlocked = (shards[target] && (!isPhase || !shards[target]->IsPhase()));
        isBlocked |= (shards[control] && !shards[control]->IsPhase());

        if (isBlocked) {
            SwitchToEngine();
        }
    }

    virtual bool CollapseSeparableShard(bitLenInt qubit)
    {
        MpsShardPtr shard = shards[qubit];
        shards[qubit] = NULL;
        real1_f prob;

        const bool isZ1 = stabilizer->M(qubit);

        if (isZ1) {
            prob = norm(shard->gate[3]);
        } else {
            prob = norm(shard->gate[2]);
        }

        bool result;
        if (prob <= ZERO_R1) {
            result = false;
        } else if (prob >= ONE_R1) {
            result = true;
        } else {
            result = (Rand() <= prob);
        }

        if (result != isZ1) {
            stabilizer->X(qubit);
        }

        return result;
    }

    virtual void FlushBuffers()
    {
        if (stabilizer) {
            for (bitLenInt i = 0; i < qubitCount; i++) {
                if (shards[i]) {
                    // This will call FlushBuffers() again after no longer stabilizer.
                    SwitchToEngine();
                    return;
                }
            }
        }

        if (stabilizer) {
            return;
        }

        for (bitLenInt i = 0; i < qubitCount; i++) {
            MpsShardPtr shard = shards[i];
            if (shard) {
                shards[i] = NULL;
                engine->Mtrx(shard->gate, i);
            }
        }
    }

    virtual void DumpBuffers()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            shards[i] = NULL;
        }
    }

    virtual bool TrimControls(
        const bitLenInt* lControls, bitLenInt lControlLen, std::vector<bitLenInt>& output, bool anti = false)
    {
        if (engine) {
            output.insert(output.begin(), lControls, lControls + lControlLen);
            return false;
        }

        for (bitLenInt i = 0; i < lControlLen; i++) {
            bitLenInt bit = lControls[i];

            if (!stabilizer->IsSeparableZ(bit)) {
                output.push_back(bit);
                continue;
            }

            if (shards[bit]) {
                if (shards[bit]->IsInvert()) {
                    InvertBuffer(bit);

                    if (!shards[bit]) {
                        if (anti == stabilizer->M(bit)) {
                            return true;
                        }
                        continue;
                    }
                }

                if (shards[bit]->IsPhase()) {
                    if (anti == stabilizer->M(bit)) {
                        return true;
                    }
                    continue;
                }

                output.push_back(bit);
            } else if (anti == stabilizer->M(bit)) {
                return true;
            }
        }

        return false;
    }

    virtual void CacheEigenstate(bitLenInt target);

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QStabilizerHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    virtual void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    virtual bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    virtual void Dump()
    {
        if (stabilizer) {
            stabilizer->Dump();
        } else {
            engine->Dump();
        }
    }

    virtual void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (engine) {
            SetConcurrency(GetConcurrencyLevel());
        }
    }

    virtual void TurnOnPaging()
    {
        if (engineTypes[0] == QINTERFACE_QPAGER) {
            return;
        }
        engineTypes.insert(engineTypes.begin(), QINTERFACE_QPAGER);

        if (engine) {
            QPagerPtr nEngine = std::dynamic_pointer_cast<QPager>(MakeEngine());
            nEngine->LockEngine(std::dynamic_pointer_cast<QEngine>(engine));
            engine = nEngine;
        }
    }

    virtual void TurnOffPaging()
    {
        if (engineTypes[0] != QINTERFACE_QPAGER) {
            return;
        }
        engineTypes.erase(engineTypes.begin());
        if (!engineTypes.size()) {
            engineTypes.push_back(QINTERFACE_OPTIMAL_BASE);
        }

        if (engine) {
            engine = std::dynamic_pointer_cast<QPager>(engine)->ReleaseEngine();
        }
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    virtual void SwitchToEngine();

    virtual bool isClifford() { return !engine; }

    virtual bool isClifford(bitLenInt qubit) { return !engine && !shards[qubit]; };

    virtual bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QInterface::Compose;
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy)
    {
        const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
        const bool isPaging = isDefaultPaging && (nQubits > maxPageQubits);
        bitLenInt toRet;

        if (isPaging) {
            TurnOnPaging();
        }

        if (engine) {
            if (isPaging) {
                toCopy->TurnOnPaging();
            }
            toCopy->SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else if (toCopy->engine) {
            if (isPaging) {
                toCopy->TurnOnPaging();
            }
            SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else {
            toRet = stabilizer->Compose(toCopy->stabilizer);
        }

        shards.insert(shards.end(), toCopy->shards.begin(), toCopy->shards.end());

        SetQubitCount(nQubits);

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
    {
        const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
        const bool isPaging = isDefaultPaging && (nQubits > maxPageQubits);
        bitLenInt toRet;

        if (isPaging) {
            TurnOnPaging();
        }

        if (engine) {
            if (isPaging) {
                toCopy->TurnOnPaging();
            }
            toCopy->SwitchToEngine();
            toRet = engine->Compose(toCopy->engine, start);
        } else if (toCopy->engine) {
            if (isPaging) {
                toCopy->TurnOnPaging();
            }
            SwitchToEngine();
            toRet = engine->Compose(toCopy->engine, start);
        } else {
            toRet = stabilizer->Compose(toCopy->stabilizer, start);
        }

        shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());

        SetQubitCount(nQubits);

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    virtual void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState)
    {
        FlushBuffers();

        if (stabilizer) {
            stabilizer->GetQuantumState(outputState);
        } else {
            engine->GetQuantumState(outputState);
        }
    }
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm)
    {
        FlushBuffers();

        if (stabilizer) {
            return stabilizer->GetAmplitude(perm);
        }

        return engine->GetAmplitude(perm);
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();

        engine = NULL;

        if (stabilizer) {
            stabilizer->SetPermutation(perm);
        } else {
            stabilizer = MakeStabilizer(perm);
        }
    }

    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2)
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

    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        if (shards[qubit1] && shards[qubit1]->IsInvert()) {
            InvertBuffer(qubit1);
        }

        if (shards[qubit2] && shards[qubit2]->IsInvert()) {
            InvertBuffer(qubit2);
        }

        if ((shards[qubit1] && !shards[qubit1]->IsPhase()) || (shards[qubit2] && !shards[qubit2]->IsPhase())) {
            FlushBuffers();
        }

        if (stabilizer) {
            stabilizer->ISwap(qubit1, qubit2);
        } else {
            engine->ISwap(qubit1, qubit2);
        }
    }

    virtual real1_f Prob(bitLenInt qubit);

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    virtual bitCapInt MAll();

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    virtual void Invert(complex topRight, complex bottomLeft, bitLenInt target);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MACPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MACInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);

    virtual void UniformlyControlledSingleBit(
        const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex, const complex* mtrxs)
    {
        if (stabilizer) {
            QInterface::UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs);
            return;
        }

        engine->UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs);
    }

    virtual void CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            QInterface::CSqrtSwap(controls, controlLen, qubit1, qubit2);
            return;
        }

        engine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            QInterface::AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
            return;
        }

        engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            QInterface::CISqrtSwap(controls, controlLen, qubit1, qubit2);
            return;
        }

        engine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            QInterface::AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
            return;
        }

        engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }

    virtual void XMask(bitCapInt mask)
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

    virtual void YMask(bitCapInt mask)
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

    virtual void ZMask(bitCapInt mask)
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

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots);
    virtual void MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned* shotsArray);

    virtual real1_f ProbParity(bitCapInt mask)
    {
        if (!mask) {
            return ZERO_R1;
        }

        if (!(mask & (mask - ONE_BCI))) {
            return Prob(log2(mask));
        }

        SwitchToEngine();
        return QINTERFACE_TO_QPARITY(engine)->ProbParity(mask);
    }
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
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
    virtual void UniformParityRZ(bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->UniformParityRZ(mask, angle);
    }
    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->CUniformParityRZ(controls, controlLen, mask, angle);
    }

#if ENABLE_ALU
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    virtual void X(bitLenInt q) { QInterface::X(q); }
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::DECC(toSub, start, length, carryIndex);
    }
    virtual void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls, controlLen);
    }

    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->PhaseFlipIfLess(greaterPerm, start, length);
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::INC(toAdd, start, length);
            return;
        }

        engine->INC(toAdd, start, length);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        if (stabilizer) {
            QInterface::CINC(toAdd, inOutStart, length, controls, controlLen);
            return;
        }

        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::INCS(toAdd, start, length, overflowIndex);
            return;
        }

        engine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (stabilizer) {
            QInterface::INCDECC(toAdd, start, length, carryIndex);
            return;
        }

        engine->INCDECC(toAdd, start, length, carryIndex);
    }
    virtual void INCDECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    virtual void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCBCD(toAdd, start, length);
    }
    virtual void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECBCDC(toAdd, start, length, carryIndex);
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedLDA(
            indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedADC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedSBC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->Hash(start, length, values);
    }
#endif

    virtual void PhaseFlip()
    {
        if (engine) {
            engine->PhaseFlip();
        }
    }
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }

    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    virtual real1_f ProbAll(bitCapInt fullRegister)
    {
        SwitchToEngine();
        return engine->ProbAll(fullRegister);
    }
    virtual real1_f ProbMask(bitCapInt mask, bitCapInt permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare));
    }

    virtual real1_f SumSqrDiff(QStabilizerHybridPtr toCompare)
    {
        // If the qubit counts are unequal, these can't be approximately equal objects.
        if (qubitCount != toCompare->qubitCount) {
            // Max square difference:
            return ONE_R1;
        }

        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        QStabilizerHybridPtr thatClone =
            toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;

        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        if (thatClone) {
            thatClone->SwitchToEngine();
        }

        QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
        QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

        return thisEngine->SumSqrDiff(thatEngine);
    }

    virtual bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), error_tol);
    }

    virtual bool ApproxCompare(QStabilizerHybridPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        FlushBuffers();
        toCompare->FlushBuffers();

        if (stabilizer && toCompare->stabilizer) {
            return stabilizer->ApproxCompare(toCompare->stabilizer);
        }

        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        QStabilizerHybridPtr thatClone =
            toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;

        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        if (thatClone) {
            thatClone->SwitchToEngine();
        }

        QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
        QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

        return thisEngine->ApproxCompare(thatEngine, error_tol);
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        if (engine) {
            engine->NormalizeState(nrm, norm_thresh, phaseArg);
        }
    }

    virtual real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, length, offset);
        }

        return engine->ExpectationBitsAll(bits, length, offset);
    }

    virtual bool TrySeparate(bitLenInt qubit)
    {
        if (qubitCount == 1U) {
            return true;
        }

        if (stabilizer) {
            return stabilizer->CanDecomposeDispose(qubit, 1);
        }

        return engine->TrySeparate(qubit);
    }
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubitCount == 2U) {
            return true;
        }

        if (stabilizer) {
            if (qubit2 < qubit1) {
                std::swap(qubit1, qubit2);
            }

            stabilizer->Swap(qubit1 + 1U, qubit2);

            const bool toRet = stabilizer->CanDecomposeDispose(qubit1, 2);

            stabilizer->Swap(qubit1 + 1U, qubit2);

            return toRet;
        }

        return engine->TrySeparate(qubit1, qubit2);
    }
    virtual bool TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol)
    {
        if (stabilizer) {
            std::vector<bitLenInt> q(length);
            std::copy(qubits, qubits + length, q.begin());
            std::sort(q.begin(), q.end());

            for (bitLenInt i = 1; i < length; i++) {
                Swap(q[0] + i, q[i]);
            }

            const bool toRet = stabilizer->CanDecomposeDispose(q[0], length);

            for (bitLenInt i = 1; i < length; i++) {
                Swap(q[0] + i, q[i]);
            }

            return toRet;
        }

        return engine->TrySeparate(qubits, length, error_tol);
    }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(int dID, bool forceReInit = false)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID, forceReInit);
        }
    }

    virtual int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }
};
} // namespace Qrack
