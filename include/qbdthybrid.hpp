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

#include "qinterface.hpp"

namespace Qrack {

class QBdtHybrid;
typedef std::shared_ptr<QBdtHybrid> QBdtHybridPtr;

/**
 * A "Qrack::QBdtHybrid" internally switched between Qrack::QBdt and Qrack::QHybrid to maximize
 * entanglement-dependent performance.
 */
#if ENABLE_ALU
class QBdtHybrid : public QAlu, public QParity, public QInterface {
#else
class QBdtHybrid : public QParity, public QInterface {
#endif
protected:
    bool useRDRAND;
    bool isSparse;
    bool useHostRam;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    int64_t devID;
    QBdtPtr qbdt;
    QEnginePtr engine;
    complex phaseFactor;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;

    /**
     * Switches between QBdt and QEngine modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically after every gate, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchMode(bool useBdt)
    {
        if (!engine == useBdt) {
            return;
        }

        QInterfacePtr nEngine = MakeSimulator(useBdt);
        std::unique_ptr<complex[]> sv(new complex[(bitCapIntOcl)maxQPower]);
        if (qbdt) {
            qbdt->GetQuantumState(sv.get());
        } else {
            engine->GetQuantumState(sv.get());
        }
        nEngine->SetQuantumState(sv.get());
        if (useBdt) {
            qbdt = std::dynamic_pointer_cast<QBdt>(nEngine);
            engine = nullptr;
        } else {
            qbdt = nullptr;
            engine = std::dynamic_pointer_cast<QEngine>(nEngine);
        }
    }

    void CheckThreshold()
    {
        if (qubitCount < 2U) {
            // Don't check QBdt below qubit threshold.
            return SwitchMode(true);
        }

        const bitLenInt strideBits = log2Ocl(GetConcurrencyLevel() * pow2Ocl(PSTRIDEPOW_DEFAULT));

        if (qubitCount <= strideBits) {
            // Don't check QBdt below qubit threshold.
            return SwitchMode(true);
        }

#if ENABLE_ENV_VARS
        const double threshold = getenv("QRACK_QBDT_HYBRID_THRESHOLD")
            ? std::stod(getenv("QRACK_QBDT_HYBRID_THRESHOLD"))
            : std::log2(strideBits - qubitCount);
#else
        const double threshold = std::log2(strideBits - qubitCount);
#endif

        if ((2.0 - threshold) <= FP_NORM_EPSILON) {
            // This definitely won't switch to state vector.
            return SwitchMode(true);
        }

        const size_t count = sizeof(QBdtHybrid) * qbdt->CountBranches() / sizeof(complex);
        if (count > (threshold * bi_to_double(maxQPower))) {
            SwitchMode(false);
        }
    }

public:
    QBdtHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QBdtHybrid(QBdtPtr q, QEnginePtr e, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount,
        const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QBdtHybrid(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh)
        : QBdtHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    QInterfacePtr MakeSimulator(
        bool isBdt, const bitCapInt& perm = ZERO_BCI, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (qbdt) {
            qbdt->SetConcurrency(GetConcurrencyLevel());
        } else {
            engine->SetConcurrency(GetConcurrencyLevel());
        }
    }

    real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation)
    {
        if (qbdt) {
            return qbdt->ProbReg(start, length, permutation);
        }

        return engine->ProbReg(start, length, permutation);
    }

    using QInterface::Compose;
    bitLenInt Compose(QBdtHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (engine) {
            return engine->Compose(toCopy->engine);
        }

        const bitLenInt toRet = qbdt->Compose(toCopy->qbdt);
        CheckThreshold();

        return toRet;
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QBdtHybrid>(toCopy)); }
    bitLenInt Compose(QBdtHybridPtr toCopy, bitLenInt start)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (engine) {
            return engine->Compose(toCopy->engine, start);
        }

        const bitLenInt toRet = qbdt->Compose(toCopy->qbdt, start);
        CheckThreshold();

        return toRet;
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBdtHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QBdtHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (engine) {
            return engine->ComposeNoClone(toCopy->engine);
        }

        const bitLenInt toRet = qbdt->ComposeNoClone(toCopy->qbdt);
        CheckThreshold();

        return toRet;
    }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QBdtHybrid>(toCopy));
    }
    using QInterface::Decompose;
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        SetQubitCount(qubitCount - length);
        QBdtPtr q{ nullptr };
        QEnginePtr e{ nullptr };
        if (qbdt) {
            q = std::dynamic_pointer_cast<QBdt>(qbdt->Decompose(start, length));
            CheckThreshold();
        } else {
            e = std::dynamic_pointer_cast<QEngine>(engine->Decompose(start, length));
        }
        return std::make_shared<QBdtHybrid>(q, e, engines, qubitCount, ZERO_BCI, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            thresholdQubits, separabilityThreshold);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QBdtHybrid>(dest));
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QBdtHybrid>(dest), error_tol);
    }
    bool TryDecompose(bitLenInt start, QBdtHybridPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        SwitchMode(false);
        dest->SwitchMode(false);
        if (engine->TryDecompose(start, dest->engine, error_tol)) {
            SetQubitCount(qubitCount - dest->qubitCount);

            return true;
        }

        return false;
    }
    void Decompose(bitLenInt start, QBdtHybridPtr dest)
    {
        SetQubitCount(qubitCount - dest->qubitCount);
        dest->SwitchMode(!engine);
        if (qbdt) {
            qbdt->Decompose(start, dest->qbdt);
            CheckThreshold();
        } else {
            engine->Decompose(start, dest->engine);
        }
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        SetQubitCount(qubitCount - length);
        if (qbdt) {
            qbdt->Dispose(start, length);
            CheckThreshold();
        } else {
            engine->Dispose(start, length);
        }
    }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        SetQubitCount(qubitCount - length);
        if (qbdt) {
            qbdt->Dispose(start, length, disposedPerm);
            CheckThreshold();
        } else {
            engine->Dispose(start, length, disposedPerm);
        }
    }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        if (qbdt) {
            qbdt->Allocate(start, length);
        } else {
            engine->Allocate(start, length);
        }
        SetQubitCount(qubitCount + length);

        return start;
    }

    void SetQuantumState(const complex* inputState)
    {
        if (qbdt) {
            qbdt->SetQuantumState(inputState);
        } else {
            engine->SetQuantumState(inputState);
        }
    }
    void GetQuantumState(complex* outputState)
    {
        if (qbdt) {
            qbdt->GetQuantumState(outputState);
        } else {
            engine->GetQuantumState(outputState);
        }
    }
    void GetProbs(real1* outputProbs)
    {
        if (qbdt) {
            qbdt->GetProbs(outputProbs);
        } else {
            engine->GetProbs(outputProbs);
        }
    }
    complex GetAmplitude(const bitCapInt& perm)
    {
        if (qbdt) {
            return qbdt->GetAmplitude(perm);
        }

        return engine->GetAmplitude(perm);
    }
    void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        if (qbdt) {
            qbdt->SetAmplitude(perm, amp);
        } else {
            engine->SetAmplitude(perm, amp);
        }
    }
    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        if (qbdt) {
            qbdt->SetPermutation(perm, phaseFac);
        } else {
            qbdt = std::dynamic_pointer_cast<QBdt>(MakeSimulator(true, perm, phaseFac));
            engine = nullptr;
        }
    }

    void Mtrx(const complex* mtrx, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Mtrx(mtrx, qubitIndex);
        } else {
            engine->Mtrx(mtrx, qubitIndex);
        }
    }
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Phase(topLeft, bottomRight, qubitIndex);
        } else {
            engine->Phase(topLeft, bottomRight, qubitIndex);
        }
    }
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Invert(topRight, bottomLeft, qubitIndex);
        } else {
            engine->Invert(topRight, bottomLeft, qubitIndex);
        }
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (qbdt) {
            qbdt->MCMtrx(controls, mtrx, target);
            CheckThreshold();
        } else {
            engine->MCMtrx(controls, mtrx, target);
        }
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (qbdt) {
            qbdt->MACMtrx(controls, mtrx, target);
            CheckThreshold();
        } else {
            engine->MACMtrx(controls, mtrx, target);
        }
    }

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt> mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask)
    {
        if (qbdt) {
            qbdt->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
            CheckThreshold();
        } else {
            engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
        }
    }

    void XMask(const bitCapInt& mask)
    {
        if (qbdt) {
            qbdt->XMask(mask);
        } else {
            engine->XMask(mask);
        }
    }
    void PhaseParity(real1_f radians, const bitCapInt& mask)
    {
        if (qbdt) {
            qbdt->PhaseParity(radians, mask);
        } else {
            engine->PhaseParity(radians, mask);
        }
    }

    real1_f CProb(bitLenInt control, bitLenInt target)
    {
        if (qbdt) {
            return qbdt->CProb(control, target);
        }

        return engine->CProb(control, target);
    }
    real1_f ACProb(bitLenInt control, bitLenInt target)
    {
        if (qbdt) {
            return qbdt->ACProb(control, target);
        }

        return engine->ACProb(control, target);
    }

    void UniformParityRZ(const bitCapInt& mask, real1_f angle)
    {
        if (qbdt) {
            qbdt->UniformParityRZ(mask, angle);
            CheckThreshold();
        } else {
            engine->UniformParityRZ(mask, angle);
        }
    }
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle)
    {
        if (qbdt) {
            qbdt->CUniformParityRZ(controls, mask, angle);
            CheckThreshold();
        } else {
            engine->CUniformParityRZ(controls, mask, angle);
        }
    }

    void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->CSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->CSwap(controls, qubit1, qubit2);
        }
    }
    void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->AntiCSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->AntiCSwap(controls, qubit1, qubit2);
        }
    }
    void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->CSqrtSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->CSqrtSwap(controls, qubit1, qubit2);
        }
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->AntiCSqrtSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->AntiCSqrtSwap(controls, qubit1, qubit2);
        }
    }
    void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->CISqrtSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->CISqrtSwap(controls, qubit1, qubit2);
        }
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qbdt) {
            qbdt->AntiCISqrtSwap(controls, qubit1, qubit2);
            CheckThreshold();
        } else {
            engine->AntiCISqrtSwap(controls, qubit1, qubit2);
        }
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        if (qbdt) {
            return qbdt->ForceM(qubit, result, doForce, doApply);
        }

        return engine->ForceM(qubit, result, doForce, doApply);
    }

    bitCapInt MAll()
    {
        if (qbdt) {
            return qbdt->MAll();
        }

        const bitCapInt toRet = engine->MAll();
        qbdt = std::dynamic_pointer_cast<QBdt>(MakeSimulator(true, toRet));
        engine = nullptr;

        return toRet;
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q)
    {
        if (qbdt) {
            return qbdt->M(q);
        }

        return engine->M(q);
    }
    using QInterface::X;
    void X(bitLenInt q)
    {
        if (qbdt) {
            qbdt->X(q);
        } else {
            engine->X(q);
        }
    }
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        if (qbdt) {
            qbdt->INC(toAdd, start, length);
        } else {
            engine->INC(toAdd, start, length);
        }
    }
    void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
    {
        if (qbdt) {
            qbdt->DEC(toSub, start, length);
        } else {
            engine->DEC(toSub, start, length);
        }
    }
    void CDEC(const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CDEC(toSub, inOutStart, length, controls);
            CheckThreshold();
        } else {
            engine->CDEC(toSub, inOutStart, length, controls);
        }
    }
    void INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (qbdt) {
            qbdt->INCDECC(toAdd, start, length, carryIndex);
        } else {
            engine->INCDECC(toAdd, start, length, carryIndex);
        }
    }
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CINC(toAdd, inOutStart, length, controls);
            CheckThreshold();
        } else {
            engine->CINC(toAdd, inOutStart, length, controls);
        }
    }
    void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (qbdt) {
            qbdt->INCC(toAdd, start, length, carryIndex);
            CheckThreshold();
        } else {
            engine->INCC(toAdd, start, length, carryIndex);
        }
    }
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (qbdt) {
            qbdt->INCS(toAdd, start, length, overflowIndex);
            CheckThreshold();
        } else {
            engine->INCS(toAdd, start, length, overflowIndex);
        }
    }
    void DECS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (qbdt) {
            qbdt->DECS(toAdd, start, length, overflowIndex);
            CheckThreshold();
        } else {
            engine->DECS(toAdd, start, length, overflowIndex);
        }
    }
    void INCSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (qbdt) {
            qbdt->DECC(toSub, start, length, carryIndex);
            CheckThreshold();
        } else {
            engine->DECC(toSub, start, length, carryIndex);
        }
    }
    void DECSC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    void DECSC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->DECSC(toSub, start, length, carryIndex);
    }
    void INCDECSC(
        const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchMode(false);
        engine->INCBCD(toAdd, start, length);
    }
    void INCBCDC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    void DECBCDC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchMode(false);
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        if (qbdt) {
            qbdt->MUL(toMul, inOutStart, carryStart, length);
        } else {
            engine->MUL(toMul, inOutStart, carryStart, length);
        }
    }
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        if (qbdt) {
            qbdt->DIV(toDiv, inOutStart, carryStart, length);
        } else {
            engine->DIV(toDiv, inOutStart, carryStart, length);
        }
    }
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        if (qbdt) {
            qbdt->MULModNOut(toMul, modN, inStart, outStart, length);
        } else {
            engine->MULModNOut(toMul, modN, inStart, outStart, length);
        }
    }
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        if (qbdt) {
            qbdt->IMULModNOut(toMul, modN, inStart, outStart, length);
        } else {
            engine->IMULModNOut(toMul, modN, inStart, outStart, length);
        }
    }
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchMode(false);
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CMUL(toMul, inOutStart, carryStart, length, controls);
            CheckThreshold();
        } else {
            engine->CMUL(toMul, inOutStart, carryStart, length, controls);
        }
    }
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CDIV(toDiv, inOutStart, carryStart, length, controls);
            CheckThreshold();
        } else {
            engine->CDIV(toDiv, inOutStart, carryStart, length, controls);
        }
    }
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
            CheckThreshold();
        } else {
            engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
        }
    }
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (qbdt) {
            qbdt->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
            CheckThreshold();
        } else {
            engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
        }
    }
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        SwitchMode(false);
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        SwitchMode(false);
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchMode(false);
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchMode(false);
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchMode(false);
        engine->Hash(start, length, values);
    }

    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchMode(false);
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchMode(false);
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->Swap(qubitIndex1, qubitIndex2);
        } else {
            engine->Swap(qubitIndex1, qubitIndex2);
        }
    }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->ISwap(qubitIndex1, qubitIndex2);
            CheckThreshold();
        } else {
            engine->ISwap(qubitIndex1, qubitIndex2);
        }
    }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->IISwap(qubitIndex1, qubitIndex2);
            CheckThreshold();
        } else {
            engine->IISwap(qubitIndex1, qubitIndex2);
        }
    }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->SqrtSwap(qubitIndex1, qubitIndex2);
            CheckThreshold();
        } else {
            engine->SqrtSwap(qubitIndex1, qubitIndex2);
        }
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->ISqrtSwap(qubitIndex1, qubitIndex2);
            CheckThreshold();
        } else {
            engine->ISqrtSwap(qubitIndex1, qubitIndex2);
        }
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (qbdt) {
            qbdt->FSim(theta, phi, qubitIndex1, qubitIndex2);
            CheckThreshold();
        } else {
            engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
        }
    }

    real1_f Prob(bitLenInt qubitIndex)
    {
        if (qbdt) {
            return qbdt->Prob(qubitIndex);
        }

        return engine->Prob(qubitIndex);
    }
    real1_f ProbAll(const bitCapInt& fullRegister)
    {
        const real1_f toRet = qbdt ? qbdt->ProbAll(fullRegister) : engine->ProbAll(fullRegister);
        if (toRet >= (ONE_R1_F - FP_NORM_EPSILON)) {
            SetPermutation(fullRegister);
        }
        return toRet;
    }
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        if (qbdt) {
            return qbdt->ProbMask(mask, permutation);
        }

        return engine->ProbMask(mask, permutation);
    }
    real1_f ProbParity(const bitCapInt& mask)
    {
        if (qbdt) {
            return qbdt->ProbParity(mask);
        }

        return engine->ProbParity(mask);
    }
    bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
    {
        if (qbdt) {
            return qbdt->ForceMParity(mask, result, doForce);
        }

        return engine->ForceMParity(mask, result, doForce);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QBdtHybrid>(toCompare)); }
    real1_f SumSqrDiff(QBdtHybridPtr toCompare)
    {
        toCompare->SwitchMode(!engine);
        if (qbdt) {
            return qbdt->SumSqrDiff(toCompare->qbdt);
        }

        return engine->SumSqrDiff(toCompare->engine);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (qbdt) {
            qbdt->UpdateRunningNorm(norm_thresh);
        } else {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        if (qbdt) {
            qbdt->NormalizeState(nrm, norm_thresh, phaseArg);
        } else {
            engine->NormalizeState(nrm, norm_thresh, phaseArg);
        }
    }

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (qbdt) {
            return qbdt->ExpectationBitsAll(bits, offset);
        }

        return engine->ExpectationBitsAll(bits, offset);
    }
    real1_f VarianceBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (qbdt) {
            return qbdt->VarianceBitsAll(bits, offset);
        }

        return engine->VarianceBitsAll(bits, offset);
    }

    void Finish()
    {
        if (qbdt) {
            qbdt->Finish();
        } else {
            engine->Finish();
        }
    }

    bool isFinished()
    {
        if (qbdt) {
            return qbdt->isFinished();
        }

        return engine->isFinished();
    }

    void Dump()
    {
        if (qbdt) {
            qbdt->Dump();
        } else {
            engine->Dump();
        }
    }

    QInterfacePtr Clone()
    {
        QBdtHybridPtr c = std::make_shared<QBdtHybrid>(engines, qubitCount, ZERO_BCI, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            thresholdQubits, separabilityThreshold);
        c->SetConcurrency(GetConcurrencyLevel());
        if (qbdt) {
            c->qbdt = std::dynamic_pointer_cast<QBdt>(qbdt->Clone());
        } else {
            c->SwitchMode(false);
            c->engine->CopyStateVec(engine);
        }

        return c;
    }

    void SetDevice(int64_t dID)
    {
        devID = dID;
        if (qbdt) {
            qbdt->SetDevice(dID);
        } else {
            engine->SetDevice(dID);
        }
    }

    void SetDeviceList(std::vector<int64_t> dIDs)
    {
        deviceIDs = dIDs;
        if (qbdt) {
            qbdt->SetDeviceList(dIDs);
        } else {
            engine->SetDeviceList(dIDs);
        }
    }

    int64_t GetDevice() { return devID; }
    std::vector<int64_t> GetDeviceList() { return deviceIDs; }

    bitCapIntOcl GetMaxSize()
    {
        if (qbdt) {
            return qbdt->GetMaxSize();
        }

        return engine->GetMaxSize();
    };
};
} // namespace Qrack
