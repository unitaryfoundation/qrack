//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QTensorNetwork is a gate-based QInterface descendant wrapping cuQuantum.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qcircuit.hpp"

namespace Qrack {

class QTensorNetwork;
typedef std::shared_ptr<QTensorNetwork> QTensorNetworkPtr;

// #if ENABLE_CUDA
// struct TensorMeta {
//     std::vector<std::vector<int32_t>> modes;
//     std::vector<std::vector<int64_t>> extents;
// };
// typedef std::vector<TensorMeta> TensorNetworkMeta;
// typedef std::shared_ptr<TensorNetworkMeta> TensorNetworkMetaPtr;
// #endif

class QTensorNetwork : public QAlu, public QInterface {
protected:
    bool useHostRam;
    bool isSparse;
    bool isReactiveSeparate;
    bool useTGadget;
    bool isNearClifford;
#if ENABLE_OPENCL || ENABLE_CUDA
    bool isCpu;
#endif
#if ENABLE_QBDT
    bool isQBdt;
#endif
    int64_t devID;
    size_t aceMb;
    bitLenInt aceQubits;
    bitLenInt qbThreshold;
    real1_f separabilityThreshold;
    real1_f ncrp;
    QInterfacePtr layerStack;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;
    QCircuitPtr circuit;

    void CheckQubitCount(bitLenInt target)
    {
        if (target >= qubitCount) {
            throw std::invalid_argument("QTensorNetwork qubit index values must be within allocated qubit bounds!");
        }
    }

    void CheckQubitCount(bitLenInt target, const std::vector<bitLenInt>& controls)
    {
        CheckQubitCount(target);
        ThrowIfQbIdArrayIsBad(
            controls, qubitCount, "QTensorNetwork qubit index values must be within allocated qubit bounds!");
    }

    void MakeLayerStack();

    template <typename Fn> void RunAsAmplitudes(Fn fn, std::set<bitLenInt> qubits = std::set<bitLenInt>())
    {
        if (qubits.size()) {
            circuit->RemovePastLightCone(qubits)->Run(layerStack);
        } else {
            circuit->Run(layerStack);
            circuit = std::make_shared<QCircuit>(true, isNearClifford);
        }

        return fn(layerStack);
    }

public:
    QTensorNetwork(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QTensorNetwork(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh)
        : QTensorNetwork({}, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
    {
    }

    void SetSdrp(real1_f sdrp)
    {
        separabilityThreshold = sdrp;
        isReactiveSeparate = (separabilityThreshold > FP_NORM_EPSILON_F);
        layerStack->SetSdrp(sdrp);
    }

    void SetReactiveSeparate(bool isAggSep) { isReactiveSeparate = isAggSep; }

    void SetTInjection(bool useGadget)
    {
        useTGadget = useGadget;
        layerStack->SetTInjection(useTGadget);
    }

    void SetNcrp(real1_f rp)
    {
        ncrp = rp;
        layerStack->SetNcrp(ncrp);
    }

    void SetAceMaxQubits(bitLenInt qb)
    {
        aceQubits = qb;
        layerStack->SetAceMaxQubits(aceQubits);
    }

    void SetSparseAceMaxMb(size_t mb)
    {
        aceMb = mb;
        layerStack->SetSparseAceMaxMb(aceMb);
    }

    double GetUnitaryFidelity()
    {
        double toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->GetUnitaryFidelity(); });
        return toRet;
    }

    bitCapInt HighestProbAll()
    {
        bitCapInt toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->HighestProbAll(); });
        return toRet;
    }

    void SetDevice(int64_t dID) { devID = dID; }
    void SetDeviceList(std::vector<int64_t> dIDs) { deviceIDs = dIDs; }
    int64_t GetDevice() { return devID; }
    std::vector<int64_t> GetDeviceList() { return deviceIDs; }

    void Finish() { layerStack->Finish(); };

    bool isFinished() { return layerStack->isFinished(); }

    void Dump() { layerStack->Dump(); }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { layerStack->UpdateRunningNorm(norm_thresh); }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        layerStack->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QTensorNetwork>(toCompare));
    }
    real1_f SumSqrDiff(QTensorNetworkPtr toCompare)
    {
        real1_f toRet;
        toCompare->RunAsAmplitudes([&](QInterfacePtr ls) {});
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->SumSqrDiff(toCompare->layerStack); });
        return toRet;
    }

    void SetPermutation(const bitCapInt& initState, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        circuit = std::make_shared<QCircuit>(true, isNearClifford);
        MakeLayerStack();
        layerStack->SetPermutation(initState, phaseFac);
    }

    QInterfacePtr Clone();

    void LossySaveStateVector(std::string f, int p = 6, int b = 4)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->LossySaveStateVector(f, p, b); });
    }
    void LossyLoadStateVector(std::string f)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->LossyLoadStateVector(f); });
    }

    void GetQuantumState(complex* state)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->GetQuantumState(state); });
    }
    void SetQuantumState(const complex* state)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->SetQuantumState(state); });
    }
    void SetQuantumState(QInterfacePtr eng)
    {
        const QTensorNetworkPtr tnEng = std::dynamic_pointer_cast<QTensorNetwork>(eng);

        if (tnEng->qubitCount != qubitCount) {
            throw std::invalid_argument("QTensorNetwork::SetQuantumState() argument must match in qubit count!");
        }

        layerStack = tnEng->layerStack->Clone();
        circuit = tnEng->circuit->Clone();
    }
    void GetProbs(real1* outputProbs)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->GetProbs(outputProbs); });
    }

    complex GetAmplitude(const bitCapInt& perm)
    {
        complex toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->GetAmplitude(perm); });
        return toRet;
    }
    void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->SetAmplitude(perm, amp); });
    }

    using QInterface::Compose;
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        bitLenInt toRet;
        std::dynamic_pointer_cast<QTensorNetwork>(toCopy)->RunAsAmplitudes([&](QInterfacePtr ls) {});
        RunAsAmplitudes([&](QInterfacePtr ls) {
            toRet = ls->Compose(std::dynamic_pointer_cast<QTensorNetwork>(toCopy)->layerStack, start);
        });
        SetQubitCount(qubitCount + toCopy->GetQubitCount());
        return toRet;
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        dest->SetPermutation(0U);
        std::dynamic_pointer_cast<QTensorNetwork>(dest)->MakeLayerStack();
        RunAsAmplitudes([&](QInterfacePtr ls) {
            ls->Decompose(start, std::dynamic_pointer_cast<QTensorNetwork>(dest)->layerStack);
        });
        SetQubitCount(qubitCount - dest->GetQubitCount());
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->Dispose(start, length); });
        SetQubitCount(qubitCount - length);
    }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->Dispose(start, length, disposedPerm); });
        SetQubitCount(qubitCount - length);
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        bool toRet;
        dest->SetPermutation(0U);
        RunAsAmplitudes([&](QInterfacePtr ls) {
            toRet = ls->TryDecompose(start, std::dynamic_pointer_cast<QTensorNetwork>(dest)->layerStack, error_tol);
        });
        SetQubitCount(qubitCount - dest->GetQubitCount());
        return toRet;
    }
    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol)
    {
        return layerStack->TrySeparate(qubits, error_tol);
    }
    bool TrySeparate(bitLenInt qubit) { return layerStack->TrySeparate(qubit); }
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2) { return layerStack->TrySeparate(qubit1, qubit2); }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (start > qubitCount) {
            throw std::invalid_argument("QTensorNetwork::Allocate() 'start' argument is out-of-bounds!");
        }

        if (!length) {
            return start;
        }

        layerStack->Allocate(length);
        const bitLenInt movedQubits = qubitCount - start;
        SetQubitCount(qubitCount + length);

        for (bitLenInt i = 0U; i < movedQubits; ++i) {
            const bitLenInt q = start + movedQubits - (i + 1U);
            Swap(q, q + length);
        }

        return start;
    }

    real1_f Prob(bitLenInt qubit)
    {
        real1_f toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->Prob(qubit); }, { qubit });
        return toRet;
    }
    real1_f ProbAll(const bitCapInt& fullRegister)
    {
        real1_f toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->ProbAll(fullRegister); });
        return toRet;
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        bool toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->ForceM(qubit, result, doForce, doApply); }, { qubit });
        return toRet;
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
    {
        std::map<bitCapInt, int> toRet;
        std::set<bitLenInt> qubits;
        for (const bitCapInt& qPow : qPowers) {
            qubits.insert(log2(qPow));
        }
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->MultiShotMeasureMask(qPowers, shots); }, qubits);
        return toRet;
    }
    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
    {
        std::map<bitCapInt, int> toRet;
        std::set<bitLenInt> qubits;
        for (const bitCapInt& qPow : qPowers) {
            qubits.insert(log2(qPow));
        }
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->MultiShotMeasureMask(qPowers, shots, shotsArray); }, qubits);
    }

    void Mtrx(const complex mtrx[4U], bitLenInt target)
    {
        CheckQubitCount(target);
        circuit->AppendGate(std::make_shared<QCircuitGate>(target, mtrx));
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex mtrx[4U], bitLenInt target)
    {
        CheckQubitCount(target, controls);
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        circuit->AppendGate(
            std::make_shared<QCircuitGate>(target, mtrx, std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex mtrx[4U], bitLenInt target)
    {
        CheckQubitCount(target, controls);
        circuit->AppendGate(std::make_shared<QCircuitGate>(
            target, mtrx, std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }
    void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        circuit->AppendGate(std::make_shared<QCircuitGate>(
            target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        circuit->AppendGate(std::make_shared<QCircuitGate>(
            target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }
    void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        circuit->AppendGate(std::make_shared<QCircuitGate>(
            target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        circuit->AppendGate(std::make_shared<QCircuitGate>(
            target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    void DECS(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CINC(toAdd, inOutStart, length, controls);
    }
    void CDEC(const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls);
    }
    void INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }

    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->INC(toAdd, start, length); });
    }
    void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->INCC(toAdd, start, length, carryIndex); });
    }
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->INCS(toAdd, start, length, overflowIndex); });
    }
    void INCDECSC(
        const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
        });
    }
    void INCDECSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->INCDECSC(toAdd, start, length, carryIndex); });
    }
    void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->DECC(toSub, start, length, carryIndex); });
    }
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->INCBCD(toAdd, start, length); });
    }
    void DECBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->DECBCD(toAdd, start, length); });
    }
    void INCDECBCDC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->INCDECBCDC(toAdd, start, length, carryIndex);
        });
    }
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->MUL(toMul, inOutStart, carryStart, length); });
    }
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        RunAsAmplitudes(
            [&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->DIV(toDiv, inOutStart, carryStart, length); });
    }
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->POWModNOut(base, modN, inStart, outStart, length);
        });
    }
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->CMUL(toMul, inOutStart, carryStart, length, controls);
        });
    }
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->CDIV(toDiv, inOutStart, carryStart, length, controls);
        });
    }
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->CPOWModNOut(base, modN, inStart, outStart, length, controls);
        });
    }
    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        bitCapInt toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) {
            toRet = std::dynamic_pointer_cast<QAlu>(ls)->IndexedLDA(
                indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });

        return toRet;
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        bitCapInt toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) {
            toRet = std::dynamic_pointer_cast<QAlu>(ls)->IndexedADC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });

        return toRet;
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        bitCapInt toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) {
            toRet = std::dynamic_pointer_cast<QAlu>(ls)->IndexedSBC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });

        return toRet;
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { std::dynamic_pointer_cast<QAlu>(ls)->Hash(start, length, values); });
    }
    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
        });
    }
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) {
            std::dynamic_pointer_cast<QAlu>(ls)->PhaseFlipIfLess(greaterPerm, start, length);
        });
    }

    /** @} */
#endif
};
} // namespace Qrack
