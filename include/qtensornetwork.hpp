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

class QTensorNetwork : public QInterface {
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
    real1_f separabilityThreshold;
    complex globalPhase;
    QInterfacePtr layerStack;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;
    std::vector<QCircuitPtr> circuit;
    std::vector<std::map<bitLenInt, bool>> measurements;

    QCircuitPtr GetCircuit(bitLenInt target, std::vector<bitLenInt> controls = std::vector<bitLenInt>())
    {
        for (size_t i = 0U; i < measurements.size(); ++i) {
            size_t l = measurements.size() - (i + 1U);
            std::map<bitLenInt, bool>& m = measurements[l];
            ++l;

            if (m.find(target) != m.end()) {
                if (circuit.size() == l) {
                    circuit.push_back(std::make_shared<QCircuit>());
                }

                return circuit[l];
            }

            for (const bitLenInt& control : controls) {
                if (m.find(control) != m.end()) {
                    if (circuit.size() == l) {
                        circuit.push_back(std::make_shared<QCircuit>());
                    }

                    return circuit[l];
                }
            }
        }

        return circuit[0U];
    }

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

    void RunMeasurmentLayer(size_t layerId)
    {
        const std::map<bitLenInt, bool>& mLayer = measurements[layerId];
        std::vector<bitLenInt> bits;
        bits.reserve(mLayer.size());
        std::vector<bool> values;
        values.reserve(mLayer.size());

        for (const auto& m : mLayer) {
            bits.push_back(m.first);
            values.push_back(m.second);
        }

        layerStack->ForceM(bits, values);
    }

    bitLenInt GetThresholdQb();

    void MakeLayerStack(std::set<bitLenInt> qubits = std::set<bitLenInt>());

    template <typename Fn> void RunAsAmplitudes(Fn fn, const std::set<bitLenInt>& qubits = std::set<bitLenInt>())
    {
        if (qubits.empty()) {
            MakeLayerStack();
            return fn(layerStack);
        }

        const bitLenInt maxQb = GetThresholdQb();
        if (qubitCount <= maxQb) {
            MakeLayerStack();
            return fn(layerStack);
        } else {
            MakeLayerStack(qubits);
            QInterfacePtr ls = layerStack;
            layerStack = nullptr;
            return fn(ls);
        }
    }

    void Copy(QInterfacePtr orig)
    {
        throw std::domain_error("Can't TryDecompose() on QTensorNetwork! (QTensorNetwork does not allow Schmidt "
                                "decomposition in general!)");
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
    }

    void SetReactiveSeparate(bool isAggSep) { isReactiveSeparate = isAggSep; }

    double GetUnitaryFidelity()
    {
        double toRet;
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->GetUnitaryFidelity(); });
        return toRet;
    }

    void SetDevice(int64_t dID) { devID = dID; }

    void Finish()
    {
        if (layerStack) {
            layerStack->Finish();
        }
    };

    bool isFinished() { return !layerStack || layerStack->isFinished(); }

    void Dump()
    {
        if (layerStack) {
            layerStack->Dump();
        }
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (layerStack) {
            layerStack->UpdateRunningNorm(norm_thresh);
        }
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        if (layerStack) {
            layerStack->NormalizeState(nrm, norm_thresh, phaseArg);
        }
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QTensorNetwork>(toCompare));
    }
    real1_f SumSqrDiff(QTensorNetworkPtr toCompare)
    {
        real1_f toRet;
        toCompare->MakeLayerStack();
        RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->SumSqrDiff(toCompare->layerStack); });
        return toRet;
    }

    void SetPermutation(const bitCapInt& initState, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        circuit.clear();
        measurements.clear();
        layerStack = nullptr;

        circuit.push_back(std::make_shared<QCircuit>());

        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            if (bi_compare_0(pow2(i) & initState) != 0) {
                X(i);
            }
        }

        if ((phaseFac == CMPLX_DEFAULT_ARG) && randGlobalPhase) {
            real1_f angle = Rand() * 2 * (real1_f)PI_R1;
            globalPhase = complex((real1)cos(angle), (real1)sin(angle));
        } else if (phaseFac == CMPLX_DEFAULT_ARG) {
            globalPhase = complex(ONE_R1, ZERO_R1);
        } else {
            globalPhase = phaseFac;
        }
    }

    QInterfacePtr Clone();

    void GetQuantumState(complex* state)
    {
        RunAsAmplitudes([&](QInterfacePtr ls) { ls->GetQuantumState(state); });
    }
    void SetQuantumState(const complex* state)
    {
        throw std::domain_error("QTensorNetwork::SetQuantumState() not implemented!");
    }
    void SetQuantumState(QInterfacePtr eng)
    {
        throw std::domain_error("QTensorNetwork::SetQuantumState() not implemented!");
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
        throw std::domain_error("QTensorNetwork::SetAmplitude() not implemented!");
    }

    using QInterface::Compose;
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        throw std::domain_error("QTensorNetwork::Compose() not implemented!");
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        throw std::domain_error("QTensorNetwork::Decompose() not implemented!");
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        throw std::domain_error("QTensorNetwork::Decompose() not implemented!");
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        throw std::domain_error("QTensorNetwork::Dispose() not implemented!");
    }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        throw std::domain_error("QTensorNetwork::Dispose() not implemented!");
    }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (start > qubitCount) {
            throw std::invalid_argument("QTensorNetwork::Allocate() 'start' argument is out-of-bounds!");
        }

        if (!length) {
            return start;
        }

        const bitLenInt movedQubits = qubitCount - start;
        SetQubitCount(qubitCount + length);
        if (!movedQubits) {
            return start;
        }

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

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll()
    {
        bitCapInt toRet = ZERO_BCI;

        const bitLenInt maxQb = GetThresholdQb();
        if (qubitCount <= maxQb) {
            MakeLayerStack();
            toRet = layerStack->MAll();
        } else {
            for (bitLenInt i = 0U; i < qubitCount; ++i) {
                if (M(i)) {
                    bi_or_ip(&toRet, pow2(i));
                }
            }
        }

        SetPermutation(toRet);

        return toRet;
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
    {
        std::map<bitCapInt, int> toRet;
        if (qubitCount <= GetThresholdQb()) {
            std::set<bitLenInt> qubits;
            for (const bitCapInt& qPow : qPowers) {
                qubits.insert(log2(qPow));
            }
            RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->MultiShotMeasureMask(qPowers, shots); }, qubits);
        } else {
            std::vector<bitLenInt> qubits;
            qubits.reserve(qPowers.size());
            for (const bitCapInt& qPow : qPowers) {
                qubits.push_back(log2(qPow));
            }
#if ENABLE_QBDT && !ENABLE_QBDT_CPU_PARALLEL
            if (isQBdt) {
                std::mutex mapMtx;
                par_for(0U, shots, [&](const bitCapIntOcl& ignored, const unsigned& ignored2) {
                    QInterfacePtr clone = Clone();
                    bitCapInt result = ZERO_BCI;
                    for (size_t i = 0U; i < qubits.size(); ++i) {
                        if (clone->M(qubits[i])) {
                            bi_or_ip(&result, pow2(i));
                        }
                    }
                    std::lock_guard<std::mutex> mapLock(mapMtx);
                    ++toRet[result];
                });

                return toRet;
            }
#endif
            for (unsigned shot = 0U; shot < shots; ++shot) {
                QInterfacePtr clone = Clone();
                bitCapInt result = ZERO_BCI;
                for (size_t i = 0U; i < qubits.size(); ++i) {
                    if (clone->M(qubits[i])) {
                        bi_or_ip(&result, pow2(i));
                    }
                }
                ++toRet[result];
            }
        }

        return toRet;
    }
    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
    {
        std::map<bitCapInt, int> toRet;
        if (qubitCount <= GetThresholdQb()) {
            std::set<bitLenInt> qubits;
            for (const bitCapInt& qPow : qPowers) {
                qubits.insert(log2(qPow));
            }
            RunAsAmplitudes([&](QInterfacePtr ls) { ls->MultiShotMeasureMask(qPowers, shots, shotsArray); }, qubits);
        } else {
            std::vector<bitLenInt> qubits;
            qubits.reserve(qPowers.size());
            for (const bitCapInt& qPow : qPowers) {
                qubits.push_back(log2(qPow));
            }
#if ENABLE_QBDT && !ENABLE_QBDT_CPU_PARALLEL
            if (isQBdt) {
                return par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& ignored) {
                    QInterfacePtr clone = Clone();
                    bitCapInt result = ZERO_BCI;
                    for (size_t i = 0U; i < qubits.size(); ++i) {
                        if (clone->M(qubits[i])) {
                            bi_or_ip(&result, pow2(i));
                        }
                    }
                    shotsArray[shot] = (bitCapIntOcl)result;
                });
            }
#endif
            for (unsigned shot = 0U; shot < shots; ++shot) {
                QInterfacePtr clone = Clone();
                bitCapInt result = ZERO_BCI;
                for (size_t i = 0U; i < qubits.size(); ++i) {
                    if (clone->M(qubits[i])) {
                        bi_or_ip(&result, pow2(i));
                    }
                }
                shotsArray[shot] = (bitCapIntOcl)result;
            }
        }
    }

    void Mtrx(const complex* mtrx, bitLenInt target)
    {
        CheckQubitCount(target);
        layerStack = nullptr;
        GetCircuit(target)->AppendGate(std::make_shared<QCircuitGate>(target, mtrx));
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, mtrx, std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, mtrx, std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }
    void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }
    void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, m));
    }
    void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        CheckQubitCount(target, controls);
        layerStack = nullptr;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        GetCircuit(target, controls)
            ->AppendGate(std::make_shared<QCircuitGate>(
                target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, ZERO_BCI));
    }

    void DepolarizingChannelWeak1Qb(bitLenInt qubit, real1_f lambda)
    {
        if (lambda <= ZERO_R1) {
            return;
        }

        // "lambda" is the overall depolarization strength.
        // ChatGPT (custom GPT "Elara") reasons that
        // \epsilon(p) = (1 - p) * \rho + (p / 3) * (X \rho X + Y \rho Y + Z \rho Z),
        // so we use lambda / 3 for 3 checks.
        lambda = lambda / 3;
        if (Rand() < lambda) {
            Z(qubit);
        }
        if (Rand() < lambda) {
            X(qubit);
        }
        if (Rand() < lambda) {
            Y(qubit);
        }
    }
};
} // namespace Qrack
