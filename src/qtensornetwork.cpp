//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QTensorNetwork is a gate-based QInterface descendant wrapping cuQuantum.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif
#if ENABLE_CUDA
#include "common/cudaengine.cuh"
#endif

#if ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

// #if ENABLE_CUDA
// #include <cuda_runtime.h>
// #include <cutensornet.h>
// #endif

namespace Qrack {

QTensorNetwork::QTensorNetwork(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState,
    qrack_rand_gen_ptr rgp, const complex& phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem,
    int64_t deviceId, bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , useHostRam(useHostMem)
    , isSparse(useSparseStateVec)
    , useTGadget(true)
    , isNearClifford(true)
#if ENABLE_OPENCL || ENABLE_CUDA
    , isCpu(false)
#endif
#if ENABLE_QBDT
    , isQBdt(false)
#endif
    , devID(deviceId)
    , separabilityThreshold(sep_thresh)
    , globalPhase(phaseFac)
    , deviceIDs(devList)
    , engines(eng)
{
    isReactiveSeparate = (separabilityThreshold > FP_NORM_EPSILON_F);

    if (engines.empty()) {
#if ENABLE_OPENCL
        engines.push_back((OCLEngine::Instance().GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#elif ENABLE_CUDA
        engines.push_back(
            (CUDAEngine::Instance().GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#else
        engines.push_back(QINTERFACE_OPTIMAL);
#endif
    }

    for (size_t i = 0U; i < engines.size(); ++i) {
        const QInterfaceEngine& et = engines[i];
        if (et == QINTERFACE_STABILIZER_HYBRID) {
            break;
        }
        if ((et == QINTERFACE_BDT_HYBRID) || (et == QINTERFACE_BDT) || (et == QINTERFACE_QPAGER) ||
            (et == QINTERFACE_HYBRID) || (et == QINTERFACE_CPU) || (et == QINTERFACE_OPENCL) ||
            (et == QINTERFACE_CUDA)) {
            isNearClifford = false;
            break;
        }
    }

#if ENABLE_OPENCL || ENABLE_CUDA
    for (size_t i = 0U; i < engines.size(); ++i) {
        const QInterfaceEngine& et = engines[i];
        if ((et == QINTERFACE_HYBRID) || (et == QINTERFACE_OPENCL) || (et == QINTERFACE_CUDA)) {
            break;
        }
        if (et == QINTERFACE_CPU) {
            isCpu = true;
            break;
        }
    }
#endif

#if ENABLE_QBDT
    for (size_t i = 0U; i < engines.size(); ++i) {
        const QInterfaceEngine& et = engines[i];
        if ((et == QINTERFACE_BDT_HYBRID) || (et == QINTERFACE_BDT)) {
            isQBdt = true;
            break;
        }
        if ((et == QINTERFACE_STABILIZER_HYBRID) || (et == QINTERFACE_QPAGER) || (et == QINTERFACE_HYBRID) ||
            (et == QINTERFACE_CPU) || (et == QINTERFACE_OPENCL) || (et == QINTERFACE_CUDA)) {
            break;
        }
    }
#endif

    SetPermutation(initState, globalPhase);
}

bitLenInt QTensorNetwork::GetThresholdQb()
{
#if ENABLE_ENV_VARS
    if (getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")) {
        return (bitLenInt)std::stoi(std::string(getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")));
    }
#endif

#if ENABLE_OPENCL || ENABLE_CUDA
    if (isCpu) {
        return QRACK_QRACK_QTENSORNETWORK_THRESHOLD_CPU_QB;
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_PAGING_QB")) {
        return (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    }
#endif

    const size_t devCount = QRACK_GPU_SINGLETON.GetDeviceCount();
    if (!devCount) {
        return QRACK_QRACK_QTENSORNETWORK_THRESHOLD_CPU_QB;
    }

    const bitLenInt perPage = log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex));

#if ENABLE_OPENCL
    if (devCount < 2U) {
        return perPage + 2U;
    }

    return perPage + log2Ocl(devCount) + 1U;
#else
    if (devCount < 2U) {
        return perPage;
    }

    return (perPage + log2Ocl(devCount)) - 1U;
#endif

#else
    return QRACK_QRACK_QTENSORNETWORK_THRESHOLD_CPU_QB;
#endif
}

void QTensorNetwork::MakeLayerStack(std::set<bitLenInt> qubits)
{
    if (layerStack) {
        // We have a cached layerStack.
        return;
    }

    // We need to prepare the layer stack (and cache it).
    layerStack = CreateQuantumInterface(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, devID, !!hardware_rand_generator, isSparse, (real1_f)amplitudeFloor, deviceIDs);
    layerStack->SetReactiveSeparate(isReactiveSeparate);
    layerStack->SetTInjection(useTGadget);

    std::vector<QCircuitPtr> c;
    if (qubits.size()) {
        for (size_t i = 0U; i < circuit.size(); ++i) {
            const size_t j = circuit.size() - (i + 1U);
            if (j < measurements.size()) {
                for (const auto& m : measurements[j]) {
                    qubits.erase(m.first);
                }
            }
            if (qubits.empty()) {
                QRACK_CONST complex pauliX[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
                c.push_back(std::make_shared<QCircuit>(true, isNearClifford));
                for (const auto& m : measurements[j]) {
                    if (m.second) {
                        c.back()->AppendGate(std::make_shared<QCircuitGate>(m.first, pauliX));
                    }
                }

                break;
            }
            c.push_back(circuit[j]->PastLightCone(qubits));
        }
        std::reverse(c.begin(), c.end());
    } else {
        c = circuit;
    }

    const size_t offset = circuit.size() - c.size();
    for (size_t i = 0U; i < c.size(); ++i) {
        c[i]->Run(layerStack);
        if (measurements.size() > (offset + i)) {
            RunMeasurmentLayer(offset + i);
        }
    }
}

QInterfacePtr QTensorNetwork::Clone()
{
    QTensorNetworkPtr clone = std::make_shared<QTensorNetwork>(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, useHostRam, devID, !!hardware_rand_generator, isSparse, (real1_f)amplitudeFloor,
        deviceIDs);

    clone->circuit.clear();
    for (size_t i = 0U; i < circuit.size(); ++i) {
        clone->circuit.push_back(circuit[i]->Clone());
    }
    clone->measurements = measurements;
    if (layerStack) {
        clone->layerStack = layerStack->Clone();
    }

    clone->SetReactiveSeparate(isReactiveSeparate);
    clone->SetTInjection(useTGadget);

    return clone;
}

bool QTensorNetwork::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    CheckQubitCount(qubit);

    bool toRet;
    RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->ForceM(qubit, result, doForce, doApply); }, { qubit });

    if (!doApply) {
        return toRet;
    }

    size_t layerId = circuit.size() - 1U;
    // Starting from latest circuit layer, if measurement commutes...
    while (layerId && !(circuit[layerId]->IsNonPhaseTarget(qubit))) {
        const QCircuitPtr& c = circuit[layerId];
        c->DeletePhaseTarget(qubit, toRet);

        if (measurements.size() <= layerId) {
            // ...Fill an earlier layer.
            --layerId;
            continue;
        }

        // We will insert a terminal measurement on this qubit, again.
        // This other measurement commutes, as it is in the same basis.
        // So, erase any redundant later measurement.
        std::map<bitLenInt, bool>& m = measurements[layerId];
        m.erase(qubit);

        // If the measurement layer is empty, telescope the layers.
        if (m.empty()) {
            measurements.erase(measurements.begin() + layerId);
            const size_t prevLayerId = layerId + 1U;
            if (prevLayerId < circuit.size()) {
                c->Combine(circuit[prevLayerId]);
                circuit.erase(circuit.begin() + prevLayerId);
            }
        }

        // ...Fill an earlier layer.
        --layerId;
    }

    // Identify whether we need a totally new measurement layer.
    if (layerId >= measurements.size()) {
        // Insert the required measurement layer.
        measurements.emplace_back();
    }

    // Insert terminal measurement.
    measurements[layerId][qubit] = toRet;

    // If no qubit in this layer is target of a non-phase gate, it can be completely telescoped into classical state
    // preparation.
    while (true) {
        std::vector<bitLenInt> nonMeasuredQubits;
        nonMeasuredQubits.reserve(qubitCount);
        for (size_t i = 0U; i < qubitCount; ++i) {
            nonMeasuredQubits.push_back(i);
        }
        std::map<bitLenInt, bool>& m = measurements[layerId];
        for (const auto& b : m) {
            nonMeasuredQubits.erase(std::find(nonMeasuredQubits.begin(), nonMeasuredQubits.end(), b.first));
        }

        const QCircuitPtr& c = circuit[layerId];
        for (const bitLenInt& q : nonMeasuredQubits) {
            if (c->IsNonPhaseTarget(q)) {
                // Nothing more to do; tell the user the result.
                return toRet;
            }
        }

        // If we did not return, this circuit layer is fully collapsed.
        QRACK_CONST complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };

        if (!layerId) {
            circuit[0U] = std::make_shared<QCircuit>();
            for (const auto& b : m) {
                if (b.second) {
                    circuit[0U]->AppendGate(std::make_shared<QCircuitGate>(b.first, pauliX));
                }
            }

            return toRet;
        }
        circuit.erase(circuit.begin() + layerId);

        const size_t layerIdMin1 = layerId - 1U;
        const std::map<bitLenInt, bool>& mMin1 = measurements[layerIdMin1];
        for (const auto& b : m) {
            const auto it = mMin1.find(b.first);
            if ((it == mMin1.end()) || (b.second == it->second)) {
                continue;
            }
            circuit[layerIdMin1]->AppendGate(std::make_shared<QCircuitGate>(b.first, pauliX));
        }
        m.insert(mMin1.begin(), mMin1.end());
        measurements.erase(measurements.begin() + (layerId - 1U));

        // ...Repeat until we reach the terminal layer.
        --layerId;
    }

    // Tell the user the result.
    return toRet;
}
} // namespace Qrack
