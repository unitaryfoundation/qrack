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
    , aceMb(0U)
    , aceQubits(0U)
    , qbThreshold(qubitThreshold)
    , separabilityThreshold(sep_thresh)
    , layerStack(nullptr)
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

    SetPermutation(initState, phaseFac);
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

void QTensorNetwork::MakeLayerStack()
{
    layerStack = nullptr;
    layerStack = CreateQuantumInterface(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, devID, !!hardware_rand_generator, isSparse, (real1_f)amplitudeFloor, deviceIDs,
        qbThreshold);
    layerStack->SetReactiveSeparate(isReactiveSeparate);
    layerStack->SetSdrp(separabilityThreshold);
    layerStack->SetNcrp(ncrp);
    layerStack->SetReactiveSeparate(isReactiveSeparate);
    layerStack->SetTInjection(useTGadget);
    if (aceQubits) {
        layerStack->SetAceMaxQubits(aceQubits);
    }
    if (aceMb) {
        layerStack->SetSparseAceMaxMb(aceMb);
    }
}

QInterfacePtr QTensorNetwork::Clone()
{
    QTensorNetworkPtr clone = std::make_shared<QTensorNetwork>(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, useHostRam, devID, !!hardware_rand_generator, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, qbThreshold);

    clone->circuit = circuit->Clone();
    clone->layerStack = layerStack->Clone();
    clone->SetSdrp(separabilityThreshold);
    clone->SetNcrp(ncrp);
    clone->SetReactiveSeparate(isReactiveSeparate);
    clone->SetTInjection(useTGadget);
    if (aceQubits) {
        clone->SetAceMaxQubits(aceQubits);
    }
    if (aceMb) {
        clone->SetSparseAceMaxMb(aceMb);
    }

    return clone;
}
} // namespace Qrack
