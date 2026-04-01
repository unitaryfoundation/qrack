//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

namespace Qrack {

QHybrid::QHybrid(bitLenInt qBitCount, const bitCapInt& initState, qrack_rand_gen_ptr rgp, const complex& phaseFac,
    bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1_f norm_thresh, std::vector<int64_t> devList, bitLenInt qubitThreshold, real1_f sep_thresh, bool useTurbo)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , isTurbo(useTurbo)
    , separabilityThreshold(sep_thresh)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , deviceIDs(devList)
{
    if (qubitThreshold) {
        gpuThresholdQubits = qubitThreshold;
    } else {
        const bitLenInt gpuQubits =
            log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;
        const bitLenInt cpuQubits = (GetStride() <= 1U) ? 0U : (log2Ocl(GetStride() - 1U) + 1U);
        gpuThresholdQubits = gpuQubits < cpuQubits ? gpuQubits : cpuQubits;
    }

    pagerThresholdQubits = log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex));
#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_PAGE_QB")) {
        pagerThresholdQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGE_QB")));
    }
#endif
    if (pagerThresholdQubits < 3U) {
        pagerThresholdQubits = 3U;
    }
    if ((pagerThresholdQubits - 1U) < gpuThresholdQubits) {
        gpuThresholdQubits = pagerThresholdQubits - 1U;
    }

    isGpu = (qubitCount >= gpuThresholdQubits);
    isPager = (qubitCount > pagerThresholdQubits);

    std::vector<QInterfaceEngine> engines;
    if (isPager) {
        engines.push_back(QINTERFACE_QPAGER);
    }
    engines.push_back(isGpu ? QRACK_GPU_ENGINE : QINTERFACE_CPU);

    engine = std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engines, qubitCount, initState, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, pagerThresholdQubits, separabilityThreshold, isTurbo));
}

QEnginePtr QHybrid::MakeEngine(bool isOpenCL)
{
    QEnginePtr toRet =
        std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(isOpenCL ? QRACK_GPU_ENGINE : QINTERFACE_CPU, 0U,
            ZERO_BCI, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse,
            (real1_f)amplitudeFloor, deviceIDs, pagerThresholdQubits, separabilityThreshold, isTurbo));
    toRet->SetQubitCount(qubitCount);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return toRet;
}
} // namespace Qrack
