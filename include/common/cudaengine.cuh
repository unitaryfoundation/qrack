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

#include "oclapi.hpp"

#if !ENABLE_CUDA
#error CUDA has not been enabled
#endif

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace Qrack {

class CUDADeviceContext;
typedef std::shared_ptr<CUDADeviceContext> DeviceContextPtr;
typedef std::vector<cudaEvent_t> EventVec;
typedef std::shared_ptr<std::vector<cudaEvent_t>> EventVecPtr;

class CUDADeviceContext {
public:
    const int64_t device_id;
    // If we have ~16 streams, this doesn't scale to single-qubit QEngineCUDA instances under QUnit.
    // However, we prefer QHybrid!
    // That said, creating these streams once per simulator becomes very expensive in the simulator is trashed and
    // recreated repeatedly.
    cudaStream_t queue;
    cudaStream_t params_queue;

private:
    size_t globalLimit;
    size_t preferredSizeMultiple;
    size_t preferredConcurrency;
    cudaDeviceProp properties;

public:
    CUDADeviceContext(int64_t dev_id, int64_t maxAlloc = -1)
        : device_id(dev_id)
        , queue(0)
        , params_queue(0)
        , globalLimit((maxAlloc >= 0) ? maxAlloc : -1)
        , preferredSizeMultiple(0U)
        , preferredConcurrency(0U)
    {
        cudaGetDeviceProperties(&properties, device_id);
#if ENABLE_OCL_MEM_GUARDS
        globalLimit = (maxAlloc >= 0) ? maxAlloc : properties.totalGlobalMem;
#endif

        cudaError_t error = cudaStreamCreate(&queue);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error code on main stream creation: " + std::to_string(error));
        }

        error = cudaStreamCreate(&params_queue);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error code on parameter stream creation: " + std::to_string(error));
        }
    }

    ~CUDADeviceContext()
    {
        // Theoretically, all user output is blocking, so don't throw in destructor.
        cudaStreamDestroy(params_queue);
        cudaStreamDestroy(queue);
    }

    size_t GetPreferredSizeMultiple()
    {
        return preferredSizeMultiple ? preferredSizeMultiple : preferredSizeMultiple = properties.warpSize;
    }

    size_t GetPreferredConcurrency()
    {
        if (preferredConcurrency) {
            return preferredConcurrency;
        }

        int hybridOffset = 3U;
#if ENABLE_ENV_VARS
        if (getenv("QRACK_GPU_OFFSET_QB")) {
            hybridOffset = std::stoi(std::string(getenv("QRACK_GPU_OFFSET_QB")));
        }
#endif

        const size_t pc = GetProcElementCount() * GetPreferredSizeMultiple();
        preferredConcurrency = 1U;
        while (preferredConcurrency < pc) {
            preferredConcurrency <<= 1U;
        }
        preferredConcurrency =
            hybridOffset > 0 ? (preferredConcurrency << hybridOffset) : (preferredConcurrency >> -hybridOffset);
        if (preferredConcurrency < 1U) {
            preferredConcurrency = 1U;
        }

        return preferredConcurrency;
    }

    size_t GetProcElementCount() { return properties.multiProcessorCount; }
    size_t GetMaxWorkItems() { return properties.maxThreadsPerMultiProcessor; }
    size_t GetMaxWorkGroupSize() { return properties.warpSize; }
    size_t GetMaxAlloc() { return properties.totalGlobalMem; }
    size_t GetGlobalSize() { return properties.totalGlobalMem; }
    size_t GetLocalSize() { return properties.sharedMemPerBlock; }
    size_t GetGlobalAllocLimit() { return globalLimit; }

    friend class CUDAEngine;
};

struct InitCUDAResult {
    std::vector<DeviceContextPtr> all_dev_contexts;
    DeviceContextPtr default_dev_context;

    InitCUDAResult()
        : all_dev_contexts()
        , default_dev_context{ nullptr }
    {
        // Intentionally left blank
    }

    InitCUDAResult(std::vector<DeviceContextPtr> adc, DeviceContextPtr ddc)
        : all_dev_contexts(adc)
        , default_dev_context(ddc)
    {
        // Intentionally left blank
    }
};

/** "Qrack::CUDAEngine" manages the single CUDA context. */
class CUDAEngine {
public:
    // See https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static CUDAEngine& Instance()
    {
        static CUDAEngine instance;
        return instance;
    }
    /// Initialize the CUDA environment. This returns a Qrack::CUDAInitResult object which should be passed to
    /// SetDeviceContextPtrVector().
    static InitCUDAResult InitCUDA(std::vector<int64_t> maxAllocVec = { -1 });

    /// Get a pointer one of the available CUDA contexts, by its index in the list of all contexts.
    DeviceContextPtr GetDeviceContextPtr(const int64_t& dev = -1);
    /// Get the list of all available devices (and their supporting objects).
    std::vector<DeviceContextPtr> GetDeviceContextPtrVector();
    /** Set the list of DeviceContextPtr object available for use. If one takes the result of
     * GetDeviceContextPtrVector(), trims items from it, and sets it with this method, (at initialization, before any
     * QEngine objects depend on them,) all resources associated with the removed items are freed.
     */
    void SetDeviceContextPtrVector(std::vector<DeviceContextPtr> vec, DeviceContextPtr dcp = nullptr);
    /// Get the count of devices in the current list.
    int GetDeviceCount() { return all_device_contexts.size(); }
    /// Get default device ID.
    size_t GetDefaultDeviceID() { return default_device_context->device_id; }
    /// Pick a default device, for QEngineCUDA instances that don't specify a preferred device.
    void SetDefaultDeviceContext(DeviceContextPtr dcp);

    size_t GetActiveAllocSize(const int64_t& dev)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            return 0U;
        }

        return (dev < 0) ? activeAllocSizes[GetDefaultDeviceID()] : activeAllocSizes[(size_t)dev];
    }
    size_t AddToActiveAllocSize(const int64_t& dev, size_t size)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            throw std::invalid_argument("OCLEngine::AddToActiveAllocSize device ID is too high!");
        }

        const size_t lDev = (dev < 0) ? GetDefaultDeviceID() : dev;

        if (size == 0) {
            return activeAllocSizes[lDev];
        }

        std::lock_guard<std::mutex> lock(allocMutex);
        activeAllocSizes[lDev] += size;

        return activeAllocSizes[lDev];
    }
    size_t SubtractFromActiveAllocSize(const int64_t& dev, size_t size)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            throw std::invalid_argument("OCLEngine::AddToActiveAllocSize device ID is too high!");
        }

        const size_t lDev = (dev < 0) ? GetDefaultDeviceID() : dev;

        if (size == 0) {
            return activeAllocSizes[lDev];
        }

        std::lock_guard<std::mutex> lock(allocMutex);
        if (size < activeAllocSizes[lDev]) {
            activeAllocSizes[lDev] -= size;
        } else {
            activeAllocSizes[lDev] = 0;
        }

        return activeAllocSizes[lDev];
    }
    void ResetActiveAllocSize(const int64_t& dev)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            throw std::invalid_argument("OCLEngine::AddToActiveAllocSize device ID is too high!");
        }
        const size_t lDev = (dev < 0) ? GetDefaultDeviceID() : dev;
        std::lock_guard<std::mutex> lock(allocMutex);
        // User code should catch std::bad_alloc and reset:
        activeAllocSizes[lDev] = 0;
    }

    CUDAEngine(CUDAEngine const&) = delete;
    void operator=(CUDAEngine const&) = delete;

private:
    std::vector<size_t> activeAllocSizes;
    std::vector<int64_t> maxActiveAllocSizes;
    std::mutex allocMutex;
    std::vector<DeviceContextPtr> all_device_contexts;
    DeviceContextPtr default_device_context;

    CUDAEngine(); // Private so that it can  not be called
};

} // namespace Qrack
