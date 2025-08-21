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

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>

#if defined(OPENCL_V3)
#include <CL/opencl.hpp>
#elif defined(__APPLE__)
#define CL_SILENCE_DEPRECATION
#include <CL/opencl.hpp>
#elif defined(_WIN32) || ENABLE_SNUCL
#include <CL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

namespace Qrack {

class OCLDeviceCall;

class OCLDeviceContext;

typedef std::shared_ptr<OCLDeviceContext> DeviceContextPtr;
typedef std::vector<cl::Event> EventVec;
typedef std::shared_ptr<EventVec> EventVecPtr;

struct OCLKernelHandle {
    OCLAPI oclapi;
    std::string kernelname;

    OCLKernelHandle(OCLAPI o, std::string kn)
        : oclapi(o)
        , kernelname(kn)
    {
    }
};

class OCLDeviceCall {
protected:
    std::lock_guard<std::mutex> guard;

public:
    // A cl::Kernel is unique object which should always be taken by reference, or the OCLDeviceContext will lose
    // ownership.
    cl::Kernel& call;
    OCLDeviceCall(const OCLDeviceCall&);

protected:
    OCLDeviceCall(std::mutex& m, cl::Kernel& c)
        : guard(m)
        , call(c)
    {
    }

    friend class OCLDeviceContext;

private:
    OCLDeviceCall& operator=(const OCLDeviceCall&) = delete;
};

class OCLDeviceContext {
public:
    const cl::Platform platform;
    const cl::Device device;
    const cl::Context context;
    const int64_t context_id;
    const int64_t device_id;
    const bool is_gpu;
    const bool is_cpu;
    const bool use_host_mem;
    cl::CommandQueue queue;
    EventVecPtr wait_events;

protected:
    std::mutex waitEventsMutex;
    std::map<OCLAPI, cl::Kernel> calls;
    std::map<OCLAPI, std::unique_ptr<std::mutex>> mutexes;

private:
    const size_t procElemCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    const size_t maxWorkItems = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
    const size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const size_t maxAlloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    const size_t globalSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const size_t localSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    size_t globalLimit;
    size_t preferredSizeMultiple;
    size_t preferredConcurrency;

public:
    OCLDeviceContext(cl::Platform& p, cl::Device& d, cl::Context& c, int64_t dev_id, int64_t cntxt_id, int64_t maxAlloc,
        bool isGpu, bool isCpu, bool useHostMem)
        : platform(p)
        , device(d)
        , context(c)
        , context_id(cntxt_id)
        , device_id(dev_id)
        , is_gpu(isGpu)
        , is_cpu(isCpu)
        , use_host_mem(useHostMem)
        , wait_events(new EventVec())
#if ENABLE_OCL_MEM_GUARDS
        , globalLimit((maxAlloc >= 0) ? maxAlloc : globalSize)
#else
        , globalLimit((maxAlloc >= 0) ? maxAlloc : -1)
#endif
        , preferredSizeMultiple(0U)
        , preferredConcurrency(0U)
    {
        cl_int error;
#if ENABLE_OOO_OCL
#if ENABLE_ENV_VARS
        if (getenv("DISABLE_OOO_OCL")) {
            queue = cl::CommandQueue(c, d, 0, &error);
            if (error != CL_SUCCESS) {
                throw std::runtime_error("Failed to create OpenCL command queue!");
            }
        } else {
            queue = cl::CommandQueue(c, d, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
            if (error != CL_SUCCESS) {
                queue = cl::CommandQueue(c, d, 0, &error);
                if (error != CL_SUCCESS) {
                    throw std::runtime_error("Failed to create OpenCL command queue!");
                }
            }
        }
#else
        queue = cl::CommandQueue(c, d, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        if (error != CL_SUCCESS) {
            queue = cl::CommandQueue(c, d, 0, &error);
            if (error != CL_SUCCESS) {
                throw std::runtime_error("Failed to create OpenCL command queue!");
            }
        }
#endif
#else
        queue = cl::CommandQueue(c, d, 0, &error);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL command queue!");
        }
#endif
    }

    OCLDeviceCall Reserve(OCLAPI call) { return OCLDeviceCall(*(mutexes[call]), calls[call]); }

    EventVecPtr ResetWaitEvents()
    {
        std::lock_guard<std::mutex> guard(waitEventsMutex);
        EventVecPtr waitVec = std::move(wait_events);
        wait_events = EventVecPtr(new EventVec());
        return waitVec;
    }

    template <typename Fn> void EmplaceEvent(Fn fn)
    {
        std::lock_guard<std::mutex> guard(waitEventsMutex);
        wait_events->emplace_back();
        fn(wait_events->back());
    }

    void WaitOnAllEvents()
    {
        std::lock_guard<std::mutex> guard(waitEventsMutex);
        if ((wait_events.get())->size()) {
            cl::Event::waitForEvents((const EventVec&)*(wait_events.get()));
            wait_events->clear();
        }
    }

    size_t GetPreferredSizeMultiple()
    {
        return preferredSizeMultiple
            ? preferredSizeMultiple
            : preferredSizeMultiple =
                  calls[OCL_API_APPLY2X2_NORM_SINGLE].getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                      device);
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

        const size_t pc = procElemCount * GetPreferredSizeMultiple();
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

    size_t GetProcElementCount() { return procElemCount; }
    size_t GetMaxWorkItems() { return maxWorkItems; }
    size_t GetMaxWorkGroupSize() { return maxWorkGroupSize; }
    size_t GetMaxAlloc() { return maxAlloc; }
    size_t GetGlobalSize() { return globalSize; }
    size_t GetLocalSize() { return localSize; }
    size_t GetGlobalAllocLimit() { return globalLimit; }

    friend class OCLEngine;
};

struct InitOClResult {
    std::vector<DeviceContextPtr> all_dev_contexts;
    DeviceContextPtr default_dev_context;

    InitOClResult()
        : all_dev_contexts()
        , default_dev_context{ nullptr }
    {
        // Intentionally left blank
    }

    InitOClResult(std::vector<DeviceContextPtr> adc, DeviceContextPtr ddc)
        : all_dev_contexts(adc)
        , default_dev_context(ddc)
    {
        // Intentionally left blank
    }
};

/** "Qrack::OCLEngine" manages the single OpenCL context. */
class OCLEngine {
public:
    // See https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLEngine& Instance()
    {
        static OCLEngine instance;
        return instance;
    }
    /// Get default location for precompiled binaries:
    static std::string GetDefaultBinaryPath()
    {
#if ENABLE_ENV_VARS
        if (getenv("QRACK_OCL_PATH")) {
            std::string toRet = std::string(getenv("QRACK_OCL_PATH"));
            if ((toRet.back() != '/') && (toRet.back() != '\\')) {
#if defined(_WIN32) && !defined(__CYGWIN__)
                toRet += "\\";
#else
                toRet += "/";
#endif
            }

            return toRet;
        }
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
        return std::string(getenv("HOMEDRIVE") ? getenv("HOMEDRIVE") : "") +
            std::string(getenv("HOMEPATH") ? getenv("HOMEPATH") : "") + "\\.qrack\\";
#else
        return std::string(getenv("HOME") ? getenv("HOME") : "") + "/.qrack/";
#endif
    }
    /// Initialize the OCL environment, with the option to save the generated binaries. Binaries will be saved/loaded
    /// from the folder path "home". This returns a Qrack::OCLInitResult object which should be passed to
    /// SetDeviceContextPtrVector().
    static InitOClResult InitOCL(bool buildFromSource = false, bool saveBinaries = false, std::string home = "*",
        std::vector<int64_t> maxAllocVec = { -1 });

    /// Get a pointer one of the available OpenCL contexts, by its index in the list of all contexts.
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
    /// Pick a default device, for QEngineOCL instances that don't specify a preferred device.
    void SetDefaultDeviceContext(DeviceContextPtr dcp);

    size_t GetActiveAllocSize(const int64_t& dev)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            throw std::invalid_argument("OCLEngine::GetActiveAllocSize device ID is too high!");
        }

        return (dev < 0) ? activeAllocSizes[GetDefaultDeviceID()] : activeAllocSizes[(size_t)dev];
    }
    size_t AddToActiveAllocSize(const int64_t& dev, size_t size)
    {
        if (dev > ((int64_t)activeAllocSizes.size())) {
            throw std::invalid_argument("OCLEngine::GetActiveAllocSize device ID is too high!");
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
            throw std::invalid_argument("OCLEngine::GetActiveAllocSize device ID is too high!");
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
            throw std::invalid_argument("OCLEngine::GetActiveAllocSize device ID is too high!");
        }
        const size_t lDev = (dev < 0) ? GetDefaultDeviceID() : dev;
        std::lock_guard<std::mutex> lock(allocMutex);
        // User code should catch std::bad_alloc and reset:
        activeAllocSizes[lDev] = 0;
    }

    OCLEngine(OCLEngine const&) = delete;
    void operator=(OCLEngine const&) = delete;

private:
    static const std::vector<OCLKernelHandle> kernelHandles;
    static const std::string binary_file_prefix;
    static const std::string binary_file_ext;

    std::vector<size_t> activeAllocSizes;
    std::vector<int64_t> maxActiveAllocSizes;
    std::mutex allocMutex;
    std::vector<DeviceContextPtr> all_device_contexts;
    DeviceContextPtr default_device_context;

    OCLEngine(); // Private so that it can  not be called

    /// Make the program, from either source or binary
    static cl::Program MakeProgram(bool buildFromSource, std::string path, std::shared_ptr<OCLDeviceContext> devCntxt);
    /// Save the program binary:
    static void SaveBinary(cl::Program program, std::string path, std::string fileName);
};

} // namespace Qrack
