//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#include <map>
#include <mutex>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Qrack {

class OCLDeviceCall;

class OCLDeviceContext;

typedef std::shared_ptr<OCLDeviceContext> DeviceContextPtr;

enum OCLAPI {
    OCL_API_UNKNOWN = 0,
    OCL_API_APPLY2X2,
    OCL_API_APPLY2X2_NORM,
    OCL_API_COHERE,
    OCL_API_DECOHEREPROB,
    OCL_API_DECOHEREAMP,
    OCL_API_DISPOSEPROB,
    OCL_API_PROB,
    OCL_API_X,
    OCL_API_SWAP,
    OCL_API_ROL,
    OCL_API_ROR,
    OCL_API_INC,
    OCL_API_DEC,
    OCL_API_INCC,
    OCL_API_DECC,
    OCL_API_INDEXEDLDA,
    OCL_API_INDEXEDADC,
    OCL_API_INDEXEDSBC,
    OCL_API_NORMALIZE,
    OCL_API_UPDATENORM,
};

class OCLDeviceCall {
protected:
    std::lock_guard<std::recursive_mutex> guard;

public:
    // A cl::Kernel is unique object which should always be taken by reference, or the OCLDeviceContext will lose
    // ownership.
    cl::Kernel& call;
    OCLDeviceCall(const OCLDeviceCall&);

protected:
    OCLDeviceCall(std::recursive_mutex& m, cl::Kernel& c)
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
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

protected:
    std::recursive_mutex mutex;
    std::map<OCLAPI, cl::Kernel> calls;

public:
    OCLDeviceContext(cl::Platform& p, cl::Device& d)
        : platform(p)
        , device(d)
        , mutex()
    {
    }
    OCLDeviceCall Reserve(OCLAPI call) { return OCLDeviceCall(mutex, calls[call]); }
    friend class OCLEngine;
};

/** "Qrack::OCLEngine" manages the single OpenCL context. */
class OCLEngine {
public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLEngine* Instance();
    /// Get a pointer to the OpenCL context
    DeviceContextPtr GetDeviceContextPtr(const int& dev = -1);
    int GetDeviceCount() { return deviceCount; }
    void SetDefaultDeviceContext(DeviceContextPtr dcp);

private:
    int deviceCount;

    std::vector<DeviceContextPtr> all_device_contexts;
    DeviceContextPtr default_device_context;

    OCLEngine(); // Private so that it can  not be called
    OCLEngine(OCLEngine const&); // copy constructor is private
    OCLEngine& operator=(OCLEngine const& rhs); // assignment operator is private
    static OCLEngine* m_pInstance;

    void InitOCL();

    unsigned long PowerOf2LessThan(unsigned long number);
};

} // namespace Qrack
