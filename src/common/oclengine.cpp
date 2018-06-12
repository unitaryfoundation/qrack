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

#include <iostream>

#include "oclengine.hpp"
#include "qenginecl.hpp"

#if ENABLE_COMPLEX8
#include "qheader_floatcl.hpp"
#else
#include "qheader_doublecl.hpp"
#endif

namespace Qrack {

/// "Qrack::OCLEngine" manages the single OpenCL context

// Public singleton methods to get pointers to various methods
cl::Context* OCLEngine::GetContextPtr() { return &context; }
cl::CommandQueue* OCLEngine::GetQueuePtr() { return &queue; }
cl::Kernel* OCLEngine::GetApply2x2Ptr() { return &apply2x2; }
cl::Kernel* OCLEngine::GetApply2x2NormPtr() { return &apply2x2norm; }
cl::Kernel* OCLEngine::GetCoherePtr() { return &cohere; }
cl::Kernel* OCLEngine::GetDecohereProbPtr() { return &decohereprob; }
cl::Kernel* OCLEngine::GetDisposeProbPtr() { return &disposeprob; }
cl::Kernel* OCLEngine::GetDecohereAmpPtr() { return &decohereamp; }
cl::Kernel* OCLEngine::GetProbPtr() { return &prob; }
cl::Kernel* OCLEngine::GetXPtr() { return &x; }
cl::Kernel* OCLEngine::GetSwapPtr() { return &swap; }
cl::Kernel* OCLEngine::GetROLPtr() { return &rol; }
cl::Kernel* OCLEngine::GetRORPtr() { return &ror; }
cl::Kernel* OCLEngine::GetINCPtr() { return &inc; }
cl::Kernel* OCLEngine::GetDECPtr() { return &dec; }
cl::Kernel* OCLEngine::GetINCCPtr() { return &incc; }
cl::Kernel* OCLEngine::GetDECCPtr() { return &decc; }
cl::Kernel* OCLEngine::GetLDAPtr() { return &indexedLda; }
cl::Kernel* OCLEngine::GetADCPtr() { return &indexedAdc; }
cl::Kernel* OCLEngine::GetSBCPtr() { return &indexedSbc; }

OCLEngine::OCLEngine() { InitOCL(0, -1); }
OCLEngine::OCLEngine(int plat, int dev) { InitOCL(plat, dev); }
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

void OCLEngine::InitOCL(int plat, int dev)
{
    // get all platforms (drivers), e.g. NVIDIA

    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    default_platform = all_platforms[plat];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // get default device (CPUs, GPUs) of the default platform
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    if (dev < 0) {
#ifdef ENABLE_COMPLEX8
        dev = all_devices.size() - 1;
#else
        dev = 0;
#endif
    }

    // use device[1] because that's a GPU; device[0] is the CPU
    default_device = all_devices[dev];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    context = cl::Context({ default_device });

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

#if ENABLE_COMPLEX8
    sources.push_back({ (const char*)qheader_float_cl, (long unsigned int)qheader_float_cl_len });
#else
    sources.push_back({ (const char*)qheader_double_cl, (long unsigned int)qheader_double_cl_len });
#endif
    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

    program = cl::Program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    queue = cl::CommandQueue(context, default_device);
    apply2x2 = cl::Kernel(program, "apply2x2");
    apply2x2norm = cl::Kernel(program, "apply2x2norm");
    x = cl::Kernel(program, "x");
    cohere = cl::Kernel(program, "cohere");
    decohereprob = cl::Kernel(program, "decohereprob");
    decohereamp = cl::Kernel(program, "decohereamp");
    disposeprob = cl::Kernel(program, "disposeprob");
    prob = cl::Kernel(program, "prob");
    swap = cl::Kernel(program, "swap");
    rol = cl::Kernel(program, "rol");
    ror = cl::Kernel(program, "ror");
    inc = cl::Kernel(program, "inc");
    dec = cl::Kernel(program, "dec");
    incc = cl::Kernel(program, "incc");
    decc = cl::Kernel(program, "decc");
    indexedLda = cl::Kernel(program, "indexedLda");
    indexedAdc = cl::Kernel(program, "indexedAdc");
    indexedSbc = cl::Kernel(program, "indexedSbc");
}

OCLEngine* OCLEngine::m_pInstance = NULL;
OCLEngine* OCLEngine::Instance()
{
    if (!m_pInstance)
        m_pInstance = new OCLEngine();
    return m_pInstance;
}

OCLEngine* OCLEngine::Instance(int plat, int dev)
{
    if (!m_pInstance) {
        m_pInstance = new OCLEngine(plat, dev);
    } else {
        std::cout << "Warning: Tried to reinitialize OpenCL environment with platform and device." << std::endl;
    }
    return m_pInstance;
}

} // namespace Qrack
