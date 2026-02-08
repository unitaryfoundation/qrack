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

#include "oclengine.hpp"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

namespace Qrack {

#if UINTPOW < 4
#include "qheader_uint8cl.hpp"
#elif UINTPOW < 5
#include "qheader_uint16cl.hpp"
#elif UINTPOW < 6
#include "qheader_uint32cl.hpp"
#else
#include "qheader_uint64cl.hpp"
#endif

#if FPPOW < 5
#include "qheader_halfcl.hpp"
#elif FPPOW < 6
#include "qheader_floatcl.hpp"
#elif FPPOW < 7
#include "qheader_doublecl.hpp"
#else
#include "qheader_quadcl.hpp"
#endif

#include "qenginecl.hpp"

#if ENABLE_ALU
#include "qheader_alucl.hpp"
#if ENABLE_BCD
#include "qheader_bcdcl.hpp"
#endif
#endif

/// "Qrack::OCLEngine" manages the single OpenCL context

// Public singleton methods to get pointers to various methods
DeviceContextPtr OCLEngine::GetDeviceContextPtr(const int64_t& dev)
{
    if ((dev >= GetDeviceCount()) || (dev < -1) || (dev >= ((int64_t)all_device_contexts.size()))) {
        throw std::invalid_argument("Invalid OpenCL device selection");
    }

    if (dev == -1) {
        return default_device_context;
    }

    return all_device_contexts[dev];
}

// clang-format off
const std::vector<OCLKernelHandle> OCLEngine::kernelHandles{
    OCLKernelHandle(OCL_API_APPLY2X2, "apply2x2"),
    OCLKernelHandle(OCL_API_APPLY2X2_SINGLE, "apply2x2single"),
    OCLKernelHandle(OCL_API_APPLY2X2_NORM_SINGLE, "apply2x2normsingle"),
    OCLKernelHandle(OCL_API_APPLY2X2_DOUBLE, "apply2x2double"),
    OCLKernelHandle(OCL_API_APPLY2X2_WIDE, "apply2x2wide"),
    OCLKernelHandle(OCL_API_APPLY2X2_SINGLE_WIDE, "apply2x2singlewide"),
    OCLKernelHandle(OCL_API_APPLY2X2_NORM_SINGLE_WIDE, "apply2x2normsinglewide"),
    OCLKernelHandle(OCL_API_APPLY2X2_DOUBLE_WIDE, "apply2x2doublewide"),
    OCLKernelHandle(OCL_API_PHASE_SINGLE, "phasesingle"),
    OCLKernelHandle(OCL_API_PHASE_SINGLE_WIDE, "phasesinglewide"),
    OCLKernelHandle(OCL_API_INVERT_SINGLE, "invertsingle"),
    OCLKernelHandle(OCL_API_INVERT_SINGLE_WIDE, "invertsinglewide"),
    OCLKernelHandle(OCL_API_UNIFORMLYCONTROLLED, "uniformlycontrolled"),
    OCLKernelHandle(OCL_API_UNIFORMPARITYRZ, "uniformparityrz"),
    OCLKernelHandle(OCL_API_UNIFORMPARITYRZ_NORM, "uniformparityrznorm"),
    OCLKernelHandle(OCL_API_CUNIFORMPARITYRZ, "cuniformparityrz"),
    OCLKernelHandle(OCL_API_X_SINGLE, "xsingle"),
    OCLKernelHandle(OCL_API_X_SINGLE_WIDE, "xsinglewide"),
    OCLKernelHandle(OCL_API_X_MASK, "xmask"),
    OCLKernelHandle(OCL_API_Z_SINGLE, "zsingle"),
    OCLKernelHandle(OCL_API_Z_SINGLE_WIDE, "zsinglewide"),
    OCLKernelHandle(OCL_API_PHASE_PARITY, "phaseparity"),
    OCLKernelHandle(OCL_API_PHASE_MASK, "phasemask"),
    OCLKernelHandle(OCL_API_COMPOSE, "compose"),
    OCLKernelHandle(OCL_API_COMPOSE_WIDE, "compose"),
    OCLKernelHandle(OCL_API_COMPOSE_MID, "composemid"),
    OCLKernelHandle(OCL_API_DECOMPOSEPROB, "decomposeprob"),
    OCLKernelHandle(OCL_API_DECOMPOSEAMP, "decomposeamp"),
    OCLKernelHandle(OCL_API_DISPOSEPROB, "disposeprob"),
    OCLKernelHandle(OCL_API_DISPOSE, "dispose"),
    OCLKernelHandle(OCL_API_PROB, "prob"),
    OCLKernelHandle(OCL_API_CPROB, "cprob"),
    OCLKernelHandle(OCL_API_PROBREG, "probreg"),
    OCLKernelHandle(OCL_API_PROBREGALL, "probregall"),
    OCLKernelHandle(OCL_API_PROBMASK, "probmask"),
    OCLKernelHandle(OCL_API_PROBMASKALL, "probmaskall"),
    OCLKernelHandle(OCL_API_PROBPARITY, "probparity"),
    OCLKernelHandle(OCL_API_FORCEMPARITY, "forcemparity"),
    OCLKernelHandle(OCL_API_EXPPERM, "expperm"),
    OCLKernelHandle(OCL_API_ROL, "rol"),
#if ENABLE_ALU
    OCLKernelHandle(OCL_API_INC, "inc"),
    OCLKernelHandle(OCL_API_CINC, "cinc"),
    OCLKernelHandle(OCL_API_INCDECC, "incdecc"),
    OCLKernelHandle(OCL_API_INCS, "incs"),
    OCLKernelHandle(OCL_API_INCDECSC_1, "incdecsc1"),
    OCLKernelHandle(OCL_API_INCDECSC_2, "incdecsc2"),
#if ENABLE_BCD
    OCLKernelHandle(OCL_API_INCBCD, "incbcd"),
    OCLKernelHandle(OCL_API_INCDECBCDC, "incdecbcdc"),
#endif
    OCLKernelHandle(OCL_API_MUL, "mul"),
    OCLKernelHandle(OCL_API_DIV, "div"),
    OCLKernelHandle(OCL_API_MULMODN_OUT, "mulmodnout"),
    OCLKernelHandle(OCL_API_IMULMODN_OUT, "imulmodnout"),
    OCLKernelHandle(OCL_API_POWMODN_OUT, "powmodnout"),
    OCLKernelHandle(OCL_API_CMUL, "cmul"),
    OCLKernelHandle(OCL_API_CDIV, "cdiv"),
    OCLKernelHandle(OCL_API_CMULMODN_OUT, "cmulmodnout"),
    OCLKernelHandle(OCL_API_CIMULMODN_OUT, "cimulmodnout"),
    OCLKernelHandle(OCL_API_CPOWMODN_OUT, "cpowmodnout"),
    OCLKernelHandle(OCL_API_FULLADD, "fulladd"),
    OCLKernelHandle(OCL_API_IFULLADD, "ifulladd"),
    OCLKernelHandle(OCL_API_INDEXEDLDA, "indexedLda"),
    OCLKernelHandle(OCL_API_INDEXEDADC, "indexedAdc"),
    OCLKernelHandle(OCL_API_INDEXEDSBC, "indexedSbc"),
    OCLKernelHandle(OCL_API_HASH, "hash"),
    OCLKernelHandle(OCL_API_CPHASEFLIPIFLESS, "cphaseflipifless"),
    OCLKernelHandle(OCL_API_PHASEFLIPIFLESS, "phaseflipifless"),
#endif
    OCLKernelHandle(OCL_API_APPROXCOMPARE, "approxcompare"),
    OCLKernelHandle(OCL_API_NORMALIZE, "nrmlze"),
    OCLKernelHandle(OCL_API_NORMALIZE_WIDE, "nrmlzewide"),
    OCLKernelHandle(OCL_API_UPDATENORM, "updatenorm"),
    OCLKernelHandle(OCL_API_APPLYM, "applym"),
    OCLKernelHandle(OCL_API_APPLYMREG, "applymreg"),
    OCLKernelHandle(OCL_API_CLEARBUFFER, "clearbuffer"),
    OCLKernelHandle(OCL_API_SHUFFLEBUFFERS, "shufflebuffers")
};
// clang-format on

const std::string OCLEngine::binary_file_prefix("qrack_ocl_dev_");
const std::string OCLEngine::binary_file_ext(".ir");

std::vector<DeviceContextPtr> OCLEngine::GetDeviceContextPtrVector() { return all_device_contexts; }
void OCLEngine::SetDeviceContextPtrVector(std::vector<DeviceContextPtr> vec, DeviceContextPtr dcp)
{
    all_device_contexts = vec;
    if (!!dcp) {
        default_device_context = dcp;
    }
}

void OCLEngine::SetDefaultDeviceContext(DeviceContextPtr dcp) { default_device_context = dcp; }

cl::Program OCLEngine::MakeProgram(bool buildFromSource, std::string path, std::shared_ptr<OCLDeviceContext> devCntxt)
{
    FILE* clBinFile;
    cl::Program program;
    cl_int buildError = -1;
    std::vector<int> binaryStatus;
    if (!buildFromSource && (clBinFile = fopen(path.c_str(), "r"))) {
        struct stat statSize;
        if (fstat(fileno(clBinFile), &statSize)) {
            std::cout << "Binary error: Invalid file fstat result. (Falling back to JIT.)" << std::endl;
        } else {
            size_t lSize = statSize.st_size;
            std::vector<unsigned char> buffer(lSize);
            size_t lSizeResult = fread(&buffer[0U], sizeof(unsigned char), lSize, clBinFile);
            fclose(clBinFile);

            if (lSizeResult != lSize) {
                std::cout << "Binary warning: Binary file size and read result length do not match. (Attempting to "
                             "build anyway.)"
                          << std::endl;
            }

#if ENABLE_SNUCL
            program = cl::Program(devCntxt->context, { devCntxt->device },
                { std::pair<const void*, size_t>(&buffer[0U], buffer.size()) }, &binaryStatus, &buildError);
#else
            program = cl::Program(devCntxt->context, { devCntxt->device }, { buffer }, &binaryStatus, &buildError);
#endif

            if ((buildError != CL_SUCCESS) || (binaryStatus[0U] != CL_SUCCESS)) {
                std::cout << "Binary error: " << buildError << ", " << binaryStatus[0U] << " (Falling back to JIT.)"
                          << std::endl;
            } else {
                std::cout << "Loaded binary from: " << path << std::endl;
            }
        }
    }

    // If, either, there are no cached binaries, or binary loading failed, then fall back to JIT.
    if (buildError == CL_SUCCESS) {
        return program;
    }

    cl::Program::Sources sources;
#if UINTPOW < 4
    sources.push_back({ (const char*)qheader_uint8_cl, (long unsigned int)qheader_uint8_cl_len });
#elif UINTPOW < 5
    sources.push_back({ (const char*)qheader_uint16_cl, (long unsigned int)qheader_uint16_cl_len });
#elif UINTPOW < 6
    sources.push_back({ (const char*)qheader_uint32_cl, (long unsigned int)qheader_uint32_cl_len });
#else
    sources.push_back({ (const char*)qheader_uint64_cl, (long unsigned int)qheader_uint64_cl_len });
#endif

#if FPPOW < 5
    sources.push_back({ (const char*)qheader_half_cl, (long unsigned int)qheader_half_cl_len });
#elif FPPOW < 6
    sources.push_back({ (const char*)qheader_float_cl, (long unsigned int)qheader_float_cl_len });
#elif FPPOW < 7
    sources.push_back({ (const char*)qheader_double_cl, (long unsigned int)qheader_double_cl_len });
#else
    sources.push_back({ (const char*)qheader_quad_cl, (long unsigned int)qheader_quad_cl_len });
#endif

    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

#if ENABLE_ALU
    sources.push_back({ (const char*)qheader_alu_cl, (long unsigned int)qheader_alu_cl_len });
#if ENABLE_BCD
    sources.push_back({ (const char*)qheader_bcd_cl, (long unsigned int)qheader_bcd_cl_len });
#endif
#endif

    program = cl::Program(devCntxt->context, sources);
    std::cout << "Building JIT." << std::endl;

    return program;
}

void OCLEngine::SaveBinary(cl::Program program, std::string path, std::string fileName)
{
    std::vector<size_t> clBinSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    size_t clBinSize = 0U;
    int64_t clBinIndex = 0;

    for (size_t i = 0U; i < clBinSizes.size(); ++i) {
        if (clBinSizes[i]) {
            clBinSize = clBinSizes[i];
            clBinIndex = i;
            break;
        }
    }

    std::cout << "Binary size:" << clBinSize << std::endl;

#if defined(_WIN32) && !defined(__CYGWIN__)
    int err = _mkdir(path.c_str());
#else
    int err = mkdir(path.c_str(), 0700);
#endif
    if (err != -1) {
        std::cout << "Making directory: " << path << std::endl;
    }

    FILE* clBinFile = fopen((path + fileName).c_str(), "w");
#if ENABLE_SNUCL
    std::vector<char*> clBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
    char* clBinary = clBinaries[clBinIndex];
    fwrite(clBinary, clBinSize, sizeof(char), clBinFile);
#else
    std::vector<std::vector<unsigned char>> clBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
    std::vector<unsigned char> clBinary = clBinaries[clBinIndex];
    fwrite(&clBinary[0U], clBinSize, sizeof(unsigned char), clBinFile);
#endif
    fclose(clBinFile);
}

InitOClResult OCLEngine::InitOCL(
    bool buildFromSource, bool saveBinaries, std::string home, std::vector<int64_t> maxAllocVec)
{
    if (home == "*") {
        home = GetDefaultBinaryPath();
    }
    // get all platforms (drivers), e.g. NVIDIA

    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> all_devices;
    std::vector<int64_t> device_platform_id;
    cl::Platform default_platform;
    cl::Device default_device;
    std::vector<DeviceContextPtr> all_dev_contexts;
    DeviceContextPtr default_dev_context;

    cl::Platform::get(&all_platforms);

    if (all_platforms.empty()) {
        std::cout << " No platforms found. Check OpenCL installation!\n";

        return InitOClResult();
    }

    // get all devices
    std::vector<cl::Platform> devPlatVec;
    std::vector<std::vector<cl::Device>> all_platforms_devices;
    std::vector<bool> all_devices_is_gpu;
    std::vector<bool> all_devices_is_cpu;
    for (size_t i = 0U; i < all_platforms.size(); ++i) {
        all_platforms_devices.push_back(std::vector<cl::Device>());
        all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &(all_platforms_devices[i]));
        for (size_t j = 0U; j < all_platforms_devices[i].size(); ++j) {
            // VirtualCL seems to break if the assignment constructor of cl::Platform is used here from the original
            // list. Assigning the object from a new query is always fine, though. (They carry the same underlying
            // platform IDs.)
            std::vector<cl::Platform> temp_platforms;
            cl::Platform::get(&temp_platforms);
            devPlatVec.push_back(temp_platforms[i]);
            device_platform_id.push_back(i);
        }
        all_devices.insert(all_devices.end(), all_platforms_devices[i].begin(), all_platforms_devices[i].end());

        // Linux implements `cl::Device` relation operators, including equality, but Mac considers OpenCL "deprecated,"
        // and other compilers might not see a strict need in OpenCL implementation standard for a `cl::Device` equality
        // operator, which would allow the use of `std::find()`.
        std::vector<cl::Device> gpu_devices;
        all_platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices);
        std::vector<bool> gpu_to_insert(all_platforms_devices[i].size(), false);
        for (size_t j = 0U; j < gpu_devices.size(); ++j) {
            for (size_t k = 0U; k < all_platforms_devices[i].size(); ++k) {
                if (gpu_devices[j].getInfo<CL_DEVICE_NAME>() == all_platforms_devices[i][j].getInfo<CL_DEVICE_NAME>()) {
                    // Assuming all devices with the same name are identical vendor, line, and model, this works.
                    gpu_to_insert[k] = true;
                }
            }
        }
        all_devices_is_gpu.insert(all_devices_is_gpu.end(), gpu_to_insert.begin(), gpu_to_insert.end());

        std::vector<cl::Device> cpu_devices;
        all_platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &cpu_devices);
        std::vector<bool> cpu_to_insert(all_platforms_devices[i].size(), false);
        for (size_t j = 0U; j < cpu_devices.size(); ++j) {
            for (size_t k = 0U; k < all_platforms_devices[i].size(); ++k) {
                if (cpu_devices[j].getInfo<CL_DEVICE_NAME>() == all_platforms_devices[i][j].getInfo<CL_DEVICE_NAME>()) {
                    // Assuming all devices with the same name are identical vendor, line, and model, this works.
                    cpu_to_insert[k] = true;
                }
            }
        }
        all_devices_is_cpu.insert(all_devices_is_cpu.end(), cpu_to_insert.begin(), cpu_to_insert.end());
    }
    if (all_devices.empty()) {
        std::cout << " No devices found. Check OpenCL installation!\n";

        return InitOClResult();
    }

    int64_t deviceCount = all_devices.size();
    // prefer the last device because that's usually a GPU or accelerator; device[0U] is usually the CPU
    int64_t dev = deviceCount - 1;
    if (getenv("QRACK_OCL_DEFAULT_DEVICE")) {
        dev = std::stoi(std::string(getenv("QRACK_OCL_DEFAULT_DEVICE")));
        if ((dev < 0) || (dev > (deviceCount - 1))) {
            std::cout << "WARNING: Invalid QRACK_OCL_DEFAULT_DEVICE selection. (Falling back to highest index device "
                         "as default.)"
                      << std::endl;
            dev = deviceCount - 1;
        }
    }

    // create the programs that we want to execute on the devices
    int64_t plat_id = -1;
    std::vector<cl::Context> all_contexts;
    std::vector<std::string> all_filenames;
    for (int64_t i = 0; i < deviceCount; ++i) {
        // a context is like a "runtime link" to the device and platform;
        // i.e. communication is possible
        if (device_platform_id[i] != plat_id) {
            plat_id = device_platform_id[i];
            all_contexts.push_back(cl::Context(all_platforms_devices[plat_id]));
        }
        const std::string devName(all_devices[i].getInfo<CL_DEVICE_NAME>());
        const bool useHostRam = all_devices_is_cpu[i] || (devName.find("Intel(R) UHD") != std::string::npos) ||
            (devName.find("Iris") != std::string::npos);
        DeviceContextPtr devCntxt =
            std::make_shared<OCLDeviceContext>(devPlatVec[i], all_devices[i], all_contexts[all_contexts.size() - 1U], i,
                plat_id, maxAllocVec[i % maxAllocVec.size()], all_devices_is_gpu[i], all_devices_is_cpu[i], useHostRam);

        std::string fileName = binary_file_prefix + all_devices[i].getInfo<CL_DEVICE_NAME>() + binary_file_ext;
        std::replace(fileName.begin(), fileName.end(), ' ', '_');
        std::string clBinName = home + fileName;

        std::cout << "Device #" << i << ", ";
        cl::Program program = MakeProgram(buildFromSource, clBinName, devCntxt);

        cl_int buildError =
            program.build({ all_devices[i] }, "-cl-strict-aliasing -cl-denorms-are-zero -cl-fast-relaxed-math");
        if (buildError != CL_SUCCESS) {
            std::cout << "Error building for device #" << i << ": " << buildError << ", "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(all_devices[i])
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[i]) << std::endl;

            // The default device was set above to be the last device in the list. If we can't compile for it, we
            // use the first device. If the default is the first device, and we can't compile for it, then we don't
            // have any devices that can compile at all, and the environment needs to be fixed by the user.
            if (i == dev) {
                default_dev_context = all_dev_contexts[0U];
                default_platform = all_platforms[0U];
                default_device = all_devices[0U];
            }

            continue;
        }

        all_dev_contexts.push_back(devCntxt);

        for (unsigned int j = 0U; j < kernelHandles.size(); ++j) {
            all_dev_contexts[i]->calls[kernelHandles[j].oclapi] =
                cl::Kernel(program, kernelHandles[j].kernelname.c_str());
            all_dev_contexts[i]->mutexes.emplace(kernelHandles[j].oclapi, new std::mutex);
        }

        std::vector<std::string>::iterator fileNameIt = std::find(all_filenames.begin(), all_filenames.end(), fileName);
        if (saveBinaries && (fileNameIt == all_filenames.end())) {
            std::cout << "OpenCL program #" << i << ", ";
            SaveBinary(program, home, fileName);
        }

        if (i == dev) {
            default_dev_context = all_dev_contexts[i];
            default_platform = all_platforms[plat_id];
            default_device = all_devices[i];
        }
    }

    // For VirtualCL support, the device info can only be accessed AFTER all contexts are created.
    std::cout << "Default platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Default device: #" << dev << ", " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    for (int64_t i = 0; i < deviceCount; ++i) {
        std::cout << "OpenCL device #" << i << ": " << all_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }

    return InitOClResult(all_dev_contexts, default_dev_context);
}

OCLEngine::OCLEngine()
    : maxActiveAllocSizes(1U, -1)
{
    if (getenv("QRACK_MAX_ALLOC_MB")) {
        std::string devListStr = std::string(getenv("QRACK_MAX_ALLOC_MB"));
        maxActiveAllocSizes.clear();
        if (devListStr.compare("")) {
            std::stringstream devListStr_stream(devListStr);
            // See
            // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
            std::regex re("[.]");
            while (devListStr_stream.good()) {
                std::string term;
                getline(devListStr_stream, term, ',');
                // the '-1' is what makes the regex split (-1 := what was not matched)
                std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
                std::vector<std::string> tokens{ first, last };
                if (tokens.size() == 1U) {
                    maxActiveAllocSizes.push_back(stoi(term));
                    if (maxActiveAllocSizes.back() >= 0) {
                        maxActiveAllocSizes.back() = maxActiveAllocSizes.back() << 20U;
                    }
                    continue;
                }
                const unsigned maxI = stoi(tokens[0U]);
                std::vector<int64_t> limits(tokens.size() - 1U);
                for (unsigned i = 1U; i < tokens.size(); ++i) {
                    limits[i - 1U] = stoi(tokens[i]);
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < limits.size(); ++j) {
                        maxActiveAllocSizes.push_back(limits[j]);
                        if (maxActiveAllocSizes.back() >= 0) {
                            maxActiveAllocSizes.back() = maxActiveAllocSizes.back() << 20U;
                        }
                    }
                }
            }
        }
    }

    InitOClResult initResult = InitOCL(false, false, "*", maxActiveAllocSizes);
    SetDeviceContextPtrVector(initResult.all_dev_contexts, initResult.default_dev_context);
    activeAllocSizes = std::vector<size_t>(initResult.all_dev_contexts.size());
}

} // namespace Qrack
