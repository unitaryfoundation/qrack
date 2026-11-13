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

// Compatibility shim that lets the single CUDA backend compile unchanged as
// either native CUDA (nvcc) or HIP (ROCm clang). When USE_HIP is defined the
// CUDA runtime surface this backend uses is aliased onto its hipXxx
// equivalents, which are 1:1. CUDA builds are unchanged: this header
// includes <cuda_runtime.h> (and <cuda_fp16.h> when half precision is active)
// and defines nothing.

#pragma once

#if defined(USE_HIP)

#include <hip/hip_runtime.h>

#if FPPOW < 5
#include <hip/hip_fp16.h>
#endif

// Runtime API
// hipDeviceProp_t exposes warpSize, multiProcessorCount, totalGlobalMem,
// sharedMemPerBlock, maxThreadsPerMultiProcessor with the same names as
// cudaDeviceProp, so CUDADeviceContext needs no field-level changes.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize

// Streams and events
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaLaunchHostFunc hipLaunchHostFunc
// Host-callback calling-convention tag; no equivalent attribute needed on HIP.
#define CUDART_CB

// Memory
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterDefault hipHostRegisterDefault

#else

#include <cuda_runtime.h>

#if FPPOW < 5
#include "cuda_fp16.h"
#endif

#endif // USE_HIP
