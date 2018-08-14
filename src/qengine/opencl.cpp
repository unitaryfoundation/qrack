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

#include <memory>

#include "oclengine.hpp"
#include "qengine_opencl.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp,
    int devID, bool partialInit, complex phaseFac)
    : QInterface(qBitCount, rgp)
    , stateVec(NULL)
    , deviceID(-1)
    , nrmArray(NULL)
{
    doNormalize = true;
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    runningNorm = partialInit ? ZERO_R1 : ONE_R1;
    SetQubitCount(qBitCount);

    stateVec = AllocStateVec(maxQPower);
    std::fill(stateVec, stateVec + maxQPower, complex(ZERO_R1, ZERO_R1));
    if (!partialInit) {
        if (phaseFac == complex(-999.0, -999.0)) {
            real1 angle = Rand() * 2.0 * PI_R1;
            stateVec[initState] = complex(cos(angle), sin(angle));
        } else {
            stateVec[initState] = phaseFac;
        }
    }

    InitOCL(devID);
}

QEngineOCL::QEngineOCL(QEngineOCLPtr toCopy)
    : QInterface(toCopy->qubitCount, toCopy->rand_generator, toCopy->doNormalize)
    , stateVec(NULL)
    , deviceID(-1)
    , nrmArray(NULL)
{
    CopyState(toCopy);
    InitOCL(toCopy->deviceID);
}

void QEngineOCL::LockSync(cl_int flags)
{
    queue.enqueueMapBuffer(*stateBuffer, CL_TRUE, flags, 0, sizeof(complex) * maxQPower);
}

void QEngineOCL::UnlockSync()
{
    queue.enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue.finish();
}

void QEngineOCL::Sync()
{
    LockSync(CL_MAP_READ);
    UnlockSync();
}

size_t QEngineOCL::FixWorkItemCount(size_t maxI, size_t wic)
{
    if (wic > maxI) {
        wic = maxI;
    }
    return wic;
}

size_t QEngineOCL::FixGroupSize(size_t wic, size_t gs)
{
    if (gs > (wic / procElemCount)) {
        gs = (wic / procElemCount);
        if (gs == 0) {
            gs = 1;
        }
    }
    size_t frac = wic / gs;
    while ((frac * gs) != wic) {
        gs++;
        frac = wic / gs;
    }
    return gs;
}

void QEngineOCL::CopyState(QInterfacePtr orig)
{
    queue.finish();

    knowIsPhaseSeparable = false;

    /* Set the size and reset the stateVec to the correct size. */
    SetQubitCount(orig->GetQubitCount());

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, nStateVec);
    ResetStateVec(nStateVec, nStateBuffer);

    QEngineOCLPtr src = std::dynamic_pointer_cast<QEngineOCL>(orig);
    runningNorm = src->runningNorm;
    src->LockSync(CL_MAP_READ);
    LockSync(CL_MAP_WRITE);
    std::copy(src->stateVec, src->stateVec + (1 << (src->qubitCount)), stateVec);
    src->UnlockSync();
    UnlockSync();
}

real1 QEngineOCL::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    complex amp[1];
    queue.finish();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, sizeof(complex) * fullRegister, sizeof(complex), amp);
    return norm(amp[0]);
}

void QEngineOCL::SetDevice(const int& dID, const bool& forceReInit)
{
    bool didInit = (nrmArray != NULL);

    if (didInit) {
        // If we're "switching" to the device we already have, don't reinitialize.
        if ((!forceReInit) && (dID == deviceID)) {
            return;
        }

        // Otherwise, we're about to switch to a new device, so finish the queue, first.
        Sync();
    }

    deviceID = dID;
    device_context = OCLEngine::Instance()->GetDeviceContextPtr(deviceID);
    context = device_context->context;
    queue = device_context->queue;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_UPDATENORM);
    bitCapInt oldNrmGroupCount = nrmGroupCount;
    nrmGroupSize = ocl.call.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_context->device);
    procElemCount = device_context->device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    nrmGroupCount = procElemCount * 64 * nrmGroupSize;
    maxWorkItems = device_context->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
    if (nrmGroupCount > maxWorkItems) {
        nrmGroupCount = maxWorkItems;
    }
    if (nrmGroupSize > (nrmGroupCount / procElemCount)) {
        nrmGroupSize = (nrmGroupCount / procElemCount);
        if (nrmGroupSize == 0) {
            nrmGroupSize = 1;
        }
    }
    size_t frac = nrmGroupCount / nrmGroupSize;
    while ((frac * nrmGroupSize) != nrmGroupCount) {
        nrmGroupSize++;
        frac = nrmGroupCount / nrmGroupSize;
    }

    size_t nrmVecAlignSize =
        ((sizeof(real1) * nrmGroupCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(real1) * nrmGroupCount);

    if (!didInit) {
#ifdef __APPLE__
        posix_memalign(&nrmArray, ALIGN_SIZE, nrmVecAlignSize);
#else
        nrmArray = (real1*)aligned_alloc(ALIGN_SIZE, nrmVecAlignSize);
#endif
    } else if (nrmGroupCount != oldNrmGroupCount) {
        delete[] nrmArray;
#ifdef __APPLE__
        posix_memalign(&nrmArray, ALIGN_SIZE, nrmVecAlignSize);
#else
        nrmArray = (real1*)aligned_alloc(ALIGN_SIZE, nrmVecAlignSize);
#endif
    }

    // create buffers on device (allocate space on GPU)
    stateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, stateVec);
    cmplxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(complex) * CMPLX_NORM_LEN);
    ulongBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * BCI_ARG_LEN);
    nrmBuffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * nrmGroupCount, nrmArray);
    // GPUs can't always tolerate uninitialized host memory, even if they're not reading from it
    queue.enqueueFillBuffer(nrmBuffer, ZERO_R1, 0, sizeof(real1) * nrmGroupCount);
    queue.finish();
}

void QEngineOCL::SetQubitCount(bitLenInt qb)
{
    qubitCount = qb;
    maxQPower = 1 << qubitCount;
}

void QEngineOCL::InitOCL(int devID) { SetDevice(devID); }

void QEngineOCL::ResetStateVec(complex* nStateVec, BufferPtr nStateBuffer)
{
    queue.finish();
    stateBuffer = nStateBuffer;
    free(stateVec);
    stateVec = nStateVec;
}

void QEngineOCL::SetPermutation(bitCapInt perm)
{
    queue.finish();
    knowIsPhaseSeparable = true;
    isPhaseSeparable = true;
    queue.enqueueFillBuffer(*stateBuffer, complex(ZERO_R1, ZERO_R1), 0, sizeof(complex) * maxQPower);
    real1 angle = Rand() * 2.0 * PI_R1;
    complex amp = complex(cos(angle), sin(angle));
    queue.finish();
    queue.enqueueFillBuffer(*stateBuffer, amp, sizeof(complex) * perm, sizeof(complex));
    runningNorm = ONE_R1;
    queue.finish();
}

void QEngineOCL::DispatchCall(
    OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], unsigned char* values, bitCapInt valuesPower, bool isParallel)
{
    queue.finish();
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, nStateVec);
    queue.enqueueFillBuffer(*nStateBuffer, complex(ZERO_R1, ZERO_R1), 0, sizeof(complex) * maxQPower);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    OCLDeviceCall ocl = device_context->Reserve(api_call);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, *nStateBuffer);
    cl::Buffer loadBuffer;
    if (values) {
        if (isParallel) {
            loadBuffer = cl::Buffer(
                context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        } else {
            loadBuffer = cl::Buffer(
                context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        }
        ocl.call.setArg(3, loadBuffer);
    }

    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();
    ResetStateVec(nStateVec, nStateBuffer);
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    complex cmplx[CMPLX_NORM_LEN];
    for (int i = 0; i < 4; i++) {
        cmplx[i] = mtrx[i];
    }
    cmplx[4] = complex(
        (doNormalize && (bitCount == 1) && (runningNorm > min_norm)) ? (ONE_R1 / sqrt(runningNorm)) : ONE_R1, ZERO_R1);
    queue.enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(complex) * CMPLX_NORM_LEN, cmplx);
    queue.flush();

    bitCapInt maxI = maxQPower >> bitCount;
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    bitCapInt bciArgs[BCI_ARG_LEN] = { bitCount, maxI, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < bitCount; i++) {
        bciArgs[4 + i] = qPowersSorted[i];
    }
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    doCalcNorm &= (bitCount == 1);

    OCLAPI api_call;
    if (doCalcNorm) {
        api_call = OCL_API_APPLY2X2_NORM;
    } else {
        api_call = OCL_API_APPLY2X2;
    }
    OCLDeviceCall ocl = device_context->Reserve(api_call);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, cmplxBuffer);
    ocl.call.setArg(2, ulongBuffer);
    if (doCalcNorm) {
        ocl.call.setArg(3, nrmBuffer);
    }
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();

    if (doCalcNorm) {
        queue.enqueueMapBuffer(nrmBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * ngc);
        runningNorm = ZERO_R1;
        for (unsigned long int i = 0; i < ngc; i++) {
            runningNorm += nrmArray[i];
        }
        queue.enqueueUnmapMemObject(nrmBuffer, nrmArray);
        queue.finish();
    }
}

void QEngineOCL::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapInt powerTest = result ? qPower : 0;

    complex cmplx[CMPLX_NORM_LEN] = { nrm, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1) };
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, qPower, powerTest, 0, 0, 0, 0, 0, 0, 0 };
    queue.enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(complex) * CMPLX_NORM_LEN, cmplx);
    queue.flush();
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPLYM);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, cmplxBuffer);
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();
}

bitLenInt QEngineOCL::Cohere(QEngineOCLPtr toCopy)
{
    bitLenInt result = qubitCount;

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    if ((toCopy->doNormalize) && (toCopy->runningNorm != ONE_R1)) {
        toCopy->NormalizeState();
    }

    bitCapInt nQubitCount = qubitCount + toCopy->qubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy->qubitCount)) - 1) << qubitCount;
    bitCapInt bciArgs[BCI_ARG_LEN] = { nMaxQPower, startMask, endMask, qubitCount, 0, 0, 0, 0, 0, 0 };

    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    SetQubitCount(nQubitCount);

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, nStateVec);

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_COHERE);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, *(toCopy->stateBuffer));
    ocl.call.setArg(2, ulongBuffer);
    ocl.call.setArg(3, *nStateBuffer);

    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();

    ResetStateVec(nStateVec, nStateBuffer);
    runningNorm = ONE_R1;

    return result;
}

void QEngineOCL::DecohereDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr destination)
{
    // "Dispose" is basically the same as decohere, except "Dispose" throws the removed bits away.

    if (length == 0) {
        return;
    }

    // Depending on whether we Decohere or Dispose, we have optimized kernels.
    OCLAPI api_call;
    if (destination != nullptr) {
        api_call = OCL_API_DECOHEREPROB;
    } else {
        api_call = OCL_API_DISPOSEPROB;
    }
    OCLDeviceCall prob_call = device_context->Reserve(api_call);
    OCLDeviceCall amp_call = device_context->Reserve(OCL_API_DECOHEREAMP);

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);
    bitCapInt bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // The "remainder" bits will always be maintained.
    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower];
    cl::Buffer probBuffer1 = cl::Buffer(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateProb);
    cl::Buffer angleBuffer1 = cl::Buffer(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateAngle);

    queue.finish();
    // These arguments are common to both kernels.
    prob_call.call.setArg(0, *stateBuffer);
    prob_call.call.setArg(1, ulongBuffer);
    prob_call.call.setArg(2, probBuffer1);
    prob_call.call.setArg(3, angleBuffer1);

    // The removed "part" is only necessary for Decohere.
    real1* partStateProb = nullptr;
    real1* partStateAngle = nullptr;
    cl::Buffer probBuffer2, angleBuffer2;
    if (destination != nullptr) {
        partStateProb = new real1[partPower]();
        partStateAngle = new real1[partPower];
        probBuffer2 =
            cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateProb);
        angleBuffer2 =
            cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateAngle);

        prob_call.call.setArg(4, probBuffer2);
        prob_call.call.setArg(5, angleBuffer2);
    }

    // Call the kernel that calculates bit probability and angle.
    queue.enqueueNDRangeKernel(prob_call.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.flush();

    if ((maxQPower - partPower) <= 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    // Wait as long as possible before joining the kernel.
    queue.finish();

    // groupSize = amp_call.call.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_context->device);

    // If we Decohere, calculate the state of the bit system removed.
    if (destination != nullptr) {
        bciArgs[0] = partPower;
        queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt), bciArgs);
        queue.flush();

        size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        queue.finish();
        amp_call.call.setArg(0, probBuffer2);
        amp_call.call.setArg(1, angleBuffer2);
        amp_call.call.setArg(2, ulongBuffer);
        amp_call.call.setArg(3, *(destination->stateBuffer));

        queue.enqueueNDRangeKernel(amp_call.call, cl::NullRange, // kernel, offset
            cl::NDRange(ngc2), // global number of work items
            cl::NDRange(ngs2)); // local number (per group)

        queue.finish();

        delete[] partStateProb;
        delete[] partStateAngle;
    }

    // If we either Decohere or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPower;
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt), bciArgs);
    queue.flush();

    ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    ngs = FixGroupSize(ngc, nrmGroupSize);

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, nStateVec);

    queue.finish();
    amp_call.call.setArg(0, probBuffer1);
    amp_call.call.setArg(1, angleBuffer1);
    amp_call.call.setArg(2, ulongBuffer);
    amp_call.call.setArg(3, *nStateBuffer);

    queue.enqueueNDRangeKernel(amp_call.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();

    ResetStateVec(nStateVec, nStateBuffer);
    runningNorm = ONE_R1;
    if (destination != nullptr) {
        destination->runningNorm = ONE_R1;
    }

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
}

void QEngineOCL::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    DecohereDispose(start, length, std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length) { DecohereDispose(start, length, (QEngineOCLPtr) nullptr); }

/// PSEUDO-QUANTUM Check whether bit phase is separable in permutation basis
bool QEngineOCL::IsPhaseSeparable(bool forceCheck)
{
    if ((!forceCheck) && knowIsPhaseSeparable) {
        return isPhaseSeparable;
    }

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    bitLenInt* isAllSame = new bitLenInt[ngc];
    std::fill(isAllSame, isAllSame + ngc, 1);
    real1* phases = new real1[ngc];
    std::fill(phases, phases + ngc, -PI_R1 * 2);

    cl::Buffer isAllSameBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(bitLenInt) * ngc, isAllSame);
    cl::Buffer phasesBuffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * ngc, phases);

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_ISPHASESEPARABLE);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, phasesBuffer);
    ocl.call.setArg(3, isAllSameBuffer);

    // Note that the global size is 1 (serial). This is because the kernel is not very easily parallelized, but we
    // ultimately want to offload all manipulation of stateVec from host code to OpenCL kernels.
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    bool toRet = true;
    queue.enqueueMapBuffer(isAllSameBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(bitLenInt) * ngc);
    for (size_t i = 0; i < ngc; i++) {
        toRet &= (isAllSame[i] == 1);
    }
    queue.enqueueUnmapMemObject(isAllSameBuffer, isAllSame);
    queue.finish();

    if (toRet) {
        queue.enqueueMapBuffer(phasesBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * ngc);
        real1 phase = -PI_R1 * 2;
        for (size_t i = 0; i < ngc; i++) {
            if (phase < (-PI_R1)) {
                if (phases[i] >= (-PI_R1)) {
                    phase = phases[i];
                }
                continue;
            }

            real1 diff = phases[i] - phase;
            if (diff < ZERO_R1) {
                diff = -diff;
            }
            if (diff > PI_R1) {
                diff = (2 * PI_R1) - diff;
            }
            if (diff > min_norm) {
                toRet = false;
                break;
            }
        }
        queue.enqueueUnmapMemObject(phasesBuffer, phases);
        queue.finish();
    }

    delete[] isAllSame;
    delete[] phases;

    knowIsPhaseSeparable = true;
    isPhaseSeparable = toRet;

    return toRet;
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1 QEngineOCL::Prob(bitLenInt qubit)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt qPower = 1 << qubit;
    real1 oneChance = ZERO_R1;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_PROB);
    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, nrmBuffer);

    // Note that the global size is 1 (serial). This is because the kernel is not very easily parallelized, but we
    // ultimately want to offload all manipulation of stateVec from host code to OpenCL kernels.
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.enqueueMapBuffer(nrmBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * ngc);
    for (size_t i = 0; i < ngc; i++) {
        oneChance += nrmArray[i];
    }
    queue.enqueueUnmapMemObject(nrmBuffer, nrmArray);
    queue.finish();

    if (oneChance > ONE_R1)
        oneChance = ONE_R1;

    return oneChance;
}

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QEngineOCL::X(bitLenInt start, bitLenInt length)
{
    if (length == 1) {
        X(start);
        return;
    }

    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ regMask;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, 0, 0, 0, 0, 0, 0, 0 };

    DispatchCall(OCL_API_X, bciArgs);
}

/// Bitwise swap
void QEngineOCL::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    if (start1 == start2) {
        return;
    }

    bitCapInt reg1Mask = ((1 << length) - 1) << start1;
    bitCapInt reg2Mask = ((1 << length) - 1) << start2;
    bitCapInt otherMask = maxQPower - 1;
    otherMask ^= reg1Mask | reg2Mask;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, reg1Mask, reg2Mask, otherMask, start1, start2, 0, 0, 0, 0 };

    DispatchCall(OCL_API_SWAP, bciArgs);
}

void QEngineOCL::ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    DispatchCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

/// "Circular shift right" - shift bits right, and carry first bits.
void QEngineOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROR, shift, start, length); }

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & ~(regMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, toMod, 0, 0, 0, 0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INT(OCL_API_INC, toAdd, start, length);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DEC(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    INT(OCL_API_DEC, toSub, start, length);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~(regMask | carryMask));

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, regMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTC(OCL_API_INCC, toAdd, start, length, carryIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTC(OCL_API_DECC, toSub, start, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{

    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ regMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, overflowMask, start, toMod, 0, 0,
        0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCS(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, toAdd, start, length, overflowIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECS(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_DECS, toSub, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt overflowIndex, const bitLenInt carryIndex)
{
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, overflowMask, carryMask,
        start, toMod, 0, 0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCSC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex,
    const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTSC(OCL_API_INCSC_1, toAdd, start, length, overflowIndex, carryIndex);
}

/** Subtract integer (with sign, with carry) */
void QEngineOCL::DECSC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex,
    const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTSC(OCL_API_DECSC_1, toSub, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCSC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTSC(OCL_API_INCSC_2, toAdd, start, length, carryIndex);
}

/** Subtract integer (with sign, with carry) */
void QEngineOCL::DECSC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTSC(OCL_API_DECSC_2, toSub, start, length, carryIndex);
}

/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ inOutMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, inOutMask, otherMask, start, toMod, nibbleCount, 0, 0, 0, 0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineOCL::INCBCD(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, toAdd, start, length);
}

/** Subtract integer (BCD) */
void QEngineOCL::DECBCD(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_DECBCD, toSub, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, carryMask, start, toMod, nibbleCount, 0, 0,
        0 };

    DispatchCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCBCDC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTBCDC(OCL_API_INCBCDC, toAdd, start, length, carryIndex);
}

/** Subtract integer (BCD, with carry) */
void QEngineOCL::DECBCDC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTBCDC(OCL_API_DECBCDC, toSub, start, length, carryIndex);
}

/** Multiply by integer */
void QEngineOCL::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bool clearCarry)
{
    if (clearCarry) {
        SetReg(carryStart, length, 0);
    }
    if ((length > 0) && (toMul != 1)) {
        bitCapInt lowMask = (1 << length) - 1;
        bitCapInt highMask = lowMask << length;
        bitCapInt inOutMask = lowMask << inOutStart;
        bitCapInt carryMask = lowMask << carryStart;
        bitCapInt otherMask = (maxQPower - 1) ^ (inOutMask | carryMask);

        bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, toMul, lowMask, highMask, inOutMask, carryMask,
            otherMask, length, inOutStart, carryStart };

        DispatchCall(OCL_API_MUL, bciArgs);
    }
}

/** Divide by integer */
void QEngineOCL::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
        throw "DIV by zero";
    }
    if ((length > 0) && (toDiv != 1)) {
        bitCapInt lowMask = (1 << length) - 1;
        bitCapInt highMask = lowMask << length;
        bitCapInt inOutMask = lowMask << inOutStart;
        bitCapInt carryMask = lowMask << carryStart;
        bitCapInt otherMask = (maxQPower - 1) ^ (inOutMask | carryMask);

        bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, toDiv, lowMask, highMask, inOutMask, carryMask,
            otherMask, length, inOutStart, carryStart };

        DispatchCall(OCL_API_DIV, bciArgs);
    }
}

/** Controlled multiplication by integer */
void QEngineOCL::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit,
    bitLenInt length, bool clearCarry)
{
    if (clearCarry) {
        SetReg(carryStart, length, 0);
    }
    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }
    if ((length > 0) && (toMul != 1)) {
        bitCapInt lowMask = (1 << length) - 1;
        bitCapInt inOutMask = lowMask << inOutStart;
        bitCapInt carryMask = lowMask << carryStart;
        bitCapInt controlPower = 1 << controlBit;
        bitCapInt otherMask = (maxQPower - 1) ^ (inOutMask | carryMask);

        bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> (length + 1), toMul, lowMask, controlPower, inOutMask,
            carryMask, otherMask, length, inOutStart, carryStart };

        DispatchCall(OCL_API_CMUL, bciArgs);
    }
}

/** Controlled division by integer */
void QEngineOCL::CDIV(
    bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit, bitLenInt length)
{
    if (toDiv == 0) {
        throw "DIV by zero";
    }
    if ((length > 0) && (toDiv != 1)) {
        bitCapInt lowMask = (1 << length) - 1;
        bitCapInt inOutMask = lowMask << inOutStart;
        bitCapInt carryMask = lowMask << carryStart;
        bitCapInt controlPower = 1 << controlBit;
        bitCapInt otherMask = (maxQPower - 1) ^ (inOutMask | carryMask);

        bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> (length + 1), toDiv, lowMask, controlPower, inOutMask,
            carryMask, otherMask, length, inOutStart, carryStart };

        DispatchCall(OCL_API_CDIV, bciArgs);
    }
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, unsigned char* values, bool isParallel)
{
    SetReg(valueStart, valueLength, 0);
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> valueLength, indexStart, inputMask, valueStart, valueBytes,
        valueLength, 0, 0, 0, 0 };

    DispatchCall(OCL_API_INDEXEDLDA, bciArgs, values, (1 << valueLength) * valueBytes, isParallel);

    real1 prob;
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    bitCapInt i, outputInt;
    LockSync(CL_MAP_READ);
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(stateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return (bitCapInt)(average + 0.5);
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::OpIndexed(OCLAPI api_call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values, bool isParallel)
{
    bool carryRes = M(carryIndex);
    // The carry has to first to be measured for its input value.
    if (carryRes) {
        /*
         * If the carry is set, we flip the carry bit. We always initially
         * clear the carry after testing for carry in.
         */
        carryIn ^= 1U;
        X(carryIndex);
    }

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt lengthPower = 1 << valueLength;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, indexStart, inputMask, valueStart, outputMask, otherMask,
        carryIn, carryMask, lengthPower, valueBytes };

    DispatchCall(api_call, bciArgs, values, (1 << valueLength) * valueBytes, isParallel);

    // At the end, just as a convenience, we return the expectation value for the addition result.
    real1 prob;
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    bitCapInt i, outputInt;
    LockSync(CL_MAP_READ);
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(stateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}

/** Add based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values, bool isParallel)
{
    return OpIndexed(
        OCL_API_INDEXEDADC, 0, indexStart, indexLength, valueStart, valueLength, carryIndex, values, isParallel);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values, bool isParallel)
{
    return OpIndexed(
        OCL_API_INDEXEDSBC, 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values, isParallel);
}

void QEngineOCL::PhaseFlip()
{
    OCLDeviceCall ocl = device_context->Reserve(OCL_API_PHASEFLIP);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queue.enqueueWriteBuffer(ulongBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);

    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(nrmGroupCount), // global number of work items
        cl::NDRange(nrmGroupSize)); // local number (per group)

    queue.finish();
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineOCL::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    knowIsPhaseSeparable = false;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_ZEROPHASEFLIP);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, (1U << start), length, 0, 0, 0, 0, 0, 0, 0 };
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    queue.finish();
    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);

    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    knowIsPhaseSeparable = false;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_CPHASEFLIPIFLESS);

    bitCapInt regMask = ((1 << length) - 1) << start;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, regMask, 1U << flagIndex, greaterPerm, start, 0, 0, 0, 0, 0 };
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue.flush();

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);

    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(ngc), // global number of work items
        cl::NDRange(ngs)); // local number (per group)

    queue.finish();
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(complex* inputState)
{
    knowIsPhaseSeparable = false;

    LockSync(CL_MAP_WRITE);
    std::copy(inputState, inputState + maxQPower, stateVec);
    runningNorm = ONE_R1;
    UnlockSync();
}

void QEngineOCL::NormalizeState(real1 nrm)
{
    if (nrm < ZERO_R1) {
        nrm = runningNorm;
    }
    if ((nrm == ONE_R1) || (runningNorm == ZERO_R1)) {
        return;
    }
    if (nrm < min_norm) {
        queue.enqueueFillBuffer(*stateBuffer, complex(ZERO_R1, ZERO_R1), 0, sizeof(complex) * maxQPower);
        queue.finish();
        runningNorm = ZERO_R1;
        return;
    }

    real1 r1_args[2] = { min_norm, (real1)sqrt(nrm) };
    cl::Buffer argsBuffer = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(real1) * 2, r1_args);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queue.finish();
    queue.enqueueWriteBuffer(ulongBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_NORMALIZE);

    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, argsBuffer);
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(nrmGroupCount), // global number of work items
        cl::NDRange(nrmGroupSize)); // local number (per group)

    queue.finish();

    runningNorm = ONE_R1;
}

void QEngineOCL::UpdateRunningNorm()
{
    OCLDeviceCall ocl = device_context->Reserve(OCL_API_UPDATENORM);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queue.enqueueWriteBuffer(ulongBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    ocl.call.setArg(0, *stateBuffer);
    ocl.call.setArg(1, ulongBuffer);
    ocl.call.setArg(2, nrmBuffer);
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(nrmGroupCount), // global number of work items
        cl::NDRange(nrmGroupSize)); // local number (per group)

    runningNorm = ZERO_R1;
    queue.enqueueMapBuffer(nrmBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * nrmGroupCount);
    for (unsigned long int i = 0; i < nrmGroupCount; i++) {
        runningNorm += nrmArray[i];
    }
    queue.enqueueUnmapMemObject(nrmBuffer, nrmArray);
    queue.finish();

    if (runningNorm < min_norm) {
        NormalizeState(ZERO_R1);
    }
}

complex* QEngineOCL::AllocStateVec(bitCapInt elemCount)
{
// elemCount is always a power of two, but might be smaller than ALIGN_SIZE
#ifdef __APPLE__
    void* toRet;
    posix_memalign(
        &toRet, ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
    return (complex*)toRet;
#else
    return (complex*)aligned_alloc(
        ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
#endif
}

} // namespace Qrack
