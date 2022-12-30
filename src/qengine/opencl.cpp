//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_opencl.hpp"

#include <algorithm>
#include <thread>

namespace Qrack {

// Mask definition for Apply2x2()
#define APPLY2X2_DEFAULT 0x00
#define APPLY2X2_NORM 0x01
#define APPLY2X2_SINGLE 0x02
#define APPLY2X2_DOUBLE 0x04
#define APPLY2X2_WIDE 0x08
#define APPLY2X2_X 0x10
#define APPLY2X2_Z 0x20
#define APPLY2X2_PHASE 0x40
#define APPLY2X2_INVERT 0x80

// These are commonly used emplace patterns, for OpenCL buffer I/O.
#define DISPATCH_BLOCK_WRITE(waitVec, buff, offset, length, array)                                                     \
    tryOcl("Failed to write buffer",                                                                                   \
        [&] { return queue.enqueueWriteBuffer(buff, CL_TRUE, offset, length, array, waitVec.get()); });                \
    wait_refs.clear();

#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)                                                       \
    tryOcl("Failed to write buffer",                                                                                   \
        [&] { return queue.enqueueWriteBuffer(buff, CL_FALSE, 0U, size, array, waitVec.get(), &clEvent); });

#define DISPATCH_LOC_WRITE(buff, size, array, clEvent)                                                                 \
    tryOcl("Failed to enqueue buffer write",                                                                           \
        [&] { return queue.enqueueWriteBuffer(buff, CL_FALSE, 0U, size, array, NULL, &clEvent); });

#define DISPATCH_WRITE(waitVec, buff, size, array)                                                                     \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    tryOcl(                                                                                                            \
        "Failed to enqueue buffer write",                                                                              \
        [&] {                                                                                                          \
            return queue.enqueueWriteBuffer(                                                                           \
                buff, CL_FALSE, 0U, size, array, waitVec.get(), &(device_context->wait_events->back()));               \
        },                                                                                                             \
        true);                                                                                                         \
    device_context->UnlockWaitEvents();

#define DISPATCH_BLOCK_READ(waitVec, buff, offset, length, array)                                                      \
    tryOcl("Failed to read buffer",                                                                                    \
        [&] { return queue.enqueueReadBuffer(buff, CL_TRUE, offset, length, array, waitVec.get()); });                 \
    wait_refs.clear();

#define WAIT_REAL1_SUM(buff, size, array, sumPtr)                                                                      \
    clFinish();                                                                                                        \
    tryOcl("Failed to enqueue buffer read",                                                                            \
        [&] { return queue.enqueueReadBuffer(buff, CL_TRUE, 0U, sizeof(real1) * size, array.get(), NULL, NULL); });    \
    *(sumPtr) = ParSum(array.get(), size);

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateBuffer) {                                                                                                \
        return;                                                                                                        \
    }

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int64_t devID, bool useHardwareRNG, bool ignored, real1_f norm_thresh,
    std::vector<int64_t> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , unlockHostMem(false)
    , callbackError(CL_SUCCESS)
    , nrmGroupSize(0U)
    , totalOclAllocSize(0U)
    , deviceID(devID)
    , wait_refs()
    , nrmArray(NULL, [](real1* r) {})
{
    InitOCL(devID);
    clFinish();
    if (qubitCount) {
        SetPermutation(initState, phaseFac);
    } else {
        ZeroAmplitudes();
    }
}

void QEngineOCL::FreeAll()
{
    ZeroAmplitudes();

    powersBuffer = NULL;
    nrmBuffer = NULL;
    nrmArray = NULL;

    SubtractAlloc(totalOclAllocSize);
}

void QEngineOCL::ZeroAmplitudes()
{
    clDump();
    runningNorm = ZERO_R1;

    if (!stateBuffer) {
        return;
    }

    ResetStateBuffer(NULL);
    FreeStateVec();

    SubtractAlloc(sizeof(complex) * maxQPowerOcl);
}

void QEngineOCL::CopyStateVec(QEnginePtr src)
{
    if (qubitCount != src->GetQubitCount()) {
        throw std::invalid_argument("QEngineOCL::CopyStateVec argument size differs from this!");
    }

    if (src->IsZeroAmplitude()) {
        ZeroAmplitudes();
        return;
    }

    if (stateBuffer) {
        clDump();
    } else {
        ReinitBuffer();
    }

    LockSync(CL_MAP_WRITE);
    src->GetQuantumState(stateVec.get());
    UnlockSync();

    runningNorm = src->GetRunningNorm();
}

void QEngineOCL::GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineOCL::GetAmplitudePage range is out-of-bounds!");
    }

    if (!stateBuffer) {
        std::fill(pagePtr, pagePtr + length, ZERO_CMPLX);
        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_BLOCK_READ(waitVec, *stateBuffer, sizeof(complex) * offset, sizeof(complex) * length, pagePtr);
}

void QEngineOCL::SetAmplitudePage(complex const* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineOCL::SetAmplitudePage range is out-of-bounds!");
    }

    if (!stateBuffer) {
        ReinitBuffer();
        if (length != maxQPowerOcl) {
            ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
        }
    }

    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_BLOCK_WRITE(waitVec, *stateBuffer, sizeof(complex) * offset, sizeof(complex) * length, pagePtr);

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::SetAmplitudePage(
    QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
{
    if (isBadPermRange(dstOffset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineOCL::SetAmplitudePage source range is out-of-bounds!");
    }

    QEngineOCLPtr pageEngineOclPtr = std::dynamic_pointer_cast<QEngineOCL>(pageEnginePtr);

    if (isBadPermRange(srcOffset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineOCL::SetAmplitudePage source range is out-of-bounds!");
    }

    BufferPtr oStateBuffer = pageEngineOclPtr->stateBuffer;

    if (!stateBuffer && !oStateBuffer) {
        return;
    }

    if (!oStateBuffer) {
        if (length == maxQPower) {
            ZeroAmplitudes();
        } else {
            ClearBuffer(stateBuffer, dstOffset, length);
            runningNorm = REAL1_DEFAULT_ARG;
        }

        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    pageEngineOclPtr->clFinish();

    if (device_context->context_id != pageEngineOclPtr->device_context->context_id) {
        // Cross-platform - can't automatically migrate buffers.
        pageEngineOclPtr->LockSync(CL_MAP_READ);
        SetAmplitudePage(pageEngineOclPtr->stateVec.get() + srcOffset, dstOffset, length);
        pageEngineOclPtr->UnlockSync();

        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();

    cl::Event copyEvent;
    tryOcl("Failed to enqueue buffer copy", [&] {
        return queue.enqueueCopyBuffer(*oStateBuffer, *stateBuffer, sizeof(complex) * srcOffset,
            sizeof(complex) * dstOffset, sizeof(complex) * length, waitVec.get(), &copyEvent);
    });
    copyEvent.wait();

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::ShuffleBuffers(QEnginePtr engine)
{
    if (qubitCount != engine->GetQubitCount()) {
        throw std::invalid_argument("QEngineOCL::ShuffleBuffers argument size differs from this!");
    }

    QEngineOCLPtr engineOcl = std::dynamic_pointer_cast<QEngineOCL>(engine);

    if (!stateBuffer && !(engineOcl->stateBuffer)) {
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    if (!(engineOcl->stateBuffer)) {
        engineOcl->ReinitBuffer();
        engineOcl->ClearBuffer(engineOcl->stateBuffer, 0U, engineOcl->maxQPowerOcl);
    }

    const bitCapIntOcl halfMaxQPower = (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI);

    if (device_context->context_id != engineOcl->device_context->context_id) {
        LockSync(CL_MAP_READ | CL_MAP_WRITE);
        engineOcl->LockSync(CL_MAP_READ | CL_MAP_WRITE);

        std::swap_ranges(
            engineOcl->stateVec.get(), engineOcl->stateVec.get() + halfMaxQPower, stateVec.get() + halfMaxQPower);

        engineOcl->UnlockSync();
        UnlockSync();

        return;
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ halfMaxQPower, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeArgsEvent);
    writeArgsEvent.wait();

    engineOcl->clFinish();
    QueueCall(OCL_API_SHUFFLEBUFFERS, nrmGroupCount, nrmGroupSize,
        { stateBuffer, engineOcl->stateBuffer, poolItem->ulongBuffer });
    engineOcl->wait_refs.emplace_back(device_context->wait_events);

    runningNorm = REAL1_DEFAULT_ARG;
    engineOcl->runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::LockSync(cl_map_flags flags)
{
    lockSyncFlags = flags;
    EventVecPtr waitVec = ResetWaitEvents();

    if (stateVec) {
        unlockHostMem = true;
        tryOcl("Failed to map buffer", [&] {
            cl_int error;
            queue.enqueueMapBuffer(
                *stateBuffer, CL_TRUE, flags, 0U, sizeof(complex) * maxQPowerOcl, waitVec.get(), NULL, &error);
            return error;
        });
        wait_refs.clear();
    } else {
        unlockHostMem = false;
        stateVec = AllocStateVec(maxQPowerOcl, true);
        if (lockSyncFlags & CL_MAP_READ) {
            DISPATCH_BLOCK_READ(waitVec, *stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, stateVec.get());
        }
    }
}

void QEngineOCL::UnlockSync()
{
    EventVecPtr waitVec = ResetWaitEvents();

    if (unlockHostMem) {
        cl::Event unmapEvent;
        tryOcl("Failed to unmap buffer",
            [&] { return queue.enqueueUnmapMemObject(*stateBuffer, stateVec.get(), waitVec.get(), &unmapEvent); });
        unmapEvent.wait();
        wait_refs.clear();
    } else {
        if (lockSyncFlags & CL_MAP_WRITE) {
            DISPATCH_BLOCK_WRITE(waitVec, *stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, stateVec.get())
        }
        FreeStateVec();
    }

    lockSyncFlags = 0;
}

void QEngineOCL::clFinish(bool doHard)
{
    if (!device_context) {
        return;
    }

    checkCallbackError();

    while (wait_queue_items.size() > 1) {
        device_context->WaitOnAllEvents();
        PopQueue(true);
        checkCallbackError();
    }

    if (doHard) {
        tryOcl("Failed to finish queue", [&] { return queue.finish(); });
    } else {
        device_context->WaitOnAllEvents();
        checkCallbackError();
    }

    wait_refs.clear();
}

void QEngineOCL::clDump()
{
    if (!device_context) {
        return;
    }

    checkCallbackError();

    while (wait_queue_items.size() > 1) {
        device_context->WaitOnAllEvents();
        PopQueue(false);
        checkCallbackError();
    }

    device_context->WaitOnAllEvents();
    checkCallbackError();

    wait_refs.clear();
}

PoolItemPtr QEngineOCL::GetFreePoolItem()
{
    std::lock_guard<std::mutex> lock(queue_mutex);

    checkCallbackError();

    while (wait_queue_items.size() >= poolItems.size()) {
        poolItems.push_back(std::make_shared<PoolItem>(context));
    }

    return poolItems[wait_queue_items.size()];
}

EventVecPtr QEngineOCL::ResetWaitEvents(bool waitQueue)
{
    if (waitQueue) {
        while (wait_queue_items.size() > 1) {
            device_context->WaitOnAllEvents();
            PopQueue(true);
            checkCallbackError();
        }
    }

    EventVecPtr waitVec = device_context->ResetWaitEvents();
    if (waitVec->size()) {
        wait_refs.emplace_back(waitVec);
    }
    return wait_refs.size() ? wait_refs.back() : std::make_shared<EventVec>();
}

void QEngineOCL::WaitCall(
    OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args, size_t localBuffSize)
{
    QueueCall(api_call, workItemCount, localGroupSize, args, localBuffSize);
    clFinish();
}

void CL_CALLBACK _PopQueue(cl_event event, cl_int type, void* user_data) { ((QEngineOCL*)user_data)->PopQueue(true); }

void QEngineOCL::PopQueue(bool isDispatch)
{
    // For lock_guard scope
    if (true) {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (poolItems.size()) {
            poolItems.front()->probArray = NULL;
            poolItems.front()->angleArray = NULL;

            if (poolItems.size() > 1) {
                rotate(poolItems.begin(), poolItems.begin() + 1, poolItems.end());
            }
        }

        if (!wait_queue_items.size()) {
            return;
        }
        SubtractAlloc(wait_queue_items.front().deallocSize);
        wait_queue_items.pop_front();
    }

    if (callbackError != CL_SUCCESS) {
        wait_queue_items.clear();
        wait_refs.clear();
        return;
    }

    if (isDispatch) {
        DispatchQueue();
    }
}

void QEngineOCL::DispatchQueue()
{
    QueueItem item;

    if (true) {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (!wait_queue_items.size()) {
            return;
        }

        item = wait_queue_items.front();

        while (item.isSetDoNorm || item.isSetRunningNorm) {
            if (item.isSetDoNorm) {
                doNormalize = item.doNorm;
            }
            if (item.isSetRunningNorm) {
                runningNorm = item.runningNorm;
            }

            wait_queue_items.pop_front();
            if (!wait_queue_items.size()) {
                return;
            }
            item = wait_queue_items.front();
        }
    }

    std::vector<BufferPtr> args = item.buffers;

    // We have to reserve the kernel, because its argument hooks are unique. The same kernel therefore can't be used by
    // other QEngineOCL instances, until we're done queueing it.
    OCLDeviceCall ocl = device_context->Reserve(item.api_call);

    // Load the arguments.
    for (unsigned int i = 0U; i < args.size(); ++i) {
        ocl.call.setArg(i, *args[i]);
    }

    // For all of our kernels, if a local memory buffer is used, there is always only one, as the last argument.
    if (item.localBuffSize) {
#if ENABLE_SNUCL
        ocl.call.setArg(args.size(), cl::__local(item.localBuffSize));
#else
        ocl.call.setArg(args.size(), cl::Local(item.localBuffSize));
#endif
    }

    // Dispatch the primary kernel, to apply the gate.
    EventVecPtr kernelWaitVec = ResetWaitEvents(false);
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    device_context->wait_events->back().setCallback(CL_COMPLETE, _PopQueue, this);
    cl_int error = queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(item.workItemCount), // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(), // vector of events to wait for
        &(device_context->wait_events->back())); // handle to wait for the kernel
    device_context->UnlockWaitEvents();
    if (error != CL_SUCCESS) {
        // We're fatally blocked, since we can't make any blocking calls like clFinish() in a callback.
        callbackError = error;
        wait_queue_items.clear();
        wait_refs.clear();
        return;
    }
    error = queue.flush();
    if (error != CL_SUCCESS) {
        // We're fatally blocked, since we can't make any blocking calls like clFinish() in a callback.
        callbackError = error;
        wait_queue_items.clear();
        wait_refs.clear();
        return;
    }
}

void QEngineOCL::SetDevice(int64_t dID)
{
    if (!(OCLEngine::Instance().GetDeviceCount())) {
        throw std::runtime_error("Tried to initialize QEngineOCL, but no available OpenCL devices.");
    }

    const bool didInit = (nrmArray != NULL);

    clFinish();

    const int64_t oldContextId = device_context ? device_context->context_id : 0;
    const DeviceContextPtr nDeviceContext = OCLEngine::Instance().GetDeviceContextPtr(dID);
    const int64_t defDevId = (int)OCLEngine::Instance().GetDefaultDeviceID();

    std::unique_ptr<complex[]> copyVec = NULL;

    if (!didInit) {
        AddAlloc(sizeof(complex) * maxQPowerOcl);
    } else if ((dID == deviceID) || ((dID == -1) && (deviceID == defDevId)) ||
        ((deviceID == -1) && (dID == defDevId))) {
        // If we're "switching" to the device we already have, don't reinitialize.
        return;
    } else if (stateBuffer && !stateVec) {
        // This copies the contents of stateBuffer to host memory, to load into a buffer in the new context.
        copyVec = std::unique_ptr<complex[]>(new complex[maxQPowerOcl]);
        GetQuantumState(copyVec.get());
    }

    device_context = nDeviceContext;
    deviceID = dID;
    context = device_context->context;
    queue = device_context->queue;

    // If the user wants not to use host RAM, but we can't allocate enough on the device, fall back to host RAM anyway.
#if ENABLE_OCL_MEM_GUARDS
    const size_t stateVecSize = maxQPowerOcl * sizeof(complex);
    // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
    if (stateVecSize > device_context->GetMaxAlloc()) {
        throw bad_alloc("VRAM limits exceeded in QEngineOCL::SetDevice()");
    }
    usingHostRam = (useHostRam || ((OclMemDenom * stateVecSize) > device_context->GetGlobalSize()));
#endif

    const bitCapIntOcl oldNrmVecAlignSize = nrmGroupSize ? (nrmGroupCount / nrmGroupSize) : 0U;
    nrmGroupCount = device_context->GetPreferredConcurrency();
    nrmGroupSize = device_context->GetPreferredSizeMultiple();
    if (nrmGroupSize > device_context->GetMaxWorkGroupSize()) {
        nrmGroupSize = device_context->GetMaxWorkGroupSize();
    }
    // constrain to a power of two
    nrmGroupSize = (size_t)pow2(log2(nrmGroupSize));

    const size_t nrmArrayAllocSize =
        (!nrmGroupSize || ((sizeof(real1) * nrmGroupCount / nrmGroupSize) < QRACK_ALIGN_SIZE))
        ? QRACK_ALIGN_SIZE
        : (sizeof(real1) * nrmGroupCount / nrmGroupSize);

    const bool doResize = (nrmGroupCount / nrmGroupSize) != oldNrmVecAlignSize;

    nrmBuffer = NULL;
    if (didInit && doResize) {
        nrmArray = NULL;
        SubtractAlloc(oldNrmVecAlignSize);
    }

    if (!didInit || doResize) {
        AddAlloc(nrmArrayAllocSize);
#if defined(__ANDROID__)
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            new real1[nrmArrayAllocSize / sizeof(real1)], [](real1* r) { delete r; });
#elif defined(__APPLE__)
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            _aligned_nrm_array_alloc(nrmArrayAllocSize), [](real1* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            (real1*)_aligned_malloc(nrmArrayAllocSize, QRACK_ALIGN_SIZE), [](real1* c) { _aligned_free(c); });
#else
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            (real1*)aligned_alloc(QRACK_ALIGN_SIZE, nrmArrayAllocSize), [](real1* c) { free(c); });
#endif
    }
    nrmBuffer = MakeBuffer(CL_MEM_READ_WRITE, nrmArrayAllocSize);

    poolItems.clear();
    poolItems.push_back(std::make_shared<PoolItem>(context));

    if (!didInit) {
        AddAlloc(sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));
    }
    powersBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));

    // If this is the same context, then all other buffers are valid.
    if (oldContextId == nDeviceContext->context_id) {
        return;
    }

    // create buffers on device (allocate space on GPU)
    if (didInit) {
        if (stateVec) {
            ResetStateBuffer(MakeStateVecBuffer(stateVec));
        } else {
            ResetStateBuffer(MakeStateVecBuffer(NULL));

            if (copyVec) {
                EventVecPtr waitVec = ResetWaitEvents();
                DISPATCH_WRITE(waitVec, *stateBuffer, sizeof(complex) * maxQPowerOcl, copyVec.get())
                tryOcl("Failed to write buffer", [&] {
                    return queue.enqueueWriteBuffer(
                        *stateBuffer, CL_TRUE, 0U, sizeof(complex) * maxQPowerOcl, copyVec.get(), NULL);
                });
                wait_refs.clear();
                copyVec.reset();
            } else {
                ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
            }
        }
    } else {
        // In this branch, the QEngineOCL is first being initialized, and no data needs to be copied between device
        // contexts.
        stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);
    }
}

real1_f QEngineOCL::ParSum(real1* toSum, bitCapIntOcl maxI)
{
    // This interface is potentially parallelizable, but, for now, better performance is probably given by implementing
    // it as a serial loop.
    real1 totSum = ZERO_R1;
    for (bitCapIntOcl i = 0U; i < maxI; ++i) {
        totSum += toSum[i];
    }

    return (real1_f)totSum;
}

void QEngineOCL::InitOCL(int64_t devID) { SetDevice(devID); }

void QEngineOCL::ResetStateBuffer(BufferPtr nStateBuffer) { stateBuffer = nStateBuffer; }

void QEngineOCL::SetPermutation(bitCapInt perm, complex phaseFac)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    ClearBuffer(stateBuffer, 0U, maxQPowerOcl);

    // If "permutationAmp" amp is in (read-only) use, this method completely supersedes that application anyway.

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        permutationAmp = GetNonunitaryPhase();
    } else {
        permutationAmp = phaseFac;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    tryOcl(
        "Failed to enqueue buffer write",
        [&] {
            return queue.enqueueWriteBuffer(*stateBuffer, CL_FALSE, sizeof(complex) * (bitCapIntOcl)perm,
                sizeof(complex), &permutationAmp, waitVec.get(), &(device_context->wait_events->back()));
        },
        true);
    device_context->UnlockWaitEvents();

    QueueSetRunningNorm(ONE_R1_F);
}

/// NOT gate, which is also Pauli x matrix
void QEngineOCL::X(bitLenInt qubit)
{
    const complex pauliX[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const bitCapIntOcl qPowers[1]{ pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::PAULIX);
}

/// Apply Pauli Z matrix to bit
void QEngineOCL::Z(bitLenInt qubit)
{
    const complex pauliZ[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const bitCapIntOcl qPowers[1]{ pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PAULIZ);
}

void QEngineOCL::Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
{
    if ((randGlobalPhase || IS_NORM_0(ONE_CMPLX - topRight)) && IS_NORM_0(topRight - bottomLeft)) {
        X(qubitIndex);
        return;
    }

    const complex pauliX[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    const bitCapIntOcl qPowers[1]{ pow2Ocl(qubitIndex) };
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::INVERT);
}

void QEngineOCL::Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
{
    if (randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft)) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            return;
        }

        if (IS_NORM_0(topLeft + bottomRight)) {
            Z(qubitIndex);
            return;
        }
    }

    const complex pauliZ[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    const bitCapIntOcl qPowers[1]{ pow2Ocl(qubitIndex) };
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PHASE);
}
void QEngineOCL::XMask(bitCapInt mask)
{
    if (!mask) {
        return;
    }

    if (!(mask & (mask - ONE_BCI))) {
        X(log2(mask));
        return;
    }

    BitMask((bitCapIntOcl)mask, OCL_API_X_MASK);
}
void QEngineOCL::PhaseParity(real1_f radians, bitCapInt mask)
{
    if (!mask) {
        return;
    }

    if (!(mask & (mask - ONE_BCI))) {
        complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    BitMask((bitCapIntOcl)mask, OCL_API_PHASE_PARITY, radians);
}

void QEngineOCL::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, complex const* mtrx, bitLenInt bitCount,
    const bitCapIntOcl* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special, real1_f norm_thresh)
{
    CHECK_ZERO_SKIP();

    if ((offset1 >= maxQPowerOcl) || (offset2 >= maxQPowerOcl)) {
        throw std::invalid_argument(
            "QEngineOCL::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
    }

    for (bitLenInt i = 0U; i < bitCount; ++i) {
        if (qPowersSorted[i] >= maxQPowerOcl) {
            throw std::invalid_argument(
                "QEngineOCL::Apply2x2 parameter qPowersSorted array values must be within allocated qubit bounds!");
        }
        if (i && (qPowersSorted[i - 1U] == qPowersSorted[i])) {
            throw std::invalid_argument("QEngineOCL::Apply2x2 parameter qPowersSorted array values cannot be "
                                        "duplicated (for control and target qubits)!");
        }
    }

    const bool skipNorm = !doNormalize || (abs(ONE_R1 - runningNorm) <= FP_NORM_EPSILON);
    const bool isXGate = skipNorm && (special == SPECIAL_2X2::PAULIX);
    const bool isZGate = skipNorm && (special == SPECIAL_2X2::PAULIZ);
    const bool isInvertGate = skipNorm && (special == SPECIAL_2X2::INVERT);
    const bool isPhaseGate = skipNorm && (special == SPECIAL_2X2::PHASE);

    // Are we going to calculate the normalization factor, on the fly? We can't, if this call doesn't iterate through
    // every single permutation amplitude.
    bool doApplyNorm = doNormalize && (bitCount == 1) && (runningNorm > ZERO_R1) && !isXGate && !isZGate &&
        !isInvertGate && !isPhaseGate;
    doCalcNorm &= doApplyNorm || (runningNorm <= ZERO_R1);
    doApplyNorm &= (runningNorm != ONE_R1);

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec = ResetWaitEvents();

    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    const bitCapIntOcl maxI = maxQPowerOcl >> bitCount;
    bitCapIntOcl bciArgs[5]{ offset2, offset1, maxI, bitCount, 0U };

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // In an efficient OpenCL kernel, every single byte loaded comes at a significant execution time premium.
    // We handle single and double bit gates as special cases, for many reasons. Given that we have already separated
    // these out as special cases, since we know the bit count, we can eliminate the qPowersSorted buffer, by loading
    // its one or two values into the bciArgs buffer, of the same type. This gives us a significant execution time
    // savings.
    size_t bciArgsSize = 4;
    if (bitCount == 1) {
        // Single bit gates offsets are always 0 and target bit power. Hence, we overwrite one of the bit offset
        // arguments.
        if (ngc == maxI) {
            bciArgsSize = 3;
            bciArgs[2] = qPowersSorted[0] - ONE_BCI;
        } else {
            bciArgsSize = 4;
            bciArgs[3] = qPowersSorted[0] - ONE_BCI;
        }
    } else if (bitCount == 2) {
        // Double bit gates include both controlled and swap gates. To reuse the code for both cases, we need two offset
        // arguments. Hence, we cannot easily overwrite either of the bit offset arguments.
        bciArgsSize = 5;
        bciArgs[3] = qPowersSorted[0] - ONE_BCI;
        bciArgs[4] = qPowersSorted[1] - ONE_BCI;
    }
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * bciArgsSize, bciArgs, writeArgsEvent);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    cmplx[4] = complex(doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1, ZERO_R1);
    cmplx[5] = (real1)norm_thresh;

    BufferPtr locCmplxBuffer;
    cl::Event writeGateEvent;
    if (!isXGate && !isZGate) {
        DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * CMPLX_NORM_LEN, cmplx, writeGateEvent);
    }

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    BufferPtr locPowersBuffer;
    cl::Event writeControlsEvent;
    if (bitCount > 2) {
        if (doCalcNorm) {
            locPowersBuffer = powersBuffer;
        } else {
            locPowersBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * bitCount);
        }
        if (sizeof(bitCapInt) == sizeof(bitCapIntOcl)) {
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent);
        } else {
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent);
        }
    }

    // We load the appropriate kernel, that does/doesn't CALCULATE the norm, and does/doesn't APPLY the norm.
    unsigned char kernelMask = APPLY2X2_DEFAULT;
    if (bitCount == 1) {
        kernelMask |= APPLY2X2_SINGLE;
        if (isXGate) {
            kernelMask |= APPLY2X2_X;
        } else if (isZGate) {
            kernelMask |= APPLY2X2_Z;
        } else if (isInvertGate) {
            kernelMask |= APPLY2X2_INVERT;
        } else if (isPhaseGate) {
            kernelMask |= APPLY2X2_PHASE;
        } else if (doCalcNorm) {
            kernelMask |= APPLY2X2_NORM;
        }
    } else if (bitCount == 2) {
        kernelMask |= APPLY2X2_DOUBLE;
    }
    if (ngc == maxI) {
        kernelMask |= APPLY2X2_WIDE;
    }

    OCLAPI api_call;
    switch (kernelMask) {
    case APPLY2X2_DEFAULT:
        api_call = OCL_API_APPLY2X2;
        break;
    case APPLY2X2_SINGLE:
        api_call = OCL_API_APPLY2X2_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_X:
        api_call = OCL_API_X_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_Z:
        api_call = OCL_API_Z_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_INVERT:
        api_call = OCL_API_INVERT_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_PHASE:
        api_call = OCL_API_PHASE_SINGLE;
        break;
    case APPLY2X2_NORM | APPLY2X2_SINGLE:
        api_call = OCL_API_APPLY2X2_NORM_SINGLE;
        break;
    case APPLY2X2_DOUBLE:
        api_call = OCL_API_APPLY2X2_DOUBLE;
        break;
    case APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_X:
        api_call = OCL_API_X_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_Z:
        api_call = OCL_API_Z_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_INVERT:
        api_call = OCL_API_INVERT_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_PHASE:
        api_call = OCL_API_PHASE_SINGLE_WIDE;
        break;
    case APPLY2X2_NORM | APPLY2X2_SINGLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_NORM_SINGLE_WIDE;
        break;
    case APPLY2X2_DOUBLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_DOUBLE_WIDE;
        break;
    default:
        throw std::runtime_error("Invalid APPLY2X2 kernel selected!");
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    if (!isXGate && !isZGate) {
        writeGateEvent.wait();
    }
    if (bitCount > 2) {
        writeControlsEvent.wait();
    }
    wait_refs.clear();

    if (isXGate || isZGate) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
    } else if (doCalcNorm) {
        if (bitCount > 2) {
            QueueCall(api_call, ngc, ngs,
                { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, locPowersBuffer, nrmBuffer },
                sizeof(real1) * ngs);
        } else {
            QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, nrmBuffer },
                sizeof(real1) * ngs);
        }
    } else {
        if (bitCount > 2) {
            QueueCall(
                api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, locPowersBuffer });
        } else {
            QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer });
        }
    }

    if (doApplyNorm) {
        QueueSetRunningNorm(ONE_R1_F);
    }

    if (!doCalcNorm) {
        return;
    }

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);
    if (runningNorm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
    }
}

void QEngineOCL::BitMask(bitCapIntOcl mask, OCLAPI api_call, real1_f phase)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::BitMask mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ mask;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, mask, otherMask, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    const bool isPhaseParity = (api_call == OCL_API_PHASE_PARITY);
    if (isPhaseParity) {
        const complex phaseFac = std::polar(ONE_R1, (real1)(phase / 2));
        const complex cmplxArray[2]{ phaseFac, ONE_CMPLX / phaseFac };
        cl::Event writePhaseEvent;
        DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), 2U * sizeof(complex), cmplxArray, writePhaseEvent);
        writePhaseEvent.wait();
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    if (isPhaseParity) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
    }
}

void QEngineOCL::UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
    complex const* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (!controls.size()) {
        Mtrx(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    if (qubitIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineOCL::UniformlyControlledSingleBit control is out-of-bounds!");

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    const bitCapIntOcl maxI = maxQPowerOcl >> ONE_BCI;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxI, pow2Ocl(qubitIndex), controls.size(), mtrxSkipPowers.size(),
        (bitCapIntOcl)mtrxSkipValueMask, 0U, 0U, 0U, 0U, 0U };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs);

    BufferPtr nrmInBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(real1));
    const real1 nrm = (runningNorm > ZERO_R1) ? ONE_R1 / (real1)sqrt(runningNorm) : ONE_R1;
    DISPATCH_WRITE(waitVec, *nrmInBuffer, sizeof(real1), &nrm);

    const size_t sizeDiff = sizeof(complex) * 4U * pow2Ocl(controls.size() + mtrxSkipPowers.size());
    AddAlloc(sizeDiff);
    BufferPtr uniformBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeDiff);

    DISPATCH_WRITE(
        waitVec, *uniformBuffer, sizeof(complex) * 4U * pow2Ocl(controls.size() + mtrxSkipPowers.size()), mtrxs);

    std::unique_ptr<bitCapIntOcl[]> qPowers(new bitCapIntOcl[controls.size() + mtrxSkipPowers.size()]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        qPowers[i] = pow2Ocl(controls[i]);
    }
    for (size_t i = 0U; i < mtrxSkipPowers.size(); ++i) {
        qPowers[controls.size() + i] = (bitCapIntOcl)mtrxSkipPowers[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    DISPATCH_WRITE(
        waitVec, *powersBuffer, sizeof(bitCapIntOcl) * (controls.size() + mtrxSkipPowers.size()), qPowers.get());

    // We call the kernel, with global buffers and one local buffer.
    WaitCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    uniformBuffer.reset();
    qPowers.reset();

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::UniformParityRZ mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, (bitCapIntOcl)mask, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };
    const real1 cosine = (real1)cos(angle);
    const real1 sine = (real1)sin(angle);
    const complex phaseFacs[3]{ complex(cosine, sine), complex(cosine, -sine),
        (runningNorm > ZERO_R1) ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent);
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * 3, &phaseFacs, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall((abs(ONE_R1 - runningNorm) <= FP_NORM_EPSILON) ? OCL_API_UNIFORMPARITYRZ : OCL_API_UNIFORMPARITYRZ_NORM,
        ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineOCL::CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle)
{
    if (!controls.size()) {
        UniformParityRZ(mask, angle);
        return;
    }

    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::CUniformParityRZ mask out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineOCL::CUniformParityRZ control is out-of-bounds!");

    CHECK_ZERO_SKIP();

    bitCapIntOcl controlMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controls.size()]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controls.size());
    BufferPtr controlBuffer = MakeBuffer(
        CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controls.size(), controlPowers.get());
    controlPowers.reset();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> controls.size()), (bitCapIntOcl)mask,
        controlMask, controls.size(), 0U, 0U, 0U, 0U, 0U, 0U };
    const real1 cosine = (real1)cos(angle);
    const real1 sine = (real1)sin(angle);
    const complex phaseFacs[2]{ complex(cosine, sine), complex(cosine, -sine) };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, writeArgsEvent);
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * 2, &phaseFacs, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall(OCL_API_CUNIFORMPARITYRZ, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer, controlBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineOCL::ApplyMx(OCLAPI api_call, const bitCapIntOcl* bciArgs, complex nrm)
{
    CHECK_ZERO_SKIP();

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent);
    BufferPtr locCmplxBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(complex));
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex), &nrm, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineOCL::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapIntOcl powerTest = result ? (bitCapIntOcl)qPower : 0U;

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), (bitCapIntOcl)qPower, powerTest,
        0U, 0U, 0U, 0U, 0U, 0U, 0U };

    ApplyMx(OCL_API_APPLYM, bciArgs, nrm);
}

void QEngineOCL::ApplyM(bitCapInt mask, bitCapInt result, complex nrm)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::ApplyM mask out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)result, 0U, 0U, 0U, 0U, 0U,
        0U, 0U };

    ApplyMx(OCL_API_APPLYMREG, bciArgs, nrm);
}

void QEngineOCL::Compose(OCLAPI apiCall, const bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy)
{
    if (!toCopy->qubitCount) {
        return;
    }

    if (!stateBuffer || !toCopy->stateBuffer) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return;
    }

    if (!qubitCount) {
        clFinish();
        SetQubitCount(toCopy->qubitCount);
        toCopy->clFinish();
        runningNorm = toCopy->runningNorm;
        stateVec = AllocStateVec(toCopy->maxQPowerOcl);
        stateBuffer = MakeStateVecBuffer(stateVec);

        if (device_context->context_id != toCopy->device_context->context_id) {
            toCopy->LockSync(CL_MAP_READ);

            EventVecPtr waitVec = ResetWaitEvents();
            DISPATCH_BLOCK_WRITE(waitVec, *stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, toCopy->stateVec.get());

            toCopy->UnlockSync();

            return;
        }

        cl::Event copyEvent;
        tryOcl("Failed to enqueue buffer copy", [&] {
            return queue.enqueueCopyBuffer(
                *(toCopy->stateBuffer), *stateBuffer, 0U, 0U, sizeof(complex) * maxQPowerOcl, NULL, &copyEvent);
        });
        copyEvent.wait();

        return;
    }

    const bitCapIntOcl oMaxQPower = maxQPowerOcl;
    const bitCapIntOcl nMaxQPower = bciArgs[0];
    const bitCapIntOcl nQubitCount = bciArgs[1] + toCopy->qubitCount;
    const size_t nStateVecSize = nMaxQPower * sizeof(complex);
#if ENABLE_OCL_MEM_GUARDS
    if (nStateVecSize > device_context->GetMaxAlloc()) {
        throw bad_alloc("VRAM limits exceeded in QEngineOCL::Compose()");
    }
#endif

    if (doNormalize) {
        NormalizeState();
    }
    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }

    const bool isMigrate = (device_context->context_id != toCopy->device_context->context_id);
    if (isMigrate) {
        toCopy->SetDevice(deviceID);
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    EventVecPtr waitVec = ResetWaitEvents();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 7, bciArgs, writeArgsEvent);

    AddAlloc(sizeof(complex) * nMaxQPower);

    SetQubitCount(nQubitCount);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);
    const bool forceAlloc = !stateVec && ((OclMemDenom * nStateVecSize) > device_context->GetGlobalSize());

    writeArgsEvent.wait();
    wait_refs.clear();

    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl, forceAlloc);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    toCopy->clFinish();

    QueueCall(apiCall, ngc, ngs, { stateBuffer, toCopy->stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    toCopy->wait_refs.emplace_back(device_context->wait_events);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeof(complex) * oMaxQPower);
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy)
{
    const bitLenInt result = qubitCount;

    const bitCapIntOcl oQubitCount = toCopy->qubitCount;
    const bitCapIntOcl nQubitCount = qubitCount + oQubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = maxQPowerOcl - ONE_BCI;
    const bitCapIntOcl endMask = (toCopy->maxQPowerOcl - ONE_BCI) << (bitCapIntOcl)qubitCount;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ nMaxQPower, qubitCount, startMask, endMask, 0U, 0U, 0U, 0U, 0U, 0U };

    OCLAPI api_call;
    if (nMaxQPower <= nrmGroupCount) {
        api_call = OCL_API_COMPOSE_WIDE;
    } else {
        api_call = OCL_API_COMPOSE;
    }

    Compose(api_call, bciArgs, toCopy);

    return result;
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QEngineOCL::Compose start index is out-of-bounds!");
    }

    const bitLenInt result = start;

    const bitLenInt oQubitCount = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + oQubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = pow2Ocl(start) - ONE_BCI;
    const bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    const bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ nMaxQPower, qubitCount, oQubitCount, startMask, midMask, endMask, start,
        0U, 0U, 0U };

    Compose(OCL_API_COMPOSE_MID, bciArgs, toCopy);

    return result;
}

void QEngineOCL::DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr destination)
{
    // "Dispose" is basically the same as decompose, except "Dispose" throws the removed bits away.

    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    if (!stateBuffer) {
        SetQubitCount(qubitCount - length);
        if (destination) {
            destination->ZeroAmplitudes();
        }
        return;
    }

    if (destination && !destination->stateBuffer) {
        // Reinitialize stateVec RAM
        destination->SetPermutation(0U);
    }

    if (doNormalize) {
        NormalizeState();
    }
    if (destination && destination->doNormalize) {
        destination->NormalizeState();
    }

    const bitLenInt nLength = qubitCount - length;

    if (!nLength) {
        if (destination != NULL) {
            destination->stateVec = stateVec;
            destination->stateBuffer = stateBuffer;
            stateBuffer = NULL;
            stateVec = NULL;
        }
        SetQubitCount(0U);
        // This will be cleared by the destructor:
        SubtractAlloc(sizeof(complex) * pow2Ocl(qubitCount));
        stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);

        return;
    }

    const bool isMigrate = destination && (device_context->context_id != destination->device_context->context_id);
    const int64_t oDevId = destination ? destination->deviceID : 0;
    if (isMigrate) {
        destination->SetDevice(deviceID);
    }

    const bitCapIntOcl partPower = pow2Ocl(length);
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);
    const bitCapIntOcl oMaxQPower = maxQPowerOcl;
    bitCapIntOcl bciArgs[BCI_ARG_LEN]{ partPower, remainderPower, start, length, 0U, 0U, 0U, 0U, 0U, 0U };

    const size_t remainderDiff = 2 * sizeof(real1) * remainderPower;
    AddAlloc(remainderDiff);

    // The "remainder" bits will always be maintained.
    BufferPtr probBuffer1 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(probBuffer1, 0U, remainderPower >> ONE_BCI);
    BufferPtr angleBuffer1 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(angleBuffer1, 0U, remainderPower >> ONE_BCI);

    // The removed "part" is only necessary for Decompose.
    BufferPtr probBuffer2, angleBuffer2;
    const size_t partDiff = 2 * sizeof(real1) * partPower;
    if (destination) {
        AddAlloc(2 * sizeof(real1) * partPower);
        probBuffer2 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(probBuffer2, 0U, partPower >> ONE_BCI);
        angleBuffer2 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(angleBuffer2, 0U, partPower >> ONE_BCI);
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    const bitCapIntOcl largerPower = partPower > remainderPower ? partPower : remainderPower;

    const size_t ngc = FixWorkItemCount(largerPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Call the kernel that calculates bit probability and angle, retaining both parts.
    if (destination) {
        QueueCall(OCL_API_DECOMPOSEPROB, ngc, ngs,
            { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1, probBuffer2, angleBuffer2 });
    } else {
        QueueCall(OCL_API_DISPOSEPROB, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1 });
    }

    SetQubitCount(nLength);

    // If we Decompose, calculate the state of the bit system removed.
    if (destination) {
        bciArgs[0] = partPower;

        destination->clFinish();

        poolItem = GetFreePoolItem();
        EventVecPtr waitVec2 = ResetWaitEvents();
        DISPATCH_WRITE(waitVec2, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

        const size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        const size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        const size_t oNStateVecSize = maxQPowerOcl * sizeof(complex);

        WaitCall(OCL_API_DECOMPOSEAMP, ngc2, ngs2,
            { probBuffer2, angleBuffer2, poolItem->ulongBuffer, destination->stateBuffer });

        probBuffer2.reset();
        angleBuffer2.reset();

        SubtractAlloc(partDiff);

        if (!(destination->useHostRam) && destination->stateVec &&
            oNStateVecSize <= destination->device_context->GetMaxAlloc() &&
            (2 * oNStateVecSize) <= destination->device_context->GetGlobalSize()) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cl::Event copyEvent;
            tryOcl("Failed to enqueue buffer copy", [&] {
                return destination->queue.enqueueCopyBuffer(*(destination->stateBuffer), *nSB, 0U, 0U,
                    sizeof(complex) * destination->maxQPowerOcl, NULL, &copyEvent);
            });
            copyEvent.wait();

            destination->stateBuffer = nSB;
            destination->stateVec = NULL;
        }

        if (isMigrate) {
            destination->SetDevice(oDevId);
        }
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPowerOcl;
    poolItem = GetFreePoolItem();
    EventVecPtr waitVec3 = ResetWaitEvents();
    DISPATCH_WRITE(waitVec3, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    const size_t ngc3 = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs3 = FixGroupSize(ngc, nrmGroupSize);

    const size_t nStateVecSize = maxQPowerOcl * sizeof(complex);

    if (!useHostRam && stateVec && ((OclMemDenom * nStateVecSize) <= device_context->GetGlobalSize())) {
        FreeStateVec();
    }
    // Drop references to state vector buffer, which we're done with.
    ResetStateBuffer(NULL);
    SubtractAlloc(sizeof(complex) * oMaxQPower);

    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    // Tell QueueCall to track deallocation:
    QueueCall(OCL_API_DECOMPOSEAMP, ngc3, ngs3, { probBuffer1, angleBuffer1, poolItem->ulongBuffer, stateBuffer }, 0U,
        remainderDiff);
}

void QEngineOCL::Decompose(bitLenInt start, QInterfacePtr destination)
{
    DecomposeDispose(start, destination->GetQubitCount(), std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QEngineOCLPtr)NULL); }

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (!length) {
        return;
    }

    if (!stateBuffer) {
        SetQubitCount(qubitCount - length);
        return;
    }

    if (length == qubitCount) {
        // This will be cleared by the destructor:
        stateVec = AllocStateVec(2);
        stateBuffer = MakeStateVecBuffer(stateVec);
        SubtractAlloc(sizeof(complex) * (pow2Ocl(qubitCount) - 2U));
        SetQubitCount(1);
        return;
    }

    if (doNormalize) {
        NormalizeState();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    const bitLenInt nLength = qubitCount - length;
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);
    const size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    const bitCapIntOcl skipMask = pow2Ocl(start) - ONE_BCI;
    const bitCapIntOcl disposedRes = (bitCapIntOcl)disposedPerm << (bitCapIntOcl)start;

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ remainderPower, length, skipMask, disposedRes, 0U, 0U, 0U, 0U, 0U, 0U };

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    SetQubitCount(nLength);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    AddAlloc(sizeof(complex) * maxQPowerOcl);
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    QueueCall(OCL_API_DISPOSE, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeDiff);
}

bitLenInt QEngineOCL::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QEngineOCLPtr nQubits = std::make_shared<QEngineOCL>(length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);
    return Compose(nQubits, start);
}

real1_f QEngineOCL::Probx(OCLAPI api_call, const bitCapIntOcl* bciArgs)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, writeArgsEvent);

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    writeArgsEvent.wait();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    return clampProb((real1_f)oneChance);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1_f QEngineOCL::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubitCount == 1) {
        return ProbAll(1);
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    const bitCapIntOcl qPower = pow2Ocl(qubit);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), qPower, 0U, 0U, 0U, 0U, 0U, 0U,
        0U, 0U };

    return Probx(OCL_API_PROB, bciArgs);
}

real1_f QEngineOCL::CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target)
{
    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    real1_f controlProb = Prob(control);
    if (!controlState) {
        controlProb = ONE_R1 - controlProb;
    }
    if (controlProb <= FP_NORM_EPSILON) {
        return ZERO_R1;
    }
    if ((ONE_R1 - controlProb) <= FP_NORM_EPSILON) {
        return Prob(target);
    }

    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QEngineOCL::CtrlOrAntiProb target index parameter must be within allocated qubit bounds!");
    }

    const bitCapIntOcl qPower = pow2Ocl(target);
    const bitCapIntOcl qControlPower = pow2Ocl(control);
    const bitCapIntOcl qControlMask = controlState ? qControlPower : 0U;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> 2U), qPower, qControlPower, qControlMask,
        0U, 0U, 0U, 0U, 0U, 0U };

    real1_f oneChance = Probx(OCL_API_CPROB, bciArgs);
    oneChance /= controlProb;

    return clampProb((real1_f)oneChance);
}

// Returns probability of permutation of the register
real1_f QEngineOCL::ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
{
    if (!start && qubitCount == length) {
        return ProbAll(permutation);
    }

    const bitCapIntOcl perm = (bitCapIntOcl)permutation << (bitCapIntOcl)start;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> length), perm, start, length, 0U, 0U, 0U,
        0U, 0U, 0U };

    return Probx(OCL_API_PROBREG, bciArgs);
}

void QEngineOCL::ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray)
{
    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl maxJ = maxQPowerOcl >> length;

    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(probsArray, probsArray + lengthPower, ZERO_R1);
        return;
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ lengthPower, maxJ, start, length, 0U, 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    AddAlloc(sizeof(real1) * lengthPower);
    BufferPtr probsBuffer = MakeBuffer(CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBREGALL, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probsBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();
    DISPATCH_BLOCK_READ(waitVec2, *probsBuffer, 0U, sizeof(real1) * lengthPower, probsArray);

    probsBuffer.reset();

    SubtractAlloc(sizeof(real1) * lengthPower);
}

// Returns probability of permutation of the register
real1_f QEngineOCL::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::ProbMask mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (length = 0U; v; ++length) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> length), (bitCapIntOcl)mask,
        (bitCapIntOcl)permutation, length, 0U, 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[length]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, skipPowers.get());
    skipPowers.reset();

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASK, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer, qPowersBuffer },
        sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    return clampProb((real1_f)oneChance);
}

void QEngineOCL::ProbMaskAll(bitCapInt mask, real1* probsArray)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::ProbMaskAll mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length;
    std::vector<bitCapIntOcl> powersVec;
    for (length = 0U; v; ++length) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        powersVec.push_back((v ^ oldV) & oldV);
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl maxJ = maxQPowerOcl >> length;

    if (!stateBuffer) {
        std::fill(probsArray, probsArray + lengthPower, ZERO_R1);
        return;
    }

    if ((lengthPower * lengthPower) < nrmGroupCount) {
        // With "lengthPower" count of threads, compared to a redundancy of "lengthPower" with full utilization, this is
        // close to the point where it becomes more efficient to rely on iterating through ProbReg calls.
        QEngine::ProbMaskAll(mask, probsArray);
        return;
    }

    v = (~(bitCapIntOcl)mask) & (maxQPowerOcl - ONE_BCI); // count the number of bits set in v
    bitCapIntOcl skipPower;
    bitLenInt skipLength = 0U; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (skipLength = 0U; v; ++skipLength) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        skipPowersVec.push_back(skipPower);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ lengthPower, maxJ, length, skipLength, 0U, 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    size_t sizeDiff = sizeof(real1) * lengthPower + sizeof(bitCapIntOcl) * length + sizeof(bitCapIntOcl) * skipLength;
    AddAlloc(sizeDiff);

    BufferPtr probsBuffer = MakeBuffer(CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    std::unique_ptr<bitCapIntOcl[]> powers(new bitCapIntOcl[length]);
    std::copy(powersVec.begin(), powersVec.end(), powers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, powers.get());
    powers.reset();

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[skipLength]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qSkipPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * skipLength, skipPowers.get());
    skipPowers.reset();

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASKALL, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probsBuffer, qPowersBuffer, qSkipPowersBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();
    DISPATCH_BLOCK_READ(waitVec2, *probsBuffer, 0U, sizeof(real1) * lengthPower, probsArray);

    probsBuffer.reset();
    qPowersBuffer.reset();
    qSkipPowersBuffer.reset();

    SubtractAlloc(sizeDiff);
}

real1_f QEngineOCL::ProbParity(bitCapInt mask)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::ProbParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (!mask) {
        return ZERO_R1_F;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, (bitCapIntOcl)mask, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    return Probx(OCL_API_PROBPARITY, bciArgs);
}

bool QEngineOCL::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineOCL::ForceMParity mask out-of-bounds!");
    }

    if (!stateBuffer || !mask) {
        return false;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return ForceM(log2(mask), result, doForce);
    }

    if (!doForce) {
        result = (Rand() <= ProbParity(mask));
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)(result ? ONE_BCI : 0U),
        0U, 0U, 0U, 0U, 0U, 0U, 0U };

    runningNorm = Probx(OCL_API_FORCEMPARITY, bciArgs);

    if (!doNormalize) {
        NormalizeState();
    }

    return result;
}

real1_f QEngineOCL::ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset)
{
    if (bits.size() == 1U) {
        return Prob(bits[0]);
    }

    if (!stateBuffer || !bits.size()) {
        return ZERO_R1_F;
    }

    if (doNormalize) {
        NormalizeState();
    }

    std::unique_ptr<bitCapIntOcl[]> bitPowers(new bitCapIntOcl[bits.size()]);
    std::transform(bits.begin(), bits.end(), bitPowers.get(), pow2Ocl);

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    BufferPtr bitMapBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * bits.size());
    DISPATCH_WRITE(waitVec, *bitMapBuffer, sizeof(bitCapIntOcl) * bits.size(), bitPowers.get());
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, bits.size(), (bitCapIntOcl)offset, 0U, 0U, 0U, 0U, 0U, 0U,
        0U };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_EXPPERM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, bitMapBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    real1_f expectation;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &expectation);

    return expectation;
}

real1_f QEngineOCL::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    LockSync(CL_MAP_READ);
    for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
        const bitCapIntOcl outputInt = (i & outputMask) >> valueStart;
        const real1 prob = norm(stateVec.get()[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return (real1_f)average;
}

void QEngineOCL::ArithmeticCall(
    OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], const unsigned char* values, bitCapIntOcl valuesPower)
{
    CArithmeticCall(api_call, bciArgs, NULL, 0U, values, valuesPower);
}
void QEngineOCL::CArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN],
    bitCapIntOcl* controlPowers, bitLenInt controlLen, const unsigned char* values, bitCapIntOcl valuesPower)
{
    CHECK_ZERO_SKIP();

    size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    if (controlLen) {
        sizeDiff += sizeof(bitCapIntOcl) * controlLen;
    }
    if (values) {
        sizeDiff += sizeof(unsigned char) * valuesPower;
    }
    AddAlloc(sizeDiff);

    EventVecPtr waitVec = ResetWaitEvents();

    // Allocate a temporary nStateVec, or use the one supplied.
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen) {
        controlBuffer =
            MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers);
    }

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen) {
        device_context->LockWaitEvents();
        device_context->wait_events->emplace_back();
        tryOcl(
            "Failed to enqueue buffer copy",
            [&] {
                return queue.enqueueCopyBuffer(*stateBuffer, *nStateBuffer, 0U, 0U, sizeof(complex) * maxQPowerOcl,
                    waitVec.get(), &(device_context->wait_events->back()));
            },
            true);
        device_context->UnlockWaitEvents();
    } else {
        ClearBuffer(nStateBuffer, 0U, maxQPowerOcl);
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * BCI_ARG_LEN, bciArgs);

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    std::vector<BufferPtr> oclArgs{ stateBuffer, poolItem->ulongBuffer, nStateBuffer };

    BufferPtr loadBuffer;
    if (values) {
        loadBuffer =
            MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, (void*)values);
        oclArgs.push_back(loadBuffer);
    }
    if (controlLen) {
        oclArgs.push_back(controlBuffer);
    }

    QueueCall(api_call, ngc, ngs, oclArgs);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::ROx range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    shift %= length;
    if (!shift) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~regMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, regMask, otherMask, lengthPower, start, shift, length, 0U,
        0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

#if ENABLE_ALU
/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INT range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & ~(regMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, regMask, otherMask, lengthPower, start, toMod, 0U, 0U, 0U,
        0U };

    ArithmeticCall(api_call, bciArgs);
}

/// Add or Subtract integer (without sign or carry, with controls)
void QEngineOCL::CINT(
    OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CINT range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineOCL::CINT control is out-of-bounds!");

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl regMask = lengthMask << start;

    bitCapIntOcl controlMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controls.size()]);
    for (size_t i = 0U; i < controls.size(); ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controls.size());

    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (regMask | controlMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> controls.size()), regMask, otherMask,
        lengthPower, start, toMod, controls.size(), controlMask, 0U, 0U };

    CArithmeticCall(api_call, bciArgs, controlPowers.get(), controls.size());
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INT(OCL_API_INC, (bitCapIntOcl)toAdd, start, length);
}

void QEngineOCL::CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        INC(toAdd, inOutStart, length);
        return;
    }

    CINT(OCL_API_CINC, (bitCapIntOcl)toAdd, inOutStart, length, controls);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(regMask | carryMask));
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), regMask, otherMask, lengthPower,
        carryMask, start, toMod, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/// Common driver method behing INCC and DECC
void QEngineOCL::INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    INTC(OCL_API_INCDECC, (bitCapIntOcl)toMod, inOutStart, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTS range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTS overflowIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ regMask;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, regMask, otherMask, lengthPower, overflowMask, start, toMod,
        0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, (bitCapIntOcl)toAdd, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex,
    bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTSC range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTSC overflowIndex is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTSC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, lengthPower,
        overflowMask, carryMask, start, toMod, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_1, (bitCapIntOcl)toAdd, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTSC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTSC carryIndex is out-of-bounds!");
    }

    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl inOutMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = pow2MaskOcl(qubitCount) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, lengthPower,
        carryMask, start, toMod, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_2, (bitCapIntOcl)toAdd, start, length, carryIndex);
}

#if ENABLE_BCD
/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTBCD range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, inOutMask, otherMask, start, toMod, nibbleCount, 0U, 0U, 0U,
        0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineOCL::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, (bitCapIntOcl)toAdd, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::INTBCDC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::INTBCDC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, carryMask,
        start, toMod, nibbleCount, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTBCDC(OCL_API_INCDECBCDC, (bitCapIntOcl)toAdd, start, length, carryIndex);
}
#endif

/** Multiply by integer */
void QEngineOCL::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(carryStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        SetReg(inOutStart, length, 0U);
        return;
    }

    MULx(OCL_API_MUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length);
}

/** Divide by integer */
void QEngineOCL::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (!toDiv) {
        throw std::runtime_error("DIV by zero");
    }

    MULx(OCL_API_DIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length);
}

/** Multiplication modulo N by integer, (out of place) */
void QEngineOCL::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(outStart, length, 0U);

    MULModx(OCL_API_MULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

void QEngineOCL::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    MULModx(OCL_API_IMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Raise a classical base to a quantum power, modulo N, (out of place) */
void QEngineOCL::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (base == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    MULModx(OCL_API_POWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Quantum analog of classical "Full Adder" gate */
void QEngineOCL::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_FULLADD);
}

/** Inverse of FullAdd */
void QEngineOCL::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_IFULLADD);
}

void QEngineOCL::FullAdx(
    bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)2U), pow2Ocl(inputBit1),
        pow2Ocl(inputBit2), pow2Ocl(carryInSumOut), pow2Ocl(carryOut), 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

/** Controlled multiplication by integer */
void QEngineOCL::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    CHECK_ZERO_SKIP();

    if (!controls.size()) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 1) {
        return;
    }

    CMULx(OCL_API_CMUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length, controls);
}

/** Controlled division by integer */
void QEngineOCL::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (!toDiv) {
        throw std::runtime_error("DIV by zero");
    }

    if (toDiv == 1) {
        return;
    }

    CMULx(OCL_API_CDIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length, controls);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    CHECK_ZERO_SKIP();

    if (!controls.size()) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        return;
    }

    CMULModx(OCL_API_CMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls);
}

void QEngineOCL::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        IMULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        return;
    }

    CMULModx(OCL_API_CIMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    CHECK_ZERO_SKIP();

    if (!controls.size()) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    CMULModx(OCL_API_CPOWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length, controls);
}

void QEngineOCL::xMULx(OCLAPI api_call, const bitCapIntOcl* bciArgs, BufferPtr controlBuffer)
{
    CHECK_ZERO_SKIP();

    EventVecPtr waitVec = ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ClearBuffer(nStateBuffer, 0U, maxQPowerOcl);

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 10U, bciArgs);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    if (controlBuffer) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer, controlBuffer });
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    }

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);
}

void QEngineOCL::MULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::MULx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::MULx range is out-of-bounds!");
    }

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << (bitCapIntOcl)inOutStart;
    const bitCapIntOcl carryMask = lowMask << (bitCapIntOcl)carryStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(carryStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inOutMask, carryMask,
        otherMask, length, inOutStart, carryStart, skipMask, 0U };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::MULModx(
    OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::MULModx range is out-of-bounds!");
    }

    if (isBadBitRange(outStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::MULModx range is out-of-bounds!");
    }

    if (!toMod) {
        return;
    }

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inMask = lowMask << (bitCapIntOcl)inStart;
    const bitCapIntOcl modMask = (isPowerOfTwo(modN) ? modN : pow2Ocl(log2(modN) + 1U)) - ONE_BCI;
    const bitCapIntOcl outMask = modMask << (bitCapIntOcl)outStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(outStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inMask, outMask, otherMask,
        length, inStart, outStart, skipMask, modN };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::CMULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart,
    bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CMULx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CMULx range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineOCL::CMULx control is out-of-bounds!");

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controls.size() + length]);
    bitCapIntOcl controlMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        skipPowers[i + controls.size()] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controls.size() + length);

    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask | controlMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)(controls.size() + length)),
        toMod, controls.size(), controlMask, inOutMask, carryMask, otherMask, length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controls.size() * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart,
    bitLenInt carryStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CMULModx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CMULModx range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineOCL::CMULModx control is out-of-bounds!");

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controls.size() + length]);
    bitCapIntOcl controlMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        skipPowers[i + controls.size()] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controls.size() + length);

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, toMod, controls.size(), controlMask, inOutMask, carryMask,
        modN, length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controls.size() * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values, bool resetValue)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::IndexedLDA range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::IndexedLDA range is out-of-bounds!");
    }

    if (!stateBuffer) {
        return 0U;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, 0U);
    }

    const bitLenInt valueBytes = (valueLength + 7) / 8;
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> valueLength), indexStart, inputMask,
        valueStart, valueBytes, valueLength, 0U, 0U, 0U, 0U };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return 0U;
#endif
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapIntOcl QEngineOCL::OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::OpIndexed range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::OpIndexed range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::OpIndexed carryIndex is out-of-bounds!");
    }

    if (!stateBuffer) {
        return 0U;
    }

    bool carryRes = M(carryIndex);
    // The carry has to first to be measured for its input value.
    if (carryRes) {
        /*
         * If the carry is set, we flip the carry bit. We always initially
         * clear the carry after testing for carry in.
         */
        carryIn ^= ONE_BCI;
        X(carryIndex);
    }

    const bitLenInt valueBytes = (valueLength + 7) / 8;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), indexStart, inputMask, valueStart,
        outputMask, otherMask, carryIn, carryMask, lengthPower, valueBytes };

    ArithmeticCall(api_call, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return 0U;
#endif
}

/** Add based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDADC, 0U, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDSBC, 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Set 8 bit register bits based on read from classical memory */
void QEngineOCL::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    const bitLenInt bytes = (length + 7) / 8;
    const bitCapIntOcl inputMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, start, inputMask, bytes, 0U, 0U, 0U, 0U, 0U, 0U };

    ArithmeticCall(OCL_API_HASH, bciArgs, values, pow2Ocl(length) * bytes);
}

void QEngineOCL::PhaseFlipX(OCLAPI api_call, const bitCapIntOcl* bciArgs)
{
    CHECK_ZERO_SKIP();

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::CPhaseFlipIfLess range is out-of-bounds!");
    }

    if (flagIndex >= qubitCount) {
        throw std::invalid_argument("QEngineOCL::CPhaseFlipIfLess flagIndex is out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        pow2Ocl(flagIndex), (bitCapIntOcl)greaterPerm, start, 0U, 0U, 0U, 0U, 0U };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineOCL::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineOCL::PhaseFlipIfLess range is out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        (bitCapIntOcl)greaterPerm, start, 0U, 0U, 0U, 0U, 0U, 0U };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}
#endif

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(complex const* inputState)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_BLOCK_WRITE(waitVec, *stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, inputState);

    UpdateRunningNorm();
}

complex QEngineOCL::GetAmplitude(bitCapInt perm)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QEngineOCL::GetAmplitude argument out-of-bounds!");
    }

    // WARNING: Does not normalize!
    if (!stateBuffer) {
        return ZERO_CMPLX;
    }

    complex amp;
    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_BLOCK_READ(waitVec, *stateBuffer, sizeof(complex) * (bitCapIntOcl)perm, sizeof(complex), &amp);

    return amp;
}

void QEngineOCL::SetAmplitude(bitCapInt perm, complex amp)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QEngineOCL::SetAmplitude argument out-of-bounds!");
    }

    if (!stateBuffer && !norm(amp)) {
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    permutationAmp = amp;

    if (runningNorm != REAL1_DEFAULT_ARG) {
        runningNorm += norm(amp) - norm(permutationAmp);
    }

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    tryOcl(
        "Failed to enqueue buffer write",
        [&] {
            return queue.enqueueWriteBuffer(*stateBuffer, CL_FALSE, sizeof(complex) * (bitCapIntOcl)perm,
                sizeof(complex), &permutationAmp, waitVec.get(), &(device_context->wait_events->back()));
        },
        true);
    device_context->UnlockWaitEvents();
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineOCL::GetQuantumState(complex* outputState)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(outputState, outputState + maxQPowerOcl, ZERO_CMPLX);
        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_BLOCK_READ(waitVec, *stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, outputState);
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineOCL::GetProbs(real1* outputProbs) { ProbRegAll(0U, qubitCount, outputProbs); }

real1_f QEngineOCL::SumSqrDiff(QEngineOCLPtr toCompare)
{
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }

    if (!stateBuffer && !toCompare->stateBuffer) {
        return ZERO_R1_F;
    }

    if (!stateBuffer) {
        toCompare->UpdateRunningNorm();
        return (real1_f)(toCompare->runningNorm);
    }

    if (!toCompare->stateBuffer) {
        UpdateRunningNorm();
        return (real1_f)runningNorm;
    }

    if (randGlobalPhase) {
        real1_f lPhaseArg = FirstNonzeroPhase();
        real1_f rPhaseArg = toCompare->FirstNonzeroPhase();
        NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, rPhaseArg - lPhaseArg);
    }

    toCompare->clFinish();

    const bool isMigrate = (device_context->context_id != toCompare->device_context->context_id);
    const int64_t oDevId = toCompare->deviceID;
    if (isMigrate) {
        toCompare->SetDevice(deviceID);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN]{ maxQPowerOcl, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    const size_t partInnerSize = ngc / ngs;

    AddAlloc(sizeof(complex) * partInnerSize);
    BufferPtr locCmplxBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(complex) * partInnerSize);

    QueueCall(OCL_API_APPROXCOMPARE, ngc, ngs,
        { stateBuffer, toCompare->stateBuffer, poolItem->ulongBuffer, locCmplxBuffer }, sizeof(complex) * ngs);

    std::unique_ptr<complex[]> partInner(new complex[partInnerSize]);

    clFinish();
    tryOcl("Failed to read buffer", [&] {
        return queue.enqueueReadBuffer(*locCmplxBuffer, CL_TRUE, 0U, sizeof(complex) * partInnerSize, partInner.get());
    });
    locCmplxBuffer.reset();
    SubtractAlloc(sizeof(complex) * partInnerSize);

    if (isMigrate) {
        toCompare->SetDevice(oDevId);
    }

    complex totInner = ZERO_CMPLX;
    for (size_t i = 0; i < partInnerSize; ++i) {
        totInner += partInner[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(totInner));
}

QInterfacePtr QEngineOCL::Clone()
{
    if (!stateBuffer) {
        return CloneEmpty();
    }

    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(qubitCount, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);

    cl::Event copyEvent;

    copyPtr->clFinish();
    clFinish();

    tryOcl("Failed to enqueue buffer copy", [&] {
        return queue.enqueueCopyBuffer(
            *stateBuffer, *(copyPtr->stateBuffer), 0U, 0U, sizeof(complex) * maxQPowerOcl, NULL, &copyEvent);
    });
    copyEvent.wait();

    copyPtr->runningNorm = runningNorm;

    return copyPtr;
}

QEnginePtr QEngineOCL::CloneEmpty()
{
    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(0U, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);

    copyPtr->SetQubitCount(qubitCount);

    return copyPtr;
}

void QEngineOCL::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    CHECK_ZERO_SKIP();

    if ((runningNorm == REAL1_DEFAULT_ARG) && (nrm == REAL1_DEFAULT_ARG)) {
        UpdateRunningNorm();
    }

    if (nrm < ZERO_R1) {
        // runningNorm can be set by OpenCL queue pop, so finish first.
        clFinish();
        nrm = (real1_f)runningNorm;
    }
    // We might avoid the clFinish().
    if (nrm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
        return;
    }
    if ((abs(ONE_R1 - nrm) <= FP_NORM_EPSILON) && ((phaseArg * phaseArg) <= FP_NORM_EPSILON)) {
        return;
    }
    // We might have async execution of gates still happening.
    clFinish();

    if (norm_thresh < ZERO_R1) {
        norm_thresh = (real1_f)amplitudeFloor;
    }
    nrm = ONE_R1_F / std::sqrt((real1_s)nrm);

    PoolItemPtr poolItem = GetFreePoolItem();

    complex c_args[2]{ complex((real1)norm_thresh, ZERO_R1), std::polar((real1)nrm, (real1)phaseArg) };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->cmplxBuffer), sizeof(complex) * 2, c_args, writeRealArgsEvent);

    bitCapIntOcl bciArgs[1]{ maxQPowerOcl };
    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeBCIArgsEvent);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    OCLAPI api_call;
    if (maxQPowerOcl == ngc) {
        api_call = OCL_API_NORMALIZE_WIDE;
    } else {
        api_call = OCL_API_NORMALIZE;
    }

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineOCL::UpdateRunningNorm(real1_f norm_thresh)
{
    if (!stateBuffer) {
        runningNorm = ZERO_R1_F;
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = (real1_f)amplitudeFloor;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    const real1 r1_args[1]{ (real1)norm_thresh };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1), r1_args, writeRealArgsEvent);

    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), &maxQPowerOcl, writeBCIArgsEvent);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    QueueCall(OCL_API_UPDATENORM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    if (runningNorm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
    }
}

#if defined(__APPLE__)
complex* _aligned_state_vec_alloc(bitCapIntOcl allocSize)
{
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
    return (complex*)toRet;
}
#endif

std::shared_ptr<complex> QEngineOCL::AllocStateVec(bitCapInt elemCount, bool doForceAlloc)
{
    // If we're not using host ram, there's no reason to allocate.
    if (!elemCount || (!doForceAlloc && !stateVec)) {
        return NULL;
    }

#if defined(__ANDROID__)
    return std::shared_ptr<complex>(elemCount);
#else
    // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
    size_t allocSize = sizeof(complex) * (size_t)elemCount;
    if (allocSize < QRACK_ALIGN_SIZE) {
        allocSize = QRACK_ALIGN_SIZE;
    }
#if defined(__APPLE__)
    return std::shared_ptr<complex>(_aligned_state_vec_alloc(allocSize), [](complex* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return std::shared_ptr<complex>(
        (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE), [](complex* c) { _aligned_free(c); });
#else
    return std::shared_ptr<complex>((complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize), [](complex* c) { free(c); });
#endif
#endif
}

BufferPtr QEngineOCL::MakeStateVecBuffer(std::shared_ptr<complex> nStateVec)
{
    if (!maxQPowerOcl) {
        return NULL;
    }

    if (nStateVec) {
        return MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl, nStateVec.get());
    } else {
        return MakeBuffer(CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl);
    }
}

void QEngineOCL::ReinitBuffer()
{
    AddAlloc(sizeof(complex) * maxQPowerOcl);
    stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
    ResetStateBuffer(MakeStateVecBuffer(stateVec));
}

void QEngineOCL::ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size)
{
    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapIntOcl bciArgs[2]{ size, offset };
    cl::Event writeArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(size, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();

    QueueCall(OCL_API_CLEARBUFFER, ngc, ngs, { buff, poolItem->ulongBuffer });
}

} // namespace Qrack
