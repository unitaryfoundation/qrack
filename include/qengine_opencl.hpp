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

#include "common/oclengine.hpp"
#include "qengine.hpp"
#include "qengine_gpu_util.hpp"

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#include <list>
#include <mutex>

#define BCI_ARG_LEN 10
#define CMPLX_NORM_LEN 6
#define REAL_ARG_LEN 2

namespace Qrack {

typedef std::shared_ptr<cl::Buffer> BufferPtr;

class QEngineOCL;
typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

struct QueueItem {
    OCLAPI api_call;
    size_t workItemCount;
    size_t localGroupSize;
    size_t deallocSize;
    std::vector<BufferPtr> buffers;
    size_t localBuffSize;
    bool isSetDoNorm;
    bool isSetRunningNorm;
    bool doNorm;
    real1 runningNorm;

    QueueItem()
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(false)
        , isSetRunningNorm(true)
        , doNorm(false)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(OCLAPI ac, size_t wic, size_t lgs, size_t ds, std::vector<BufferPtr> b, size_t lbs)
        : api_call(ac)
        , workItemCount(wic)
        , localGroupSize(lgs)
        , deallocSize(ds)
        , buffers(b)
        , localBuffSize(lbs)
        , isSetDoNorm(false)
        , isSetRunningNorm(false)
        , doNorm(false)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(bool doNrm)
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(true)
        , isSetRunningNorm(false)
        , doNorm(doNrm)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(real1_f runningNrm)
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(false)
        , isSetRunningNorm(true)
        , doNorm(false)
        , runningNorm(runningNrm)
    {
    }
};

class PoolItem {
public:
    BufferPtr cmplxBuffer;
    BufferPtr realBuffer;
    BufferPtr ulongBuffer;

    std::shared_ptr<real1> probArray;
    std::shared_ptr<real1> angleArray;

    PoolItem(cl::Context& context)
        : probArray{ nullptr }
        , angleArray{ nullptr }
    {
        cmplxBuffer = MakeBuffer(context, sizeof(complex) * CMPLX_NORM_LEN);
        realBuffer = MakeBuffer(context, sizeof(real1) * REAL_ARG_LEN);
        ulongBuffer = MakeBuffer(context, sizeof(bitCapIntOcl) * BCI_ARG_LEN);
    }

    ~PoolItem() {}

protected:
    BufferPtr MakeBuffer(const cl::Context& context, size_t size)
    {
        cl_int error;
        BufferPtr toRet = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, size, (void*)nullptr, &error);
        if (error != CL_SUCCESS) {
            if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                throw bad_alloc("CL_MEM_OBJECT_ALLOCATION_FAILURE in PoolItem::MakeBuffer()");
            }
            if (error == CL_OUT_OF_HOST_MEMORY) {
                throw bad_alloc("CL_OUT_OF_HOST_MEMORY in PoolItem::MakeBuffer()");
            }
            if (error == CL_INVALID_BUFFER_SIZE) {
                throw bad_alloc("CL_INVALID_BUFFER_SIZE in PoolItem::MakeBuffer()");
            }
            throw std::runtime_error("OpenCL error code on buffer allocation attempt: " + std::to_string(error));
        }

        return toRet;
    }
};

typedef std::shared_ptr<PoolItem> PoolItemPtr;

/**
 * OpenCL enhanced QEngineCPU implementation.
 *
 * QEngineOCL exposes asynchronous void-return public methods, wherever possible. While QEngine public methods run on a
 * secondary accelerator, such as a GPU, other code can be executed on the CPU at the same time. If only one (CPU)
 * OpenCL device is available, this engine type is still compatible with most CPUs, and this implementation will still
 * usually give a very significant performance boost over the non-OpenCL QEngineCPU implementation.
 *
 * Each QEngineOCL queues an independent event list of chained asynchronous methods. Multiple QEngineOCL instances may
 * share a single device. Any one QEngineOCL instance (or QEngineCPU instance) is NOT safe to access from multiple
 * threads, but different QEngineOCL instances may be accessed in respective threads. When a public method with a
 * non-void return type is called, (such as Prob() or M() variants,) the engine wait list of OpenCL events will first be
 * finished, then the return value will be calculated based on all public method calls dispatched up to that point.
 * Asynchronous method dispatch is "transparent," in the sense that no explicit consideration for synchronization should
 * be necessary. The programmer benefits from knowing that void-return methods attempt asynchronous execution, but
 * asynchronous methods are always joined, in order of dispatch, before any and all non-void-return methods give their
 * results.
 */
class QEngineOCL : public QEngine {
protected:
    bool didInit;
    bool usingHostRam;
    bool unlockHostMem;
    cl_int callbackError;
    size_t nrmGroupCount;
    size_t nrmGroupSize;
    size_t totalOclAllocSize;
    int64_t deviceID;
    cl_map_flags lockSyncFlags;
    complex permutationAmp;
    std::shared_ptr<complex> stateVec;
    std::mutex queue_mutex;
    cl::CommandQueue queue;
    cl::Context context;
    // stateBuffer is allocated as a shared_ptr, because it's the only buffer that will be acted on outside of
    // QEngineOCL itself, specifically by QEngineOCLMulti.
    BufferPtr stateBuffer;
    BufferPtr nrmBuffer;
    DeviceContextPtr device_context;
    std::vector<EventVecPtr> wait_refs;
    std::list<QueueItem> wait_queue_items;
    std::vector<PoolItemPtr> poolItems;
    std::unique_ptr<real1[], void (*)(real1*)> nrmArray;

#if defined(__APPLE__)
    real1* _aligned_nrm_array_alloc(bitCapIntOcl allocSize)
    {
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
        return (real1*)toRet;
    }
#endif

    void checkCallbackError()
    {
        if (callbackError == CL_SUCCESS) {
            return;
        }

        wait_queue_items.clear();
        wait_refs.clear();

        throw std::runtime_error("Failed to enqueue kernel, error code: " + std::to_string(callbackError));
    }

    // For std::function, cl_int use might discard int qualifiers.
    void tryOcl(std::string message, std::function<int()> oclCall)
    {
        checkCallbackError();

        if (oclCall() == CL_SUCCESS) {
            // Success
            return;
        }

        // Soft finish (just for this QEngineOCL)
        clFinish();

        if (oclCall() == CL_SUCCESS) {
            // Success after clearing QEngineOCL queue
            return;
        }

        // Hard finish (for the unique OpenCL device)
        clFinish(true);

        cl_int error = oclCall();
        if (error == CL_SUCCESS) {
            // Success after clearing all queues for the OpenCL device
            return;
        }

        wait_queue_items.clear();
        wait_refs.clear();

        // We're fatally blocked. Throw to exit.
        throw std::runtime_error(message + ", error code: " + std::to_string(error));
    }

    using QEngine::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QEngineOCL>(orig)); }
    void Copy(QEngineOCLPtr orig)
    {
        QEngine::Copy(std::dynamic_pointer_cast<QEngine>(orig));
        didInit = orig->didInit;
        usingHostRam = orig->usingHostRam;
        unlockHostMem = orig->unlockHostMem;
        callbackError = orig->callbackError;
        nrmGroupCount = orig->nrmGroupCount;
        nrmGroupSize = orig->nrmGroupSize;
        AddAlloc(orig->totalOclAllocSize);
        deviceID = orig->deviceID;
        lockSyncFlags = orig->lockSyncFlags;
        permutationAmp = orig->permutationAmp;
        stateVec = orig->stateVec;
        // queue_mutex = orig->queue_mutex;
        queue = orig->queue;
        context = orig->context;
        stateBuffer = orig->stateBuffer;
        nrmBuffer = orig->nrmBuffer;
        device_context = orig->device_context;
        wait_refs = orig->wait_refs;
        wait_queue_items = orig->wait_queue_items;
        poolItems = orig->poolItems;
    }

public:
    /// 1 / OclMemDenom is the maximum fraction of total OCL device RAM that a single state vector should occupy, by
    /// design of the QEngine.
    static const bitCapIntOcl OclMemDenom = 3U;

    /**
     * Initialize a Qrack::QEngineOCL object. Specify the number of qubits and an initial permutation state.
     * Additionally, optionally specify a pointer to a random generator engine object, a device ID from the list of
     * devices in the OCLEngine singleton, and a boolean that is set to "true" to initialize the state vector of the
     * object to zero norm.
     *
     * "devID" is the index of an OpenCL device in the OCLEngine singleton, to select the device to run this engine on.
     * If "useHostMem" is set false, as by default, the QEngineOCL will attempt to allocate the state vector object
     * only on device memory. If "useHostMem" is set true, general host RAM will be used for the state vector buffers.
     * If the state vector is too large to allocate only on device memory, the QEngineOCL will attempt to fall back to
     * allocating it in general host RAM.
     *
     * \warning "useHostMem" is not conscious of allocation by other QEngineOCL instances on the same device. Attempting
     * to allocate too much device memory across too many QEngineOCL instances, for which each instance would have
     * sufficient device resources on its own, will probably cause the program to crash (and may lead to general system
     * instability). For safety, "useHostMem" can be turned on.
     */

    QEngineOCL(bitLenInt qBitCount, const bitCapInt& initState, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t devID = -1, bool useHardwareRNG = true, bool ignored = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored2 = {}, bitLenInt ignored4 = 0U,
        real1_f ignored3 = _qrack_qunit_sep_thresh);

    ~QEngineOCL()
    {
        // Theoretically, all user output is blocking, so don't throw in destructor.
        callbackError = CL_SUCCESS;
        // Make sure we track device allocation.
        FreeAll();
    }

    virtual bool isOpenCL() { return true; }

    bool IsZeroAmplitude() { return !stateBuffer; }
    real1_f FirstNonzeroPhase()
    {
        if (!stateBuffer) {
            return ZERO_R1_F;
        }

        return QInterface::FirstNonzeroPhase();
    }

    void ZeroAmplitudes();
    void CopyStateVec(QEnginePtr src);

    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length);
    void ShuffleBuffers(QEnginePtr engine);
    QEnginePtr CloneEmpty();

    bitCapIntOcl GetMaxSize() { return device_context->GetMaxAlloc() / sizeof(complex); };

    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    using QEngine::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask);
    void UniformParityRZ(const bitCapInt& mask, real1_f angle);
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle);

    /* Operations that have an improved implementation. */
    using QEngine::X;
    void X(bitLenInt target);
    using QEngine::Z;
    void Z(bitLenInt target);
    using QEngine::Invert;
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubitIndex);
    using QEngine::Phase;
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubitIndex);
    void XMask(const bitCapInt& mask);
    void PhaseParity(real1_f radians, const bitCapInt& mask);
    void PhaseRootNMask(bitLenInt n, const bitCapInt& mask);

    using QEngine::Compose;
    bitLenInt Compose(QEngineOCLPtr toCopy);
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    bitLenInt Compose(QEngineOCLPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy), start);
    }
    using QEngine::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm);
    using QEngine::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

#if ENABLE_ALU
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls);
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true);
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);

    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length);
#endif

    real1_f Prob(bitLenInt qubit);
    real1_f CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target);
    real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation);
    void ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray);
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    void ProbMaskAll(const bitCapInt& mask, real1* probsArray);
    real1_f ProbParity(const bitCapInt& mask);
    bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI);

    void SetDevice(int64_t dID);
    int64_t GetDevice() { return deviceID; }
    void SetDeviceList(std::vector<int64_t> dIDs);
    std::vector<int64_t> GetDeviceList() { return std::vector<int64_t>{ deviceID }; }

    void SetQuantumState(const complex* inputState);
    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    bitCapInt MAll();
    complex GetAmplitude(const bitCapInt& perm);
    void SetAmplitude(const bitCapInt& perm, const complex& amp);

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QEngineOCL>(toCompare)); }
    real1_f SumSqrDiff(QEngineOCLPtr toCompare);

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    void Finish() { clFinish(); };
    bool isFinished() { return wait_queue_items.empty(); };

    QInterfacePtr Clone();
    QInterfacePtr Copy();

    void PopQueue(bool isDispatch);
    void DispatchQueue();

protected:
    void AddAlloc(size_t size)
    {
        size_t currentAlloc = OCLEngine::Instance().AddToActiveAllocSize(deviceID, size);
        if (device_context && (currentAlloc > device_context->GetGlobalAllocLimit())) {
            OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
            throw bad_alloc("VRAM limits exceeded in QEngineOCL::AddAlloc()");
        }
        totalOclAllocSize += size;
    }
    void SubtractAlloc(size_t size)
    {
        OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
        totalOclAllocSize -= size;
    }

    BufferPtr MakeBuffer(cl_mem_flags flags, size_t size, void* host_ptr = nullptr)
    {
        checkCallbackError();

        cl_int error;
        BufferPtr toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error == CL_SUCCESS) {
            // Success
            return toRet;
        }

        // Soft finish (just for this QEngineOCL)
        clFinish();

        toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error == CL_SUCCESS) {
            // Success after clearing QEngineOCL queue
            return toRet;
        }

        // Hard finish (for the unique OpenCL device)
        clFinish(true);

        toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error != CL_SUCCESS) {
            if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                throw bad_alloc("CL_MEM_OBJECT_ALLOCATION_FAILURE in QEngineOCL::MakeBuffer()");
            }
            if (error == CL_OUT_OF_HOST_MEMORY) {
                throw bad_alloc("CL_OUT_OF_HOST_MEMORY in QEngineOCL::MakeBuffer()");
            }
            if (error == CL_INVALID_BUFFER_SIZE) {
                throw bad_alloc("CL_INVALID_BUFFER_SIZE in QEngineOCL::MakeBuffer()");
            }
            throw std::runtime_error("OpenCL error code on buffer allocation attempt: " + std::to_string(error));
        }

        return toRet;
    }

    void SwitchHostPtr(bool useHostMem)
    {
        if (useHostMem == usingHostRam) {
            return;
        }

        std::shared_ptr<complex> copyVec = AllocStateVec(maxQPowerOcl, true);
        GetQuantumState(copyVec.get());

        if (useHostMem) {
            stateVec = copyVec;
            stateBuffer = MakeStateVecBuffer(stateVec);
        } else {
            stateVec = nullptr;
            stateBuffer = MakeStateVecBuffer(stateVec);
            tryOcl("Failed to write buffer", [&] {
                return queue.enqueueWriteBuffer(
                    *stateBuffer, CL_TRUE, 0U, sizeof(complex) * maxQPowerOcl, copyVec.get(), ResetWaitEvents().get());
            });
            wait_refs.clear();
            copyVec = nullptr;
        }

        usingHostRam = useHostMem;
    }

    void QueueCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0U, size_t deallocSize = 0U)
    {
        if (localBuffSize > device_context->GetLocalSize()) {
            throw bad_alloc("Local memory limits exceeded in QEngineOCL::QueueCall()");
        }
        AddQueueItem(QueueItem(api_call, workItemCount, localGroupSize, deallocSize, args, localBuffSize));
    }

    void QueueSetDoNormalize(bool doNorm) { AddQueueItem(QueueItem(doNorm)); }
    void QueueSetRunningNorm(real1_f runningNrm) { AddQueueItem(QueueItem(runningNrm)); }
    void AddQueueItem(const QueueItem& item)
    {
        bool isBase;
        // For lock_guard:
        if (true) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            checkCallbackError();
            isBase = wait_queue_items.empty();
            wait_queue_items.push_back(item);
        }

        if (isBase) {
            DispatchQueue();
        }
    }

    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength);

    std::shared_ptr<complex> AllocStateVec(bitCapIntOcl elemCount, bool doForceAlloc = false);
    void FreeStateVec() { stateVec = nullptr; }
    void FreeAll();
    void ResetStateBuffer(BufferPtr nStateBuffer);
    BufferPtr MakeStateVecBuffer(std::shared_ptr<complex> nStateVec);
    void ReinitBuffer();

    void Compose(OCLAPI apiCall, const bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy);

    void InitOCL(int64_t devID);
    PoolItemPtr GetFreePoolItem();

    real1_f ParSum(real1* toSum, bitCapIntOcl maxI);

    /**
     * Locks synchronization between the state vector buffer and general RAM, so the state vector can be directly read
     * and/or written to.
     *
     * OpenCL buffers, even when allocated on "host" general RAM, are not safe to read from or write to unless "mapped."
     * When mapped, a buffer cannot be used by OpenCL kernels. If the state vector needs to be directly manipulated, it
     * needs to be temporarily mapped, and this can be accomplished with LockSync(). When direct reading from or writing
     * to the state vector is done, before performing other OpenCL operations on it, it must be unmapped with
     * UnlockSync().
     */
    void LockSync(cl_map_flags flags = (CL_MAP_READ | CL_MAP_WRITE));
    /**
     * Unlocks synchronization between the state vector buffer and general RAM, so the state vector can be operated on
     * with OpenCL kernels and operations.
     *
     * OpenCL buffers, even when allocated on "host" general RAM, are not safe to read from or write to unless "mapped."
     * When mapped, a buffer cannot be used by OpenCL kernels. If the state vector needs to be directly manipulated, it
     * needs to be temporarily mapped, and this can be accomplished with LockSync(). When direct reading from or writing
     * to the state vector is done, before performing other OpenCL operations on it, it must be unmapped with
     * UnlockSync().
     */
    void UnlockSync();

    /**
     * Finishes the asynchronous wait event list or queue of OpenCL events.
     *
     * By default (doHard = false) only the wait event list of this engine is finished. If doHard = true, the entire
     * device queue is finished, (which might be shared by other QEngineOCL instances).
     */
    void clFinish(bool doHard = false);

    /**
     * Dumps the remaining asynchronous wait event list or queue of OpenCL events, for the current queue.
     */
    void clDump();

    size_t FixWorkItemCount(size_t maxI, size_t wic)
    {
        if (wic > maxI) {
            // Guaranteed to be a power of two
            return maxI;
        }

        // Otherwise, clamp to a power of two
        return pow2Ocl(log2Ocl(wic));
    }

    size_t FixGroupSize(size_t wic, size_t gs)
    {
        if (gs > wic) {
            return wic;
        }

        return gs - (wic % gs);
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);

    using QEngine::Apply2x2;
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, SPECIAL_2X2::NONE, norm_thresh);
    }
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special,
        real1_f norm_thresh = REAL1_DEFAULT_ARG);

    void BitMask(bitCapIntOcl mask, OCLAPI api_call, real1_f phase = (real1_f)PI_R1);

    void ApplyM(const bitCapInt& mask, bool result, const complex& nrm);
    void ApplyM(const bitCapInt& mask, const bitCapInt& result, const complex& nrm);

    /* Utility functions used by the operations above. */
    void WaitCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0U);
    EventVecPtr ResetWaitEvents(bool waitQueue = true);
    void ApplyMx(OCLAPI api_call, const bitCapIntOcl* bciArgs, const complex& nrm);
    real1_f Probx(OCLAPI api_call, const bitCapIntOcl* bciArgs);

    void ArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN],
        const unsigned char* values = nullptr, bitCapIntOcl valuesLength = 0U);
    void CArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], bitCapIntOcl* controlPowers,
        bitLenInt controlLen, const unsigned char* values = nullptr, bitCapIntOcl valuesLength = 0U);
    void ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length);

#if ENABLE_ALU
    void INCDECC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(
        const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCDECBCDC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif

    void INT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length);
    void CINT(
        OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls);
    void INTC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INTS(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex,
        bitLenInt carryIndex);
#if ENABLE_BCD
    void INTBCD(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length);
    void INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif
    void xMULx(OCLAPI api_call, const bitCapIntOcl* bciArgs, BufferPtr controlBuffer);
    void MULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart, bitLenInt carryStart,
        bitLenInt length);
    void CMULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart, bitLenInt carryStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void FullAdx(
        bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call);
    void PhaseFlipX(OCLAPI api_call, const bitCapIntOcl* bciArgs);

    bitCapIntOcl OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
#endif

    void ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size);
};

} // namespace Qrack
