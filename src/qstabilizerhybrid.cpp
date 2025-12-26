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

#include <iomanip>
#include <thread>

#if ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

#define IS_REAL_1(r) (abs(ONE_R1 - r) <= FP_NORM_EPSILON)
#define IS_CTRLED_CLIFFORD(top, bottom)                                                                                \
    ((IS_REAL_1(std::real(top)) || IS_REAL_1(std::imag(bottom))) && (IS_SAME(top, bottom) || IS_SAME(top, -bottom)))
#define IS_CLIFFORD_PHASE_INVERT(top, bottom)                                                                          \
    (IS_SAME(top, bottom) || IS_SAME(top, -bottom) || IS_SAME(top, I_CMPLX * bottom) || IS_SAME(top, -I_CMPLX * bottom))
#define IS_CLIFFORD(mtrx)                                                                                              \
    ((IS_PHASE(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[0], mtrx[3])) ||                                                 \
        (IS_INVERT(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[1], mtrx[2])) ||                                             \
        ((IS_SAME(mtrx[0U], mtrx[1U]) || IS_SAME(mtrx[0U], -mtrx[1U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) ||      \
             IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                                                                \
            (IS_SAME(mtrx[0U], mtrx[2U]) || IS_SAME(mtrx[0U], -mtrx[2U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                                                             \
            (IS_SAME(mtrx[0U], mtrx[3U]) || IS_SAME(mtrx[0U], -mtrx[3U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U]))))
#define IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState,
    qrack_rand_gen_ptr rgp, const complex& phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem,
    int64_t deviceId, bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , useHostRam(useHostMem)
    , doNormalize(doNorm)
    , useTGadget(true)
    , isRoundingFlushed(false)
    , thresholdQubits(qubitThreshold)
    , ancillaCount(0U)
    , deadAncillaCount(0U)
    , maxEngineQubitCount(27U)
    , maxAncillaCount(28U)
    , origMaxAncillaCount(28U)
    , separabilityThreshold(sep_thresh)
    , roundingThreshold(FP_NORM_EPSILON_F)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , logFidelity(0.0)
    , engine{ nullptr }
    , rdmClone{ nullptr }
    , deviceIDs(devList)
    , engineTypes(eng)
    , cloneEngineTypes(eng)
    , shards(qubitCount)
    , stateMapCache{ nullptr }
    , prng(std::random_device{}())
{
#if ENABLE_OPENCL || ENABLE_CUDA
    const size_t devCount = QRACK_GPU_SINGLETON.GetDeviceCount();
    const bool isQPager = (engineTypes[0U] == QINTERFACE_HYBRID) || (engineTypes[0U] == QINTERFACE_OPENCL);
    if (devCount &&
        (isQPager ||
            ((engineTypes[0U] == QINTERFACE_QPAGER) &&
                ((engineTypes.size() == 1U) || (engineTypes[1U] == QINTERFACE_OPENCL))))) {
        DeviceContextPtr devContext = QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID);
        maxEngineQubitCount = log2Ocl(devContext->GetMaxAlloc() / sizeof(complex));
        maxAncillaCount = maxEngineQubitCount;
        if (isQPager) {
            --maxEngineQubitCount;
            const bitLenInt perPage =
                log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) - 1U;
#if ENABLE_OPENCL
            maxAncillaCount = (devCount < 2U) ? (perPage + 3U) : (perPage + log2Ocl(devCount) + 2U);
#else
            maxAncillaCount = perPage + log2Ocl(devCount);
#endif
            if (QRACK_MAX_PAGE_QB_DEFAULT < maxEngineQubitCount) {
                maxEngineQubitCount = QRACK_MAX_PAGE_QB_DEFAULT;
            }
#if ENABLE_OPENCL
            else {
                maxEngineQubitCount = (maxEngineQubitCount > 1U) ? (maxEngineQubitCount - 1U) : 1U;
            }
#endif
            if (QRACK_MAX_PAGING_QB_DEFAULT < maxAncillaCount) {
                maxAncillaCount = QRACK_MAX_PAGING_QB_DEFAULT;
            }
        }
    } else {
        maxEngineQubitCount = QRACK_MAX_CPU_QB_DEFAULT;
        maxAncillaCount = maxEngineQubitCount;
    }
#endif

    UpdateRoundingThreshold();
    maxStateMapCacheQubitCount = QRACK_MAX_CPU_QB_DEFAULT - ((QBCAPPOW < FPPOW) ? 1U : (1U + QBCAPPOW - FPPOW));
    stabilizer = MakeStabilizer(initState);
}

QUnitCliffordPtr QStabilizerHybrid::MakeStabilizer(const bitCapInt& perm)
{
    return std::make_shared<QUnitClifford>(qubitCount + ancillaCount + deadAncillaCount, perm, rand_generator,
        ONE_CMPLX, false, randGlobalPhase, false, -1, useRDRAND);
}
QInterfacePtr QStabilizerHybrid::MakeEngine(const bitCapInt& perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineTypes, qubitCount, perm, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs,
        thresholdQubits, separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return toRet;
}
QInterfacePtr QStabilizerHybrid::MakeEngine(const bitCapInt& perm, bitLenInt qbCount)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineTypes, qbCount, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return toRet;
}

void QStabilizerHybrid::InvertBuffer(bitLenInt qubit)
{
    QRACK_CONST complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
    pauliShard->Compose(shards[qubit]->gate);
    shards[qubit] = pauliShard->IsIdentity() ? nullptr : pauliShard;
    stabilizer->X(qubit);
}

void QStabilizerHybrid::FlushH(bitLenInt qubit)
{
    QRACK_CONST complex h[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    MpsShardPtr shard = std::make_shared<MpsShard>(h);
    shard->Compose(shards[qubit]->gate);
    shards[qubit] = shard->IsIdentity() ? nullptr : shard;
    stabilizer->H(qubit);
}

void QStabilizerHybrid::FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase)
{
    if (engine) {
        return;
    }

    const MpsShardPtr& cshard = shards[control];
    if (cshard && (cshard->IsHPhase() || cshard->IsHInvert())) {
        FlushH(control);
    }
    if (cshard && cshard->IsInvert()) {
        InvertBuffer(control);
    }
    if (cshard && !cshard->IsPhase()) {
        return SwitchToEngine();
    }

    const MpsShardPtr& tshard = shards[target];
    if (tshard && (tshard->IsHPhase() || tshard->IsHInvert())) {
        FlushH(target);
    }
    if (tshard && tshard->IsInvert()) {
        InvertBuffer(target);
    }

    if (!tshard) {
        return;
    }
    // Shard is definitely non-nullptr.

    if (!(tshard->IsPhase())) {
        return SwitchToEngine();
    }
    // Shard is definitely a phase gate.

    if (isPhase) {
        // The previously potentially blocked gate commutes.
        return;
    }
    // The gate payload is definitely not a phase gate.
    // Since the blocked gate does not commute, we must flush.
    // This is the new case we can handle with the "reverse gadget" for t-injection in this PRX Quantum article, in
    // Appendix A: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020361
    // Hakop Pashayan, Oliver Reardon-Smith, Kamil Korzekwa, and Stephen D. Bartlett
    // PRX Quantum 3, 020361 – Published 23 June 2022

    if (!useTGadget || (ancillaCount >= maxAncillaCount)) {
        // The option to optimize this case is off.
        return SwitchToEngine();
    }

    const MpsShardPtr shard = shards[target];
    shards[target] = nullptr;

    const real1 angle = (real1)(FractionalRzAngleWithFlush(target, std::arg(shard->gate[3U] / shard->gate[0U])) / 2);
    if ((2 * abs(angle)) <= (FP_NORM_EPSILON * PI_R1)) {
        return;
    }
    const real1 angleCos = (real1)cos(angle);
    const real1 angleSin = (real1)sin(angle);
    shard->gate[0U] = complex(angleCos, -angleSin);
    shard->gate[3U] = complex(angleCos, angleSin);

    // Form a representation of state that can entangle a new (or reused) ancilla.
    bitLenInt ancillaIndex = deadAncillaCount ? (qubitCount + ancillaCount) : stabilizer->Allocate(1U);
    ++ancillaCount;
    shards.emplace_back(nullptr);
    if (deadAncillaCount) {
        --deadAncillaCount;
    }

    // Use reverse t-injection gadget.
    stabilizer->CNOT(target, ancillaIndex);
    Mtrx(shard->gate, ancillaIndex);
    H(ancillaIndex);

    // When we eventually measure, we act postselection, but not yet.
    // ForceM(ancillaIndex, false, true, true);
    // Ancilla is separable after measurement.
    // Dispose(ancillaIndex, 1U);
}

bool QStabilizerHybrid::CollapseSeparableShard(bitLenInt qubit)
{
    MpsShardPtr shard = shards[qubit];
    shards[qubit] = nullptr;

    const bool isZ1 = stabilizer->M(qubit);
    const real1_f prob = (real1_f)(isZ1 ? norm(shard->gate[3U]) : norm(shard->gate[2U]));

    bool result;
    if (prob <= ZERO_R1) {
        result = false;
    } else if (prob >= ONE_R1) {
        result = true;
    } else {
        result = (Rand() <= prob);
    }

    if (result != isZ1) {
        stabilizer->X(qubit);
    }

    return result;
}

void QStabilizerHybrid::FlushBuffers()
{
    if (stabilizer) {
        if (IsBuffered()) {
            // This will call FlushBuffers() again after no longer stabilizer.
            SwitchToEngine();
        }

        return;
    }

    for (size_t i = 0U; i < shards.size(); ++i) {
        const MpsShardPtr shard = shards[i];
        if (shard) {
            shards[i] = nullptr;
            engine->Mtrx(shard->gate, i);
        }
    }
}

bool QStabilizerHybrid::TrimControls(const std::vector<bitLenInt>& lControls, std::vector<bitLenInt>& output, bool anti)
{
    if (engine) {
        output.insert(output.begin(), lControls.begin(), lControls.end());

        return false;
    }

    for (size_t i = 0U; i < lControls.size(); ++i) {
        bitLenInt bit = lControls[i];
        if (!stabilizer->IsSeparableZ(bit)) {
            output.push_back(bit);
            continue;
        }

        const MpsShardPtr& shard = shards[bit];
        if (shard) {
            if (shard->IsInvert()) {
                if (anti != stabilizer->M(bit)) {
                    return true;
                }

                continue;
            }

            if (shard->IsPhase()) {
                if (anti == stabilizer->M(bit)) {
                    return true;
                }

                continue;
            }

            output.push_back(bit);
        } else if (anti == stabilizer->M(bit)) {
            return true;
        }
    }

    return false;
}

void QStabilizerHybrid::CacheEigenstate(bitLenInt target)
{
    if (engine) {
        return;
    }

    MpsShardPtr toRet{ nullptr };
    // If in PauliX or PauliY basis, compose gate with conversion from/to PauliZ basis.
    stabilizer->H(target);
    if (stabilizer->IsSeparableZ(target)) {
        // X eigenstate
        const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(SQRT1_2_R1, ZERO_R1), complex(-SQRT1_2_R1, ZERO_R1) };
        toRet = std::make_shared<MpsShard>(mtrx);
    } else {
        stabilizer->H(target);
        stabilizer->IS(target);
        stabilizer->H(target);
        if (stabilizer->IsSeparableZ(target)) {
            // Y eigenstate
            const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
                complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };
            toRet = std::make_shared<MpsShard>(mtrx);
        } else {
            stabilizer->H(target);
            stabilizer->S(target);
        }
    }

    if (!toRet) {
        return;
    }

    MpsShardPtr& shard = shards[target];
    if (shard) {
        toRet->Compose(shard->gate);
    }
    shard = toRet;
}

QInterfacePtr QStabilizerHybrid::CloneBody(bool isCopy)
{
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(cloneEngineTypes, qubitCount, ZERO_BCI, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);

    if (engine) {
        // Clone and set engine directly.
        c->engine = isCopy ? engine->Copy() : engine->Clone();
        c->stabilizer = nullptr;

        return c;
    }

    // Otherwise, stabilizer
    c->engine = nullptr;
    c->stabilizer = std::dynamic_pointer_cast<QUnitClifford>(stabilizer->Clone());
    c->rdmClone = rdmClone ? std::dynamic_pointer_cast<QStabilizerHybrid>(rdmClone->Clone()) : nullptr;
    c->shards.resize(shards.size());
    c->ancillaCount = ancillaCount;
    c->deadAncillaCount = deadAncillaCount;
    c->roundingThreshold = roundingThreshold;
    for (size_t i = 0U; i < shards.size(); ++i) {
        if (shards[i]) {
            c->shards[i] = shards[i]->Clone();
        }
    }

    return c;
}

real1_f QStabilizerHybrid::ProbAllRdm(bool roundRz, const bitCapInt& fullRegister)
{
    if (engine || !ancillaCount) {
        return ProbAll(fullRegister);
    }

    if (!roundRz) {
        return stabilizer->ProbPermRdm(fullRegister, qubitCount);
    }

    return RdmCloneHelper()->stabilizer->ProbPermRdm(fullRegister, qubitCount);
}

real1_f QStabilizerHybrid::ProbMaskRdm(bool roundRz, const bitCapInt& mask, const bitCapInt& permutation)
{
    if (bi_compare(maxQPower - ONE_BCI, mask) == 0) {
        return ProbAllRdm(roundRz, permutation);
    }

    if (engine || !ancillaCount) {
        return ProbMask(mask, permutation);
    }

    if (!roundRz) {
        return stabilizer->ProbMask(mask, permutation);
    }

    return RdmCloneHelper()->stabilizer->ProbMask(mask, permutation);
}

void QStabilizerHybrid::SwitchToEngine()
{
    if (engine) {
        return;
    }

    rdmClone = nullptr;

    if ((qubitCount + ancillaCount + deadAncillaCount) > maxEngineQubitCount) {
        QInterfacePtr e = MakeEngine(ZERO_BCI);
#if ENABLE_PTHREAD
        const unsigned numCores = GetConcurrencyLevel();
        std::vector<QStabilizerHybridPtr> clones;
        for (unsigned i = 0U; i < numCores; ++i) {
            clones.push_back(std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()));
        }
        bitCapInt i = ZERO_BCI;
        while (i < maxQPower) {
            const bitCapInt p = i;
            std::vector<std::future<complex>> futures;
            for (unsigned j = 0U; j < numCores; ++j) {
                futures.push_back(
                    std::async(std::launch::async, [j, p, &clones]() { return clones[j]->GetAmplitude(j + p); }));
                bi_increment(&i, 1U);
                if (bi_compare(i, maxQPower) >= 0) {
                    break;
                }
            }
            for (size_t j = 0U; j < futures.size(); ++j) {
                e->SetAmplitude(j + p, futures[j].get());
            }
        }
        clones.clear();
#else
        for (bitCapInt i = 0U; bi_compare(i, maxQPower) < 0; bi_increment(&i, 1U)) {
            e->SetAmplitude(i, GetAmplitude(i));
        }
#endif

        stabilizer = nullptr;
        engine = e;

        engine->UpdateRunningNorm();
        if (!doNormalize) {
            engine->NormalizeState();
        }
        // We have extra "gate fusion" shards leftover.
        shards.erase(shards.begin() + qubitCount, shards.end());
        // We're done with ancillae.
        ancillaCount = 0U;
        deadAncillaCount = 0U;

        return;
    }

    engine = MakeEngine(ZERO_BCI, stabilizer->GetQubitCount());
    stabilizer->GetQuantumState(engine);
    stabilizer = nullptr;
    FlushBuffers();

    if (!ancillaCount && !deadAncillaCount) {
        return;
    }

    // When we measure, we act postselection on reverse T-gadgets.
    if (ancillaCount) {
        engine->ForceMReg(qubitCount, ancillaCount, ZERO_BCI, true, true);
    }
    // Ancillae are separable after measurement.
    engine->Dispose(qubitCount, ancillaCount + deadAncillaCount);
    // We have extra "gate fusion" shards leftover.
    shards.erase(shards.begin() + qubitCount, shards.end());
    // We're done with ancillae.
    ancillaCount = 0U;
    deadAncillaCount = 0U;
}

bitLenInt QStabilizerHybrid::ComposeEither(QStabilizerHybridPtr toCopy, bool willDestroy)
{
    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    const bitLenInt nQubits = qubitCount + toCopy->qubitCount;

    if ((ancillaCount + toCopy->ancillaCount) > maxAncillaCount) {
        SwitchToEngine();
    }

    bitLenInt toRet;
    if (engine) {
        toCopy->SwitchToEngine();
        toRet = willDestroy ? engine->ComposeNoClone(toCopy->engine) : engine->Compose(toCopy->engine);
    } else if (toCopy->engine) {
        SwitchToEngine();
        toRet = willDestroy ? engine->ComposeNoClone(toCopy->engine) : engine->Compose(toCopy->engine);
    } else {
        toRet = stabilizer->Compose(toCopy->stabilizer, qubitCount);
        stabilizer->ROR(deadAncillaCount, qubitCount + ancillaCount,
            deadAncillaCount + toCopy->ancillaCount + toCopy->deadAncillaCount);
        ancillaCount += toCopy->ancillaCount;
        deadAncillaCount += toCopy->deadAncillaCount;
    }

    // Resize the shards buffer.
    shards.insert(shards.begin() + qubitCount, toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (size_t i = qubitCount; i < shards.size(); ++i) {
        if (shards[i]) {
            shards[i] = shards[i]->Clone();
        }
    }

    SetQubitCount(nQubits);

    return toRet;
}

bitLenInt QStabilizerHybrid::Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    if (toCopy->ancillaCount || toCopy->deadAncillaCount) {
        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, qubitCount);
        const bitLenInt result = Compose(toCopy);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }

    const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
    bitLenInt toRet;

    if (engine) {
        toCopy->SwitchToEngine();
        toRet = engine->Compose(toCopy->engine, start);
    } else if (toCopy->engine) {
        SwitchToEngine();
        toRet = engine->Compose(toCopy->engine, start);
    } else {
        toRet = stabilizer->Compose(toCopy->stabilizer, start);
        stabilizer->ROR(deadAncillaCount, qubitCount + ancillaCount,
            deadAncillaCount + toCopy->ancillaCount + toCopy->deadAncillaCount);
        ancillaCount += toCopy->ancillaCount;
        deadAncillaCount += toCopy->deadAncillaCount;
    }

    // Resize the shards buffer.
    shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (bitLenInt i = 0U; i < toCopy->qubitCount; ++i) {
        if (shards[start + i]) {
            shards[start + i] = shards[start + i]->Clone();
        }
    }

    SetQubitCount(nQubits);

    return toRet;
}

QInterfacePtr QStabilizerHybrid::Decompose(bitLenInt start, bitLenInt length)
{
    QStabilizerHybridPtr dest = std::make_shared<QStabilizerHybrid>(engineTypes, length, ZERO_BCI, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);
    Decompose(start, dest);
    return dest;
}

void QStabilizerHybrid::Decompose(bitLenInt start, QStabilizerHybridPtr dest)
{
    const bitLenInt length = dest->qubitCount;

    if (!length) {
        return;
    }

    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        dest->SwitchToEngine();
        engine->Decompose(start, dest->engine);

        return SetQubitCount(qubitCount - length);
    }

    if (dest->engine) {
        dest->engine.reset();
        dest->stabilizer = dest->MakeStabilizer(ZERO_BCI);
    }

    stabilizer->Decompose(start, dest->stabilizer);
    std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length)
{
    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        engine->Dispose(start, length);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
{
    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

bitLenInt QStabilizerHybrid::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QStabilizerHybridPtr nQubits = std::make_shared<QStabilizerHybrid>(cloneEngineTypes, length, ZERO_BCI,
        rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, false,
        (real1_f)amplitudeFloor, std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);

    return Compose(nQubits, start);
}

void QStabilizerHybrid::GetQuantumState(complex* outputState)
{
    if (engine) {
        return engine->GetQuantumState(outputState);
    }

    if (!IsBuffered()) {
        return stabilizer->GetQuantumState(outputState);
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->SwitchToEngine();
    clone->GetQuantumState(outputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    if (engine) {
        return engine->GetProbs(outputProbs);
    }

    if (!IsProbBuffered()) {
        return stabilizer->GetProbs(outputProbs);
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->SwitchToEngine();
    clone->GetProbs(outputProbs);
}

complex QStabilizerHybrid::GetAmplitudeOrProb(const bitCapInt& perm, bool isProb)
{
    if (engine) {
        return engine->GetAmplitude(perm);
    }

    UpdateRoundingThreshold();
    const bool isRounded = roundingThreshold > FP_NORM_EPSILON;
    const QUnitCliffordPtr origStabilizer =
        isRounded ? std::dynamic_pointer_cast<QUnitClifford>(stabilizer->Clone()) : nullptr;
    const bitLenInt origAncillaCount = ancillaCount;
    const bitLenInt origDeadAncillaCount = deadAncillaCount;
    std::vector<MpsShardPtr> origShards = isRounded ? shards : std::vector<MpsShardPtr>();
    if (isRounded) {
        for (MpsShardPtr& origShard : origShards) {
            if (origShard) {
                origShard = origShard->Clone();
            }
        }
        RdmCloneFlush(roundingThreshold);
    }

    if ((isProb && !ancillaCount && !IsLogicalProbBuffered()) || !IsBuffered()) {
        const complex toRet = stabilizer->GetAmplitude(perm);

        if (isRounded) {
            stabilizer = origStabilizer;
            ancillaCount = origAncillaCount;
            deadAncillaCount = origDeadAncillaCount;
            shards = origShards;
        }

        return toRet;
    }

    std::vector<bitLenInt> indices;
    std::vector<bitCapInt> perms{ perm };
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (!shards[i]) {
            continue;
        }
        indices.push_back(i);
        perms.push_back(perm ^ pow2(i));
    }

    if (!ancillaCount) {
        std::vector<complex> amps;
        amps.reserve(perms.size());
        if (stateMapCache) {
            for (size_t i = 0U; i < perms.size(); ++i) {
                amps.push_back(stateMapCache->get(perms[i]));
            }
        } else {
            amps = stabilizer->GetAmplitudes(perms);
        }
        complex amp = amps[0U];
        for (size_t i = 1U; i < amps.size(); ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            if (bi_and_1(perm >> j)) {
                amp = mtrx[2U] * amps[i] + mtrx[3U] * amp;
            } else {
                amp = mtrx[0U] * amp + mtrx[1U] * amps[i];
            }
        }

        if (isRounded) {
            stabilizer = origStabilizer;
            ancillaCount = origAncillaCount;
            deadAncillaCount = origDeadAncillaCount;
            shards = origShards;
        }

        return amp;
    }

    const bitLenInt aStride = indices.size() + 1U;
    const bitCapIntOcl ancillaPow = pow2Ocl(ancillaCount);
    for (bitCapIntOcl i = 1U; i < ancillaPow; ++i) {
        const bitCapInt ancillaPerm = i << qubitCount;
        for (size_t j = 0U; j < aStride; ++j) {
            perms.push_back(perms[j] | ancillaPerm);
        }
    }

    std::vector<complex> amps;
    amps.reserve(perms.size());
    if (stateMapCache) {
        for (size_t i = 0U; i < perms.size(); ++i) {
            amps.push_back(stateMapCache->get(perms[i]));
        }
    } else {
        amps = stabilizer->GetAmplitudes(perms);
    }

    std::vector<QInterfaceEngine> et = engineTypes;
    for (size_t i = 0U; i < et.size(); ++i) {
        const size_t j = et.size() - (i + 1U);
        if ((et[j] == QINTERFACE_BDT_HYBRID) || (et[j] == QINTERFACE_BDT)) {
            et.erase(et.begin() + j);
        }
    }
    if (et.empty()) {
        et.push_back(QINTERFACE_OPTIMAL_BASE);
    }
    QEnginePtr aEngine = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(et, ancillaCount, ZERO_BCI, rand_generator, ONE_CMPLX, false, false, useHostRam, devID,
            useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits, separabilityThreshold));

#if ENABLE_COMPLEX_X2
    std::vector<complex2> top;
    std::vector<complex2> bottom;
    // For variable scoping, only:
    if (true) {
        complex amp = amps[0U];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            top.emplace_back(mtrx[0U], mtrx[1U]);
            bottom.emplace_back(mtrx[3U], mtrx[2U]);
            complex2 amp2(amp, amps[i]);
            if (bi_and_1(perm >> j)) {
                amp2 = amp2 * bottom.back();
            } else {
                amp2 = amp2 * top.back();
            }
            amp = amp2.c(0U) + amp2.c(1U);
        }
        aEngine->SetAmplitude(0U, amp);
    }
    for (bitCapIntOcl a = 1U; a < ancillaPow; ++a) {
        const bitCapIntOcl offset = a * aStride;
        complex amp = amps[offset];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            complex2 amp2(amp, amps[i]);
            if (bi_and_1(perm >> j)) {
                amp2 = amp2 * bottom[j];
            } else {
                amp2 = amp2 * top[j];
            }
            amp = amp2.c(0U) + amp2.c(1U);
        }
        aEngine->SetAmplitude(a, amp);
    }
    top.clear();
    bottom.clear();
#else
    for (bitCapIntOcl a = 0U; a < ancillaPow; ++a) {
        const bitCapIntOcl offset = a * aStride;
        complex amp = amps[offset];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            const complex oAmp = amps[i + offset];
            if (bi_and_1(perm >> j)) {
                amp = mtrx[3U] * amp + mtrx[2U] * oAmp;
            } else {
                amp = mtrx[0U] * amp + mtrx[1U] * oAmp;
            }
        }
        aEngine->SetAmplitude(a, amp);
    }
#endif
    amps.clear();

    for (bitLenInt i = 0U; i < ancillaCount; ++i) {
        const MpsShardPtr& shard = shards[i + qubitCount];
        if (shard) {
            aEngine->Mtrx(shard->gate, i);
        }
    }

    if (isRounded) {
        stabilizer = origStabilizer;
        ancillaCount = origAncillaCount;
        deadAncillaCount = origDeadAncillaCount;
        shards = origShards;
    }

    return (real1)pow(SQRT2_R1, (real1)ancillaCount) * aEngine->GetAmplitude(ZERO_BCI);
}

void QStabilizerHybrid::SetQuantumState(const complex* inputState)
{
    DumpBuffers();

    if (qubitCount > 1U) {
        ancillaCount = 0U;
        deadAncillaCount = 0U;
        shards.resize(qubitCount);
        if (stabilizer) {
            engine = MakeEngine();
            stabilizer = nullptr;
        }

        return engine->SetQuantumState(inputState);
    }

    // Otherwise, we're preparing 1 qubit.
    engine = nullptr;

    if (stabilizer && !ancillaCount && !deadAncillaCount) {
        stabilizer->SetPermutation(ZERO_BCI);
    } else {
        ancillaCount = 0U;
        deadAncillaCount = 0U;
        stabilizer = MakeStabilizer(ZERO_BCI);
        shards.clear();
        shards.resize(qubitCount);
    }

    const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
    const real1 sqrtProb = sqrt(prob);
    const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
    const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
    const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
    const complex mtrx[4U]{ sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
    Mtrx(mtrx, 0);
}

void QStabilizerHybrid::SetPermutation(const bitCapInt& perm, const complex& phaseFac)
{
    DumpBuffers();

    engine = nullptr;

    if (stabilizer && !ancillaCount && !deadAncillaCount) {
        stabilizer->SetPermutation(perm);
    } else {
        ancillaCount = 0U;
        deadAncillaCount = 0U;
        stabilizer = MakeStabilizer(perm);
        shards.clear();
        shards.resize(qubitCount);
    }

    isRoundingFlushed = false;
}

void QStabilizerHybrid::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    std::swap(shards[qubit1], shards[qubit2]);

    if (stabilizer) {
        rdmClone = nullptr;
        stabilizer->Swap(qubit1, qubit2);
    } else {
        engine->Swap(qubit1, qubit2);
    }
}
void QStabilizerHybrid::CSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (stabilizer) {
        std::vector<bitLenInt> controls;

        if (TrimControls(lControls, controls, false)) {
            return;
        }

        if (controls.empty()) {
            rdmClone = nullptr;

            return stabilizer->Swap(qubit1, qubit2);
        }

        SwitchToEngine();
    }

    engine->CSwap(lControls, qubit1, qubit2);
}
void QStabilizerHybrid::CSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (stabilizer) {
        std::vector<bitLenInt> controls;

        if (TrimControls(lControls, controls, false)) {
            return;
        }

        if (controls.empty()) {
            return QInterface::SqrtSwap(qubit1, qubit2);
        }

        SwitchToEngine();
    }

    engine->CSqrtSwap(lControls, qubit1, qubit2);
}
void QStabilizerHybrid::AntiCSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (stabilizer) {
        std::vector<bitLenInt> controls;

        if (TrimControls(lControls, controls, true)) {
            return;
        }

        if (controls.empty()) {
            return QInterface::SqrtSwap(qubit1, qubit2);
        }

        SwitchToEngine();
    }

    engine->AntiCSqrtSwap(lControls, qubit1, qubit2);
}
void QStabilizerHybrid::CISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (stabilizer) {
        std::vector<bitLenInt> controls;

        if (TrimControls(lControls, controls, false)) {
            return;
        }

        if (controls.empty()) {
            return QInterface::ISqrtSwap(qubit1, qubit2);
        }

        SwitchToEngine();
    }

    engine->CISqrtSwap(lControls, qubit1, qubit2);
}
void QStabilizerHybrid::AntiCISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2)
{
    if (stabilizer) {
        std::vector<bitLenInt> controls;

        if (TrimControls(lControls, controls, true)) {
            return;
        }

        if (controls.empty()) {
            return QInterface::ISqrtSwap(qubit1, qubit2);
        }

        SwitchToEngine();
    }

    engine->AntiCISqrtSwap(lControls, qubit1, qubit2);
}

void QStabilizerHybrid::XMask(const bitCapInt& _mask)
{
    if (engine) {
        return engine->XMask(_mask);
    }

    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        X(log2(mask ^ v));
        mask = v;
    }
}
void QStabilizerHybrid::YMask(const bitCapInt& _mask)
{
    if (engine) {
        return engine->YMask(_mask);
    }

    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        Y(log2(mask ^ v));
        mask = v;
    }
}
void QStabilizerHybrid::ZMask(const bitCapInt& _mask)
{
    if (engine) {
        return engine->ZMask(_mask);
    }

    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        Z(log2(mask ^ v));
        mask = v;
    }
}

void QStabilizerHybrid::Mtrx(const complex* lMtrx, bitLenInt target)
{
    MpsShardPtr shard = shards[target];
    shards[target] = nullptr;
    const bool wasCached = (bool)shard;
    complex mtrx[4U];
    if (!wasCached) {
        std::copy(lMtrx, lMtrx + 4U, mtrx);
    } else if (!engine && useTGadget && (target < qubitCount) && (ancillaCount < maxAncillaCount) && !IS_PHASE(lMtrx) &&
        !IS_INVERT(lMtrx) && (shard->IsPhase() || shard->IsInvert() || shard->IsHPhase() || shard->IsHInvert())) {

        if (shard->IsHPhase() || shard->IsHInvert()) {
            complex hGate[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
            MpsShardPtr hShard = std::make_shared<MpsShard>(hGate);
            hShard->Compose(shard->gate);
            shard = hShard->IsIdentity() ? nullptr : hShard;
            rdmClone = nullptr;
            stabilizer->H(target);
        }

        if (shard && shard->IsInvert()) {
            complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
            MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
            pauliShard->Compose(shard->gate);
            shard = pauliShard->IsIdentity() ? nullptr : pauliShard;
            rdmClone = nullptr;
            stabilizer->X(target);
        }

        if (shard) {
            const real1 angle =
                (real1)(FractionalRzAngleWithFlush(target, std::arg(shard->gate[3U] / shard->gate[0U])) / 2);
            if ((2 * abs(angle)) > (FP_NORM_EPSILON * PI_R1)) {
                // We're adding an ancilla, so drop any rdmClone.
                rdmClone = nullptr;

                const real1 angleCos = cos(angle);
                const real1 angleSin = sin(angle);
                shard->gate[0U] = complex(angleCos, -angleSin);
                shard->gate[3U] = complex(angleCos, angleSin);

                // Form potentially entangled representation, with this.
                bitLenInt ancillaIndex = deadAncillaCount ? (qubitCount + ancillaCount) : stabilizer->Allocate(1U);
                ++ancillaCount;
                shards.emplace_back(nullptr);
                if (deadAncillaCount) {
                    --deadAncillaCount;
                }

                // Use reverse t-injection gadget.
                stabilizer->CNOT(target, ancillaIndex);
                Mtrx(shard->gate, ancillaIndex);
                H(ancillaIndex);
            }
        }

        std::copy(lMtrx, lMtrx + 4U, mtrx);
    } else {
        shard->Compose(lMtrx);
        std::copy(shard->gate, shard->gate + 4U, mtrx);
    }

    if (engine) {
        return engine->Mtrx(mtrx, target);
    }

    if (IS_CLIFFORD(mtrx) || ((IS_PHASE(mtrx) || IS_INVERT(mtrx)) && stabilizer->IsSeparableZ(target))) {
        rdmClone = nullptr;

        return stabilizer->Mtrx(mtrx, target);
    }

    shards[target] = std::make_shared<MpsShard>(mtrx);
    if (!wasCached) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCMtrx(const std::vector<bitLenInt>& lControls, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        return MCPhase(lControls, mtrx[0U], mtrx[3U], target);
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        return MCInvert(lControls, mtrx[1U], mtrx[2U], target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls)) {
        return;
    }

    if (controls.empty()) {
        return Mtrx(mtrx, target);
    }

    SwitchToEngine();
    engine->MCMtrx(lControls, mtrx, target);
}

void QStabilizerHybrid::MCPhase(
    const std::vector<bitLenInt>& lControls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    if (engine) {
        return engine->MCPhase(lControls, topLeft, bottomRight, target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls)) {
        return;
    }

    if (controls.empty()) {
        return Phase(topLeft, bottomRight, target);
    }

    if (IS_NORM_0(topLeft - ONE_CMPLX) || IS_NORM_0(bottomRight - ONE_CMPLX)) {
        real1_f prob = ProbRdm(target);

        if (IS_NORM_0(topLeft - ONE_CMPLX) && (prob <= FP_NORM_EPSILON)) {
            return;
        }

        if (IS_NORM_0(bottomRight - ONE_CMPLX) && ((ONE_R1 - prob) <= FP_NORM_EPSILON)) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target, true);
    }

    if (engine) {
        return engine->MCPhase(lControls, topLeft, bottomRight, target);
    }

    const bitLenInt control = controls[0U];
    rdmClone = nullptr;
    stabilizer->MCPhase(controls, topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCInvert(
    const std::vector<bitLenInt>& lControls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (engine) {
        return engine->MCInvert(lControls, topRight, bottomLeft, target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls)) {
        return;
    }

    if (controls.empty()) {
        return Invert(topRight, bottomLeft, target);
    }

    if ((controls.size() > 1U) && IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
        H(target);
        const real1_f prob = ProbRdm(target);
        H(target);

        if (prob <= FP_NORM_EPSILON) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        return engine->MCInvert(lControls, topRight, bottomLeft, target);
    }

    const bitLenInt control = controls[0U];
    rdmClone = nullptr;
    stabilizer->MCInvert(controls, topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACMtrx(const std::vector<bitLenInt>& lControls, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        return MACPhase(lControls, mtrx[0U], mtrx[3U], target);
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        return MACInvert(lControls, mtrx[1U], mtrx[2U], target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (controls.empty()) {
        return Mtrx(mtrx, target);
    }

    SwitchToEngine();
    engine->MACMtrx(lControls, mtrx, target);
}

void QStabilizerHybrid::MACPhase(
    const std::vector<bitLenInt>& lControls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (engine) {
        return engine->MACPhase(lControls, topLeft, bottomRight, target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (controls.empty()) {
        return Phase(topLeft, bottomRight, target);
    }

    if (IS_NORM_0(topLeft - ONE_CMPLX) || IS_NORM_0(bottomRight - ONE_CMPLX)) {
        real1_f prob = ProbRdm(target);

        if (IS_NORM_0(topLeft - ONE_CMPLX) && (prob <= FP_NORM_EPSILON)) {
            return;
        }

        if (IS_NORM_0(bottomRight - ONE_CMPLX) && ((ONE_R1 - prob) <= FP_NORM_EPSILON)) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target, true);
    }

    if (engine) {
        return engine->MACPhase(lControls, topLeft, bottomRight, target);
    }

    const bitLenInt control = controls[0U];
    rdmClone = nullptr;
    stabilizer->MACPhase(controls, topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACInvert(
    const std::vector<bitLenInt>& lControls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (engine) {
        return engine->MACInvert(lControls, topRight, bottomLeft, target);
    }

    std::vector<bitLenInt> controls;

    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (controls.empty()) {
        return Invert(topRight, bottomLeft, target);
    }

    if ((controls.size() > 1U) && IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
        H(target);
        const real1_f prob = ProbRdm(target);
        H(target);

        if (prob <= FP_NORM_EPSILON) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        return engine->MACInvert(lControls, topRight, bottomLeft, target);
    }

    const bitLenInt control = controls[0U];
    rdmClone = nullptr;
    stabilizer->MACInvert(controls, topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

real1_f QStabilizerHybrid::Prob(bitLenInt qubit)
{
    if (ancillaCount && !(stabilizer->IsSeparable(qubit))) {
        if (qubitCount <= maxEngineQubitCount) {
            QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
            clone->SwitchToEngine();

            return clone->Prob(qubit);
        }

        if (stabilizer->PermCount() < pow2(maxStateMapCacheQubitCount)) {
            stateMapCache = stabilizer->GetDecomposedQuantumState();
        }

        const bitCapInt qPower = pow2(qubit);
        const bitCapInt maxLcv = maxQPower >> 1U;
        real1_f partProb = ZERO_R1_F;
#if ENABLE_PTHREAD
        const unsigned numCores =
            (bi_compare(maxLcv, GetConcurrencyLevel()) < 0) ? (unsigned)maxLcv : GetConcurrencyLevel();
        std::vector<QStabilizerHybridPtr> clones;
        for (unsigned i = 0U; i < numCores; ++i) {
            clones.push_back(std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()));
        }
        bitCapInt i = ZERO_BCI;
        while (bi_compare(i, maxLcv) < 0) {
            const bitCapInt p = i;
            std::vector<std::future<real1>> futures;
            for (unsigned j = 0U; j < numCores; ++j) {
                futures.push_back(std::async(std::launch::async, [j, p, qPower, &clones]() {
                    bitCapInt k = (j + p) & (qPower - 1U);
                    bi_or_ip(&k, ((j + p) ^ k) << 1U);
                    return norm(clones[j]->GetAmplitude(k | qPower));
                }));
                bi_increment(&i, 1U);
                if (bi_compare(i, maxLcv) >= 0) {
                    break;
                }
            }
            for (std::future<real1>& future : futures) {
                partProb += future.get();
            }
        }
        stateMapCache = nullptr;
#else
        for (bitCapInt i = 0U; bi_compare(i, maxLcv) < 0; bi_increment(&i, 1U)) {
            bitCapInt j = i & (qPower - 1U);
            bi_or_ip(&j, (i ^ j) << 1U);
            partProb += norm(GetAmplitude(j | qPower));
        }
        stateMapCache = nullptr;
#endif

        return partProb;
    }

    if (engine) {
        return engine->Prob(qubit);
    }

    const MpsShardPtr& shard = shards[qubit];

    if (shard && shard->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shard && !shard->IsPhase()) {
        // Bit was already rotated to Z basis, if separable.
        if (stabilizer->IsSeparableZ(qubit)) {
            if (stabilizer->M(qubit)) {
                return (real1_f)norm(shard->gate[3U]);
            }

            return (real1_f)norm(shard->gate[2U]);
        }

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        return HALF_R1_F;
    }

    if (stabilizer->IsSeparableZ(qubit)) {
        return stabilizer->M(qubit) ? ONE_R1_F : ZERO_R1_F;
    }

    // Otherwise, state appears locally maximally mixed.
    return HALF_R1_F;
}

bool QStabilizerHybrid::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (engine) {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    MpsShardPtr& shard = shards[qubit];

    if (shard && shard->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shard && !shard->IsPhase()) {
        if (stabilizer->IsSeparableZ(qubit)) {
            if (doForce) {
                if (doApply) {
                    if (result != stabilizer->ForceM(qubit, result, true, true)) {
                        // Sorry to throw, but the requested forced result is definitely invalid.
                        throw std::invalid_argument(
                            "QStabilizerHybrid::ForceM() forced a measurement result with 0 probability!");
                    }
                    shard = nullptr;
                }

                return result;
            }

            // Bit was already rotated to Z basis, if separable.
            return CollapseSeparableShard(qubit);
        }

        // Otherwise, we have non-Clifford measurement.
        SwitchToEngine();

        return engine->ForceM(qubit, result, doForce, doApply);
    }
    shard = nullptr;

    if (stabilizer->IsSeparable(qubit)) {
        return stabilizer->ForceM(qubit, result, doForce, doApply);
    }

    FlushCliffordFromBuffers();

    if (ancillaCount) {
        SwitchToEngine();

        return engine->ForceM(qubit, result, doForce, doApply);
    }

    return stabilizer->ForceM(qubit, result, doForce, doApply);
}

#define ADD_SHOT_PROB(m)                                                                                               \
    if (prob > FP_NORM_EPSILON) {                                                                                      \
        d = (m);                                                                                                       \
    }                                                                                                                  \
    partProb += prob;

#define CHECK_NARROW_SHOT()                                                                                            \
    const real1_f prob = norm(GetAmplitude(m));                                                                        \
    ADD_SHOT_PROB(m)                                                                                                   \
    if (resProb < partProb) {                                                                                          \
        foundM = true;                                                                                                 \
        break;                                                                                                         \
    }

#define CHECK_WIDE_SHOT(j, k)                                                                                          \
    const real1 prob = futures[j].get();                                                                               \
    if (foundM) {                                                                                                      \
        continue;                                                                                                      \
    }                                                                                                                  \
    ADD_SHOT_PROB(k)                                                                                                   \
    if (resProb < partProb) {                                                                                          \
        m = (k);                                                                                                       \
        foundM = true;                                                                                                 \
    }

#define FIX_OVERPROB_SHOT_AND_FINISH()                                                                                 \
    if (!foundM) {                                                                                                     \
        m = d;                                                                                                         \
    }                                                                                                                  \
    SetPermutation(m);                                                                                                 \
    stateMapCache = nullptr;
bitCapInt QStabilizerHybrid::MAll()
{
    if (engine) {
        const bitCapInt toRet = engine->MAll();
        SetPermutation(toRet);

        return toRet;
    }

    UpdateRoundingThreshold();

    if (roundingThreshold > FP_NORM_EPSILON) {
        RdmCloneFlush(roundingThreshold);
    }

    if (!IsProbBuffered()) {
        const bitCapInt toRet = stabilizer->MAll();
        SetPermutation(toRet);

        return toRet;
    }

    if (stabilizer->PermCount() < pow2(maxStateMapCacheQubitCount)) {
        stateMapCache = stabilizer->GetDecomposedQuantumState();
    }

#if ENABLE_PTHREAD
    const real1_f resProb = Rand();
    real1_f partProb = ZERO_R1;
    bitCapInt d = ZERO_BCI;
    bitCapInt m;
    bool foundM = false;

    const unsigned numCores =
        (bi_compare(maxQPower, GetConcurrencyLevel()) < 0) ? (unsigned)maxQPower : GetConcurrencyLevel();

    std::vector<QStabilizerHybridPtr> clones;
    for (unsigned i = 0U; i < numCores; ++i) {
        clones.push_back(std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()));
    }
    bitCapInt i = ZERO_BCI;
    while (i < maxQPower) {
        const bitCapInt p = i;
        std::vector<std::future<real1>> futures;
        for (unsigned j = 0U; j < numCores; ++j) {
            futures.push_back(
                std::async(std::launch::async, [j, p, &clones]() { return norm(clones[j]->GetAmplitude(j + p)); }));
            bi_increment(&i, 1U);
            if (bi_compare(i, maxQPower) >= 0) {
                break;
            }
        }
        for (size_t j = 0U; j < futures.size(); ++j) {
            CHECK_WIDE_SHOT(j, j + p)
        }
        if (foundM) {
            break;
        }
    }
#else
    const real1 resProb = (real1)Rand();
    real1 partProb = ZERO_R1;
    bitCapInt d = 0U;
    bitCapInt m;
    bool foundM = false;
    for (m = 0U; bi_compare(m, maxQPower) < 0; bi_increment(&m, 1U)) {
        CHECK_NARROW_SHOT()
    }
#endif

    FIX_OVERPROB_SHOT_AND_FINISH()
    isRoundingFlushed = false;

    return m;
}

void QStabilizerHybrid::UniformlyControlledSingleBit(
    const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, const complex* mtrxs)
{
    if (stabilizer) {
        return QInterface::UniformlyControlledSingleBit(controls, qubitIndex, mtrxs);
    }

    engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs);
}

#define FILL_REMAINING_MAP_SHOTS()                                                                                     \
    if (rng.size()) {                                                                                                  \
        results[d] += shots - rng.size();                                                                              \
    }                                                                                                                  \
    stateMapCache = nullptr;

#define ADD_SHOTS_PROB(m)                                                                                              \
    if (rng.empty()) {                                                                                                 \
        continue;                                                                                                      \
    }                                                                                                                  \
    ADD_SHOT_PROB(m)

#define CHECK_SHOTS(m, lm)                                                                                             \
    ADD_SHOT_PROB(m)                                                                                                   \
    CheckShots(shots, m, partProb, qPowers, rng, lm);

#define CHECK_SHOTS_IF_ANY(m, lm)                                                                                      \
    ADD_SHOTS_PROB(m)                                                                                                  \
    CheckShots(shots, m, partProb, qPowers, rng, lm);

std::map<bitCapInt, int> QStabilizerHybrid::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    if (engine) {
        return engine->MultiShotMeasureMask(qPowers, shots);
    }

    FlushCliffordFromBuffers();

    UpdateRoundingThreshold();

    if (!isRoundingFlushed && (roundingThreshold > FP_NORM_EPSILON)) {
        QStabilizerHybridPtr roundedClone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        roundedClone->RdmCloneFlush(roundingThreshold);

        return roundedClone->MultiShotMeasureMask(qPowers, shots);
    }

    std::map<bitCapInt, int> results;

    if (!IsProbBuffered()) {
        std::mutex resultsMutex;
        par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
            const bitCapInt sample = SampleClone(qPowers);
            std::lock_guard<std::mutex> lock(resultsMutex);
            ++(results[sample]);
        });

        return results;
    }

    std::vector<real1_f> rng = GenerateShotProbs(shots);
    const auto shotFunc = [&](bitCapInt sample, unsigned unused) { ++(results[sample]); };
    real1 partProb = ZERO_R1;
    bitCapInt d = ZERO_BCI;

    if (stabilizer->PermCount() < pow2(maxStateMapCacheQubitCount)) {
        stateMapCache = stabilizer->GetDecomposedQuantumState();
    }

#if ENABLE_PTHREAD
    const unsigned numCores =
        (bi_compare(maxQPower, GetConcurrencyLevel()) < 0) ? (unsigned)maxQPower : GetConcurrencyLevel();

    std::vector<QStabilizerHybridPtr> clones;
    for (unsigned i = 0U; i < numCores; ++i) {
        clones.push_back(std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()));
    }
    bitCapInt i = ZERO_BCI;
    while (i < maxQPower) {
        const bitCapInt p = i;
        std::vector<std::future<real1>> futures;
        for (unsigned j = 0U; j < numCores; ++j) {
            futures.push_back(
                std::async(std::launch::async, [j, p, &clones]() { return norm(clones[j]->GetAmplitude(j + p)); }));
            bi_increment(&i, 1U);
            if (bi_compare(i, maxQPower) >= 0) {
                break;
            }
        }
        for (size_t j = 0U; j < futures.size(); ++j) {
            const real1 prob = futures[j].get();
            CHECK_SHOTS_IF_ANY(j + p, shotFunc);
        }
        if (rng.empty()) {
            break;
        }
    }
#else
    for (bitCapInt m = 0U; bi_compare(m, maxQPower) < 0; bi_increment(&m, 1U)) {
        const real1 prob = norm(GetAmplitude(m));
        CHECK_SHOTS(m, shotFunc);
    }
#endif

    FILL_REMAINING_MAP_SHOTS()

    return results;
}

#ifdef __SIZEOF_INT128__
#define FILL_REMAINING_ARRAY_SHOTS()                                                                                   \
    if (rng.size()) {                                                                                                  \
        const unsigned long long _d = (unsigned long long)(unsigned __int128)d;                                        \
        for (size_t shot = 0U; shot < rng.size(); ++shot) {                                                            \
            shotsArray[shot + (shots - rng.size())] = _d;                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    std::shuffle(shotsArray, shotsArray + shots, prng);                                                                \
    stateMapCache = nullptr;
#else
#define FILL_REMAINING_ARRAY_SHOTS()                                                                                   \
    if (rng.size()) {                                                                                                  \
        const unsigned long long _d = (unsigned long long)(uint64_t)d;                                                 \
        for (size_t shot = 0U; shot < rng.size(); ++shot) {                                                            \
            shotsArray[shot + (shots - rng.size())] = _d;                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    std::shuffle(shotsArray, shotsArray + shots, prng);                                                                \
    stateMapCache = nullptr;
#endif

void QStabilizerHybrid::MultiShotMeasureMask(
    const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    if (engine) {
        return engine->MultiShotMeasureMask(qPowers, shots, shotsArray);
    }

    FlushCliffordFromBuffers();

    UpdateRoundingThreshold();

    if (!isRoundingFlushed && (roundingThreshold > FP_NORM_EPSILON)) {
        QStabilizerHybridPtr roundedClone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        roundedClone->RdmCloneFlush(roundingThreshold);

        return roundedClone->MultiShotMeasureMask(qPowers, shots, shotsArray);
    }

    if (!IsProbBuffered()) {
        return par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
            shotsArray[shot] = (bitCapIntOcl)SampleClone(qPowers);
        });
    }

    std::vector<real1_f> rng = GenerateShotProbs(shots);
    const auto shotFunc = [&](bitCapInt sample, unsigned shot) { shotsArray[shot] = (bitCapIntOcl)sample; };
    real1 partProb = ZERO_R1;
    bitCapInt d = ZERO_BCI;

    if (stabilizer->PermCount() < pow2(maxStateMapCacheQubitCount)) {
        stateMapCache = stabilizer->GetDecomposedQuantumState();
    }

#if ENABLE_PTHREAD
    const unsigned numCores =
        (bi_compare(maxQPower, GetConcurrencyLevel()) < 0) ? (unsigned)maxQPower : GetConcurrencyLevel();

    std::vector<QStabilizerHybridPtr> clones;
    for (unsigned i = 0U; i < numCores; ++i) {
        clones.push_back(std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()));
    }
    bitCapInt i = ZERO_BCI;
    while (bi_compare(i, maxQPower) < 0) {
        const bitCapInt p = i;
        std::vector<std::future<real1>> futures;
        for (unsigned j = 0U; j < numCores; ++j) {
            futures.push_back(
                std::async(std::launch::async, [j, p, &clones]() { return norm(clones[j]->GetAmplitude(j + p)); }));
            bi_increment(&i, 1U);
            if (bi_compare(i, maxQPower) >= 0) {
                break;
            }
        }
        for (size_t j = 0U; j < futures.size(); ++j) {
            const real1 prob = futures[j].get();
            CHECK_SHOTS_IF_ANY(j + p, shotFunc);
        }
        if (rng.empty()) {
            break;
        }
    }
#else
    for (bitCapInt m = 0U; bi_compare(m, maxQPower) < 0; bi_increment(&m, 1U)) {
        const real1 prob = norm(GetAmplitude(m));
        CHECK_SHOTS(m, shotFunc);
    }
#endif

    FILL_REMAINING_ARRAY_SHOTS()
}

real1_f QStabilizerHybrid::ProbParity(const bitCapInt& mask)
{
    if (bi_compare_0(mask) == 0) {
        return ZERO_R1_F;
    }

    if (isPowerOfTwo(mask)) {
        return Prob(log2(mask));
    }

    SwitchToEngine();

    return QINTERFACE_TO_QPARITY(engine)->ProbParity(mask);
}
bool QStabilizerHybrid::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    // If no bits in mask:
    if (bi_compare_0(mask) == 0) {
        return false;
    }

    // If only one bit in mask:
    if (isPowerOfTwo(mask)) {
        return ForceM(log2(mask), result, doForce);
    }

    SwitchToEngine();

    return QINTERFACE_TO_QPARITY(engine)->ForceMParity(mask, result, doForce);
}

/// Flush non-Clifford phase gate gadgets with angle below a threshold.
void QStabilizerHybrid::RdmCloneFlush(real1_f threshold)
{
    QRACK_CONST complex h[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    for (size_t i = shards.size() - 1U; i >= qubitCount; --i) {
        // We're going to start by non-destructively "simulating" measurement collapse.
        MpsShardPtr nShard = shards[i]->Clone();

        for (int p = 0; p < 2; ++p) {
            // Say that we hypothetically collapse ancilla index "i" into state |p>...
            QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
            clone->stabilizer->H(i);
            clone->stabilizer->ForceM(i, p);

            // Do any other ancillae collapse?
            nShard->Compose(h);
            bool isCorrected = p;
            for (size_t j = clone->shards.size() - 1U; j >= clone->qubitCount; --j) {
                if (i == j) {
                    continue;
                }
                const real1_f prob = clone->stabilizer->Prob(j);
                const MpsShardPtr& oShard = clone->shards[j];
                oShard->Compose(h);
                if (prob < QUARTER_R1_F) {
                    // Collapsed to 0 - combine buffers
                    nShard->Compose(oShard->gate);
                } else if (prob > (3 * QUARTER_R1_F)) {
                    // Collapsed to 1 - combine buffers, with Z correction
                    isCorrected = !isCorrected;
                    nShard->Compose(oShard->gate);
                }
            }

            // Calculate the near-Clifford gate phase angle, but don't change the state:
            const real1 angle =
                (real1)FractionalRzAngleWithFlush(i, std::arg(nShard->gate[3U] / nShard->gate[0U]), true);
            if ((2 * abs(angle)) > (threshold * PI_R1)) {
                // The gate phase angle is too significant to flush.
                continue;
            }

            const complex phaseFac = nShard->gate[3U] / nShard->gate[0U];
            logFidelity += (double)log(0.25 * norm(ONE_CMPLX + phaseFac));

            // We're round the gates to 0, and we eliminate the ancillae.
            FractionalRzAngleWithFlush(i, std::arg(phaseFac));
            if (isCorrected) {
                stabilizer->Z(i);
            }
            stabilizer->H(i);
            stabilizer->ForceM(i, p);
            ClearAncilla(i);

            // All observable effects of qubits becoming separable have been accounted.
            // (Hypothetical newly-arising X-basis separable states have no effect.)
            for (size_t j = shards.size() - 1U; j >= qubitCount; --j) {
                if (stabilizer->IsSeparable(j)) {
                    ClearAncilla(j);
                }
            }

            // Start the loop condition over entirely, less the ancillae we removed.
            i = shards.size();

            break;
        }
    }

    isRoundingFlushed = true;
}

void QStabilizerHybrid::PruneAncillae(bool gaussian)
{
    if (engine) {
        return;
    }

    FlushCliffordFromBuffers();

    const bitLenInt maxI = stabilizer->GetQubitCount();
    std::set<bitLenInt> sepAncilla;
    for (bitLenInt i = qubitCount; i < maxI; ++i) {
        const std::vector<bitLenInt> eqb = stabilizer->EntangledQubits(i, gaussian);
        bool isSep = true;
        for (size_t j = 0U; j < eqb.size(); ++j) {
            if (eqb[j] < qubitCount) {
                isSep = false;
                break;
            }
        }
        if (!isSep) {
            continue;
        }
        std::copy(eqb.begin(), eqb.end(), std::inserter(sepAncilla, sepAncilla.end()));
    }

    if (sepAncilla.empty()) {
        return;
    }

    const bitLenInt liveCount = qubitCount + ancillaCount;
    bitLenInt i = liveCount;
    bitLenInt deadCount = 0U;
    for (const bitLenInt& qb : sepAncilla) {
        if (qb >= liveCount) {
            // Dead ancilla
            stabilizer->Swap(liveCount + deadCount, qb);
            ++deadCount;
        } else {
            // Live ancilla
            --i;
            stabilizer->Swap(i, qb);
            std::swap(shards[i], shards[qb]);
        }
    }

    const bitLenInt aCount = (sepAncilla.size() - deadCount);
    stabilizer->Dispose(i, sepAncilla.size());
    shards.resize(shards.size() - aCount);
    ancillaCount -= aCount;
    deadAncillaCount -= deadCount;
}

real1_f QStabilizerHybrid::ApproxCompareHelper(QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol)
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

    QStabilizerHybridPtr thisClone{ stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : nullptr };
    QStabilizerHybridPtr thatClone{
        toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare->Clone()) : nullptr
    };

    if (thisClone) {
        thisClone->FlushBuffers();
    }

    if (thatClone) {
        thatClone->FlushBuffers();
    }

    if (thisClone && thisClone->stabilizer && thatClone && thatClone->stabilizer) {
        if (randGlobalPhase) {
            thisClone->stabilizer->ResetPhaseOffset();
        }
        if (toCompare->randGlobalPhase) {
            thatClone->stabilizer->ResetPhaseOffset();
        }

        if (isDiscreteBool) {
            return thisClone->stabilizer->ApproxCompare(thatClone->stabilizer, error_tol) ? ZERO_R1_F : ONE_R1_F;
        }

        return thisClone->stabilizer->SumSqrDiff(thatClone->stabilizer);
    }

    if (thisClone) {
        thisClone->SwitchToEngine();
    }

    if (thatClone) {
        thatClone->SwitchToEngine();
    }

    QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
    QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

    const real1_f toRet = isDiscreteBool ? (thisEngine->ApproxCompare(thatEngine, error_tol) ? ZERO_R1_F : ONE_R1_F)
                                         : thisEngine->SumSqrDiff(thatEngine);

    if (toRet > TRYDECOMPOSE_EPSILON) {
        return toRet;
    }

    if (engine && toCompare->stabilizer) {
        SetPermutation(ZERO_BCI);
        stabilizer = std::dynamic_pointer_cast<QUnitClifford>(toCompare->stabilizer->Clone());
        shards.resize(toCompare->shards.size());
        ancillaCount = toCompare->ancillaCount;
        deadAncillaCount = toCompare->deadAncillaCount;
        for (size_t i = 0U; i < shards.size(); ++i) {
            shards[i] = toCompare->shards[i] ? toCompare->shards[i]->Clone() : nullptr;
        }
    } else if (stabilizer && !toCompare->stabilizer) {
        toCompare->SetPermutation(ZERO_BCI);
        toCompare->stabilizer = std::dynamic_pointer_cast<QUnitClifford>(stabilizer->Clone());
        toCompare->shards.resize(shards.size());
        toCompare->ancillaCount = ancillaCount;
        toCompare->deadAncillaCount = deadAncillaCount;
        for (size_t i = 0U; i < shards.size(); ++i) {
            toCompare->shards[i] = shards[i] ? shards[i]->Clone() : nullptr;
        }
    }

    return toRet;
}

void QStabilizerHybrid::ISwapHelper(bitLenInt qubit1, bitLenInt qubit2, bool inverse)
{
    if (qubit1 == qubit2) {
        return;
    }

    FlushIfBlocked(qubit1, qubit2, true);
    FlushIfBlocked(qubit2, qubit1, true);

    std::swap(shards[qubit1], shards[qubit2]);

    if (stabilizer) {
        rdmClone = nullptr;
        if (inverse) {
            return stabilizer->IISwap(qubit1, qubit2);
        }

        return stabilizer->ISwap(qubit1, qubit2);
    }

    if (inverse) {
        return engine->IISwap(qubit1, qubit2);
    }

    return engine->ISwap(qubit1, qubit2);
}

void QStabilizerHybrid::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    if ((nrm > ZERO_R1) && (abs(ONE_R1 - nrm) > FP_NORM_EPSILON)) {
        SwitchToEngine();
    }

    if (stabilizer) {
        rdmClone = nullptr;
        stabilizer->NormalizeState(REAL1_DEFAULT_ARG, norm_thresh, phaseArg);
    } else {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }
}

bool QStabilizerHybrid::TrySeparate(bitLenInt qubit)
{
    if (qubitCount == 1U) {
        if (ancillaCount || deadAncillaCount) {
            SwitchToEngine();
            complex sv[2];
            engine->GetQuantumState(sv);
            SetQuantumState(sv);
        }
        return true;
    }

    if (stabilizer) {
        rdmClone = nullptr;

        return stabilizer->TrySeparate(qubit);
    }

    return engine->TrySeparate(qubit);
}
bool QStabilizerHybrid::TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
{
    if ((qubitCount == 2U) && !ancillaCount && !deadAncillaCount) {
        return true;
    }

    if (engine) {
        return engine->TrySeparate(qubit1, qubit2);
    }

    rdmClone = nullptr;

    const bool toRet = stabilizer->TrySeparate(qubit1, qubit2);

    return toRet;
}
bool QStabilizerHybrid::TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol)
{
    if (engine) {
        return engine->TrySeparate(qubits, error_tol);
    }

    rdmClone = nullptr;

    return stabilizer->TrySeparate(qubits, error_tol);
}

std::ostream& operator<<(std::ostream& os, const QStabilizerHybridPtr s)
{
    if (s->engine) {
        throw std::logic_error("QStabilizerHybrid can only stream out when in Clifford format!");
    }

    s->PruneAncillae(true);

    os << (size_t)s->qubitCount << std::endl;

    os << s->stabilizer;

    const complex id[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
    const std::vector<MpsShardPtr>& shards = s->shards;
    for (size_t i = 0U; i < shards.size(); ++i) {
        const complex* mtrx = !shards[i] ? id : shards[i]->gate;
        for (size_t j = 0U; j < 3U; ++j) {
            os << mtrx[j] << " ";
        }
        os << mtrx[3U] << std::endl;
    }

    return os;
}

std::istream& operator>>(std::istream& is, const QStabilizerHybridPtr s)
{
    s->SetPermutation(ZERO_BCI);

    size_t qbCount;
    is >> qbCount;
    s->SetQubitCount(qbCount);

    is >> s->stabilizer;

    s->ancillaCount = s->stabilizer->GetQubitCount() - qbCount;
    s->shards.resize(s->stabilizer->GetQubitCount());

    std::vector<MpsShardPtr>& shards = s->shards;
    for (size_t i = 0U; i < shards.size(); ++i) {
        MpsShardPtr shard = std::make_shared<MpsShard>();
        for (size_t j = 0U; j < 4U; ++j) {
            is >> shard->gate[j];
        }
        if (!shard->IsIdentity()) {
            shards[i] = shard;
        }
    }

    const int64_t minLcv = (int64_t)qbCount;
    for (int64_t i = shards.size() - 1U; i >= minLcv; --i) {
        if (s->stabilizer->TrySeparate(i)) {
            s->stabilizer->Dispose(i, 1U);
            shards.erase(shards.begin() + i);
        }
    }

    return is;
}
} // namespace Qrack
