//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbdt.hpp"
#include "qfactory.hpp"

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)

namespace Qrack {

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds, bitLenInt qubitThreshold,
    real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , devID(deviceId)
    , root(NULL)
    , deviceIDs(devIds)
    , engines(eng)
    , shards(qubitCount)
{
    Init();

    SetPermutation(initState, phaseFac);
}

void QBdt::Init()
{
#if ENABLE_PTHREAD
    SetConcurrency(std::thread::hardware_concurrency());
#endif

    bdtStride = (GetStride() + 1U) >> 1U;
    if (!bdtStride) {
        bdtStride = 1U;
    }

    bitLenInt engineLevel = 0U;
    if (!engines.size()) {
        engines.push_back(QINTERFACE_OPTIMAL_BASE);
    }
    QInterfaceEngine rootEngine = engines[0U];
    while ((engines.size() < engineLevel) && (rootEngine != QINTERFACE_CPU) && (rootEngine != QINTERFACE_OPENCL) &&
        (rootEngine != QINTERFACE_HYBRID)) {
        ++engineLevel;
        rootEngine = engines[engineLevel];
    }
}

QEnginePtr QBdt::MakeQEngine(bitLenInt qbCount, bitCapInt perm)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX,
        doNormalize, false, false, devID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor, deviceIDs));
}

void QBdt::par_for_qbdt(const bitCapInt& end, bitLenInt maxQubit, BdtFunc fn, bool branch)
{
    if (branch) {
        root->Branch(maxQubit);
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    const bitCapInt Stride = bdtStride;
    unsigned underThreads = (unsigned)(pow2(qubitCount - (maxQubit + 1U)) / Stride);
    if (underThreads == 1U) {
        underThreads = 0U;
    }
    const unsigned nmCrs = (unsigned)(GetConcurrencyLevel() / (underThreads + 1U));
    unsigned threads = (unsigned)(end / Stride);
    if (threads > nmCrs) {
        threads = nmCrs;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            j |= fn(j);
        }
        if (branch) {
            root->Prune(maxQubit);
        }

        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.push_back(std::async(std::launch::async, [&myMutex, &idx, &end, &Stride, fn]() {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                bitCapInt j;
                for (j = 0U; j < maxJ; ++j) {
                    bitCapInt k = j + l;
                    k |= fn(k);
                    j = k - l;
                    if (j >= maxJ) {
                        std::lock_guard<std::mutex> lock(myMutex);
                        idx |= j / Stride;
                        break;
                    }
                }
            }
        }));
    }

    for (unsigned cpu = 0U; cpu < futures.size(); ++cpu) {
        futures[cpu].get();
    }
#else
    for (bitCapInt j = 0U; j < end; ++j) {
        j |= fn(j);
    }
#endif

    if (branch) {
        root->Prune(maxQubit);
    }
}

void QBdt::_par_for(const bitCapInt& end, ParallelFuncBdt fn)
{
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    const bitCapInt Stride = bdtStride;
    const unsigned nmCrs = GetConcurrencyLevel();
    unsigned threads = (unsigned)(end / Stride);
    if (threads > nmCrs) {
        threads = nmCrs;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            fn(j, 0U);
        }
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.push_back(std::async(std::launch::async, [&myMutex, &idx, &end, &Stride, cpu, fn]() {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                for (bitCapInt j = 0U; j < maxJ; ++j) {
                    fn(j + l, cpu);
                }
            }
        }));
    }

    for (unsigned cpu = 0U; cpu < futures.size(); ++cpu) {
        futures[cpu].get();
    }
#else
    for (bitCapInt j = 0U; j < end; ++j) {
        fn(j, 0U);
    }
#endif
}

size_t QBdt::CountBranches()
{
    const bitLenInt maxQubitIndex = qubitCount - 1U;
    std::set<QBdtNodeInterface*> nodes;
    std::mutex mtx;
    nodes.insert(root.get());
    par_for_qbdt(
        maxQPower, maxQubitIndex,
        [&](const bitCapInt& i) {
            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < maxQubitIndex; ++j) {
                if (IS_NODE_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(maxQubitIndex - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, maxQubitIndex - (j + 1U))];
                std::lock_guard<std::mutex> lock(mtx);
                nodes.insert(leaf.get());
            }

            return (bitCapInt)0U;
        },
        false);

    return nodes.size();
}

void QBdt::SetPermutation(bitCapInt initState, complex phaseFac)
{
    DumpBuffers();

    if (!qubitCount) {
        return;
    }

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * (real1_f)PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = ONE_CMPLX;
        }
    }

    root = std::make_shared<QBdtNode>(phaseFac);
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt qubit = 0U; qubit < qubitCount; ++qubit) {
        const size_t bit = SelectBit(initState, qubit);
        leaf->branches[bit] = std::make_shared<QBdtNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }
}

QInterfacePtr QBdt::Clone()
{
    QBdtPtr c = std::make_shared<QBdt>(engines, 0U, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, false,
        -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    c->root = root ? root->ShallowClone() : NULL;
    c->shards.resize(shards.size());
    c->SetQubitCount(qubitCount);
    for (size_t i = 0U; i < shards.size(); ++i) {
        if (shards[i]) {
            c->shards[i] = shards[i]->Clone();
        }
    }

    return c;
}

real1_f QBdt::SumSqrDiff(QBdtPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    FlushBuffers();
    toCompare->FlushBuffers();

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> projectionBuff(new complex[numCores]());

    if (randGlobalPhase) {
        real1_f lPhaseArg = FirstNonzeroPhase();
        real1_f rPhaseArg = toCompare->FirstNonzeroPhase();
        root->scale *= std::polar(ONE_R1, (real1)(rPhaseArg - lPhaseArg));
    }

    _par_for(maxQPower, [&](const bitCapInt& i, const unsigned& cpu) {
        projectionBuff[cpu] += conj(toCompare->GetAmplitude(i)) * GetAmplitude(i);
    });

    complex projection = ZERO_CMPLX;
    for (unsigned i = 0U; i < numCores; ++i) {
        projection += projectionBuff[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(projection));
}

complex QBdt::GetAmplitude(bitCapInt perm)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QBdt::GetAmplitude argument out-of-bounds!");
    }

    FlushBuffers();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        if (IS_NODE_0(leaf->scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    return scale;
}

bitLenInt QBdt::Compose(QBdtPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QBdt::Compose start index is out-of-bounds!");
    }

    if (!toCopy->qubitCount) {
        return start;
    }

    root->InsertAtDepth(toCopy->root->ShallowClone(), start, toCopy->qubitCount);

    // Resize the shards buffer.
    shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (bitLenInt i = 0; i < toCopy->qubitCount; ++i) {
        if (shards[start + i]) {
            shards[start + i] = shards[start + i]->Clone();
        }
    }

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

QInterfacePtr QBdt::Decompose(bitLenInt start, bitLenInt length)
{
    QBdtPtr dest = std::make_shared<QBdt>(engines, length, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    Decompose(start, dest);

    return dest;
}

void QBdt::DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QBdt::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    if (dest) {
        dest->root = root->RemoveSeparableAtDepth(start, length)->ShallowClone();
        std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    } else {
        root->RemoveSeparableAtDepth(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);

    SetQubitCount(qubitCount - length);
    root->Prune(qubitCount);
}

bitLenInt QBdt::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QBdtPtr nQubits = std::make_shared<QBdt>(engines, length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);
    nQubits->root->InsertAtDepth(root, length, qubitCount);
    root = nQubits->root;
    shards.insert(shards.begin() + start, nQubits->shards.begin(), nQubits->shards.end());
    SetQubitCount(qubitCount + length);
    ROR(length, 0U, start + length);

    return start;
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const MpsShardPtr shard = shards[qubit];
    if (shard && !shard->IsPhase()) {
        shards[qubit] = NULL;
        ApplySingle(shard->gate, qubit);
    }

    const bitCapInt qPower = pow2(qubit);
    const unsigned numCores = GetConcurrencyLevel();
    std::map<QEnginePtr, real1> qiProbs;
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    _par_for(qPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < qubit; ++j) {
            if (IS_NODE_0(leaf->scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (IS_NODE_0(leaf->scale)) {
            return;
        }

        oneChanceBuff[cpu] += norm(scale * leaf->branches[1U]->scale);
    });

    real1 oneChance = ZERO_R1;
    for (unsigned i = 0U; i < numCores; ++i) {
        oneChance += oneChanceBuff[i];
    }

    return clampProb((real1_f)oneChance);
}

real1_f QBdt::ProbAll(bitCapInt perm)
{
    FlushBuffers();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;

    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        if (IS_NODE_0(leaf->scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    return clampProb((real1_f)norm(scale));
}

bool QBdt::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const real1_f oneChance = Prob(qubit);
    if (oneChance >= ONE_R1) {
        result = true;
    } else if (oneChance <= ZERO_R1) {
        result = false;
    } else if (!doForce) {
        result = (Rand() <= oneChance);
    }

    if (!doApply) {
        return result;
    }

    shards[qubit] = NULL;

    const bitCapInt qPower = pow2(qubit);
    root->scale = GetNonunitaryPhase();

    _par_for(qPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0U; j < qubit; ++j) {
            if (IS_NODE_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        std::lock_guard<std::mutex> lock(leaf->mtx);

        if (IS_NODE_0(leaf->scale)) {
            return;
        }

        leaf->Branch();

        QBdtNodeInterfacePtr& b0 = leaf->branches[0U];
        QBdtNodeInterfacePtr& b1 = leaf->branches[1U];

        if (result) {
            if (IS_NODE_0(b1->scale)) {
                b1->SetZero();
                return;
            }
            b0->SetZero();
            b1->scale /= abs(b1->scale);
        } else {
            if (IS_NODE_0(b0->scale)) {
                b0->SetZero();
                return;
            }
            b0->scale /= abs(b0->scale);
            b1->SetZero();
        }
    });

    root->Prune(qubit);

    return result;
}

bitCapInt QBdt::MAll()
{
    bitCapInt result = 0U;
    QBdtNodeInterfacePtr leaf = root;

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        const MpsShardPtr shard = shards[i];
        if (shard && !shard->IsPhase()) {
            ApplySingle(shard->gate, i);
        }
        shards[i] = NULL;
    }

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        real1_f oneChance = clampProb((real1_f)norm(leaf->branches[1U]->scale));
        bool bitResult;
        if (oneChance >= ONE_R1) {
            bitResult = true;
        } else if (oneChance <= ZERO_R1) {
            bitResult = false;
        } else {
            bitResult = (Rand() <= oneChance);
        }

        // We might share this node with a clone:
        leaf->Branch();

        if (bitResult) {
            leaf->branches[0U]->SetZero();
            leaf->branches[1U]->scale = ONE_CMPLX;
            leaf = leaf->branches[1U];
            result |= pow2(i);
        } else {
            leaf->branches[0U]->scale = ONE_CMPLX;
            leaf->branches[1U]->SetZero();
            leaf = leaf->branches[0U];
        }
    }

    return result;
}

void QBdt::ApplySingle(const complex* mtrx, bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QBdt::ApplySingle target parameter must be within allocated qubit bounds!");
    }

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]) && IS_NORM_0(mtrx[0U] - mtrx[3U]) &&
        (randGlobalPhase || IS_NORM_0(ONE_CMPLX - mtrx[0U]))) {
        return;
    }

    const bitCapInt qPower = pow2(target);

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
    const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);

    const complex2 mtrxCol1Shuff = mtrxColShuff(mtrxCol1);
    const complex2 mtrxCol2Shuff = mtrxColShuff(mtrxCol2);
#endif

    par_for_qbdt(qPower, target,
#if ENABLE_COMPLEX_X2
        [this, target, &mtrxCol1, &mtrxCol2, &mtrxCol1Shuff, &mtrxCol2Shuff](const bitCapInt& i) {
#else
        [this, target, mtrx](const bitCapInt& i) {
#endif
            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < target; ++j) {
                if (IS_NODE_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(target - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, target - (j + 1U))];
            }

            std::lock_guard<std::mutex> lock(leaf->mtx);

            if (IS_NODE_0(leaf->scale)) {
                return (bitCapInt)0U;
            }

#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubitCount - target);
#else
            leaf->Apply2x2(mtrx, qubitCount - target);
#endif

            return (bitCapInt)0U;
        });
}

void QBdt::ApplyControlledSingle(const complex* mtrx, std::vector<bitLenInt> controls, bitLenInt target, bool isAnti)
{
    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QBdt::ApplyControlledSingle target parameter must be within allocated qubit bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount,
        "QBdt::ApplyControlledSingle parameter controls array values must be within allocated qubit bounds!");

    const bool isPhase = IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]) &&
        (isAnti ? IS_NORM_0(ONE_CMPLX - mtrx[3U]) : IS_NORM_0(ONE_CMPLX - mtrx[0U]));
    if (isPhase && IS_NORM_0(ONE_CMPLX - mtrx[0U]) && IS_NORM_0(ONE_CMPLX - mtrx[3U])) {
        return;
    }

    std::sort(controls.begin(), controls.end());
    if (target < controls.back()) {
        std::swap(target, controls.back());
        if (!isPhase) {
            // We need the target at back, for QBdt, if this isn't symmetric.
            Swap(target, controls.back());
            ApplyControlledSingle(mtrx, controls, target, isAnti);
            Swap(target, controls.back());

            return;
        }
        // Otherwise, the gate is symmetric in target and controls, so we can continue.
    }

    const bitCapInt qPower = pow2(target);
    bitCapInt controlMask = 0U;
    for (size_t c = 0U; c < controls.size(); ++c) {
        const bitLenInt control = controls[c];
        controlMask |= pow2(target - (control + 1U));
    }
    const bitCapInt controlPerm = isAnti ? 0U : controlMask;

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
    const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);

    const complex2 mtrxCol1Shuff = mtrxColShuff(mtrxCol1);
    const complex2 mtrxCol2Shuff = mtrxColShuff(mtrxCol2);
#endif

    par_for_qbdt(qPower, target,
#if ENABLE_COMPLEX_X2
        [this, controlMask, controlPerm, target, &mtrxCol1, &mtrxCol2, &mtrxCol1Shuff, &mtrxCol2Shuff](
            const bitCapInt& i) {
#else
        [this, controlMask, controlPerm, target, mtrx](const bitCapInt& i) {
#endif
            if ((i & controlMask) != controlPerm) {
                return (bitCapInt)(controlMask - ONE_BCI);
            }

            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < target; ++j) {
                if (IS_NODE_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(target - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, target - (j + 1U))];
            }

            std::lock_guard<std::mutex> lock(leaf->mtx);

            if (IS_NODE_0(leaf->scale)) {
                return (bitCapInt)0U;
            }

#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubitCount - target);
#else
            leaf->Apply2x2(mtrx, qubitCount - target);
#endif

            return (bitCapInt)0U;
        });
}

void QBdt::Mtrx(const complex* mtrx, bitLenInt target)
{
    MpsShardPtr& shard = shards[target];
    if (shard) {
        shard->Compose(mtrx);
    } else {
        shard = std::make_shared<MpsShard>(mtrx);
    }
}

void QBdt::MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{
    if (!controls.size()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        FlushNonPhaseBuffers();
        FlushIfBlocked(target, controls);
        ApplyControlledSingle(mtrx, controls, target, false);
    }
}

void QBdt::MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{

    if (!controls.size()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        FlushNonPhaseBuffers();
        FlushIfBlocked(target, controls);
        ApplyControlledSingle(mtrx, controls, target, true);
    }
}

void QBdt::MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (!controls.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (!IS_NORM_0(ONE_CMPLX - topLeft)) {
        FlushNonPhaseBuffers();
        ApplyControlledSingle(mtrx, controls, target, false);
        return;
    }

    if (IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    std::vector<bitLenInt> lControls(controls);
    lControls.push_back(target);
    std::sort(lControls.begin(), lControls.end());
    target = lControls.back();
    lControls.pop_back();

    FlushNonPhaseBuffers();
    ApplyControlledSingle(mtrx, lControls, target, false);
}

void QBdt::MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target)
{
    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (!IS_NORM_0(ONE_CMPLX - topRight) || !IS_NORM_0(ONE_CMPLX - bottomLeft)) {
        FlushNonPhaseBuffers();
        FlushIfBlocked(target, controls);
        ApplyControlledSingle(mtrx, controls, target, false);
        return;
    }

    std::vector<bitLenInt> lControls(controls);
    std::sort(lControls.begin(), lControls.end());

    if (lControls.back() < target) {
        FlushNonPhaseBuffers();
        FlushIfBlocked(target, lControls);
        ApplyControlledSingle(mtrx, lControls, target, false);
        return;
    }

    H(target);
    MCPhase(lControls, ONE_CMPLX, -ONE_CMPLX, target);
    H(target);
}

void QBdt::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const std::vector<bitLenInt> controls{ qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
        MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));

    const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
    if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
    if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON) {
        IISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubit1, qubit2); });
}
} // namespace Qrack
