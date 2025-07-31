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

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState, qrack_rand_gen_ptr rgp,
    const complex& phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devIds,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , devID(deviceId)
    , root{ nullptr }
    , deviceIDs(devIds)
    , engines(eng)
{
    Init();

    SetPermutation(initState, phaseFac);
}

void QBdt::Init()
{
    bdtStride = (GetStride() + 1U) >> 1U;
    if (!bdtStride) {
        bdtStride = 1U;
    }

    if (engines.empty()) {
        engines.push_back(QINTERFACE_OPTIMAL_BASE);
    }
    QInterfaceEngine rootEngine = engines[0U];
    bitLenInt engineLevel = 0U;
    while ((engines.size() < engineLevel) && (rootEngine != QINTERFACE_CPU) && (rootEngine != QINTERFACE_OPENCL) &&
        (rootEngine != QINTERFACE_HYBRID)) {
        ++engineLevel;
        rootEngine = engines[engineLevel];
    }
}

QEnginePtr QBdt::MakeQEngine(bitLenInt qbCount, const bitCapInt& perm)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX,
        doNormalize, false, false, devID, !!hardware_rand_generator, false, (real1_f)amplitudeFloor, deviceIDs));
}

void QBdt::par_for_qbdt(const bitCapInt& end, bitLenInt maxQubit, BdtFunc fn, bool branch)
{
    if (branch) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(root->mtx);
#endif
        try {
            root->Branch(maxQubit);
        } catch (const std::bad_alloc&) {
            root->Prune();

            throw bad_alloc("RAM limits exceeded in QBdt::par_for_qbdt()");
        }
    }

    for (bitCapInt j = 0U; bi_compare(j, end) < 0; bi_increment(&j, 1U)) {
        bi_or_ip(&j, fn(j));
    }

    if (branch) {
        root->Prune(maxQubit);
    }
}

void QBdt::_par_for(const bitCapInt& end, ParallelFuncBdt fn)
{
    for (bitCapInt j = 0U; bi_compare(j, end) < 0; bi_increment(&j, 1U)) {
        fn(j, 0U);
    }
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
                leaf = leaf->branches[SelectBit(i, maxQubitIndex - (j + 1U))];

                if (!leaf) {
                    return (bitCapInt)(pow2(maxQubitIndex - j) - ONE_BCI);
                }

                std::lock_guard<std::mutex> lock(mtx);
                nodes.insert(leaf.get());
            }

            return ZERO_BCI;
        },
        false);

    return nodes.size();
}

void QBdt::SetPermutation(const bitCapInt& initState, const complex& _phaseFac)
{
    if (!qubitCount) {
        return;
    }

    complex phaseFac = _phaseFac;
    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            phaseFac = std::polar(ONE_R1, (real1)(Rand() * 2 * PI_R1));
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
    QBdtPtr c = std::make_shared<QBdt>(engines, 0U, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, !hardware_rand_generator ? false : true, false, (real1_f)amplitudeFloor);

    c->root = root ? root->ShallowClone() : nullptr;
    c->SetQubitCount(qubitCount);

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

complex QBdt::GetAmplitude(const bitCapInt& perm)
{
    if (bi_compare(perm, maxQPower) >= 0) {
        throw std::invalid_argument("QBdt::GetAmplitude argument out-of-bounds!");
    }

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        leaf = leaf->branches[SelectBit(perm, j)];
        if (!leaf) {
            break;
        }
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

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (true) {
        QBdtNodeInterfacePtr _root = root;
        std::lock_guard<std::mutex> lock(_root->mtx);
        root->InsertAtDepth(toCopy->root->ShallowClone(), start, toCopy->qubitCount);
    }
#else
    root->InsertAtDepth(toCopy->root->ShallowClone(), start, toCopy->qubitCount);
#endif

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

QInterfacePtr QBdt::Decompose(bitLenInt start, bitLenInt length)
{
    QBdtPtr dest = std::make_shared<QBdt>(engines, length, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, !hardware_rand_generator ? false : true, false, (real1_f)amplitudeFloor);

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
        QBdtNodeInterfacePtr _root = root;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(_root->mtx);
#endif
        dest->root = root->RemoveSeparableAtDepth(start, length);
    } else {
        QBdtNodeInterfacePtr _root = root;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(_root->mtx);
#endif
        root->RemoveSeparableAtDepth(start, length);
    }

    SetQubitCount(qubitCount - length);
    root->Prune(qubitCount);
}

bool QBdt::IsSeparable(bitLenInt start)
{
    if (!start || (start >= qubitCount)) {
        throw std::invalid_argument(
            "QBdt::IsSeparable() start parameter must be at least 1 and less than the QBdt qubit width!");
    }

    // If the tree has been fully reduced, this should ALWAYS be the same for ALL branches
    // (that have nonzero amplitude), if-and-only-if the state is separable.
    QBdtNodeInterfacePtr subsystemPtr{ nullptr };

    const bitCapInt qPower = pow2(start);
    bool result = true;

    par_for_qbdt(
        qPower, start,
        [this, start, &subsystemPtr, &result](const bitCapInt& i) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0U; j < start; ++j) {
                leaf = leaf->branches[SelectBit(i, start - (j + 1U))];

                if (!leaf) {
                    // The immediate parent of "leaf" has 0 amplitude.
                    return (bitCapInt)(pow2(start - j) - ONE_BCI);
                }
            }

            if (!leaf->branches[0U] || !leaf->branches[1U]) {
                // "leaf" is a 0-amplitude branch.
                return ZERO_BCI;
            }

            // "leaf" is nonzero.
            // Every such instance must be identical.

            if (!subsystemPtr) {
                // Even if another thread "clobbers" this assignment,
                // then the equality check afterward will fail.
                subsystemPtr = leaf;
            }

            if (subsystemPtr != leaf) {
                // There are at least two distinct possible subsystem states for the "high-index" subsystem,
                // depending specifically on which dimension of the "low-index" subsystem we're inspecting.
                result = false;

                return (bitCapInt)(pow2(start) - ONE_BCI);
            }

            return ZERO_BCI;
        },
        false);

    return result;
}

bitLenInt QBdt::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QBdtPtr nQubits = std::make_shared<QBdt>(engines, length, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, !hardware_rand_generator ? false : true, false, (real1_f)amplitudeFloor);
    nQubits->root->InsertAtDepth(root, length, qubitCount);
    root = nQubits->root;
    SetQubitCount(qubitCount + length);
    ROR(length, 0U, start + length);

    return start;
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QBdt::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    const bitCapInt qPower = pow2(qubit);
    const unsigned numCores = GetConcurrencyLevel();
    std::map<QEnginePtr, real1> qiProbs;
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    _par_for(qPower, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0U; j < qubit; ++j) {
            leaf = leaf->branches[SelectBit(i, j)];
            if (!leaf) {
                break;
            }
            scale *= leaf->scale;
        }

        if (!leaf || !leaf->branches[1U]) {
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

real1_f QBdt::ProbAll(const bitCapInt& perm)
{
    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;

    for (bitLenInt j = 0U; j < qubitCount; ++j) {
        leaf = leaf->branches[SelectBit(perm, j)];
        if (!leaf) {
            break;
        }
        scale *= leaf->scale;
    }

    return clampProb((real1_f)norm(scale));
}

bitCapInt QBdt::MAllOptionalCollapse(bool isCollapsing)
{
    bitCapInt result = ZERO_BCI;
    QBdtNodeInterfacePtr leaf = root;

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        real1_f oneChance = clampProb(norm(leaf->branches[1U]->scale));
        bool bitResult;
        if (oneChance >= ONE_R1) {
            bitResult = true;
        } else if (oneChance <= ZERO_R1) {
            bitResult = false;
        } else {
            bitResult = (Rand() <= oneChance);
        }

        if (isCollapsing) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> lock(leaf->mtx);
#endif
            // We might share this node with a clone:
            leaf->Branch();
        }

        if (bitResult) {
            if (isCollapsing) {
                leaf->branches[0U]->SetZero();
                leaf->branches[1U]->scale = ONE_CMPLX;
            }
            leaf = leaf->branches[1U];
            bi_or_ip(&result, pow2(i));
        } else {
            if (isCollapsing) {
                leaf->branches[0U]->scale = ONE_CMPLX;
                leaf->branches[1U]->SetZero();
            }
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
                leaf = leaf->branches[SelectBit(i, target - (j + 1U))];

                if (!leaf) {
                    return (bitCapInt)(pow2(target - j) - ONE_BCI);
                }
            }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> lock(leaf->mtx);
#endif

            if (!leaf->branches[0U] || !leaf->branches[1U]) {
                leaf->SetZero();

                return ZERO_BCI;
            }

#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubitCount - target);
#else
            leaf->Apply2x2(mtrx, qubitCount - target);
#endif

            return ZERO_BCI;
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
            return Swap(target, controls.back());
        }
        // Otherwise, the gate is symmetric in target and controls, so we can continue.
    }

    const bitCapInt qPower = pow2(target);
    bitCapInt controlMask = ZERO_BCI;
    for (const bitLenInt& control : controls) {
        bi_or_ip(&controlMask, pow2(target - (control + 1U)));
    }
    const bitCapInt controlPerm = isAnti ? ZERO_BCI : controlMask;

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
            if (bi_compare((i & controlMask), controlPerm) != 0) {
                return controlMask - ONE_BCI;
            }

            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0U; j < target; ++j) {
                leaf = leaf->branches[SelectBit(i, target - (j + 1U))];

                if (!leaf) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(target - j) - ONE_BCI);
                }
            }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> lock(leaf->mtx);
#endif

            if (!leaf->branches[0U] || !leaf->branches[1U]) {
                leaf->SetZero();

                return ZERO_BCI;
            }

#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubitCount - target);
#else
            leaf->Apply2x2(mtrx, qubitCount - target);
#endif

            return ZERO_BCI;
        });
}

void QBdt::Mtrx(const complex* mtrx, bitLenInt target) { ApplySingle(mtrx, target); }

void QBdt::MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{
    if (controls.empty()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        ApplyControlledSingle(mtrx, controls, target, false);
    }
}

void QBdt::MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
{

    if (controls.empty()) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(controls, mtrx[0U], mtrx[3U], target);
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(controls, mtrx[1U], mtrx[2U], target);
    } else {
        ApplyControlledSingle(mtrx, controls, target, true);
    }
}

void QBdt::MCPhase(
    const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
{
    if (controls.empty()) {
        return Phase(topLeft, bottomRight, target);
    }

    const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (!IS_NORM_0(ONE_CMPLX - topLeft)) {
        return ApplyControlledSingle(mtrx, controls, target, false);
    }

    if (IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    std::vector<bitLenInt> lControls(controls);
    lControls.push_back(target);
    std::sort(lControls.begin(), lControls.end());
    target = lControls.back();
    lControls.pop_back();

    ApplyControlledSingle(mtrx, lControls, target, false);
}

void QBdt::MCInvert(
    const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
{
    if (controls.empty()) {
        return Invert(topRight, bottomLeft, target);
    }

    const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (!IS_NORM_0(ONE_CMPLX - topRight) || !IS_NORM_0(ONE_CMPLX - bottomLeft)) {
        return ApplyControlledSingle(mtrx, controls, target, false);
    }

    std::vector<bitLenInt> lControls(controls);
    std::sort(lControls.begin(), lControls.end());

    if (lControls.back() < target) {
        return ApplyControlledSingle(mtrx, lControls, target, false);
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
    const real1_f sinTheta = sin(theta);
    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON_F) {
        return MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));
    const real1_f sinThetaDiffNeg = ONE_R1_F + sinTheta;
    if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON_F) {
        ISwap(qubit1, qubit2);

        return MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
    }

    const real1_f sinThetaDiffPos = ONE_R1_F - sinTheta;
    if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON_F) {
        IISwap(qubit1, qubit2);

        return MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
    }

    ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubit1, qubit2); });
}
} // namespace Qrack
