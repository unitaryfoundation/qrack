//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// When we allocate a quantum register, all bits are in a (re)set state. At this point,
// we know they are separable, in the sense of full Schmidt decomposability into qubits
// in the "natural" or "permutation" basis of the register. Many operations can be
// traced in terms of fewer qubits that the full "Schr\{"o}dinger representation."
//
// Based on experimentation, QUnit is designed to avoid increasing representational
// entanglement for its primary action, and only try to decrease it when inquiries
// about probability need to be made otherwise anyway. Avoiding introducing the cost of
// basically any entanglement whatsoever, rather than exponentially costly "garbage
// collection," should be the first and ultimate concern, in the authors' experience.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#include <ctime>
#include <initializer_list>
#include <map>

#define DIRTY(shard) (shard.isPhaseDirty || shard.isProbDirty)
#define IS_AMP_0(c) (norm(c) <= separabilityThreshold)
#define IS_0_R1(r) (abs(r) <= REAL1_EPSILON)
#define IS_1_R1(r) (abs(ONE_R1 - r) <= REAL1_EPSILON)
#define IS_1_CMPLX(c) (norm(ONE_CMPLX - (c)) <= FP_NORM_EPSILON)
#define SHARD_STATE(shard) ((2 * norm(shard.amp0)) < ONE_R1)
#define QUEUED_PHASE(shard)                                                                                            \
    ((shard.targetOfShards.size() != 0U) || (shard.controlsShards.size() != 0U) ||                                     \
        (shard.antiTargetOfShards.size() != 0U) || (shard.antiControlsShards.size() != 0U))
#define CACHED_X(shard) ((shard.pauliBasis == PauliX) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_X_OR_Y(shard) ((shard.pauliBasis != PauliZ) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_Z(shard) ((shard.pauliBasis == PauliZ) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_ZERO(shard) (CACHED_Z(shard) && IS_AMP_0(shard.amp1))
#define CACHED_ONE(shard) (CACHED_Z(shard) && IS_AMP_0(shard.amp0))
#define CACHED_PLUS(shard) (CACHED_X(shard) && IS_AMP_0(shard.amp1))
/* "UNSAFE" variants here do not check whether the bit has cached 2-qubit gates.*/
#define UNSAFE_CACHED_ZERO_OR_ONE(shard)                                                                               \
    (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define UNSAFE_CACHED_X(shard)                                                                                         \
    (!shard.isProbDirty && (shard.pauliBasis == PauliX) && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define UNSAFE_CACHED_ONE(shard) (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && IS_AMP_0(shard.amp0))
#define UNSAFE_CACHED_ZERO(shard) (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && IS_AMP_0(shard.amp1))
#define IS_SAME_UNIT(shard1, shard2) (shard1.unit && (shard1.unit == shard2.unit))
#define ARE_CLIFFORD(shard1, shard2)                                                                                   \
    ((engines[0U] == QINTERFACE_STABILIZER_HYBRID) && shard1.isClifford() && shard2.isClifford())
#define BLOCKED_SEPARATE(shard) (shard.unit && shard.unit->isClifford() && !shard.unit->TrySeparate(shard.mapped))
#define IS_PHASE_OR_INVERT(mtrx)                                                                                       \
    ((IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) || (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])))

namespace Qrack {

QUnit::QUnit(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceID, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList, bitLenInt qubitThreshold,
    real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , isSparse(useSparseStateVec)
    , freezeBasis2Qb(false)
    , isReactiveSeparate(true)
    , useTGadget(true)
    , thresholdQubits(qubitThreshold)
    , separabilityThreshold(sep_thresh)
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , deviceIDs(devList)
    , engines(eng)
{
    if (!engines.size()) {
        engines.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")) {
        separabilityThreshold = (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")));
    }
#endif

    if (qubitCount) {
        SetPermutation(initState);
    }
}

QInterfacePtr QUnit::MakeEngine(bitLenInt length, bitCapInt perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    toRet->SetTInjection(useTGadget);

    return toRet;
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Dump();

    shards = QEngineShardMap();

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        bool bitState = ((perm >> (bitCapIntOcl)i) & ONE_BCI) != 0U;
        shards.push_back(QEngineShard(bitState, GetNonunitaryPhase()));
    }
}

void QUnit::SetQuantumState(const complex* inputState)
{
    Dump();

    if (qubitCount == 1U) {
        QEngineShard& shard = shards[0U];
        shard.unit = NULL;
        shard.mapped = 0U;
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.amp0 = inputState[0U];
        shard.amp1 = inputState[1U];
        shard.pauliBasis = PauliZ;
        if (IS_AMP_0(shard.amp0 - shard.amp1)) {
            shard.pauliBasis = PauliX;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0(shard.amp0 + shard.amp1)) {
            shard.pauliBasis = PauliX;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0U]) - inputState[1U])) {
            shard.pauliBasis = PauliY;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0U]) + inputState[1U])) {
            shard.pauliBasis = PauliY;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        }
        return;
    }

    QInterfacePtr unit = MakeEngine(qubitCount, 0U);
    unit->SetQuantumState(inputState);

    for (bitLenInt idx = 0U; idx < qubitCount; ++idx) {
        shards[idx] = QEngineShard(unit, idx);
    }
}

void QUnit::GetQuantumState(complex* outputState)
{
    if (qubitCount == 1U) {
        RevertBasis1Qb(0U);
        if (!shards[0U].unit) {
            outputState[0U] = shards[0U].amp0;
            outputState[1U] = shards[0U].amp1;

            return;
        }
    }

    QUnitPtr thisCopyShared;
    QUnit* thisCopy;

    if (shards[0U].GetQubitCount() == qubitCount) {
        ToPermBasisAll();
        OrderContiguous(shards[0U].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    thisCopy->shards[0U].unit->GetQuantumState(outputState);
}

void QUnit::GetProbs(real1* outputProbs)
{
    if (qubitCount == 1U) {
        RevertBasis1Qb(0U);
        if (!shards[0U].unit) {
            outputProbs[0U] = norm(shards[0U].amp0);
            outputProbs[1U] = norm(shards[0U].amp1);

            return;
        }
    }

    QUnitPtr thisCopyShared;
    QUnit* thisCopy;

    if (shards[0U].GetQubitCount() == qubitCount) {
        ToPermBasisProb();
        OrderContiguous(shards[0U].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll(true);
        thisCopy = thisCopyShared.get();
    }

    thisCopy->shards[0U].unit->GetProbs(outputProbs);
}

complex QUnit::GetAmplitude(bitCapInt perm) { return GetAmplitudeOrProb(perm, false); }

complex QUnit::GetAmplitudeOrProb(bitCapInt perm, bool isProb)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QUnit::GetAmplitudeOrProb argument out-of-bounds!");
    }

    if (isProb) {
        ToPermBasisProb();
    } else {
        ToPermBasisAll();
    }

    complex result(ONE_R1, ZERO_R1);

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        QEngineShard& shard = shards[i];

        if (!shard.unit) {
            result *= ((perm >> (bitCapIntOcl)i) & ONE_BCI) ? shard.amp1 : shard.amp0;
            continue;
        }

        if (perms.find(shard.unit) == perms.end()) {
            perms[shard.unit] = 0U;
        }
        if ((perm >> (bitCapIntOcl)i) & ONE_BCI) {
            perms[shard.unit] |= pow2(shard.mapped);
        }
    }

    for (auto&& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
        if (IS_AMP_0(result)) {
            break;
        }
    }

    if ((shards[0U].GetQubitCount() > 1) && (norm(result) >= (ONE_R1 - FP_NORM_EPSILON)) &&
        (randGlobalPhase || IS_AMP_0(result - ONE_CMPLX))) {
        SetPermutation(perm);
    }

    return result;
}

void QUnit::SetAmplitude(bitCapInt perm, complex amp)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QUnit::SetAmplitude argument out-of-bounds!");
    }

    EntangleAll();
    shards[0U].unit->SetAmplitude(perm, amp);
}

bitLenInt QUnit::Compose(QUnitPtr toCopy) { return Compose(toCopy, qubitCount); }

/*
 * Append QInterface in the middle of QUnit.
 */
bitLenInt QUnit::Compose(QUnitPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QUnit::Compose start index is out-of-bounds!");
    }

    /* Create a clone of the quantum state in toCopy. */
    QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(toCopy->Clone());

    /* Insert the new shards in the middle */
    shards.insert(start, clone->shards);

    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    return start;
}

void QUnit::Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::Detach range is out-of-bounds!");
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis2Qb(start + i);
    }

    // Move "emulated" bits immediately into the destination, which is initialized.
    // Find a set of shard "units" to order contiguously. Also count how many bits to decompose are in each subunit.
    std::map<QInterfacePtr, bitLenInt> subunits;
    for (bitLenInt i = 0U; i < length; ++i) {
        QEngineShard& shard = shards[start + i];
        if (shard.unit) {
            ++(subunits[shard.unit]);
        } else if (dest) {
            dest->shards[i] = shard;
        }
    }

    // Order the subsystem units contiguously. (They might be entangled at random with bits not involed in the
    // operation.)
    if (length > 1U) {
        for (auto subunit = subunits.begin(); subunit != subunits.end(); ++subunit) {
            OrderContiguous(subunit->first);
        }
    }

    // After ordering all subunits contiguously, since the top level mapping is a contiguous array, all subunit sets are
    // also contiguous. From the lowest index bits, they are mapped simply for the length count of bits involved in the
    // entire subunit.
    std::map<QInterfacePtr, bitLenInt> decomposedUnits;
    for (bitLenInt i = 0U; i < length; ++i) {
        QEngineShard& shard = shards[start + i];
        QInterfacePtr unit = shard.unit;

        if (unit == NULL) {
            continue;
        }

        if (decomposedUnits.find(unit) == decomposedUnits.end()) {
            decomposedUnits[unit] = start + i;
            bitLenInt subLen = subunits[unit];
            bitLenInt origLen = unit->GetQubitCount();
            if (subLen != origLen) {
                if (dest) {
                    QInterfacePtr nUnit = MakeEngine(subLen, 0U);
                    shard.unit->Decompose(shard.mapped, nUnit);
                    shard.unit = nUnit;
                } else {
                    shard.unit->Dispose(shard.mapped, subLen);
                }

                if ((subLen == 1U) && dest) {
                    complex amps[2U];
                    shard.unit->GetQuantumState(amps);
                    shard.amp0 = amps[0U];
                    shard.amp1 = amps[1U];
                    shard.isProbDirty = false;
                    shard.isPhaseDirty = false;
                    shard.unit = NULL;
                    shard.mapped = 0U;
                    shard.ClampAmps();
                }

                if (subLen == (origLen - 1U)) {
                    bitLenInt mapped = shards[decomposedUnits[unit]].mapped;
                    if (!mapped) {
                        mapped += subLen;
                    } else {
                        mapped = 0U;
                    }
                    for (bitLenInt i = 0U; i < shards.size(); ++i) {
                        if (!((shards[i].unit == unit) && (shards[i].mapped == mapped))) {
                            continue;
                        }

                        QEngineShard* pShard = &shards[i];
                        complex amps[2U];
                        pShard->unit->GetQuantumState(amps);
                        pShard->amp0 = amps[0U];
                        pShard->amp1 = amps[1U];
                        pShard->isProbDirty = false;
                        pShard->isPhaseDirty = false;
                        pShard->unit = NULL;
                        pShard->mapped = 0U;
                        pShard->ClampAmps();

                        break;
                    }
                }
            }
        } else {
            shard.unit = shards[decomposedUnits[unit]].unit;
        }

        if (dest) {
            dest->shards[i] = shard;
        }
    }

    /* Find the rest of the qubits. */
    for (auto&& shard : shards) {
        auto subunit = subunits.find(shard.unit);
        if (subunit != subunits.end() &&
            shard.mapped >= (shards[decomposedUnits[shard.unit]].mapped + subunit->second)) {
            shard.mapped -= subunit->second;
        }
    }

    shards.erase(start, start + length);
    SetQubitCount(qubitCount - length);
}

void QUnit::Decompose(bitLenInt start, QUnitPtr dest) { Detach(start, dest->GetQubitCount(), dest); }

QInterfacePtr QUnit::Decompose(bitLenInt start, bitLenInt length)
{
    QUnitPtr dest = std::make_shared<QUnit>(engines, length, 0U, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);

    Decompose(start, dest);

    return dest;
}

void QUnit::Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }

// The optimization of this method is redundant with other optimizations in QUnit.
void QUnit::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) { Detach(start, length, nullptr); }

QInterfacePtr QUnit::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    for (auto bit = first; bit < last; ++bit) {
        EndEmulation(**bit);
    }

    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    /* Walk through all of the supplied bits and create a unique list to compose. */
    for (auto bit = first; bit < last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    while (units.size() > 1U) {
        // Work odd unit into collapse sequence:
        if (units.size() & 1U) {
            QInterfacePtr consumed = units[1U];
            bitLenInt offset = unit1->ComposeNoClone(consumed);
            units.erase(units.begin() + 1U);

            for (auto&& shard : shards) {
                if (shard.unit == consumed) {
                    shard.mapped += offset;
                    shard.unit = unit1;
                }
            }
        }

        std::vector<QInterfacePtr> nUnits;
        std::map<QInterfacePtr, bitLenInt> offsets;
        std::map<QInterfacePtr, QInterfacePtr> offsetPartners;

        for (size_t i = 0U; i < units.size(); i += 2U) {
            QInterfacePtr retained = units[i];
            QInterfacePtr consumed = units[i + 1U];
            nUnits.push_back(retained);
            offsets[consumed] = retained->ComposeNoClone(consumed);
            offsetPartners[consumed] = retained;
        }

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto&& shard : shards) {
            auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = offsetPartners[shard.unit];
            }
        }

        units = nUnits;
    }

    /* Change the source parameters to the correct newly mapped bit indexes. */
    for (auto bit = first; bit < last; ++bit) {
        **bit = shards[**bit].mapped;
    }

    return unit1;
}

bitLenInt QUnit::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QUnitPtr nQubits = std::make_shared<QUnit>(engines, length, 0U, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    nQubits->SetReactiveSeparate(isReactiveSeparate);
    return Compose(nQubits, start);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt> bits)
{
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (bitLenInt i = 0U; i < (bitLenInt)ebits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    return Entangle(ebits);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt*> bits)
{
    for (bitLenInt i = 0U; i < (bitLenInt)bits.size(); ++i) {
        ToPermBasis(*(bits[i]));
    }
    return EntangleInCurrentBasis(bits.begin(), bits.end());
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length, bool isForProb)
{
    if (isForProb) {
        ToPermBasisProb(start, length);
    } else {
        ToPermBasis(start, length);
    }

    if (length == 1U) {
        EndEmulation(start);
        return shards[start].unit;
    }

    std::vector<bitLenInt> bits(length);
    std::vector<bitLenInt*> ebits(length);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = i + start;
        ebits[i] = &bits[i];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(toRet);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    ToPermBasis(start1, length1);
    ToPermBasis(start2, length2);

    std::vector<bitLenInt> bits(length1 + length2);
    std::vector<bitLenInt*> ebits(length1 + length2);

    if (start2 < start1) {
        std::swap(start1, start2);
        std::swap(length1, length2);
    }

    for (bitLenInt i = 0U; i < length1; ++i) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (bitLenInt i = 0U; i < length2; ++i) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(toRet);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(
    bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3)
{
    ToPermBasis(start1, length1);
    ToPermBasis(start2, length2);
    ToPermBasis(start3, length3);

    std::vector<bitLenInt> bits(length1 + length2 + length3);
    std::vector<bitLenInt*> ebits(length1 + length2 + length3);

    if (start2 < start1) {
        std::swap(start1, start2);
        std::swap(length1, length2);
    }

    if (start3 < start1) {
        std::swap(start1, start3);
        std::swap(length1, length3);
    }

    if (start3 < start2) {
        std::swap(start2, start3);
        std::swap(length2, length3);
    }

    for (bitLenInt i = 0U; i < length1; ++i) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (bitLenInt i = 0U; i < length2; ++i) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    for (bitLenInt i = 0U; i < length3; ++i) {
        bits[i + length1 + length2] = i + start3;
        ebits[i + length1 + length2] = &bits[i + length1 + length2];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(toRet);
    return toRet;
}

bool QUnit::TrySeparateClifford(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];
    if (!shard.unit->TrySeparate(shard.mapped)) {
        return false;
    }

    // If TrySeparate() == true, this bit can be decomposed.
    QInterfacePtr sepUnit = shard.unit->Decompose(shard.mapped, 1U);
    const bool isPair = (shard.unit->GetQubitCount() == 1U);

    bitLenInt oQubit = 0U;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if ((shard.unit == shards[i].unit) && (shard.mapped != shards[i].mapped)) {
            oQubit = i;
            if (shard.mapped < shards[i].mapped) {
                --(shards[i].mapped);
            }
        }
    }

    shard.mapped = 0U;
    shard.unit = sepUnit;

    ProbBase(qubit);
    if (isPair) {
        ProbBase(oQubit);
    }

    return true;
}

bool QUnit::TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol)
{
    ThrowIfQbIdArrayIsBad(qubits, length, qubitCount,
        "QUnit::TrySeparate parameter controls array values must be within allocated qubit bounds!");

    if (length == 1U) {
        bitLenInt qubit = qubits[0U];
        QEngineShard& shard = shards[qubit];

        if (shard.GetQubitCount() == 1U) {
            if (shard.unit) {
                ProbBase(qubit);
            }
            return true;
        }

        if (BLOCKED_SEPARATE(shard)) {
            return false;
        }

        bitLenInt mapped = shard.mapped;
        QInterfacePtr oUnit = shard.unit;
        QInterfacePtr nUnit = MakeEngine(1U, 0U);
        if (oUnit->TryDecompose(mapped, nUnit, error_tol)) {
            for (bitLenInt i = 0; i < qubitCount; ++i) {
                if ((shards[i].unit == oUnit) && (shards[i].mapped > mapped)) {
                    --(shards[i].mapped);
                }
            }

            shard.unit = nUnit;
            shard.mapped = 0U;
            shard.MakeDirty();
            ProbBase(qubit);

            if (oUnit->GetQubitCount() == 1U) {
                return true;
            }

            for (bitLenInt i = 0U; i < qubitCount; ++i) {
                if (shard.unit == oUnit) {
                    ProbBase(i);
                    break;
                }
            }

            return true;
        }

        return false;
    }

    std::vector<bitLenInt> q(length);
    std::copy(qubits, qubits + length, q.begin());
    std::sort(q.begin(), q.end());

    // Swap gate is free, so just bring into the form of the contiguous overload.
    for (bitLenInt i = 0U; i < length; ++i) {
        Swap(i, q[i]);
    }

    QUnitPtr dest = std::dynamic_pointer_cast<QUnit>(std::make_shared<QUnit>(
        engines, length, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam));

    bool toRet = TryDecompose(0U, dest, error_tol);
    if (toRet) {
        if (length == 1U) {
            dest->ProbBase(0U);
        }
        Compose(dest, 0U);
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        Swap(i, q[i]);
    }

    return toRet;
}

bool QUnit::TrySeparate(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[qubit];

    if (shard.GetQubitCount() == 1U) {
        if (shard.unit) {
            ProbBase(qubit);
        }
        return true;
    }

    if (shard.unit && shard.unit->isClifford()) {
        return TrySeparateClifford(qubit);
    }

    real1_f prob;
    real1_f x = ZERO_R1_F;
    real1_f y = ZERO_R1_F;
    real1_f z = ZERO_R1_F;

    for (bitLenInt i = 0U; i < 3U; ++i) {
        prob = ONE_R1_F - 2 * ProbBase(qubit);

        if (!shard.unit) {
            return true;
        }

        if (shard.pauliBasis == PauliZ) {
            z = prob;
        } else if (shard.pauliBasis == PauliX) {
            x = prob;
        } else {
            y = prob;
        }

        if (i >= 2) {
            continue;
        }

        if (shard.pauliBasis == PauliZ) {
            ConvertZToX(qubit);
        } else if (shard.pauliBasis == PauliX) {
            ConvertXToY(qubit);
        } else {
            ConvertYToZ(qubit);
        }
    }

    const real1_f r = sqrt(x * x + y * y + z * z);
    if (((ONE_R1 - r) > separabilityThreshold) || (r > (ONE_R1 + separabilityThreshold))) {
        return false;
    }

    // Permute axes for logical equivalence.
    if (shard.pauliBasis == PauliX) {
        RevertBasis1Qb(qubit);
    } else if (shard.pauliBasis == PauliY) {
        std::swap(x, z);
        std::swap(y, z);
    }

    const real1_f inclination = atan2(sqrt(x * x + y * y), z);
    const real1_f azimuth = atan2(y, x);

    shard.unit->IAI(shard.mapped, azimuth, inclination);
    prob = shard.unit->Prob(shard.mapped);

    if (prob > separabilityThreshold) {
        shard.unit->AI(shard.mapped, azimuth, inclination);
        return false;
    }

    SeparateBit(false, qubit);
    ShardAI(qubit, azimuth, inclination);

    return true;
}

bool QUnit::TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::TrySeparate target parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (freezeBasis2Qb || !shard1.unit || !shard2.unit || (shard1.unit != shard2.unit)) {
        // Both shards have non-null units, and we've tried everything, if they're not the same unit.
        const bool isShard1Sep = TrySeparate(qubit1);
        const bool isShard2Sep = TrySeparate(qubit2);
        return isShard1Sep && isShard2Sep;
    }

    const QInterfacePtr unit = shard1.unit;
    bitLenInt mapped1 = shard1.mapped;
    bitLenInt mapped2 = shard2.mapped;

    // Both shards are in the same unit.
    if (unit->isClifford() && !unit->TrySeparate(mapped1, mapped2)) {
        return false;
    }

    if (QUEUED_PHASE(shard1) || QUEUED_PHASE(shard2)) {
        // Both shards have non-null units, and we've tried everything, if they're not the same unit.
        const bool isShard1Sep = TrySeparate(qubit1);
        const bool isShard2Sep = TrySeparate(qubit2);
        return isShard1Sep && isShard2Sep;
    }

    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);

    // "Controlled inverse state preparation"
    const complex mtrx[4U] = { complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1),
        complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, SQRT1_2_R1) };
    const bitLenInt controls[1U] = { mapped1 };

    real1_f z = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->CH(shard1.mapped, shard2.mapped);
    real1_f x = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->CS(shard1.mapped, shard2.mapped);
    real1_f y = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->MCMtrx(controls, 1U, mtrx, mapped2);
    const real1_f inclination = atan2(sqrt(x * x + y * y), z);
    const real1_f azimuth = atan2(y, x);
    unit->CIAI(mapped1, mapped2, azimuth, inclination);

    z = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->AntiCH(shard1.mapped, shard2.mapped);
    x = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->AntiCS(shard1.mapped, shard2.mapped);
    y = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->MACMtrx(controls, 1U, mtrx, mapped2);
    const real1_f inclinationAnti = atan2(sqrt(x * x + y * y), z);
    const real1_f azimuthAnti = atan2(y, z);
    unit->AntiCIAI(mapped1, mapped2, azimuthAnti, inclinationAnti);

    shard1.MakeDirty();
    shard2.MakeDirty();

    const bool isShard1Sep = TrySeparate(qubit1);
    const bool isShard2Sep = TrySeparate(qubit2);

    AntiCAI(qubit1, qubit2, azimuthAnti, inclinationAnti);
    CAI(qubit1, qubit2, azimuth, inclination);

    return isShard1Sep && isShard2Sep;
}

void QUnit::OrderContiguous(QInterfacePtr unit)
{
    /* Before we call OrderContinguous, when we are cohering lists of shards, we should always proactively sort the
     * order in which we compose qubits into a single engine. This is a cheap way to reduce the need for costly qubit
     * swap gates, later. */

    if (!unit || (unit->GetQubitCount() == 1U)) {
        return;
    }

    /* Create a sortable collection of all of the bits that are in the unit. */
    std::vector<QSortEntry> bits(unit->GetQubitCount());

    bitLenInt j = 0U;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i].unit == unit) {
            bits[j].mapped = shards[i].mapped;
            bits[j].bit = i;
            ++j;
        }
    }

    SortUnit(unit, bits, 0U, bits.size() - 1U);
}

/* Sort a container of bits, calling Swap() on each. */
void QUnit::SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high)
{
    bitLenInt i = low, j = high;
    if (i == (j - 1U)) {
        if (bits[j] < bits[i]) {
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); /* Change the global mapping. */
            std::swap(bits[i].mapped, bits[j].mapped); /* Change the contents of the sorting array. */
        }
        return;
    }
    QSortEntry pivot = bits[(low + high) / 2U];

    while (i <= j) {
        while (bits[i] < pivot) {
            ++i;
        }
        while (bits[j] > pivot) {
            --j;
        }
        if (i < j) {
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); /* Change the global mapping. */
            std::swap(bits[i].mapped, bits[j].mapped); /* Change the contents of the sorting array. */
            ++i;
            --j;
        } else if (i == j) {
            ++i;
            --j;
        }
    }
    if (low < j) {
        SortUnit(unit, bits, low, j);
    }
    if (i < high) {
        SortUnit(unit, bits, i, high);
    }
}

/// Check if all qubits in the range have cached probabilities indicating that they are in permutation basis
/// eigenstates, for optimization.
bool QUnit::CheckBitsPermutation(bitLenInt start, bitLenInt length)
{
    // Certain optimizations become obvious, if all bits in a range are in permutation basis eigenstates.
    // Then, operations can often be treated as classical, instead of quantum.

    ToPermBasisProb(start, length);
    for (bitLenInt i = 0U; i < length; ++i) {
        QEngineShard& shard = shards[start + i];
        if (!UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
            return false;
        }
    }

    return true;
}

/// Assuming all bits in the range are in cached |0>/|1> eigenstates, read the unsigned integer value of the range.
bitCapInt QUnit::GetCachedPermutation(bitLenInt start, bitLenInt length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0U; i < length; ++i) {
        if (SHARD_STATE(shards[start + i])) {
            res |= pow2(i);
        }
    }
    return res;
}

bitCapInt QUnit::GetCachedPermutation(const bitLenInt* bitArray, bitLenInt length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0U; i < length; ++i) {
        if (SHARD_STATE(shards[bitArray[i]])) {
            res |= pow2(i);
        }
    }
    return res;
}

bool QUnit::CheckBitsPlus(bitLenInt qubitIndex, bitLenInt length)
{
    bool isHBasis = true;
    for (bitLenInt i = 0U; i < length; ++i) {
        QEngineShard& shard = shards[qubitIndex + i];
        if (!CACHED_PLUS(shard)) {
            isHBasis = false;
            break;
        }
    }

    return isHBasis;
}

real1_f QUnit::ProbBase(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];

    if (shard.unit && (shard.unit->GetQubitCount() == 1U)) {
        RevertBasis1Qb(qubit);
        complex amps[2U];
        shard.unit->GetQuantumState(amps);

        if (IS_AMP_0(amps[0U] - amps[1U])) {
            shard.pauliBasis = PauliX;
            amps[0U] = amps[0U] / abs(amps[0U]);
            amps[1U] = ZERO_CMPLX;
        } else if (IS_AMP_0(amps[0U] + amps[1U])) {
            shard.pauliBasis = PauliX;
            amps[1U] = amps[0U] / abs(amps[0U]);
            amps[0U] = ZERO_CMPLX;
        } else if (IS_AMP_0((I_CMPLX * amps[0U]) - amps[1U])) {
            shard.pauliBasis = PauliY;
            amps[0U] = amps[0U] / abs(amps[0U]);
            amps[1U] = ZERO_CMPLX;
        } else if (IS_AMP_0((I_CMPLX * amps[0U]) + amps[1U])) {
            shard.pauliBasis = PauliY;
            amps[1U] = amps[0U] / abs(amps[0U]);
            amps[0U] = ZERO_CMPLX;
        }

        shard.amp0 = amps[0U];
        shard.amp1 = amps[1U];
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.unit = NULL;
        shard.mapped = 0U;
        shard.ClampAmps();

        return (real1_f)norm(shard.amp1);
    }

    if (!shard.unit || !shard.isProbDirty) {
        return clampProb((real1_f)norm(shard.amp1));
    }

    shard.isProbDirty = false;

    QInterfacePtr unit = shard.unit;
    bitLenInt mapped = shard.mapped;
    real1_f prob = unit->Prob(mapped);
    shard.amp1 = complex((real1)sqrt(prob), ZERO_R1);
    shard.amp0 = complex((real1)sqrt(ONE_R1 - prob), ZERO_R1);

    if (IS_NORM_0(shard.amp1)) {
        SeparateBit(false, qubit);
    } else if (IS_NORM_0(shard.amp0)) {
        SeparateBit(true, qubit);
    }

    return prob;
}

real1_f QUnit::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QUnit::Prob target parameter must be within allocated qubit bounds!");
    }

    ToPermBasisProb(qubit);
    return ProbBase(qubit);
}

real1_f QUnit::ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset)
{
    ThrowIfQbIdArrayIsBad(bits, length, qubitCount,
        "QUnit::ExpectationBitsAll parameter controls array values must be within allocated qubit bounds!");

    if ((length == 1U) || (shards[0U].GetQubitCount() != qubitCount)) {
        return QInterface::ExpectationBitsAll(bits, length, offset);
    }

    ToPermBasisProb();
    OrderContiguous(shards[0U].unit);

    return shards[0U].unit->ExpectationBitsAll(bits, length, offset);
}

real1_f QUnit::ProbAll(bitCapInt perm) { return clampProb((real1_f)norm(GetAmplitudeOrProb(perm, true))); }

void QUnit::PhaseParity(real1 radians, bitCapInt mask)
{
    if (mask >= maxQPower) {
        throw std::invalid_argument("QUnit::PhaseParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (!mask) {
        return;
    }

    complex phaseFac = complex((real1)cos(radians / 2), (real1)sin(radians / 2));

    if (!(mask & (mask - ONE_BCI))) {
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0U; i < (bitLenInt)qIndices.size(); ++i) {
        QEngineShard& shard = shards[qIndices[i]];

        if (UNSAFE_CACHED_ZERO(shard)) {
            continue;
        }

        if (UNSAFE_CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (!eIndices.size()) {
        if (flipResult) {
            Phase(phaseFac, phaseFac, 0U);
        } else {
            Phase(ONE_CMPLX / phaseFac, ONE_CMPLX / phaseFac, 0U);
        }
        return;
    }

    if (eIndices.size() == 1U) {
        if (flipResult) {
            Phase(phaseFac, ONE_CMPLX / phaseFac, log2(mask));
        } else {
            Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        }
        return;
    }

    QInterfacePtr unit = Entangle(eIndices);

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i].unit == unit) {
            shards[i].MakeDirty();
        }
    }

    bitCapInt mappedMask = 0U;
    for (bitLenInt i = 0U; i < (bitLenInt)eIndices.size(); ++i) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    unit->PhaseParity((real1_f)(flipResult ? -radians : radians), mappedMask);
}

real1_f QUnit::ProbParity(bitCapInt mask)
{
    if (mask >= maxQPower) {
        throw std::invalid_argument("QUnit::ProbParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (!mask) {
        return ZERO_R1_F;
    }

    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));

        RevertBasis2Qb(qIndices.back(), ONLY_INVERT, ONLY_TARGETS);

        QEngineShard& shard = shards[qIndices.back()];
        if (shard.unit && QUEUED_PHASE(shard)) {
            RevertBasis1Qb(qIndices.back());
        }
    }

    std::map<QInterfacePtr, bitCapInt> units;
    real1 oddChance = ZERO_R1;
    real1 nOddChance;
    for (bitLenInt i = 0U; i < (bitLenInt)qIndices.size(); ++i) {
        QEngineShard& shard = shards[qIndices[i]];
        if (!(shard.unit)) {
            nOddChance = (shard.pauliBasis != PauliZ) ? norm(SQRT1_2_R1 * (shard.amp0 - shard.amp1)) : shard.Prob();
            oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
            continue;
        }

        RevertBasis1Qb(qIndices[i]);

        units[shard.unit] |= pow2(shard.mapped);
    }

    if (!qIndices.size()) {
        return (real1_f)oddChance;
    }

    std::map<QInterfacePtr, bitCapInt>::iterator unit;
    for (unit = units.begin(); unit != units.end(); ++unit) {
        nOddChance = std::dynamic_pointer_cast<QParity>(unit->first)->ProbParity(unit->second);
        oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
    }

    return (real1_f)oddChance;
}

bool QUnit::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    if (mask >= maxQPower) {
        throw std::invalid_argument("QUnit::ForceMParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (!mask) {
        return false;
    }

    if (!(mask & (mask - ONE_BCI))) {
        return ForceM(log2(mask), result, doForce);
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0U; i < (bitLenInt)qIndices.size(); ++i) {
        QEngineShard& shard = shards[qIndices[i]];

        if (UNSAFE_CACHED_ZERO(shard)) {
            continue;
        }

        if (UNSAFE_CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (!eIndices.size()) {
        return flipResult;
    }

    if (eIndices.size() == 1U) {
        return flipResult ^ ForceM(eIndices[0U], result ^ flipResult, doForce);
    }

    QInterfacePtr unit = Entangle(eIndices);

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i].unit == unit) {
            shards[i].MakeDirty();
        }
    }

    bitCapInt mappedMask = 0U;
    for (bitLenInt i = 0U; i < (bitLenInt)eIndices.size(); ++i) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    return flipResult ^
        (std::dynamic_pointer_cast<QParity>(unit)->ForceMParity(mappedMask, result ^ flipResult, doForce));
}

void QUnit::CUniformParityRZ(const bitLenInt* cControls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
{
    if (mask >= maxQPower) {
        throw std::invalid_argument("QUnit::CUniformParityRZ mask out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(cControls, controlLen, qubitCount,
        "QUnit::CUniformParityRZ parameter controls array values must be within allocated qubit bounds!");

    std::vector<bitLenInt> controls;
    if (TrimControls(cControls, controlLen, controls, false)) {
        return;
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0U; i < (bitLenInt)qIndices.size(); ++i) {
        ToPermBasis(qIndices[i]);
        QEngineShard& shard = shards[qIndices[i]];

        if (CACHED_ZERO(shard)) {
            continue;
        }

        if (CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (!eIndices.size()) {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac;
        if (flipResult) {
            phaseFac = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, -sine);
        }
        if (!controls.size()) {
            return Phase(phaseFac, phaseFac, 0U);
        } else {
            return MCPhase(&(controls[0U]), controls.size(), phaseFac, phaseFac, 0U);
        }
    }

    if (eIndices.size() == 1U) {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac, phaseFacAdj;
        if (flipResult) {
            phaseFac = complex(cosine, -sine);
            phaseFacAdj = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, sine);
            phaseFacAdj = complex(cosine, -sine);
        }
        if (!controls.size()) {
            return Phase(phaseFacAdj, phaseFac, eIndices[0U]);
        } else {
            return MCPhase(&(controls[0U]), controls.size(), phaseFacAdj, phaseFac, eIndices[0U]);
        }
    }

    for (bitLenInt i = 0U; i < (bitLenInt)eIndices.size(); ++i) {
        shards[eIndices[i]].isPhaseDirty = true;
    }

    QInterfacePtr unit = Entangle(eIndices);

    bitCapInt mappedMask = 0U;
    for (bitLenInt i = 0U; i < (bitLenInt)eIndices.size(); ++i) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    if (!controls.size()) {
        std::dynamic_pointer_cast<QParity>(unit)->UniformParityRZ(mappedMask, flipResult ? -angle : angle);
    } else {
        std::vector<bitLenInt*> ebits(controls.size());
        for (bitLenInt i = 0U; i < (bitLenInt)controls.size(); ++i) {
            ebits[i] = &controls[i];
        }

        Entangle(ebits);
        unit = Entangle({ controls[0U], eIndices[0U] });

        std::vector<bitLenInt> controlsMapped(controls.size());
        for (bitLenInt i = 0U; i < (bitLenInt)controls.size(); ++i) {
            QEngineShard& cShard = shards[controls[i]];
            controlsMapped[i] = cShard.mapped;
            cShard.isPhaseDirty = true;
        }

        std::dynamic_pointer_cast<QParity>(unit)->CUniformParityRZ(
            &(controlsMapped[0U]), controlsMapped.size(), mappedMask, flipResult ? -angle : angle);
    }
}

bool QUnit::SeparateBit(bool value, bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];
    QInterfacePtr unit = shard.unit;
    bitLenInt mapped = shard.mapped;

    if (unit && unit->isClifford() && !unit->TrySeparate(mapped)) {
        // This conditional coaxes the unit into separable form, so this should never actually happen.
        return false;
    }

    shard.unit = NULL;
    shard.mapped = 0U;
    shard.isProbDirty = false;
    shard.isPhaseDirty = false;
    shard.amp0 = value ? ZERO_CMPLX : GetNonunitaryPhase();
    shard.amp1 = value ? GetNonunitaryPhase() : ZERO_CMPLX;

    if (!unit || (unit->GetQubitCount() == 1U)) {
        return true;
    }

    real1_f prob = unit->Prob(shard.mapped);
    unit->Dispose(mapped, 1U, value ? ONE_BCI : 0U);

    prob = ONE_R1_F / 2 - prob;
    if (!unit->isBinaryDecisionTree() && ((ONE_R1 / 2 - abs(prob)) > FP_NORM_EPSILON)) {
        unit->UpdateRunningNorm();
        if (!doNormalize) {
            unit->NormalizeState();
        }
    }

    /* Update the mappings. */
    for (auto&& s : shards) {
        if ((s.unit == unit) && (s.mapped > mapped)) {
            --(s.mapped);
        }
    }

    if (unit->GetQubitCount() != 1U) {
        return true;
    }

    for (bitLenInt partnerIndex = 0U; partnerIndex < qubitCount; ++partnerIndex) {
        QEngineShard& partnerShard = shards[partnerIndex];
        if (unit == partnerShard.unit) {
            ProbBase(partnerIndex);
            break;
        }
    }

    return true;
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce, bool doApply)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QUnit::ForceM target parameter must be within allocated qubit bounds!");
    }

    if (doApply) {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
    } else {
        ToPermBasisMeasure(qubit);
    }

    QEngineShard& shard = shards[qubit];

    bool result;
    if (!shard.unit) {
        real1_f prob = (real1_f)norm(shard.amp1);
        if (doForce) {
            result = res;
        } else if (prob >= ONE_R1) {
            result = true;
        } else if (prob <= ZERO_R1) {
            result = false;
        } else {
            result = (Rand() <= prob);
        }
    } else {
        // ALWAYS collapse unit before Decompose()/Dispose(), for maximum consistency.
        result = shard.unit->ForceM(shard.mapped, res, doForce, doApply);
    }

    if (!doApply) {
        return result;
    }

    shard.isProbDirty = false;
    shard.isPhaseDirty = false;
    shard.amp0 = result ? ZERO_CMPLX : GetNonunitaryPhase();
    shard.amp1 = result ? GetNonunitaryPhase() : ZERO_CMPLX;

    if (shard.GetQubitCount() == 1U) {
        shard.unit = NULL;
        shard.mapped = 0U;
        if (result) {
            Flush1Eigenstate(qubit);
        } else {
            Flush0Eigenstate(qubit);
        }
        return result;
    }

    // This is critical: it's the "nonlocal correlation" of "wave function collapse".
    if (shard.unit) {
        for (bitLenInt i = 0U; i < qubit; ++i) {
            if (shards[i].unit == shard.unit) {
                shards[i].MakeDirty();
            }
        }
        for (bitLenInt i = qubit + 1U; i < qubitCount; ++i) {
            if (shards[i].unit == shard.unit) {
                shards[i].MakeDirty();
            }
        }
        SeparateBit(result, qubit);
    }

    if (result) {
        Flush1Eigenstate(qubit);
    } else {
        Flush0Eigenstate(qubit);
    }

    return result;
}

bitCapInt QUnit::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::ForceMReg range is out-of-bounds!");
    }

    if (!doForce && doApply && (length == qubitCount)) {
        return MAll();
    }

    // This will discard all buffered gates that don't affect Z basis probability,
    // so it's safe to call ToPermBasis() without performance penalty, afterward.
    if (!doApply) {
        ToPermBasisMeasure(start, length);
    }

    return QInterface::ForceMReg(start, length, result, doForce, doApply);
}

bitCapInt QUnit::MAll()
{
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        RevertBasis1Qb(i);
    }
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        QEngineShard& shard = shards[i];
        shard.DumpPhaseBuffers();
        shard.ClearInvertPhase();
    }

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i].IsInvertControl()) {
            // Measurement commutes with control
            M(i);
        }
    }

    bitCapInt toRet = 0U;

    std::vector<QInterfacePtr> units;
    std::map<QInterfacePtr, bitCapInt> partResult;

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        QInterfacePtr toFind = shards[i].unit;
        if (!toFind) {
            real1_f prob = (real1_f)norm(shards[i].amp1);
            if ((prob >= ONE_R1) || ((prob > ZERO_R1) && (Rand() <= prob))) {
                shards[i].amp0 = ZERO_CMPLX;
                shards[i].amp1 = GetNonunitaryPhase();
                toRet |= pow2(i);
            } else {
                shards[i].amp0 = GetNonunitaryPhase();
                shards[i].amp1 = ZERO_CMPLX;
            }
        } else if (M(i)) {
            toRet |= pow2(i);
        }
    }

    SetPermutation(toRet);

    return toRet;
}

std::map<bitCapInt, int> QUnit::MultiShotMeasureMask(const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    ToPermBasisProb();

    bitLenInt index;
    std::vector<bitLenInt> qIndices(qPowerCount);
    std::map<bitLenInt, bitCapInt> iQPowers;
    for (bitLenInt i = 0U; i < qPowerCount; ++i) {
        index = log2(qPowers[i]);
        qIndices[i] = index;
        iQPowers[index] = pow2(i);
    }

    ThrowIfQbIdArrayIsBad(&(qIndices[0]), qPowerCount, qubitCount,
        "QInterface::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");

    std::map<QInterfacePtr, std::vector<bitCapInt>> subQPowers;
    std::map<QInterfacePtr, std::vector<bitCapInt>> subIQPowers;
    std::vector<bitLenInt> singleBits;

    for (bitLenInt i = 0U; i < qPowerCount; ++i) {
        index = qIndices[i];
        QEngineShard& shard = shards[index];

        if (!shard.unit) {
            singleBits.push_back(index);
            continue;
        }

        subQPowers[shard.unit].push_back(pow2(shard.mapped));
        subIQPowers[shard.unit].push_back(iQPowers[index]);
    }

    std::map<bitCapInt, int> combinedResults;
    combinedResults[0U] = (int)shots;

    for (auto subQPowersIt = subQPowers.begin(); subQPowersIt != subQPowers.end(); ++subQPowersIt) {
        QInterfacePtr unit = subQPowersIt->first;
        std::map<bitCapInt, int> unitResults =
            unit->MultiShotMeasureMask(&(subQPowersIt->second[0U]), subQPowersIt->second.size(), shots);
        std::map<bitCapInt, int> topLevelResults;
        for (auto mapIter = unitResults.begin(); mapIter != unitResults.end(); ++mapIter) {
            bitCapInt mask = 0U;
            for (bitLenInt i = 0U; i < (bitLenInt)subQPowersIt->second.size(); ++i) {
                if ((mapIter->first >> i) & 1U) {
                    mask |= subIQPowers[unit][i];
                }
            }
            topLevelResults[mask] = mapIter->second;
        }
        // Release unitResults memory:
        unitResults = std::map<bitCapInt, int>();

        // If either map is fully |0>, nothing changes (after the swap).
        if (!topLevelResults.begin()->first && (topLevelResults[0U] == (int)shots)) {
            continue;
        }
        if (!combinedResults.begin()->first && (combinedResults[0U] == (int)shots)) {
            std::swap(topLevelResults, combinedResults);
            continue;
        }

        // Swap if needed, so topLevelResults.size() is smaller.
        if (combinedResults.size() < topLevelResults.size()) {
            std::swap(topLevelResults, combinedResults);
        }
        // (Since swapped...)

        std::map<bitCapInt, int> nCombinedResults;

        // If either map has exactly 1 key, (therefore with `shots` value,) pass it through without a "shuffle."
        if (topLevelResults.size() == 1U) {
            auto pickIter = topLevelResults.begin();
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); ++mapIter) {
                nCombinedResults[mapIter->first | pickIter->first] = mapIter->second;
            }
            combinedResults = nCombinedResults;
            continue;
        }

        // ... Otherwise, we've committed to simulating a random pairing selection from either side, (but
        // `topLevelResults` has fewer or the same count of keys).
        int shotsLeft = shots;
        for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); ++mapIter) {
            for (int shot = 0; shot < mapIter->second; ++shot) {
                int pick = (int)(shotsLeft * Rand());
                if (shotsLeft <= pick) {
                    pick = shotsLeft - 1;
                }
                --shotsLeft;

                auto pickIter = topLevelResults.begin();
                int count = pickIter->second;
                while (pick > count) {
                    ++pickIter;
                    count += pickIter->second;
                }

                ++(nCombinedResults[mapIter->first | pickIter->first]);

                --(pickIter->second);
                if (!pickIter->second) {
                    topLevelResults.erase(pickIter);
                }
            }
        }
        combinedResults = nCombinedResults;
    }

    for (bitLenInt i = 0U; i < (bitLenInt)singleBits.size(); ++i) {
        index = singleBits[i];

        real1_f prob = clampProb((real1_f)norm(shards[index].amp1));
        if (prob == ZERO_R1) {
            continue;
        }

        std::map<bitCapInt, int> nCombinedResults;
        if (prob == ONE_R1) {
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); ++mapIter) {
                nCombinedResults[mapIter->first | iQPowers[index]] = mapIter->second;
            }
        } else {
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); ++mapIter) {
                bitCapInt zeroPerm = mapIter->first;
                bitCapInt onePerm = mapIter->first | iQPowers[index];
                for (int shot = 0; shot < mapIter->second; ++shot) {
                    if (Rand() > prob) {
                        ++(nCombinedResults[zeroPerm]);
                    } else {
                        ++(nCombinedResults[onePerm]);
                    }
                }
            }
        }
        combinedResults = nCombinedResults;
    }

    return combinedResults;
}

void QUnit::MultiShotMeasureMask(
    const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    ToPermBasisProb();

    QInterfacePtr unit = shards[log2(qPowers[0U])].unit;
    if (unit) {
        std::unique_ptr<bitCapInt[]> mappedIndices(new bitCapInt[qPowerCount]);
        for (bitLenInt j = 0U; j < qubitCount; ++j) {
            if (qPowers[0U] == pow2(j)) {
                mappedIndices[0U] = pow2(shards[j].mapped);
                break;
            }
        }
        for (bitLenInt i = 1U; i < qPowerCount; ++i) {
            const size_t qubit = log2(qPowers[i]);
            if (qubit >= qubitCount) {
                throw std::invalid_argument("QUnit::MultiShotMeasureMask parameter qPowers array values must be within "
                                            "allocated qubit bounds!");
            }
            if (unit != shards[qubit].unit) {
                unit = NULL;
                break;
            }
            for (bitLenInt j = 0U; j < qubitCount; ++j) {
                if (qPowers[i] == pow2(j)) {
                    mappedIndices[i] = pow2(shards[j].mapped);
                    break;
                }
            }
        }

        if (unit) {
            unit->MultiShotMeasureMask(mappedIndices.get(), qPowerCount, shots, shotsArray);
            return;
        }
    }

    std::map<bitCapInt, int> results = MultiShotMeasureMask(qPowers, qPowerCount, shots);

    size_t j = 0U;
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end() && (j < shots)) {
        for (int i = 0; i < it->second; ++i) {
            shotsArray[j] = (unsigned)it->first;
            ++j;
        }

        ++it;
    }
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    for (bitLenInt i = 0U; i < length; ++i) {
        bool bitState = ((value >> (bitCapIntOcl)i) & ONE_BCI) != 0U;
        shards[i + start] = QEngineShard(bitState, GetNonunitaryPhase());
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::Swap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::Swap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit1 == qubit2) {
        return;
    }

    // Simply swap the bit mapping.
    shards.swap(qubit1, qubit2);
}

void QUnit::EitherISwap(bitLenInt qubit1, bitLenInt qubit2, bool isInverse)
{
    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::EitherISwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::EitherISwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit1 == qubit2) {
        return;
    }

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (IS_SAME_UNIT(shard1, shard2) || ARE_CLIFFORD(shard1, shard2)) {
        QInterfacePtr unit = Entangle({ qubit1, qubit2 });
        if (isInverse) {
            unit->IISwap(shard1.mapped, shard2.mapped);
        } else {
            unit->ISwap(shard1.mapped, shard2.mapped);
        }
        shard1.MakeDirty();
        shard2.MakeDirty();
        return;
    }

    if (isInverse) {
        QInterface::IISwap(qubit1, qubit2);
    } else {
        QInterface::ISwap(qubit1, qubit2);
    }
}

void QUnit::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::SqrtSwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::SqrtSwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    Entangle({ qubit1, qubit2 })->SqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::ISqrtSwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::ISqrtSwap qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    Entangle({ qubit1, qubit2 })->ISqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    bitLenInt controls[1U] = { qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if (IS_0_R1(sinTheta)) {
        MCPhase(controls, 1U, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    if (IS_1_R1(-sinTheta)) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, 1U, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    if (qubit1 >= qubitCount) {
        throw std::invalid_argument("QUnit::FSim qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubit2 >= qubitCount) {
        throw std::invalid_argument("QUnit::FSim qubit index parameter must be within allocated qubit bounds!");
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    Entangle({ qubit1, qubit2 })->FSim(theta, phi, shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    // If there are no controls, this is equivalent to the single bit gate.
    if (!controlLen) {
        Mtrx(mtrxs, qubitIndex);
        return;
    }

    if (qubitIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(
        controls, controlLen, qubitCount, "QUnit::UniformlyControlledSingleBit control is out-of-bounds!");

    std::vector<bitLenInt> trimmedControls;
    std::vector<bitCapInt> skipPowers;
    bitCapInt skipValueMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        if (!CheckBitsPermutation(controls[i])) {
            trimmedControls.push_back(controls[i]);
        } else {
            skipPowers.push_back(pow2(i));
            skipValueMask |= (SHARD_STATE(shards[controls[i]]) ? pow2(i) : 0U);
        }
    }

    // If all controls are in eigenstates, we can avoid entangling them.
    if (!trimmedControls.size()) {
        bitCapInt controlPerm = GetCachedPermutation(controls, controlLen);
        complex mtrx[4U];
        std::copy(
            mtrxs + (bitCapIntOcl)(controlPerm * 4UL), mtrxs + (bitCapIntOcl)((controlPerm + ONE_BCI) * 4U), mtrx);
        Mtrx(mtrx, qubitIndex);
        return;
    }

    std::vector<bitLenInt> bits(trimmedControls.size() + 1U);
    for (bitLenInt i = 0U; i < (bitLenInt)trimmedControls.size(); ++i) {
        bits[i] = trimmedControls[i];
    }
    bits[trimmedControls.size()] = qubitIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(trimmedControls.size() + 1U);
    for (bitLenInt i = 0U; i < (bitLenInt)bits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    std::unique_ptr<bitLenInt[]> mappedControls(new bitLenInt[trimmedControls.size()]);
    for (bitLenInt i = 0U; i < (bitLenInt)trimmedControls.size(); ++i) {
        mappedControls[i] = shards[trimmedControls[i]].mapped;
        shards[trimmedControls[i]].isPhaseDirty = true;
    }

    unit->UniformlyControlledSingleBit(mappedControls.get(), trimmedControls.size(), shards[qubitIndex].mapped, mtrxs,
        &(skipPowers[0U]), skipPowers.size(), skipValueMask);

    shards[qubitIndex].MakeDirty();
}

void QUnit::H(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::H qubit index parameter must be within allocated qubit bounds!");
    }

    RevertBasisY(target);
    CommuteH(target);

    QEngineShard& shard = shards[target];
    shard.pauliBasis = (shard.pauliBasis == PauliZ) ? PauliX : PauliZ;
}

void QUnit::S(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::S qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    shard.CommutePhase(ONE_CMPLX, I_CMPLX);

    if (shard.pauliBasis == PauliY) {
        shard.pauliBasis = PauliX;
        XBase(target);
        return;
    }

    if (shard.pauliBasis == PauliX) {
        shard.pauliBasis = PauliY;
        return;
    }

    if (shard.unit) {
        shard.unit->S(shard.mapped);
    }

    shard.amp1 = I_CMPLX * shard.amp1;
}

void QUnit::IS(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::IS qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    shard.CommutePhase(ONE_CMPLX, -I_CMPLX);

    if (shard.pauliBasis == PauliY) {
        shard.pauliBasis = PauliX;
        return;
    }

    if (shard.pauliBasis == PauliX) {
        shard.pauliBasis = PauliY;
        XBase(target);
        return;
    }

    if (shard.unit) {
        shard.unit->IS(shard.mapped);
    }

    shard.amp1 = -I_CMPLX * shard.amp1;
}

void QUnit::XBase(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::XBase qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->X(shard.mapped);
    }

    std::swap(shard.amp0, shard.amp1);
}

void QUnit::YBase(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::YBase qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Y(shard.mapped);
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = -I_CMPLX * shard.amp1;
    shard.amp1 = I_CMPLX * Y0;
}

void QUnit::ZBase(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::ZBase qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Z(shard.mapped);
    }

    shard.amp1 = -shard.amp1;
}

void QUnit::TransformX2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] + mtrxIn[1U] + mtrxIn[2U] + mtrxIn[3U]);
    mtrxOut[1U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] - mtrxIn[1U] + mtrxIn[2U] - mtrxIn[3U]);
    mtrxOut[2U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] + mtrxIn[1U] - mtrxIn[2U] - mtrxIn[3U]);
    mtrxOut[3U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] - mtrxIn[1U] - mtrxIn[2U] + mtrxIn[3U]);
}

void QUnit::TransformXInvert(complex topRight, complex bottomLeft, complex* mtrxOut)
{
    mtrxOut[0U] = (real1)(ONE_R1 / 2) * (complex)(topRight + bottomLeft);
    mtrxOut[1U] = (real1)(ONE_R1 / 2) * (complex)(-topRight + bottomLeft);
    mtrxOut[2U] = -mtrxOut[1U];
    mtrxOut[3U] = -mtrxOut[0U];
}

void QUnit::TransformY2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] + I_CMPLX * (mtrxIn[1U] - mtrxIn[2U]) + mtrxIn[3U]);
    mtrxOut[1U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] - I_CMPLX * (mtrxIn[1U] + mtrxIn[2U]) - mtrxIn[3U]);
    mtrxOut[2U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] + I_CMPLX * (mtrxIn[1U] + mtrxIn[2U]) - mtrxIn[3U]);
    mtrxOut[3U] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0U] - I_CMPLX * (mtrxIn[1U] - mtrxIn[2U]) + mtrxIn[3U]);
}

void QUnit::TransformYInvert(complex topRight, complex bottomLeft, complex* mtrxOut)
{
    mtrxOut[0U] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(topRight - bottomLeft);
    mtrxOut[1U] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(-topRight - bottomLeft);
    mtrxOut[2U] = -mtrxOut[1U];
    mtrxOut[3U] = -mtrxOut[0U];
}

void QUnit::TransformPhase(complex topLeft, complex bottomRight, complex* mtrxOut)
{
    mtrxOut[0U] = (real1)(ONE_R1 / 2) * (complex)(topLeft + bottomRight);
    mtrxOut[1U] = (real1)(ONE_R1 / 2) * (complex)(topLeft - bottomRight);
    mtrxOut[2U] = mtrxOut[1U];
    mtrxOut[3U] = mtrxOut[0U];
}

#define CTRLED_GEN_WRAP(ctrld)                                                                                         \
    ApplyEitherControlled(                                                                                             \
        controlVec, { target },                                                                                        \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            complex trnsMtrx[4U] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                 \
            if (shards[target].pauliBasis == PauliX) {                                                                 \
                TransformX2x2(mtrx, trnsMtrx);                                                                         \
            } else if (shards[target].pauliBasis == PauliY) {                                                          \
                TransformY2x2(mtrx, trnsMtrx);                                                                         \
            } else {                                                                                                   \
                std::copy(mtrx, mtrx + 4U, trnsMtrx);                                                                  \
            }                                                                                                          \
            unit->ctrld;                                                                                               \
        },                                                                                                             \
        false);

#define CTRLED_PHASE_INVERT_WRAP(ctrld, ctrldgen, isInvert, top, bottom)                                               \
    ApplyEitherControlled(                                                                                             \
        controlVec, { target },                                                                                        \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            if (shards[target].pauliBasis == PauliX) {                                                                 \
                complex trnsMtrx[4U] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                             \
                if (isInvert) {                                                                                        \
                    TransformXInvert(top, bottom, trnsMtrx);                                                           \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            } else if (shards[target].pauliBasis == PauliY) {                                                          \
                complex trnsMtrx[4U] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                             \
                if (isInvert) {                                                                                        \
                    TransformYInvert(top, bottom, trnsMtrx);                                                           \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            } else {                                                                                                   \
                unit->ctrld;                                                                                           \
            }                                                                                                          \
        },                                                                                                             \
        !isInvert);

#define CTRLED_SWAP_WRAP(ctrld, bare, anti)                                                                            \
    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,                                                            \
        "QUnit Swap variant parameter controls array values must be within allocated qubit bounds!");                  \
    if (qubit1 >= qubitCount) {                                                                                        \
        throw std::invalid_argument(                                                                                   \
            "QUnit Swap variant qubit index parameter must be within allocated qubit bounds!");                        \
    }                                                                                                                  \
    if (qubit2 >= qubitCount) {                                                                                        \
        throw std::invalid_argument(                                                                                   \
            "QUnit Swap variant qubit index parameter must be within allocated qubit bounds!");                        \
    }                                                                                                                  \
    if (qubit1 == qubit2) {                                                                                            \
        return;                                                                                                        \
    }                                                                                                                  \
    std::vector<bitLenInt> controlVec;                                                                                 \
    if (TrimControls(controls, controlLen, controlVec, anti)) {                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    if (!controlVec.size()) {                                                                                          \
        bare;                                                                                                          \
        return;                                                                                                        \
    }                                                                                                                  \
    ApplyEitherControlled(                                                                                             \
        controlVec, { qubit1, qubit2 },                                                                                \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, false)
#define CTRL_GEN_ARGS &(mappedControls[0U]), mappedControls.size(), trnsMtrx, shards[target].mapped
#define CTRL_S_ARGS &(mappedControls[0U]), mappedControls.size(), shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS &(mappedControls[0U]), mappedControls.size(), topLeft, bottomRight, shards[target].mapped
#define CTRL_I_ARGS &(mappedControls[0U]), mappedControls.size(), topRight, bottomLeft, shards[target].mapped

void QUnit::Phase(complex topLeft, complex bottomRight, bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::Phase qubit index parameter must be within allocated qubit bounds!");
    }

    if (randGlobalPhase || IS_1_CMPLX(topLeft)) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            return;
        }

        if (IS_NORM_0((I_CMPLX * topLeft) - bottomRight)) {
            S(target);
            return;
        }

        if (IS_NORM_0((I_CMPLX * topLeft) + bottomRight)) {
            IS(target);
            return;
        }
    }

    QEngineShard& shard = shards[target];

    shard.CommutePhase(topLeft, bottomRight);

    if (shard.pauliBasis == PauliZ) {
        if (shard.unit) {
            shard.unit->Phase(topLeft, bottomRight, shard.mapped);
        }

        shard.amp0 *= topLeft;
        shard.amp1 *= bottomRight;

        return;
    }

    complex mtrx[4U] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
    TransformPhase(topLeft, bottomRight, mtrx);

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }

    if (DIRTY(shard)) {
        shard.isProbDirty |= !IS_PHASE_OR_INVERT(mtrx);
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(target);
}

void QUnit::Invert(complex topRight, complex bottomLeft, bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::Invert qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];

    shard.FlipPhaseAnti();
    shard.CommutePhase(topRight, bottomLeft);

    if (shard.pauliBasis == PauliZ) {
        if (shard.unit) {
            shard.unit->Invert(topRight, bottomLeft, shard.mapped);
        }

        const complex tempAmp1 = bottomLeft * shard.amp0;
        shard.amp0 = topRight * shard.amp1;
        shard.amp1 = tempAmp1;

        return;
    }

    complex mtrx[4U] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
    if (shard.pauliBasis == PauliX) {
        TransformXInvert(topRight, bottomLeft, mtrx);
    } else {
        TransformYInvert(topRight, bottomLeft, mtrx);
    }

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }

    if (DIRTY(shard)) {
        shard.isProbDirty |= !IS_PHASE_OR_INVERT(mtrx);
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(target);
}

void QUnit::MCPhase(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    ThrowIfQbIdArrayIsBad(lControls, lControlLen, qubitCount,
        "QUnit::MCPhase parameter controls array values must be within allocated qubit bounds!");

    if (IS_1_CMPLX(topLeft) && IS_1_CMPLX(bottomRight)) {
        return;
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, lControlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if ((controlVec.size() == 1U) && IS_NORM_0(topLeft - bottomRight)) {
        Phase(ONE_CMPLX, bottomRight, controlVec[0U]);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MCPhase qubit index parameter must be within allocated qubit bounds!");
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !((IS_SAME(ONE_CMPLX, topLeft) || IS_SAME(-ONE_CMPLX, topLeft)) &&
                    (IS_SAME(ONE_CMPLX, bottomRight) || IS_SAME(-ONE_CMPLX, bottomRight))))) {
            tShard.AddPhaseAngles(&cShard, topLeft, bottomRight);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MCPhase(CTRL_P_ARGS), MCMtrx(CTRL_GEN_ARGS), false, topLeft, bottomRight);
}

void QUnit::MACPhase(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    ThrowIfQbIdArrayIsBad(lControls, lControlLen, qubitCount,
        "QUnit::MACPhase parameter controls array values must be within allocated qubit bounds!");

    if (IS_1_CMPLX(topLeft) && IS_1_CMPLX(bottomRight)) {
        return;
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, lControlLen, controlVec, true)) {
        return;
    }

    if (!controlVec.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if ((controlVec.size() == 1U) && IS_NORM_0(topLeft - bottomRight)) {
        Phase(topLeft, ONE_CMPLX, controlVec[0U]);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MACPhase qubit index parameter must be within allocated qubit bounds!");
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !((IS_SAME(ONE_CMPLX, topLeft) || IS_SAME(-ONE_CMPLX, topLeft)) &&
                    (IS_SAME(ONE_CMPLX, bottomRight) || IS_SAME(-ONE_CMPLX, bottomRight))))) {
            tShard.AddAntiPhaseAngles(&cShard, bottomRight, topLeft);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MACPhase(CTRL_P_ARGS), MACMtrx(CTRL_GEN_ARGS), false, topLeft, bottomRight);
}

void QUnit::MCInvert(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    ThrowIfQbIdArrayIsBad(lControls, lControlLen, qubitCount,
        "QUnit::MCInvert parameter controls array values must be within allocated qubit bounds!");

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MCInvert qubit index parameter must be within allocated qubit bounds!");
    }

    if (IS_1_CMPLX(topRight) && IS_1_CMPLX(bottomLeft)) {
        QEngineShard& tShard = shards[target];
        if (CACHED_PLUS(tShard)) {
            return;
        }
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, lControlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        const bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !(((IS_SAME(ONE_CMPLX, topRight) || IS_SAME(-ONE_CMPLX, topRight)) &&
                      (IS_SAME(ONE_CMPLX, bottomLeft) || IS_SAME(-ONE_CMPLX, bottomLeft))) ||
                    (((IS_SAME(I_CMPLX, topRight) || IS_SAME(-I_CMPLX, topRight)) &&
                        (IS_SAME(I_CMPLX, bottomLeft) || IS_SAME(-I_CMPLX, bottomLeft))))))) {
            tShard.AddInversionAngles(&cShard, topRight, bottomLeft);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MCInvert(CTRL_I_ARGS), MCMtrx(CTRL_GEN_ARGS), true, topRight, bottomLeft);
}

void QUnit::MACInvert(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    ThrowIfQbIdArrayIsBad(lControls, lControlLen, qubitCount,
        "QUnit::MACInvert parameter controls array values must be within allocated qubit bounds!");

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MACInvert qubit index parameter must be within allocated qubit bounds!");
    }

    if (IS_1_CMPLX(topRight) && IS_1_CMPLX(bottomLeft)) {
        QEngineShard& tShard = shards[target];
        if (CACHED_PLUS(tShard)) {
            return;
        }
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, lControlLen, controlVec, true)) {
        return;
    }

    if (!controlVec.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        const bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !(((IS_SAME(ONE_CMPLX, topRight) || IS_SAME(-ONE_CMPLX, topRight)) &&
                      (IS_SAME(ONE_CMPLX, bottomLeft) || IS_SAME(-ONE_CMPLX, bottomLeft))) ||
                    (((IS_SAME(I_CMPLX, topRight) || IS_SAME(-I_CMPLX, topRight)) &&
                        (IS_SAME(I_CMPLX, bottomLeft) || IS_SAME(-I_CMPLX, bottomLeft))))))) {
            tShard.AddAntiInversionAngles(&cShard, bottomLeft, topRight);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MACInvert(CTRL_I_ARGS), MACMtrx(CTRL_GEN_ARGS), true, topRight, bottomLeft);
}

void QUnit::Mtrx(const complex* mtrx, bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        Phase(mtrx[0U], mtrx[3U], target);
        return;
    }
    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        Invert(mtrx[1U], mtrx[2U], target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0U], (complex)SQRT1_2_R1)) && IS_SAME(mtrx[0U], mtrx[1U]) &&
        IS_SAME(mtrx[0U], mtrx[2U]) && IS_SAME(mtrx[0U], -mtrx[3U])) {
        H(target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0U], (complex)SQRT1_2_R1)) && IS_SAME(mtrx[0U], mtrx[1U]) &&
        IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U]) && IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U])) {
        H(target);
        S(target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0U], (complex)SQRT1_2_R1)) && IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) &&
        IS_SAME(mtrx[0U], mtrx[2U]) && IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U])) {
        IS(target);
        H(target);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::Mtrx qubit index parameter must be within allocated qubit bounds!");
    }

    RevertBasis2Qb(target);

    complex trnsMtrx[4U];
    if (shard.pauliBasis == PauliY) {
        TransformY2x2(mtrx, trnsMtrx);
    } else if (shard.pauliBasis == PauliX) {
        TransformX2x2(mtrx, trnsMtrx);
    } else {
        std::copy(mtrx, mtrx + 4U, trnsMtrx);
    }

    if (shard.unit) {
        shard.unit->Mtrx(trnsMtrx, shard.mapped);
    }

    if (DIRTY(shard)) {
        shard.isProbDirty |= !IS_PHASE_OR_INVERT(trnsMtrx);
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = (trnsMtrx[0U] * Y0) + (trnsMtrx[1U] * shard.amp1);
    shard.amp1 = (trnsMtrx[2U] * Y0) + (trnsMtrx[3U] * shard.amp1);
    ClampShard(target);
}

void QUnit::MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(controls, controlLen, mtrx[1U], mtrx[2U], target);
        return;
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::MCMtrx parameter controls array values must be within allocated qubit bounds!");

    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        Mtrx(mtrx, target);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MCMtrx qubit index parameter must be within allocated qubit bounds!");
    }

    CTRLED_GEN_WRAP(MCMtrx(CTRL_GEN_ARGS));
}

void QUnit::MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(controls, controlLen, mtrx[1U], mtrx[2U], target);
        return;
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::MACMtrx parameter controls array values must be within allocated qubit bounds!");

    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, true)) {
        return;
    }

    if (!controlVec.size()) {
        Mtrx(mtrx, target);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MACMtrx qubit index parameter must be within allocated qubit bounds!");
    }

    CTRLED_GEN_WRAP(MACMtrx(CTRL_GEN_ARGS));
}

void QUnit::CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), false);
}

void QUnit::AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), true);
}

void QUnit::CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), true);
}

void QUnit::CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), true);
}

bool QUnit::TrimControls(const bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>& controlVec, bool anti)
{
    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have
    // total or no probability of altering the targets, such that we can still keep them separate.

    if (!controlLen) {
        // (If we were passed 0 controls, the target functions as a gate without controls.)
        return false;
    }

    // First, no probability checks or buffer flushing.
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        QEngineShard& shard = shards[controls[i]];
        if ((anti && CACHED_ONE(shard)) || (!anti && CACHED_ZERO(shard))) {
            // This gate does nothing, so return without applying anything.
            return true;
        }
    }

    // Next, probability checks, but no buffer flushing.
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        QEngineShard& shard = shards[controls[i]];

        if ((shard.pauliBasis != PauliZ) || shard.IsInvertTarget()) {
            continue;
        }

        ProbBase(controls[i]);

        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_AMP_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (!anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        } else if (IS_AMP_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        }
    }

    // Next, just 1qb buffer flushing.
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        QEngineShard& shard = shards[controls[i]];

        if ((shard.pauliBasis == PauliZ) || shard.IsInvertTarget()) {
            continue;
        }
        RevertBasis1Qb(controls[i]);

        ProbBase(controls[i]);

        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_AMP_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (!anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        } else if (IS_AMP_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        }
    }

    // Finally, full buffer flushing, (last resort).
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        QEngineShard& shard = shards[controls[i]];

        ToPermBasisProb(controls[i]);

        ProbBase(controls[i]);

        bool isEigenstate = false;
        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_AMP_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (!anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
            // This control has 100% chance to "fire," so don't entangle it.
            isEigenstate = true;
        } else if (IS_AMP_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (anti) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
            // This control has 100% chance to "fire," so don't entangle it.
            isEigenstate = true;
        }

        if (!isEigenstate) {
            controlVec.push_back(controls[i]);
        }
    }

    return false;
}

template <typename CF>
void QUnit::ApplyEitherControlled(
    std::vector<bitLenInt> controlVec, const std::vector<bitLenInt> targets, CF cfn, bool isPhase)
{
    // If we've made it this far, we have to form the entangled representation and apply the gate.

    for (bitLenInt i = 0U; i < (bitLenInt)controlVec.size(); ++i) {
        ToPermBasisProb(controlVec[i]);
    }

    if (targets.size() > 1U) {
        for (bitLenInt i = 0U; i < (bitLenInt)targets.size(); ++i) {
            ToPermBasis(targets[i]);
        }
    } else if (isPhase) {
        RevertBasis2Qb(targets[0U], ONLY_INVERT, ONLY_TARGETS);
    } else {
        RevertBasis2Qb(targets[0U]);
    }

    std::vector<bitLenInt> allBits(controlVec.size() + targets.size());
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlVec.size());
    std::sort(allBits.begin(), allBits.end());
    std::vector<bitLenInt> allBitsMapped(allBits);

    std::vector<bitLenInt*> ebits(allBitsMapped.size());
    for (bitLenInt i = 0U; i < (bitLenInt)allBitsMapped.size(); ++i) {
        ebits[i] = &allBitsMapped[i];
    }

    QInterfacePtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());

    for (bitLenInt i = 0U; i < (bitLenInt)controlVec.size(); ++i) {
        shards[controlVec[i]].isPhaseDirty = true;
        controlVec[i] = shards[controlVec[i]].mapped;
    }
    for (bitLenInt i = 0U; i < (bitLenInt)targets.size(); ++i) {
        QEngineShard& shard = shards[targets[i]];
        shard.isPhaseDirty = true;
        shard.isProbDirty |= (shard.pauliBasis != PauliZ) || !isPhase;
    }

    // This is the original method with the maximum number of non-entangled controls excised, (potentially leaving a
    // target bit in X or Y basis and acting as if Z basis by commutation).
    cfn(unit, controlVec);

    if (!isReactiveSeparate || freezeBasis2Qb) {
        return;
    }

    // Skip 2-qubit-at-once check for 2 total qubits.
    if (allBits.size() == 2U) {
        TrySeparate(allBits[0U]);
        TrySeparate(allBits[1U]);
        return;
    }

    // Otherwise, we can try all 2-qubit combinations.
    for (bitLenInt i = 0U; i < (bitLenInt)(allBits.size() - 1U); ++i) {
        for (bitLenInt j = i + 1U; j < (bitLenInt)allBits.size(); ++j) {
            TrySeparate(allBits[i], allBits[j]);
        }
    }
}

#if ENABLE_ALU
void QUnit::CINC(bitCapInt toMod, bitLenInt start, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CINC range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::CINC parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        INC(toMod, start, length);
        return;
    }

    INT(toMod, start, length, (bitLenInt)(-1), false, controlVec);
}

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INCx range is out-of-bounds!");
    }

    if (flagIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::INCx flagIndex parameter must be within allocated qubit bounds!");
    }

    DirtyShardRange(start, length);
    DirtyShardRangePhase(start, length);
    shards[flagIndex].MakeDirty();

    EntangleRange(start, length);
    QInterfacePtr unit = Entangle({ start, flagIndex });
    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(toMod, shards[start].mapped, length, shards[flagIndex].mapped);
}

void QUnit::INCxx(
    INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INCxx range is out-of-bounds!");
    }

    if (flag1Index >= qubitCount) {
        throw std::invalid_argument("QUnit::INCxx flag1Index parameter must be within allocated qubit bounds!");
    }

    if (flag2Index >= qubitCount) {
        throw std::invalid_argument("QUnit::INCxx flag2Index parameter must be within allocated qubit bounds!");
    }

    /* Make sure the flag bits are entangled in the same QU. */
    DirtyShardRange(start, length);
    DirtyShardRangePhase(start, length);
    shards[flag1Index].MakeDirty();
    shards[flag2Index].MakeDirty();

    EntangleRange(start, length);
    QInterfacePtr unit = Entangle({ start, flag1Index, flag2Index });

    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(
        toMod, shards[start].mapped, length, shards[flag1Index].mapped, shards[flag2Index].mapped);
}

/// Check if overflow arithmetic can be optimized
bool QUnit::INTSOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex)
{
    return INTSCOptimize(toMod, start, length, isAdd, (bitLenInt)(-1), overflowIndex);
}

/// Check if carry arithmetic can be optimized
bool QUnit::INTCOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex)
{
    return INTSCOptimize(toMod, start, length, isAdd, carryIndex, (bitLenInt)(-1));
}

/// Check if arithmetic with both carry and overflow can be optimized
bool QUnit::INTSCOptimize(
    bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex, bitLenInt overflowIndex)
{
    if (!CheckBitsPermutation(start, length)) {
        return false;
    }

    const bool carry = (carryIndex != (bitLenInt)(-1));
    const bool carryIn = carry && M(carryIndex);
    if (carry && (carryIn == isAdd)) {
        ++toMod;
    }

    const bitCapInt lengthPower = pow2(length);
    const bitCapInt signMask = pow2(length - 1U);
    const bitCapInt inOutInt = GetCachedPermutation(start, length);
    const bitCapInt inInt = toMod;

    bool isOverflow;
    bitCapInt outInt;
    if (isAdd) {
        isOverflow = (overflowIndex != (bitLenInt)(-1)) && isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        outInt = inOutInt + toMod;
    } else {
        isOverflow = (overflowIndex != (bitLenInt)(-1)) && isOverflowSub(inOutInt, inInt, signMask, lengthPower);
        outInt = (inOutInt + lengthPower) - toMod;
    }

    bool carryOut = (outInt >= lengthPower);
    if (carryOut) {
        outInt &= (lengthPower - ONE_BCI);
    }
    if (carry && (carryIn != carryOut)) {
        X(carryIndex);
    }

    SetReg(start, length, outInt);

    if (isOverflow) {
        Z(overflowIndex);
    }

    return true;
}

void QUnit::INT(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
    std::vector<bitLenInt> controlVec)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INT range is out-of-bounds!");
    }

    if (hasCarry && carryIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::INT carryIndex parameter must be within allocated qubit bounds!");
    }

    if (controlVec.size()) {
        ThrowIfQbIdArrayIsBad(&(controlVec[0]), controlVec.size(), qubitCount,
            "QUnit::INT parameter controls array values must be within allocated qubit bounds!");
    }

    // Keep the bits separate, if cheap to do so:
    toMod &= pow2Mask(length);
    if (!toMod) {
        return;
    }

    if (!hasCarry && CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    // All cached classical control bits have been removed from controlVec.
    const bitLenInt controlLen = controlVec.size();
    std::unique_ptr<bitLenInt[]> controls(new bitLenInt[controlLen]);
    std::copy(controlVec.begin(), controlVec.end(), controls.get());

    std::vector<bitLenInt> allBits(controlLen + 1U);
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::sort(allBits.begin(), allBits.begin() + controlLen);

    std::vector<bitLenInt*> ebits(allBits.size());
    for (bitLenInt i = 0; i < (bitLenInt)(ebits.size() - 1U); ++i) {
        ebits[i] = &allBits[i];
    }

    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen]);

    // Try ripple addition, to avoid entanglement.
    const bitLenInt origLength = length;
    bool carry = false;
    bitLenInt i = 0U;
    while (i < origLength) {
        bool toAdd = (toMod & ONE_BCI) != 0U;

        if (toAdd == carry) {
            toMod >>= ONE_BCI;
            ++start;
            --length;
            ++i;
            // Nothing is changed, in this bit. (The carry gets promoted to the next bit.)
            continue;
        }

        if (CheckBitsPermutation(start)) {
            const bool inReg = SHARD_STATE(shards[start]);
            int total = (toAdd ? 1 : 0) + (inReg ? 1 : 0) + (carry ? 1 : 0);
            if (inReg != (total & 1)) {
                MCInvert(controls.get(), controlLen, ONE_CMPLX, ONE_CMPLX, start);
            }
            carry = (total > 1);

            toMod >>= ONE_BCI;
            ++start;
            --length;
            ++i;
        } else {
            // The carry-in is classical.
            if (carry) {
                carry = false;
                ++toMod;
            }

            if (length < 2U) {
                // We need at least two quantum bits left to try to achieve further separability.
                break;
            }

            // We're blocked by needing to add 1 to a bit in an indefinite state, which would superpose the
            // carry-out. However, if we hit another index where the qubit is known and toAdd == inReg, the
            // carry-out is guaranteed not to be superposed.

            // Load the first bit:
            bitCapInt bitMask = ONE_BCI;
            bitCapInt partMod = toMod & bitMask;
            bitLenInt partLength = 1U;
            bitLenInt partStart;
            ++i;

            do {
                // Guaranteed to need to load the second bit
                ++partLength;
                ++i;
                bitMask <<= ONE_BCI;

                toAdd = (toMod & bitMask) != 0U;
                partMod |= toMod & bitMask;

                partStart = start + partLength - ONE_BCI;
                if (!CheckBitsPermutation(partStart)) {
                    // If the quantum bit at this position is superposed, then we can't determine that the carry
                    // won't be superposed. Advance the loop.
                    continue;
                }

                const bool inReg = SHARD_STATE(shards[partStart]);
                if (toAdd != inReg) {
                    // If toAdd != inReg, the carry out might be superposed. Advance the loop.
                    continue;
                }

                // If toAdd == inReg, this prevents superposition of the carry-out. The carry out of the truth table
                // is independent of the superposed output value of the quantum bit.
                DirtyShardRange(start, partLength);
                EntangleRange(start, partLength);
                if (controlLen) {
                    allBits[controlLen] = start;
                    ebits[controlLen] = &allBits[controlLen];
                    DirtyShardIndexVector(allBits);
                    QInterfacePtr unit = Entangle(ebits);
                    for (bitLenInt cIndex = 0U; cIndex < controlLen; ++cIndex) {
                        lControls[cIndex] = shards[cIndex].mapped;
                    }
                    unit->CINC(partMod, shards[start].mapped, partLength, lControls.get(), controlLen);
                } else {
                    shards[start].unit->INC(partMod, shards[start].mapped, partLength);
                }

                carry = toAdd;
                toMod >>= (bitCapIntOcl)partLength;
                start += partLength;
                length -= partLength;

                // Break out of the inner loop and return to the flow of the containing loop.
                // (Otherwise, we hit the "continue" calls above.)
                break;
            } while (i < origLength);
        }
    }

    if (!toMod && !length) {
        // We were able to avoid entangling the carry.
        if (hasCarry && carry) {
            MCInvert(controls.get(), controlLen, ONE_CMPLX, ONE_CMPLX, carryIndex);
        }
        return;
    }

    // Otherwise, we have one unit left that needs to be entangled, plus carry bit.
    if (hasCarry) {
        if (controlLen) {
            // NOTE: This case is not actually exposed by the public API. It would only become exposed if
            // "CINCC"/"CDECC" were implemented in the public interface, in which case it would become "trivial" to
            // implement, once the QEngine methods were in place.
            throw std::logic_error("ERROR: Controlled-with-carry arithmetic is not implemented!");
        } else {
            DirtyShardRange(start, length);
            shards[carryIndex].MakeDirty();
            EntangleRange(start, length);
            QInterfacePtr unit = Entangle({ start, carryIndex });
            unit->INCC(toMod, shards[start].mapped, length, shards[carryIndex].mapped);
        }
    } else {
        DirtyShardRange(start, length);
        EntangleRange(start, length);
        if (controlLen) {
            allBits[controlLen] = start;
            ebits[controlLen] = &allBits[controlLen];
            QInterfacePtr unit = Entangle(ebits);
            DirtyShardIndexVector(allBits);
            for (bitLenInt cIndex = 0U; cIndex < controlLen; ++cIndex) {
                lControls[cIndex] = shards[cIndex].mapped;
            }
            unit->CINC(toMod, shards[start].mapped, length, lControls.get(), controlLen);
        } else {
            shards[start].unit->INC(toMod, shards[start].mapped, length);
        }
    }
}

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    INT(toMod, start, length, (bitLenInt)(-1), false);
}

/// Add integer (without sign, with carry)
void QUnit::INCC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
        ++toAdd;
    }

    INT(toAdd, inOutStart, length, carryIndex, true);
}

/// Subtract integer (without sign, with carry)
void QUnit::DECC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
    } else {
        ++toSub;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INT(invToSub, inOutStart, length, carryIndex, true);
}

void QUnit::INTS(
    bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex, bool hasCarry)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INT range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::INT overflowIndex parameter must be within allocated qubit bounds!");
    }

    if (hasCarry && carryIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::INT carryIndex parameter must be within allocated qubit bounds!");
    }

    toMod &= pow2Mask(length);
    if (!toMod) {
        return;
    }

    const bitLenInt signBit = start + length - 1U;
    const bool knewFlagSet = CheckBitsPermutation(overflowIndex);
    const bool flagSet = SHARD_STATE(shards[overflowIndex]);

    if (knewFlagSet && !flagSet) {
        // Overflow detection is disabled
        INT(toMod, start, length, carryIndex, hasCarry);
        return;
    }

    const bool addendNeg = (toMod & pow2(length - 1U)) != 0;
    const bool knewSign = CheckBitsPermutation(signBit);
    const bool quantumNeg = SHARD_STATE(shards[signBit]);

    if (knewSign && (addendNeg != quantumNeg)) {
        // No chance of overflow
        INT(toMod, start, length, carryIndex, hasCarry);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    if (hasCarry) {
        // Keep the bits separate, if cheap to do so:
        if (INTSCOptimize(toMod, start, length, true, carryIndex, overflowIndex)) {
            return;
        }
        INCxx(&QAlu::INCSC, toMod, start, length, overflowIndex, carryIndex);
    } else {
        // Keep the bits separate, if cheap to do so:
        if (INTSOptimize(toMod, start, length, true, overflowIndex)) {
            return;
        }
        INCx(&QAlu::INCS, toMod, start, length, overflowIndex);
    }
}

void QUnit::INCS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INTS(toMod, start, length, overflowIndex, (bitLenInt)(-1), false);
}

void QUnit::INCDECSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INTS(toAdd, inOutStart, length, overflowIndex, carryIndex, true);
}

void QUnit::INCDECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QAlu::INCSC, toMod, start, length, carryIndex);
}

#if ENABLE_BCD
void QUnit::INCBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INCBCD range is out-of-bounds!");
    }

    // BCD variants are low priority for optimization, for the time being.
    DirtyShardRange(start, length);
    std::dynamic_pointer_cast<QAlu>(EntangleRange(start, length))->INCBCD(toMod, shards[start].mapped, length);
}

void QUnit::DECBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INCBCD range is out-of-bounds!");
    }

    // BCD variants are low priority for optimization, for the time being.
    DirtyShardRange(start, length);
    std::dynamic_pointer_cast<QAlu>(EntangleRange(start, length))->DECBCD(toMod, shards[start].mapped, length);
}

void QUnit::INCDECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // BCD variants are low priority for optimization, for the time being.
    INCx(&QAlu::INCDECBCDC, toMod, start, length, carryIndex);
}
#endif

void QUnit::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL carryStart range is out-of-bounds!");
    }

    // Keep the bits separate, if cheap to do so:
    if (!toMul) {
        SetReg(inOutStart, length, 0U);
        SetReg(carryStart, length, 0U);
        return;
    } else if (toMul == ONE_BCI) {
        SetReg(carryStart, length, 0U);
        return;
    }

    if (CheckBitsPermutation(inOutStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt res = GetCachedPermutation(inOutStart, length) * toMul;
        SetReg(inOutStart, length, res & lengthMask);
        SetReg(carryStart, length, (res >> (bitCapIntOcl)length) & lengthMask);
        return;
    }

    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    // Otherwise, form the potentially entangled representation:
    std::dynamic_pointer_cast<QAlu>(EntangleRange(inOutStart, length, carryStart, length))
        ->MUL(toMul, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL carryStart range is out-of-bounds!");
    }

    // Keep the bits separate, if cheap to do so:
    if (toDiv == ONE_BCI) {
        return;
    }

    if (CheckBitsPermutation(inOutStart, length) && CheckBitsPermutation(carryStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt origRes =
            GetCachedPermutation(inOutStart, length) | (GetCachedPermutation(carryStart, length) << length);
        const bitCapInt res = origRes / toDiv;
        if (origRes == (res * toDiv)) {
            SetReg(inOutStart, length, res & lengthMask);
            SetReg(carryStart, length, (res >> (bitCapIntOcl)length) & lengthMask);
        }
        return;
    }

    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    // Otherwise, form the potentially entangled representation:
    std::dynamic_pointer_cast<QAlu>(EntangleRange(inOutStart, length, carryStart, length))
        ->DIV(toDiv, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL inStart range is out-of-bounds!");
    }

    if (isBadBitRange(outStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::MUL outStart range is out-of-bounds!");
    }

    if (toMod == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        const bitCapInt res = intPow(toMod, GetCachedPermutation(inStart, length)) % modN;
        SetReg(outStart, length, res);
        return;
    }

    SetReg(outStart, length, 0);

    // Otherwise, form the potentially entangled representation:
    std::dynamic_pointer_cast<QAlu>(EntangleRange(inStart, length, outStart, length))
        ->POWModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    DirtyShardRangePhase(inStart, length);
    DirtyShardRange(outStart, length);
}

QInterfacePtr QUnit::CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt>* controlsMapped)
{
    DirtyShardRangePhase(start, length);
    DirtyShardRange(carryStart, length);
    EntangleRange(start, length);
    EntangleRange(carryStart, length);

    std::vector<bitLenInt> bits(controlVec.size() + 2U);
    for (bitLenInt i = 0U; i < (bitLenInt)controlVec.size(); ++i) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    bits[controlVec.size() + 1U] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (bitLenInt i = 0U; i < (bitLenInt)ebits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    if (controlVec.size()) {
        controlsMapped->resize(controlVec.size());
        for (bitLenInt i = 0U; i < (bitLenInt)controlVec.size(); ++i) {
            (*controlsMapped)[i] = shards[controlVec[i]].mapped;
            shards[controlVec[i]].isPhaseDirty = true;
        }
    }

    return unit;
}

void QUnit::CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
    std::vector<bitLenInt> controlVec)
{
    // Otherwise, we have to "dirty" the register.
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(toMod, shards[start].mapped, shards[carryStart].mapped, length,
        controlVec.size() ? &(controlsMapped[0U]) : NULL, controlVec.size());

    DirtyShardRange(start, length);
}

void QUnit::CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt> controlVec)
{
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(toMod, modN, shards[start].mapped, shards[carryStart].mapped, length,
        controlVec.size() ? &(controlsMapped[0U]) : NULL, controlVec.size());

    DirtyShardRangePhase(start, length);
}

void QUnit::CMUL(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
    bitLenInt controlLen)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CMUL inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CMUL carryStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::CMUL parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QAlu::CMUL, toMod, start, carryStart, length, controlVec);
}

void QUnit::CDIV(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
    bitLenInt controlLen)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CDIV inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CDIV carryStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::CDIV parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, false)) {
        return;
    }

    if (!controlVec.size()) {
        DIV(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QAlu::CDIV, toMod, start, carryStart, length, controlVec);
}

void QUnit::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CPOWModNOut inStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount,
        "QUnit::CPOWModNOut parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlLen, controlVec, false)) {
        return;
    }

    CMULModx(&QAlu::CPOWModNOut, toMod, modN, inStart, outStart, length, controlVec);
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(indexStart, indexLength);
    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt value = 0U;
    for (bitCapIntOcl j = 0U; j < valueBytes; ++j) {
        value |= (bitCapInt)values[indexInt * valueBytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(start, length);
    const bitLenInt bytes = (length + 7U) / 8U;
    bitCapInt value = 0U;
    for (bitCapIntOcl j = 0U; j < bytes; ++j) {
        value |= (bitCapInt)values[indexInt * bytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    const unsigned char* values, bool resetValue)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedLDA indexStart range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedLDA valueStart range is out-of-bounds!");
    }

    // TODO: Index bits that have exactly 0 or 1 probability can be optimized out of the gate.
    // This could follow the logic of UniformlyControlledSingleBit().
    // In the meantime, checking if all index bits are in eigenstates takes very little overhead.
    if (CheckBitsPermutation(indexStart, indexLength)) {
        const bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        SetReg(valueStart, valueLength, value);
#if ENABLE_VM6502Q_DEBUG
        return value;
#else
        return 0U;
#endif
    }

    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    const bitCapInt toRet = std::dynamic_pointer_cast<QAlu>(shards[indexStart].unit)
                                ->IndexedLDA(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
                                    valueLength, values, resetValue);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);

    return toRet;
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedADC indexStart range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedADC valueStart range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::IndexedADC carryIndex is out-of-bounds!");
    }

#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) + value;
        const bitCapInt valueMask = pow2Mask(valueLength);
        bool carry = false;
        if (value > valueMask) {
            value &= valueMask;
            carry = true;
        }
        SetReg(valueStart, valueLength, value);
        if (carry) {
            X(carryIndex);
        }
        return value;
    }
#else
    if (CheckBitsPermutation(indexStart, indexLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        INCC(value, valueStart, valueLength, carryIndex);
        return 0U;
    }
#endif
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    const bitCapInt toRet = std::dynamic_pointer_cast<QAlu>(shards[indexStart].unit)
                                ->IndexedADC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
                                    valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedSBC indexStart range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QUnit::IndexedSBC valueStart range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::IndexedSBC carryIndex is out-of-bounds!");
    }

#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) - value;
        const bitCapInt valueMask = pow2Mask(valueLength);
        bool carry = false;
        if (value > valueMask) {
            value &= valueMask;
            carry = true;
        }
        SetReg(valueStart, valueLength, value);
        if (carry) {
            X(carryIndex);
        }
        return value;
    }
#else
    if (CheckBitsPermutation(indexStart, indexLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        DECC(value, valueStart, valueLength, carryIndex);
        return 0U;
    }
#endif
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    const bitCapInt toRet = std::dynamic_pointer_cast<QAlu>(shards[indexStart].unit)
                                ->IndexedSBC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
                                    valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

void QUnit::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::Hash range is out-of-bounds!");
    }

    if (CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    if (CheckBitsPermutation(start, length)) {
        const bitCapInt value = GetIndexedEigenstate(start, length, values);
        SetReg(start, length, value);
        return;
    }

    DirtyShardRange(start, length);
    std::dynamic_pointer_cast<QAlu>(EntangleRange(start, length))->Hash(shards[start].mapped, length, values);
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::PhaseFlipIfLess range is out-of-bounds!");
    }

    if (CheckBitsPermutation(start, length)) {
        const bitCapInt value = GetCachedPermutation(start, length);
        if (value < greaterPerm) {
            PhaseFlip();
        }

        return;
    }

    DirtyShardRange(start, length);
    std::dynamic_pointer_cast<QAlu>(EntangleRange(start, length))
        ->PhaseFlipIfLess(greaterPerm, shards[start].mapped, length);
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CPhaseFlipIfLess range is out-of-bounds!");
    }

    if (flagIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::CPhaseFlipIfLess flagIndex is out-of-bounds!");
    }

    if (CheckBitsPermutation(flagIndex, 1)) {
        if (SHARD_STATE(shards[flagIndex])) {
            PhaseFlipIfLess(greaterPerm, start, length);
        }

        return;
    }

    DirtyShardRange(start, length);
    shards[flagIndex].isPhaseDirty = true;
    EntangleRange(start, length);
    std::dynamic_pointer_cast<QAlu>(Entangle({ start, flagIndex }))
        ->CPhaseFlipIfLess(greaterPerm, shards[start].mapped, length, shards[flagIndex].mapped);
}
#endif

bool QUnit::ParallelUnitApply(ParallelUnitFn fn, real1_f param1, real1_f param2, real1_f param3, int64_t param4)
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0U; i < shards.size(); ++i) {
        QInterfacePtr toFind = shards[i].unit;
        if (toFind && (find(units.begin(), units.end(), toFind) == units.end())) {
            units.push_back(toFind);
            if (!fn(toFind, param1, param2, param3, param4)) {
                return false;
            }
        }
    }

    return true;
}

void QUnit::UpdateRunningNorm(real1_f norm_thresh)
{
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f norm_thresh, real1_f unused2, real1_f unused3, int64_t unused4) {
            unit->UpdateRunningNorm(norm_thresh);
            return true;
        },
        norm_thresh);
}

void QUnit::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f nrm, real1_f norm_thresh, real1_f phaseArg, int64_t unused) {
            unit->NormalizeState(nrm, norm_thresh, phaseArg);
            return true;
        },
        nrm, norm_thresh, phaseArg);
}

void QUnit::Finish()
{
    ParallelUnitApply([](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3, int64_t unused4) {
        unit->Finish();
        return true;
    });
}

bool QUnit::isFinished()
{
    return ParallelUnitApply([](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3,
                                 int64_t unused4) { return unit->isFinished(); });
}

void QUnit::SetDevice(int64_t dID)
{
    devID = dID;
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f unused1, real1_f forceReInit, real1_f unused2, int64_t dID) {
            unit->SetDevice(dID);
            return true;
        },
        ZERO_R1_F, ZERO_R1_F, ZERO_R1_F, dID);
}

real1_f QUnit::SumSqrDiff(QUnitPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    if (qubitCount == 1U) {
        RevertBasis1Qb(0U);
        toCompare->RevertBasis1Qb(0U);

        complex mAmps[2U], oAmps[2U];
        if (shards[0U].unit) {
            shards[0U].unit->GetQuantumState(mAmps);
        } else {
            mAmps[0U] = shards[0U].amp0;
            mAmps[1U] = shards[0U].amp1;
        }
        if (toCompare->shards[0U].unit) {
            toCompare->shards[0U].unit->GetQuantumState(oAmps);
        } else {
            oAmps[0U] = toCompare->shards[0U].amp0;
            oAmps[1U] = toCompare->shards[0U].amp1;
        }

        return (real1_f)(norm(mAmps[0U] - oAmps[0U]) + norm(mAmps[1U] - oAmps[1U]));
    }

    if (CheckBitsPermutation(0U, qubitCount) && toCompare->CheckBitsPermutation(0U, qubitCount)) {
        if (GetCachedPermutation((bitLenInt)0U, qubitCount) ==
            toCompare->GetCachedPermutation((bitLenInt)0U, qubitCount)) {
            return ZERO_R1_F;
        }

        // Necessarily max difference:
        return ONE_R1_F;
    }

    QUnitPtr thisCopyShared, thatCopyShared;
    QUnit* thisCopy;
    QUnit* thatCopy;

    if (shards[0U].GetQubitCount() == qubitCount) {
        ToPermBasisAll();
        OrderContiguous(shards[0U].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    if (toCompare->shards[0U].GetQubitCount() == qubitCount) {
        toCompare->ToPermBasisAll();
        toCompare->OrderContiguous(toCompare->shards[0U].unit);
        thatCopy = toCompare.get();
    } else {
        thatCopyShared = std::dynamic_pointer_cast<QUnit>(toCompare->Clone());
        thatCopyShared->EntangleAll();
        thatCopy = thatCopyShared.get();
    }

    return thisCopy->shards[0U].unit->SumSqrDiff(thatCopy->shards[0U].unit);
}

QInterfacePtr QUnit::Clone()
{
    // TODO: Copy buffers instead of flushing?
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        RevertBasis2Qb(i);
    }

    QUnitPtr copyPtr = std::make_shared<QUnit>(engines, qubitCount, 0U, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);

    copyPtr->SetReactiveSeparate(isReactiveSeparate);

    return CloneBody(copyPtr);
}

QInterfacePtr QUnit::CloneBody(QUnitPtr copyPtr)
{
    std::map<QInterfacePtr, QInterfacePtr> dupeEngines;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        copyPtr->shards[i] = QEngineShard(shards[i]);

        QInterfacePtr unit = shards[i].unit;
        if (!unit) {
            continue;
        }

        if (dupeEngines.find(unit) == dupeEngines.end()) {
            dupeEngines[unit] = unit->Clone();
        }

        copyPtr->shards[i].unit = dupeEngines[unit];
    }

    return copyPtr;
}

void QUnit::ApplyBuffer(PhaseShardPtr phaseShard, bitLenInt control, bitLenInt target, bool isAnti)
{
    const bitLenInt controls[1U] = { control };

    const complex polarDiff = phaseShard->cmplxDiff;
    const complex polarSame = phaseShard->cmplxSame;

    freezeBasis2Qb = true;
    if (phaseShard->isInvert) {
        if (isAnti) {
            MACInvert(controls, 1U, polarSame, polarDiff, target);
        } else {
            MCInvert(controls, 1U, polarDiff, polarSame, target);
        }
    } else {
        if (isAnti) {
            MACPhase(controls, 1U, polarSame, polarDiff, target);
        } else {
            MCPhase(controls, 1U, polarDiff, polarSame, target);
        }
    }
    freezeBasis2Qb = false;
}

void QUnit::ApplyBufferMap(bitLenInt bitIndex, ShardToPhaseMap bufferMap, RevertExclusivity exclusivity, bool isControl,
    bool isAnti, const std::set<bitLenInt>& exceptPartners, bool dumpSkipped)
{
    QEngineShard& shard = shards[bitIndex];

    ShardToPhaseMap::iterator phaseShard;

    while (bufferMap.size()) {
        phaseShard = bufferMap.begin();
        QEngineShardPtr partner = phaseShard->first;

        if (((exclusivity == ONLY_INVERT) && !phaseShard->second->isInvert) ||
            ((exclusivity == ONLY_PHASE) && phaseShard->second->isInvert)) {
            bufferMap.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemoveTarget(partner);
            }
            continue;
        }

        bitLenInt partnerIndex = FindShardIndex(partner);

        if (exceptPartners.find(partnerIndex) != exceptPartners.end()) {
            bufferMap.erase(phaseShard);
            if (dumpSkipped) {
                if (isControl) {
                    if (isAnti) {
                        shard.RemoveAntiTarget(partner);
                    } else {
                        shard.RemoveTarget(partner);
                    }
                } else {
                    if (isAnti) {
                        shard.RemoveAntiControl(partner);
                    } else {
                        shard.RemoveControl(partner);
                    }
                }
            }
            continue;
        }

        if (isControl) {
            if (isAnti) {
                shard.RemoveAntiTarget(partner);
            } else {
                shard.RemoveTarget(partner);
            }
            ApplyBuffer(phaseShard->second, bitIndex, partnerIndex, isAnti);
        } else {
            if (isAnti) {
                shard.RemoveAntiControl(partner);
            } else {
                shard.RemoveControl(partner);
            }
            ApplyBuffer(phaseShard->second, partnerIndex, bitIndex, isAnti);
        }

        bufferMap.erase(phaseShard);
    }
}

void QUnit::RevertBasis2Qb(bitLenInt i, RevertExclusivity exclusivity, RevertControl controlExclusivity,
    RevertAnti antiExclusivity, const std::set<bitLenInt>& exceptControlling,
    const std::set<bitLenInt>& exceptTargetedBy, bool dumpSkipped, bool skipOptimize)
{
    QEngineShard& shard = shards[i];

    if (freezeBasis2Qb || !QUEUED_PHASE(shard)) {
        // Recursive call that should be blocked,
        // or already in target basis.
        return;
    }

    shard.CombineGates();

    if (!skipOptimize && (controlExclusivity == ONLY_CONTROLS) && (exclusivity != ONLY_INVERT)) {
        if (antiExclusivity != ONLY_ANTI) {
            shard.OptimizeControls();
        }
        if (antiExclusivity != ONLY_CTRL) {
            shard.OptimizeAntiControls();
        }
    } else if (!skipOptimize && (controlExclusivity == ONLY_TARGETS) && (exclusivity != ONLY_INVERT)) {
        if (antiExclusivity == CTRL_AND_ANTI) {
            shard.OptimizeBothTargets();
        } else if (antiExclusivity == ONLY_CTRL) {
            shard.OptimizeTargets();
        } else if (antiExclusivity == ONLY_ANTI) {
            shard.OptimizeAntiTargets();
        }
    }

    if (controlExclusivity != ONLY_TARGETS) {
        if (antiExclusivity != ONLY_ANTI) {
            ApplyBufferMap(i, shard.controlsShards, exclusivity, true, false, exceptControlling, dumpSkipped);
        }
        if (antiExclusivity != ONLY_CTRL) {
            ApplyBufferMap(i, shard.antiControlsShards, exclusivity, true, true, exceptControlling, dumpSkipped);
        }
    }

    if (controlExclusivity == ONLY_CONTROLS) {
        return;
    }

    if (antiExclusivity != ONLY_ANTI) {
        ApplyBufferMap(i, shard.targetOfShards, exclusivity, false, false, exceptTargetedBy, dumpSkipped);
    }
    if (antiExclusivity != ONLY_CTRL) {
        ApplyBufferMap(i, shard.antiTargetOfShards, exclusivity, false, true, exceptTargetedBy, dumpSkipped);
    }
}

void QUnit::CommuteH(bitLenInt bitIndex)
{
    QEngineShard& shard = shards[bitIndex];

    if (!QUEUED_PHASE(shard)) {
        return;
    }

    ShardToPhaseMap controlsShards = shard.controlsShards;

    for (auto phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); ++phaseShard) {
        PhaseShardPtr buffer = phaseShard->second;
        QEngineShardPtr partner = phaseShard->first;

        if (buffer->isInvert) {
            continue;
        }

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddAntiPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    controlsShards = shard.antiControlsShards;

    for (auto phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); ++phaseShard) {
        PhaseShardPtr buffer = phaseShard->second;
        QEngineShardPtr partner = phaseShard->first;

        if (buffer->isInvert) {
            continue;
        }

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemoveAntiTarget(partner);
            shard.AddAntiPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemoveAntiTarget(partner);
            shard.AddPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    RevertBasis2Qb(bitIndex, INVERT_AND_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, {}, {}, false, true);

    ShardToPhaseMap targetOfShards = shard.targetOfShards;

    for (auto phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); ++phaseShard) {
        PhaseShardPtr buffer = phaseShard->second;

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard->first;

        if (IS_SAME(polarDiff, polarSame)) {
            continue;
        }

        if (buffer->isInvert && IS_OPPOSITE(polarDiff, polarSame)) {
            continue;
        }

        bitLenInt control = FindShardIndex(partner);
        shard.RemoveControl(partner);
        ApplyBuffer(buffer, control, bitIndex, false);
    }

    targetOfShards = shard.antiTargetOfShards;

    for (auto phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); ++phaseShard) {
        PhaseShardPtr buffer = phaseShard->second;

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard->first;

        if (IS_SAME(polarDiff, polarSame)) {
            continue;
        }

        if (buffer->isInvert && IS_OPPOSITE(polarDiff, polarSame)) {
            continue;
        }

        const bitLenInt control = FindShardIndex(partner);
        shard.RemoveAntiControl(partner);
        ApplyBuffer(buffer, control, bitIndex, true);
    }

    shard.CommuteH();
}

void QUnit::OptimizePairBuffers(bitLenInt control, bitLenInt target, bool anti)
{
    QEngineShard& cShard = shards[control];
    QEngineShard& tShard = shards[target];

    ShardToPhaseMap& targets = anti ? tShard.antiTargetOfShards : tShard.targetOfShards;
    ShardToPhaseMap::iterator phaseShard = targets.find(&cShard);
    if (phaseShard == targets.end()) {
        return;
    }

    PhaseShardPtr buffer = phaseShard->second;

    if (!buffer->isInvert) {
        if (anti) {
            if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
                tShard.RemoveAntiControl(&cShard);
                return;
            }
            if (IS_SAME_UNIT(cShard, tShard)) {
                tShard.RemoveAntiControl(&cShard);
                ApplyBuffer(buffer, control, target, true);
                return;
            }
        } else {
            if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
                tShard.RemoveControl(&cShard);
                return;
            }
            if (IS_SAME_UNIT(cShard, tShard)) {
                tShard.RemoveControl(&cShard);
                ApplyBuffer(buffer, control, target, false);
                return;
            }
        }
    }

    ShardToPhaseMap& antiTargets = anti ? tShard.targetOfShards : tShard.antiTargetOfShards;
    ShardToPhaseMap::iterator antiShard = antiTargets.find(&cShard);
    if (antiShard == antiTargets.end()) {
        return;
    }

    PhaseShardPtr aBuffer = antiShard->second;

    if (buffer->isInvert != aBuffer->isInvert) {
        return;
    }

    if (anti) {
        std::swap(buffer, aBuffer);
    }

    const bool isInvert = buffer->isInvert;
    if (isInvert) {
        if (tShard.pauliBasis == PauliY) {
            YBase(target);
        } else if (tShard.pauliBasis == PauliX) {
            ZBase(target);
        } else {
            XBase(target);
        }

        buffer->isInvert = false;
        aBuffer->isInvert = false;
    }

    if (IS_NORM_0(buffer->cmplxDiff - aBuffer->cmplxSame) && IS_NORM_0(buffer->cmplxSame - aBuffer->cmplxDiff)) {
        tShard.RemoveControl(&cShard);
        tShard.RemoveAntiControl(&cShard);
        Phase(buffer->cmplxDiff, buffer->cmplxSame, target);
    } else if (isInvert) {
        if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
            tShard.RemoveControl(&cShard);
        }
        if (IS_1_CMPLX(aBuffer->cmplxDiff) && IS_1_CMPLX(aBuffer->cmplxSame)) {
            tShard.RemoveAntiControl(&cShard);
        }
    }
}

} // namespace Qrack
