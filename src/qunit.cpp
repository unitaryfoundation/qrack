//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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
#define IS_AMP_0(c) ((2 * norm(c)) <= separabilityThreshold)
#define IS_1_CMPLX(c) (norm(ONE_CMPLX - (c)) <= FP_NORM_EPSILON)
#define SHARD_STATE(shard) ((2 * norm(shard.amp0)) < ONE_R1)
#define QUEUED_PHASE(shard)                                                                                            \
    (shard.targetOfShards.size() || shard.controlsShards.size() || shard.antiTargetOfShards.size() ||                  \
        shard.antiControlsShards.size())
#define CACHED_X(shard) ((shard.pauliBasis == PauliX) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_X_OR_Y(shard) ((shard.pauliBasis != PauliZ) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_Z(shard) ((shard.pauliBasis == PauliZ) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_ZERO(q)                                                                                                 \
    (CACHED_Z(shards[q]) && !(shards[q].unit && shards[q].unit->isClifford() && shards[q].unit->GetTInjection()) &&    \
        (ProbBase(q) <= FP_NORM_EPSILON))
#define CACHED_ONE(q)                                                                                                  \
    (CACHED_Z(shards[q]) && !(shards[q].unit && shards[q].unit->isClifford() && shards[q].unit->GetTInjection()) &&    \
        ((ONE_R1_F - ProbBase(q)) <= FP_NORM_EPSILON))
#define CACHED_PLUS(q)                                                                                                 \
    (CACHED_X(shards[q]) && !(shards[q].unit && shards[q].unit->isClifford() && shards[q].unit->GetTInjection()) &&    \
        (ProbBase(q) <= FP_NORM_EPSILON))
// "UNSAFE" variants here do not check whether the bit has cached 2-qubit gates.
#define UNSAFE_CACHED_ZERO_OR_ONE(shard)                                                                               \
    (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define UNSAFE_CACHED_X(shard)                                                                                         \
    (!shard.isProbDirty && (shard.pauliBasis == PauliX) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define UNSAFE_CACHED_ONE(shard) (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && IS_NORM_0(shard.amp0))
#define UNSAFE_CACHED_ZERO(shard) (!shard.isProbDirty && (shard.pauliBasis == PauliZ) && IS_NORM_0(shard.amp1))
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
    , freezeBasis2Qb(false)
    , useHostRam(useHostMem)
    , isSparse(useSparseStateVec)
    , useTGadget(true)
    , thresholdQubits(qubitThreshold)
    , separabilityThreshold(sep_thresh)
    , logFidelity(0.0)
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , deviceIDs(devList)
    , engines(eng)
{
    if (engines.empty()) {
        engines.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")) {
        separabilityThreshold = (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")));
    }
#endif
    isReactiveSeparate = (separabilityThreshold > FP_NORM_EPSILON_F);

    if (qubitCount) {
        SetPermutation(initState);
    }
}

QInterfacePtr QUnit::MakeEngine(bitLenInt length, bitCapInt perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    toRet->SetTInjection(useTGadget);
    toRet->SetNcrp(roundingThreshold);

    return toRet;
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Dump();

    logFidelity = 0.0;

    shards = QEngineShardMap();

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        shards.push_back(QEngineShard(bi_and_1(perm >> i) != 0U, GetNonunitaryPhase()));
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
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp0 - shard.amp1)));
            shard.pauliBasis = PauliX;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0(shard.amp0 + shard.amp1)) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp0 + shard.amp1)));
            shard.pauliBasis = PauliX;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0U]) - inputState[1U])) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm((I_CMPLX * inputState[0U]) - inputState[1U])));
            shard.pauliBasis = PauliY;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0U]) + inputState[1U])) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm((I_CMPLX * inputState[0U]) - inputState[1U])));
            shard.pauliBasis = PauliY;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        }
        return;
    }

    QInterfacePtr unit = MakeEngine(qubitCount, ZERO_BCI);
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
    if (bi_compare(perm, maxQPower) >= 0) {
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
            result *= bi_and_1(perm >> i) ? shard.amp1 : shard.amp0;
            continue;
        }

        if (perms.find(shard.unit) == perms.end()) {
            perms[shard.unit] = ZERO_BCI;
        }
        if (bi_and_1(perm >> i)) {
            bi_or_ip(&(perms[shard.unit]), pow2(shard.mapped));
        }
    }

    for (const auto& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
        if (IS_AMP_0(result)) {
            break;
        }
    }

    return result;
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
        for (const auto& subunit : subunits) {
            OrderContiguous(subunit.first);
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
                    QInterfacePtr nUnit = MakeEngine(subLen, ZERO_BCI);
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
                    for (size_t i = 0U; i < shards.size(); ++i) {
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

    // Find the rest of the qubits.
    for (auto&& shard : shards) {
        const auto subunit = subunits.find(shard.unit);
        if (subunit != subunits.end() &&
            shard.mapped >= (shards[decomposedUnits[shard.unit]].mapped + subunit->second)) {
            shard.mapped -= subunit->second;
        }
    }

    shards.erase(start, start + length);
    SetQubitCount(qubitCount - length);
}

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

    // Walk through all of the supplied bits and create a unique list to compose.
    for (auto bit = first; bit < last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    // Collapse all of the other units into unit1, returning a map to the new bit offset.
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

        // Since each unit will be collapsed in-order, one set of bits at a time.
        for (auto&& shard : shards) {
            const auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = offsetPartners[shard.unit];
            }
        }

        units = nUnits;
    }

    // Change the source parameters to the correct newly mapped bit indexes.
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

    QUnitPtr nQubits = std::make_shared<QUnit>(engines, length, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    nQubits->SetReactiveSeparate(isReactiveSeparate);
    nQubits->SetTInjection(useTGadget);
    nQubits->SetNcrp(roundingThreshold);

    return Compose(nQubits, start);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt> bits)
{
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (size_t i = 0U; i < ebits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    return Entangle(ebits);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt*> bits)
{
    for (size_t i = 0U; i < bits.size(); ++i) {
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

bool QUnit::TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol)
{
    ThrowIfQbIdArrayIsBad(qubits, qubitCount,
        "QUnit::TrySeparate parameter controls array values must be within allocated qubit bounds!");

    if (qubits.size() == 1U) {
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
        QInterfacePtr nUnit = MakeEngine(1U, ZERO_BCI);
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

    std::vector<bitLenInt> q(qubits.size());
    std::copy(qubits.begin(), qubits.end(), q.begin());
    std::sort(q.begin(), q.end());

    // Swap gate is free, so just bring into the form of the contiguous overload.
    for (size_t i = 0U; i < q.size(); ++i) {
        Swap(i, q[i]);
    }

    QUnitPtr dest = std::dynamic_pointer_cast<QUnit>(std::make_shared<QUnit>(
        engines, q.size(), ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam));

    bool toRet = TryDecompose(0U, dest, error_tol);
    if (toRet) {
        if (q.size() == 1U) {
            dest->ProbBase(0U);
        }
        Compose(dest, 0U);
    }

    for (size_t i = 0U; i < q.size(); ++i) {
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

    if (shard.unit->isClifford()) {
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

    const double oneMinR = 1.0 - sqrt((double)(x * x + y * y + z * z));
    if (oneMinR > separabilityThreshold) {
        return false;
    }

    // Adjust the qubit basis to the Pauli-Z basis, if necessary, for logical equivalence.
    if (shard.pauliBasis == PauliX) {
        RevertBasis1Qb(qubit);
    } else if (shard.pauliBasis == PauliY) {
        std::swap(x, z);
        std::swap(y, z);
    }

    const real1_f inclination = atan2(sqrt(x * x + y * y), z);
    const real1_f azimuth = atan2(y, x);

    shard.unit->IAI(shard.mapped, azimuth, inclination);
    prob = 2 * shard.unit->Prob(shard.mapped);

    if (prob > separabilityThreshold) {
        shard.unit->AI(shard.mapped, azimuth, inclination);
        return false;
    }

    SeparateBit(false, qubit);
    ShardAI(qubit, azimuth, inclination);

    logFidelity += (double)log(clampProb(1.0 - oneMinR / 2));

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
    const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(ZERO_R1, SQRT1_2_R1) };
    const std::vector<bitLenInt> controls{ mapped1 };

    real1_f z = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->CH(shard1.mapped, shard2.mapped);
    real1_f x = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->CS(shard1.mapped, shard2.mapped);
    real1_f y = ONE_R1_F - 2 * unit->CProb(mapped1, mapped2);
    unit->MCMtrx(controls, mtrx, mapped2);
    const real1_f inclination = atan2(sqrt(x * x + y * y), z);
    const real1_f azimuth = atan2(y, x);
    unit->CIAI(mapped1, mapped2, azimuth, inclination);

    z = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->AntiCH(shard1.mapped, shard2.mapped);
    x = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->AntiCS(shard1.mapped, shard2.mapped);
    y = ONE_R1_F - 2 * unit->ACProb(mapped1, mapped2);
    unit->MACMtrx(controls, mtrx, mapped2);
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
    // Before we call OrderContinguous, when we are cohering lists of shards, we should always proactively sort the
    // order in which we compose qubits into a single engine. This is a cheap way to reduce the need for costly qubit
    // swap gates, later.

    if (!unit || (unit->GetQubitCount() == 1U)) {
        return;
    }

    // Create a sortable collection of all of the bits that are in the unit.
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

/// Sort a container of bits, calling Swap() on each.
void QUnit::SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high)
{
    bitLenInt i = low, j = high;
    if (i == (j - 1U)) {
        if (bits[j] < bits[i]) {
            unit->Swap(bits[i].mapped, bits[j].mapped); // Change the location in the QE itself.
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); // Change the global mapping.
            std::swap(bits[i].mapped, bits[j].mapped); // Change the contents of the sorting array.
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
            unit->Swap(bits[i].mapped, bits[j].mapped); // Change the location in the QE itself.
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); // Change the global mapping.
            std::swap(bits[i].mapped, bits[j].mapped); // Change the contents of the sorting array.
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
    bitCapInt res = ZERO_BCI;
    for (bitLenInt i = 0U; i < length; ++i) {
        if (SHARD_STATE(shards[start + i])) {
            bi_or_ip(&res, pow2(i));
        }
    }
    return res;
}

bitCapInt QUnit::GetCachedPermutation(const std::vector<bitLenInt>& bitArray)
{
    bitCapInt res = ZERO_BCI;
    for (size_t i = 0U; i < bitArray.size(); ++i) {
        if (SHARD_STATE(shards[bitArray[i]])) {
            bi_or_ip(&res, pow2(i));
        }
    }
    return res;
}

bool QUnit::CheckBitsPlus(bitLenInt qubitIndex, bitLenInt length)
{
    bool isHBasis = true;
    for (bitLenInt i = 0U; i < length; ++i) {
        if (!CACHED_PLUS(qubitIndex + i)) {
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
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(amps[0U] - amps[1U])));
            shard.pauliBasis = PauliX;
            amps[0U] = amps[0U] / abs(amps[0U]);
            amps[1U] = ZERO_CMPLX;
        } else if (IS_AMP_0(amps[0U] + amps[1U])) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(amps[0U] + amps[1U])));
            shard.pauliBasis = PauliX;
            amps[1U] = amps[0U] / abs(amps[0U]);
            amps[0U] = ZERO_CMPLX;
        } else if (IS_AMP_0((I_CMPLX * amps[0U]) - amps[1U])) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm((I_CMPLX * amps[0U]) - amps[1U])));
            shard.pauliBasis = PauliY;
            amps[0U] = amps[0U] / abs(amps[0U]);
            amps[1U] = ZERO_CMPLX;
        } else if (IS_AMP_0((I_CMPLX * amps[0U]) + amps[1U])) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm((I_CMPLX * amps[0U]) + amps[1U])));
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

    if (shard.unit && shard.isProbDirty) {
        shard.isProbDirty = false;

        QInterfacePtr unit = shard.unit;
        bitLenInt mapped = shard.mapped;
        real1_f prob = unit->Prob(mapped);
        shard.amp1 = complex((real1)sqrt(prob), ZERO_R1);
        shard.amp0 = complex((real1)sqrt(ONE_R1 - prob), ZERO_R1);
        ClampShard(qubit);
    }

    if (IS_NORM_0(shard.amp1)) {
        logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp1)));
        SeparateBit(false, qubit);
    } else if (IS_NORM_0(shard.amp0)) {
        logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp0)));
        SeparateBit(true, qubit);
    }

    return clampProb((real1_f)norm(shard.amp1));
}

void QUnit::PhaseParity(real1 radians, bitCapInt mask)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QUnit::PhaseParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (bi_compare_0(mask) == 0) {
        return;
    }

    complex phaseFac = complex((real1)cos(radians / 2), (real1)sin(radians / 2));

    if (isPowerOfTwo(mask)) {
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; bi_compare_0(v) != 0; v = nV) {
        bi_and_ip(&nV, v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (size_t i = 0U; i < qIndices.size(); ++i) {
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

    if (eIndices.empty()) {
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

    bitCapInt mappedMask = ZERO_BCI;
    for (size_t i = 0U; i < eIndices.size(); ++i) {
        bi_or_ip(&mappedMask, pow2(shards[eIndices[i]].mapped));
    }

    unit->PhaseParity((real1_f)(flipResult ? -radians : radians), mappedMask);
}

real1_f QUnit::ProbParity(bitCapInt mask)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QUnit::ProbParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (bi_compare_0(mask) == 0) {
        return ZERO_R1_F;
    }

    if (isPowerOfTwo(mask)) {
        return Prob(log2(mask));
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; bi_compare_0(v) != 0; v = nV) {
        bi_and_ip(&nV, v - ONE_BCI); // clear the least significant bit set
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
    for (size_t i = 0U; i < qIndices.size(); ++i) {
        QEngineShard& shard = shards[qIndices[i]];
        if (!(shard.unit)) {
            nOddChance = (shard.pauliBasis != PauliZ) ? norm(SQRT1_2_R1 * (shard.amp0 - shard.amp1)) : shard.Prob();
            oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
            continue;
        }

        RevertBasis1Qb(qIndices[i]);

        bi_or_ip(&(units[shard.unit]), pow2(shard.mapped));
    }

    if (qIndices.empty()) {
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
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QUnit::ForceMParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (bi_compare_0(mask) == 0) {
        return false;
    }

    if (isPowerOfTwo(mask)) {
        return ForceM(log2(mask), result, doForce);
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; bi_compare_0(v) != 0; v = nV) {
        bi_and_ip(&nV, v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (size_t i = 0U; i < qIndices.size(); ++i) {
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

    if (eIndices.empty()) {
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

    bitCapInt mappedMask = ZERO_BCI;
    for (size_t i = 0U; i < eIndices.size(); ++i) {
        bi_or_ip(&mappedMask, pow2(shards[eIndices[i]].mapped));
    }

    return flipResult ^
        (std::dynamic_pointer_cast<QParity>(unit)->ForceMParity(mappedMask, result ^ flipResult, doForce));
}

void QUnit::CUniformParityRZ(const std::vector<bitLenInt>& cControls, bitCapInt mask, real1_f angle)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QUnit::CUniformParityRZ mask out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(cControls, qubitCount,
        "QUnit::CUniformParityRZ parameter controls array values must be within allocated qubit bounds!");

    std::vector<bitLenInt> controls;
    bitCapInt _perm = pow2(cControls.size());
    bi_decrement(&_perm, 1U);
    if (TrimControls(cControls, controls, &_perm)) {
        return;
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; bi_compare_0(v) != 0; v = nV) {
        bi_and_ip(&nV, v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (size_t i = 0U; i < qIndices.size(); ++i) {
        ToPermBasis(qIndices[i]);

        if (CACHED_ZERO(qIndices[i])) {
            continue;
        }

        if (CACHED_ONE(qIndices[i])) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (eIndices.empty()) {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac;
        if (flipResult) {
            phaseFac = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, -sine);
        }
        if (controls.empty()) {
            return Phase(phaseFac, phaseFac, 0U);
        } else {
            return MCPhase(controls, phaseFac, phaseFac, 0U);
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
        if (controls.empty()) {
            return Phase(phaseFacAdj, phaseFac, eIndices[0U]);
        } else {
            return MCPhase(controls, phaseFacAdj, phaseFac, eIndices[0U]);
        }
    }

    for (size_t i = 0U; i < eIndices.size(); ++i) {
        shards[eIndices[i]].isPhaseDirty = true;
    }

    QInterfacePtr unit = Entangle(eIndices);

    bitCapInt mappedMask = ZERO_BCI;
    for (size_t i = 0U; i < eIndices.size(); ++i) {
        bi_or_ip(&mappedMask, pow2(shards[eIndices[i]].mapped));
    }

    if (controls.empty()) {
        std::dynamic_pointer_cast<QParity>(unit)->UniformParityRZ(mappedMask, flipResult ? -angle : angle);
    } else {
        std::vector<bitLenInt*> ebits(controls.size());
        for (size_t i = 0U; i < controls.size(); ++i) {
            ebits[i] = &controls[i];
        }

        Entangle(ebits);
        unit = Entangle({ controls[0U], eIndices[0U] });

        std::vector<bitLenInt> controlsMapped(controls.size());
        for (size_t i = 0U; i < controls.size(); ++i) {
            QEngineShard& cShard = shards[controls[i]];
            controlsMapped[i] = cShard.mapped;
            cShard.isPhaseDirty = true;
        }

        std::dynamic_pointer_cast<QParity>(unit)->CUniformParityRZ(
            controlsMapped, mappedMask, flipResult ? -angle : angle);
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

    const real1_f prob = ONE_R1_F / 2 - unit->Prob(shard.mapped);
    unit->Dispose(mapped, 1U, value ? ONE_BCI : ZERO_BCI);

    if (!unit->isBinaryDecisionTree() && ((ONE_R1 / 2 - abs(prob)) > FP_NORM_EPSILON)) {
        unit->UpdateRunningNorm();
        if (!doNormalize) {
            unit->NormalizeState();
        }
    }

    // Update the mappings.
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

    logFidelity = log(GetUnitaryFidelity());

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
    if (useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID)) {
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            QEngineShard& shard = shards[i];
            if (shard.unit && shard.unit->isClifford()) {
                shard.unit->MAll();
            }
        }
    }
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i].IsInvertControl()) {
            // Measurement commutes with control
            M(i);
        }
    }

    bitCapInt toRet = ZERO_BCI;

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        QInterfacePtr toFind = shards[i].unit;
        if (!toFind) {
            real1_f prob = (real1_f)norm(shards[i].amp1);
            if ((prob >= ONE_R1) || ((prob > ZERO_R1) && (Rand() <= prob))) {
                shards[i].amp0 = ZERO_CMPLX;
                shards[i].amp1 = GetNonunitaryPhase();
                bi_or_ip(&toRet, pow2(i));
            } else {
                shards[i].amp0 = GetNonunitaryPhase();
                shards[i].amp1 = ZERO_CMPLX;
            }
        } else if (M(i)) {
            bi_or_ip(&toRet, pow2(i));
        }
    }

    const double origFidelity = logFidelity;
    SetPermutation(toRet);
    logFidelity = origFidelity;

    return toRet;
}

std::map<bitCapInt, int> QUnit::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    if (qPowers.size() == shards.size()) {
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            RevertBasis1Qb(i);
        }
    } else {
        ToPermBasisProb();
    }

    bitLenInt index;
    std::vector<bitLenInt> qIndices(qPowers.size());
    std::map<bitLenInt, bitCapInt> iQPowers;
    for (size_t i = 0U; i < qPowers.size(); ++i) {
        index = log2(qPowers[i]);
        qIndices[i] = index;
        iQPowers[index] = pow2(i);
    }

    ThrowIfQbIdArrayIsBad(qIndices, qubitCount,
        "QInterface::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");

    std::map<QInterfacePtr, std::vector<bitCapInt>> subQPowers;
    std::map<QInterfacePtr, std::vector<bitCapInt>> subIQPowers;
    std::vector<bitLenInt> singleBits;

    for (size_t i = 0U; i < qPowers.size(); ++i) {
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
    combinedResults[ZERO_BCI] = (int)shots;

    for (const auto& subQPower : subQPowers) {
        QInterfacePtr unit = subQPower.first;
        std::map<bitCapInt, int> unitResults = unit->MultiShotMeasureMask(subQPower.second, shots);
        std::map<bitCapInt, int> topLevelResults;
        for (const auto& unitResult : unitResults) {
            bitCapInt mask = ZERO_BCI;
            for (size_t i = 0U; i < subQPower.second.size(); ++i) {
                if (bi_and_1(unitResult.first >> i)) {
                    bi_or_ip(&mask, subIQPowers[unit][i]);
                }
            }
            topLevelResults[mask] = unitResult.second;
        }
        // Release unitResults memory:
        unitResults = std::map<bitCapInt, int>();

        // If either map is fully |0>, nothing changes (after the swap).
        if ((bi_compare_0(topLevelResults.begin()->first) == 0) && (topLevelResults[ZERO_BCI] == (int)shots)) {
            continue;
        }
        if ((bi_compare_0(combinedResults.begin()->first) == 0) && (combinedResults[ZERO_BCI] == (int)shots)) {
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
            const auto pickIter = topLevelResults.begin();
            for (const auto& combinedResult : combinedResults) {
                nCombinedResults[combinedResult.first | pickIter->first] = combinedResult.second;
            }
            combinedResults = nCombinedResults;
            continue;
        }

        // ... Otherwise, we've committed to simulating a random pairing selection from either side, (but
        // `topLevelResults` has fewer or the same count of keys).
        int shotsLeft = shots;
        for (const auto& combinedResult : combinedResults) {
            for (int shot = 0; shot < combinedResult.second; ++shot) {
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

                ++(nCombinedResults[combinedResult.first | pickIter->first]);

                --(pickIter->second);
                if (!pickIter->second) {
                    topLevelResults.erase(pickIter);
                }
            }
        }
        combinedResults = nCombinedResults;
    }

    for (size_t i = 0U; i < singleBits.size(); ++i) {
        index = singleBits[i];

        real1_f prob = clampProb((real1_f)norm(shards[index].amp1));
        if (prob == ZERO_R1) {
            continue;
        }

        std::map<bitCapInt, int> nCombinedResults;
        if (prob == ONE_R1) {
            for (const auto& combinedResult : combinedResults) {
                nCombinedResults[combinedResult.first | iQPowers[index]] = combinedResult.second;
            }
        } else {
            for (const auto& combinedResult : combinedResults) {
                bitCapInt zeroPerm = combinedResult.first;
                bitCapInt onePerm = combinedResult.first | iQPowers[index];
                for (int shot = 0; shot < combinedResult.second; ++shot) {
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

    if (qPowers.size() != shards.size()) {
        return combinedResults;
    }

    std::map<bitCapInt, int> toRet;
    for (const auto& combinedResult : combinedResults) {
        bitCapInt perm = combinedResult.first;

        for (size_t i = 0U; i < qIndices.size(); ++i) {
            QEngineShard& shard = shards[qIndices[i]];
            ShardToPhaseMap controlsShards = bi_and_1(perm >> i) ? shard.controlsShards : shard.antiControlsShards;
            for (const auto& phaseShard : controlsShards) {
                if (!phaseShard.second->isInvert) {
                    continue;
                }

                QEngineShardPtr partner = phaseShard.first;
                const bitLenInt target = FindShardIndex(partner);

                for (size_t j = 0U; j < qIndices.size(); ++j) {
                    if (qIndices[j] == target) {
                        bi_xor_ip(&perm, pow2(j));
                        break;
                    }
                }
            }
        }

        toRet[perm] += combinedResult.second;
    }

    return toRet;
}

void QUnit::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    if (qPowers.size() != shards.size()) {
        ToPermBasisProb();

        QInterfacePtr unit = shards[log2(qPowers[0U])].unit;
        if (unit) {
            std::vector<bitCapInt> mappedIndices(qPowers.size());
            for (bitLenInt j = 0U; j < qubitCount; ++j) {
                if (bi_compare(qPowers[0U], pow2(j)) == 0) {
                    mappedIndices[0U] = pow2(shards[j].mapped);
                    break;
                }
            }
            for (size_t i = 1U; i < qPowers.size(); ++i) {
                const size_t qubit = log2(qPowers[i]);
                if (qubit >= qubitCount) {
                    throw std::invalid_argument(
                        "QUnit::MultiShotMeasureMask parameter qPowers array values must be within "
                        "allocated qubit bounds!");
                }
                if (unit != shards[qubit].unit) {
                    unit = NULL;
                    break;
                }
                for (bitLenInt j = 0U; j < qubitCount; ++j) {
                    if (bi_compare(qPowers[i], pow2(j)) == 0) {
                        mappedIndices[i] = pow2(shards[j].mapped);
                        break;
                    }
                }
            }

            if (unit) {
                unit->MultiShotMeasureMask(mappedIndices, shots, shotsArray);
                return;
            }
        }
    }

    std::map<bitCapInt, int> results = MultiShotMeasureMask(qPowers, shots);

    size_t j = 0U;
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end() && (j < shots)) {
        for (int i = 0; i < it->second; ++i) {
            shotsArray[j] = (unsigned)(bitCapIntOcl)it->first;
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
        shards[i + start] = QEngineShard(bi_and_1(value >> i) != 0U, GetNonunitaryPhase());
    }
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

    const bool isSameUnit = IS_SAME_UNIT(shard1, shard2);

    if (isSameUnit || ARE_CLIFFORD(shard1, shard2)) {
        QInterfacePtr unit = Entangle({ qubit1, qubit2 });
        if (isInverse) {
            unit->IISwap(shard1.mapped, shard2.mapped);
        } else {
            unit->ISwap(shard1.mapped, shard2.mapped);
        }
        shard1.MakeDirty();
        shard2.MakeDirty();

        if (isSameUnit && !ARE_CLIFFORD(shard1, shard2)) {
            TrySeparate(qubit1);
            TrySeparate(qubit2);
        }
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

    const bool isSameUnit = IS_SAME_UNIT(shard1, shard2);
    Entangle({ qubit1, qubit2 })->SqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();

    if (isSameUnit) {
        TrySeparate(qubit1);
        TrySeparate(qubit2);
    }
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

    const bool isSameUnit = IS_SAME_UNIT(shard1, shard2);
    Entangle({ qubit1, qubit2 })->ISqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();

    if (isSameUnit) {
        TrySeparate(qubit1);
        TrySeparate(qubit2);
    }
}

void QUnit::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    const std::vector<bitLenInt> controls{ qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
        MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));
    const bool wasSameUnit = IS_SAME_UNIT(shards[qubit1], shards[qubit2]) &&
        (!ARE_CLIFFORD(shards[qubit1], shards[qubit2]) || !(IS_1_CMPLX(expIPhi) || IS_1_CMPLX(-expIPhi)));

    const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
    if (!wasSameUnit && ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON)) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
    if (!wasSameUnit && ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON)) {
        IISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
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

    const bool isSameUnit = IS_SAME_UNIT(shard1, shard2);
    Entangle({ qubit1, qubit2 })->FSim(theta, phi, shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();

    if (isSameUnit && !ARE_CLIFFORD(shard1, shard2)) {
        TrySeparate(qubit1);
        TrySeparate(qubit2);
    }
}

void QUnit::UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
    const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
{
    // If there are no controls, this is equivalent to the single bit gate.
    if (controls.empty()) {
        Mtrx(mtrxs, qubitIndex);
        return;
    }

    if (qubitIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QUnit::UniformlyControlledSingleBit control is out-of-bounds!");

    std::vector<bitLenInt> trimmedControls;
    std::vector<bitCapInt> skipPowers;
    bitCapInt skipValueMask = ZERO_BCI;
    for (size_t i = 0U; i < controls.size(); ++i) {
        if (!CheckBitsPermutation(controls[i])) {
            trimmedControls.push_back(controls[i]);
        } else {
            skipPowers.push_back(pow2(i));
            if (SHARD_STATE(shards[controls[i]])) {
                bi_or_ip(&skipValueMask, pow2(i));
            }
        }
    }

    // If all controls are in eigenstates, we can avoid entangling them.
    if (trimmedControls.empty()) {
        bitCapInt controlPerm = GetCachedPermutation(controls);
        complex mtrx[4U];
        std::copy(mtrxs + ((bitCapIntOcl)controlPerm << 2U), mtrxs + (((bitCapIntOcl)controlPerm + 1U) << 2U), mtrx);
        Mtrx(mtrx, qubitIndex);
        return;
    }

    std::vector<bitLenInt> bits(trimmedControls.size() + 1U);
    for (size_t i = 0U; i < trimmedControls.size(); ++i) {
        bits[i] = trimmedControls[i];
    }
    bits[trimmedControls.size()] = qubitIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(trimmedControls.size() + 1U);
    for (size_t i = 0U; i < bits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    std::vector<bitLenInt> mappedControls(trimmedControls.size());
    for (size_t i = 0U; i < trimmedControls.size(); ++i) {
        mappedControls[i] = shards[trimmedControls[i]].mapped;
        shards[trimmedControls[i]].isPhaseDirty = true;
    }

    unit->UniformlyControlledSingleBit(mappedControls, shards[qubitIndex].mapped, mtrxs, skipPowers, skipValueMask);

    shards[qubitIndex].MakeDirty();

    if (!isReactiveSeparate || freezeBasis2Qb) {
        return;
    }

    // Skip 2-qubit-at-once check for 2 total qubits.
    if (bits.size() == 2U) {
        TrySeparate(bits[0U]);
        TrySeparate(bits[1U]);
        return;
    }

    // Otherwise, we can try all 2-qubit combinations.
    for (size_t i = 0U; i < (bits.size() - 1U); ++i) {
        for (size_t j = i + 1U; j < bits.size(); ++j) {
            TrySeparate(bits[i], bits[j]);
        }
    }
}

void QUnit::H(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::H qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];
    const bool isClifford =
        useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID) && (!shard.unit || shard.unit->isClifford());

    if (isClifford) {
        RevertBasis1Qb(target);
        RevertBasis2Qb(target);
    } else {
        RevertBasisY(target);
        CommuteH(target);
    }

    shard.pauliBasis = (shard.pauliBasis == PauliZ) ? PauliX : PauliZ;

    if (isClifford) {
        RevertBasis1Qb(target);
    }
}

void QUnit::S(bitLenInt target)
{
    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::S qubit index parameter must be within allocated qubit bounds!");
    }

    QEngineShard& shard = shards[target];
    const bool isClifford =
        useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID) && (!shard.unit || shard.unit->isClifford());

    if (isClifford) {
        RevertBasis1Qb(target);
        RevertBasis2Qb(target);
    } else {
        shard.CommutePhase(ONE_CMPLX, I_CMPLX);
    }

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
    const bool isClifford =
        useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID) && (!shard.unit || shard.unit->isClifford());

    if (isClifford) {
        RevertBasis1Qb(target);
        RevertBasis2Qb(target);
    } else {
        shard.CommutePhase(ONE_CMPLX, -I_CMPLX);
    }

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

#define CTRLED_GEN_WRAP(ctrld)                                                                                         \
    ApplyEitherControlled(                                                                                             \
        controlVec, { target },                                                                                        \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            complex trnsMtrx[4U]{ ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                    \
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
                complex trnsMtrx[4U]{ ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                \
                if (isInvert) {                                                                                        \
                    TransformXInvert(top, bottom, trnsMtrx);                                                           \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            } else if (shards[target].pauliBasis == PauliY) {                                                          \
                complex trnsMtrx[4U]{ ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                \
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
    ThrowIfQbIdArrayIsBad(controls, qubitCount,                                                                        \
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
    bitCapInt _perm = anti ? ZERO_BCI : (pow2(controls.size()) - ONE_BCI);                                             \
    if (TrimControls(controls, controlVec, &_perm)) {                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    if (controlVec.empty()) {                                                                                          \
        bare;                                                                                                          \
        return;                                                                                                        \
    }                                                                                                                  \
    ApplyEitherControlled(                                                                                             \
        controlVec, { qubit1, qubit2 },                                                                                \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, false)
#define CTRL_GEN_ARGS mappedControls, trnsMtrx, shards[target].mapped, controlPerm
#define CTRL_S_ARGS mappedControls, shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS mappedControls, topLeft, bottomRight, shards[target].mapped, controlPerm
#define CTRL_I_ARGS mappedControls, topRight, bottomLeft, shards[target].mapped, controlPerm

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
    const bool isClifford =
        useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID) && (!shard.unit || shard.unit->isClifford());

    if (isClifford) {
        RevertBasis1Qb(target);
        RevertBasis2Qb(target);
    } else {
        shard.CommutePhase(topLeft, bottomRight);
    }

    if (shard.pauliBasis == PauliZ) {
        if (shard.unit) {
            shard.unit->Phase(topLeft, bottomRight, shard.mapped);
        }

        shard.amp0 *= topLeft;
        shard.amp1 *= bottomRight;

        return;
    }

    complex mtrx[4U]{ ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
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
    const bool isClifford =
        useTGadget && (engines[0U] == QINTERFACE_STABILIZER_HYBRID) && (!shard.unit || shard.unit->isClifford());

    if (isClifford) {
        RevertBasis1Qb(target);
        RevertBasis2Qb(target);
    } else {
        shard.FlipPhaseAnti();
        shard.CommutePhase(topRight, bottomLeft);
    }

    if (shard.pauliBasis == PauliZ) {
        if (shard.unit) {
            shard.unit->Invert(topRight, bottomLeft, shard.mapped);
        }

        const complex tempAmp1 = bottomLeft * shard.amp0;
        shard.amp0 = topRight * shard.amp1;
        shard.amp1 = tempAmp1;

        return;
    }

    complex mtrx[4U]{ ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
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

void QUnit::UCPhase(const std::vector<bitLenInt>& lControls, complex topLeft, complex bottomRight, bitLenInt target,
    bitCapInt controlPerm)
{
    ThrowIfQbIdArrayIsBad(
        lControls, qubitCount, "QUnit::UCPhase parameter controls array values must be within allocated qubit bounds!");

    if (IS_1_CMPLX(topLeft) && IS_1_CMPLX(bottomRight)) {
        return;
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, controlVec, &controlPerm)) {
        return;
    }

    if (controlVec.empty()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if ((controlVec.size() == 1U) && IS_NORM_0(topLeft - bottomRight)) {
        if (bi_compare_0(controlPerm) != 0) {
            Phase(ONE_CMPLX, bottomRight, controlVec[0U]);
        } else {
            Phase(topLeft, ONE_CMPLX, controlVec[0U]);
        }
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::UCPhase qubit index parameter must be within allocated qubit bounds!");
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);

        const bool isNonzeroCtrlPerm = bi_compare_0(controlPerm) != 0;
        if (isNonzeroCtrlPerm) {
            RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI);
            RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL, {}, { control });
        } else {
            RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL);
            RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI, {}, { control });
        }

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !((IS_SAME(ONE_CMPLX, topLeft) || IS_SAME(-ONE_CMPLX, topLeft)) &&
                    (IS_SAME(ONE_CMPLX, bottomRight) || IS_SAME(-ONE_CMPLX, bottomRight))))) {
            if (isNonzeroCtrlPerm) {
                tShard.AddPhaseAngles(&cShard, topLeft, bottomRight);
                OptimizePairBuffers(control, target, false);
            } else {
                tShard.AddAntiPhaseAngles(&cShard, bottomRight, topLeft);
                OptimizePairBuffers(control, target, true);
            }

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(UCPhase(CTRL_P_ARGS), UCMtrx(CTRL_GEN_ARGS), false, topLeft, bottomRight);
}

void QUnit::UCInvert(const std::vector<bitLenInt>& lControls, complex topRight, complex bottomLeft, bitLenInt target,
    bitCapInt controlPerm)
{
    ThrowIfQbIdArrayIsBad(lControls, qubitCount,
        "QUnit::UCInvert parameter controls array values must be within allocated qubit bounds!");

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::UCInvert qubit index parameter must be within allocated qubit bounds!");
    }

    if (IS_1_CMPLX(topRight) && IS_1_CMPLX(bottomLeft)) {
        if (CACHED_PLUS(target)) {
            return;
        }
    }

    std::vector<bitLenInt> controlVec;
    if (TrimControls(lControls, controlVec, &controlPerm)) {
        return;
    }

    if (controlVec.empty()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (!freezeBasis2Qb && (controlVec.size() == 1U)) {
        const bitLenInt control = controlVec[0U];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        const bool isNonzeroCtrlPerm = bi_compare_0(controlPerm) != 0;
        if (isNonzeroCtrlPerm) {
            RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
            RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL, {}, { control });
        } else {
            RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
            RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI, {}, { control });
        }

        if (!IS_SAME_UNIT(cShard, tShard) &&
            (!ARE_CLIFFORD(cShard, tShard) ||
                !(((IS_SAME(ONE_CMPLX, topRight) || IS_SAME(-ONE_CMPLX, topRight)) &&
                      (IS_SAME(ONE_CMPLX, bottomLeft) || IS_SAME(-ONE_CMPLX, bottomLeft))) ||
                    (((IS_SAME(I_CMPLX, topRight) || IS_SAME(-I_CMPLX, topRight)) &&
                        (IS_SAME(I_CMPLX, bottomLeft) || IS_SAME(-I_CMPLX, bottomLeft))))))) {
            if (isNonzeroCtrlPerm) {
                tShard.AddInversionAngles(&cShard, topRight, bottomLeft);
                OptimizePairBuffers(control, target, false);
            } else {
                tShard.AddAntiInversionAngles(&cShard, bottomLeft, topRight);
                OptimizePairBuffers(control, target, true);
            }

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(UCInvert(CTRL_I_ARGS), UCMtrx(CTRL_GEN_ARGS), true, topRight, bottomLeft);
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

void QUnit::UCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, bitCapInt controlPerm)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        UCPhase(controls, mtrx[0U], mtrx[3U], target, controlPerm);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        UCInvert(controls, mtrx[1U], mtrx[2U], target, controlPerm);
        return;
    }

    ThrowIfQbIdArrayIsBad(
        controls, qubitCount, "QUnit::UCMtrx parameter controls array values must be within allocated qubit bounds!");

    std::vector<bitLenInt> controlVec;
    if (TrimControls(controls, controlVec, &controlPerm)) {
        return;
    }

    if (controlVec.empty()) {
        Mtrx(mtrx, target);
        return;
    }

    if (target >= qubitCount) {
        throw std::invalid_argument("QUnit::MCMtrx qubit index parameter must be within allocated qubit bounds!");
    }

    CTRLED_GEN_WRAP(UCMtrx(CTRL_GEN_ARGS));
}

void QUnit::CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), false);
}

void QUnit::AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), true);
}

void QUnit::CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), true);
}

void QUnit::CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), true);
}

bool QUnit::TrimControls(const std::vector<bitLenInt>& controls, std::vector<bitLenInt>& controlVec, bitCapInt* perm)
{
    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have
    // total or no probability of altering the targets, such that we can still keep them separate.

    if (controls.empty()) {
        // (If we were passed 0 controls, the target functions as a gate without controls.)
        return false;
    }

    // First, no probability checks or buffer flushing.
    for (size_t i = 0U; i < controls.size(); ++i) {
        const bool anti = !bi_and_1(*perm >> i);
        if ((anti && CACHED_ONE(controls[i])) || (!anti && CACHED_ZERO(controls[i]))) {
            // This gate does nothing, so return without applying anything.
            return true;
        }
    }

    // Next, probability checks, but no buffer flushing.
    for (size_t i = 0U; i < controls.size(); ++i) {
        QEngineShard& shard = shards[controls[i]];

        if ((shard.pauliBasis != PauliZ) || shard.IsInvertTarget()) {
            continue;
        }

        ProbBase(controls[i]);

        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_NORM_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        } else if (IS_NORM_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (!bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        }
    }

    // Next, just 1qb buffer flushing.
    for (size_t i = 0U; i < controls.size(); ++i) {
        QEngineShard& shard = shards[controls[i]];

        if ((shard.pauliBasis == PauliZ) || shard.IsInvertTarget()) {
            continue;
        }

        RevertBasis1Qb(controls[i]);

        ProbBase(controls[i]);

        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_NORM_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        } else if (IS_NORM_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (!bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
        }
    }

    // Finally, full buffer flushing, (last resort).
    bitCapInt outPerm = ZERO_BCI;
    for (size_t i = 0U; i < controls.size(); ++i) {
        QEngineShard& shard = shards[controls[i]];

        ToPermBasisProb(controls[i]);

        ProbBase(controls[i]);

        bool isEigenstate = false;
        // This might determine that we can just skip out of the whole gate, in which case we return.
        if (IS_NORM_0(shard.amp1)) {
            Flush0Eigenstate(controls[i]);
            if (bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
            // This control has 100% chance to "fire," so don't entangle it.
            isEigenstate = true;
        } else if (IS_NORM_0(shard.amp0)) {
            Flush1Eigenstate(controls[i]);
            if (!bi_and_1(*perm >> i)) {
                // This gate does nothing, so return without applying anything.
                return true;
            }
            // This control has 100% chance to "fire," so don't entangle it.
            isEigenstate = true;
        }

        if (!isEigenstate) {
            bi_or_ip(&outPerm, bi_and_1(*perm >> i) << controlVec.size());
            controlVec.push_back(controls[i]);
        }
    }

    *perm = outPerm;

    return false;
}

template <typename CF>
void QUnit::ApplyEitherControlled(
    std::vector<bitLenInt> controlVec, const std::vector<bitLenInt> targets, CF cfn, bool isPhase)
{
    // If we've made it this far, we have to form the entangled representation and apply the gate.

    for (size_t i = 0U; i < controlVec.size(); ++i) {
        ToPermBasisProb(controlVec[i]);
    }

    if (targets.size() > 1U) {
        for (size_t i = 0U; i < targets.size(); ++i) {
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
    for (size_t i = 0U; i < allBitsMapped.size(); ++i) {
        ebits[i] = &allBitsMapped[i];
    }

    QInterfacePtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());

    for (size_t i = 0U; i < controlVec.size(); ++i) {
        bitLenInt& c = controlVec[i];
        shards[c].isPhaseDirty = true;
        c = shards[c].mapped;
    }
    for (size_t i = 0U; i < targets.size(); ++i) {
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
    for (size_t i = 0U; i < (allBits.size() - 1U); ++i) {
        for (size_t j = i + 1U; j < allBits.size(); ++j) {
            TrySeparate(allBits[i], allBits[j]);
        }
    }
}

void QUnit::ConvertZToX(bitLenInt i)
{
    QEngineShard& shard = shards[i];

    // WARNING: Might be called when shard is in either Z or X basis
    shard.pauliBasis = (shard.pauliBasis == PauliX) ? PauliZ : PauliX;

    if (shard.unit) {
        shard.unit->H(shard.mapped);
    }

    if (shard.isPhaseDirty || shard.isProbDirty) {
        shard.isProbDirty = true;
        return;
    }

    complex tempAmp1 = SQRT1_2_R1 * (shard.amp0 - shard.amp1);
    shard.amp0 = SQRT1_2_R1 * (shard.amp0 + shard.amp1);
    shard.amp1 = tempAmp1;
    ClampShard(i);
}
void QUnit::ConvertXToY(bitLenInt i)
{
    QEngineShard& shard = shards[i];

    shard.pauliBasis = PauliY;

    const complex mtrx[4U]{ ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX),
        ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX), ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX),
        ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX) };

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }

    if (shard.isPhaseDirty || shard.isProbDirty) {
        shard.isProbDirty = true;
        return;
    }

    complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(i);
}
void QUnit::ConvertYToZ(bitLenInt i)
{
    QEngineShard& shard = shards[i];

    shard.pauliBasis = PauliZ;

    const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, SQRT1_2_R1),
        complex(ZERO_R1, -SQRT1_2_R1) };

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }

    if (shard.isPhaseDirty || shard.isProbDirty) {
        shard.isProbDirty = true;
        return;
    }

    complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(i);
}
void QUnit::ConvertZToY(bitLenInt i)
{
    QEngineShard& shard = shards[i];

    shard.pauliBasis = PauliY;

    const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(ZERO_R1, SQRT1_2_R1) };

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }

    if (shard.isPhaseDirty || shard.isProbDirty) {
        shard.isProbDirty = true;
        return;
    }

    complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(i);
}
void QUnit::ShardAI(bitLenInt qubit, real1_f azimuth, real1_f inclination)
{
    real1 cosineA = (real1)cos(azimuth);
    real1 sineA = (real1)sin(azimuth);
    real1 cosineI = (real1)cos(inclination / 2);
    real1 sineI = (real1)sin(inclination / 2);
    complex expA = complex(cosineA, sineA);
    complex expNegA = complex(cosineA, -sineA);
    complex mtrx[4U]{ cosineI, -expNegA * sineI, expA * sineI, cosineI };

    QEngineShard& shard = shards[qubit];

    const complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * shard.amp1);
    shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * shard.amp1);
    ClampShard(qubit);
}

void QUnit::Flush0Eigenstate(bitLenInt i)
{
    shards[i].DumpControlOf();
    if (randGlobalPhase) {
        shards[i].DumpSamePhaseAntiControlOf();
    }
    RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
}
void QUnit::Flush1Eigenstate(bitLenInt i)
{
    shards[i].DumpAntiControlOf();
    if (randGlobalPhase) {
        shards[i].DumpSamePhaseControlOf();
    }
    RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
}
void QUnit::ToPermBasis(bitLenInt i)
{
    RevertBasis1Qb(i);
    RevertBasis2Qb(i);
}
void QUnit::ToPermBasis(bitLenInt start, bitLenInt length)
{
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis1Qb(start + i);
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis2Qb(start + i);
    }
}
void QUnit::ToPermBasisProb(bitLenInt qubit)
{
    RevertBasis1Qb(qubit);
    RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
}
void QUnit::ToPermBasisProb(bitLenInt start, bitLenInt length)
{
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis1Qb(start + i);
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis2Qb(start + i, ONLY_INVERT, ONLY_TARGETS);
    }
}
void QUnit::ToPermBasisMeasure(bitLenInt qubit)
{
    RevertBasis1Qb(qubit);
    RevertBasis2Qb(qubit, ONLY_INVERT);
    RevertBasis2Qb(qubit, ONLY_PHASE, ONLY_CONTROLS);

    shards[qubit].DumpMultiBit();
}
void QUnit::ToPermBasisMeasure(bitLenInt start, bitLenInt length)
{
    if (!start && (length == qubitCount)) {
        ToPermBasisAllMeasure();
        return;
    }

    std::set<bitLenInt> exceptBits;
    for (bitLenInt i = 0U; i < length; ++i) {
        exceptBits.insert(start + i);
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis1Qb(start + i);
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        RevertBasis2Qb(start + i, ONLY_INVERT);
        RevertBasis2Qb(start + i, ONLY_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, exceptBits);
        shards[start + i].DumpMultiBit();
    }
}
void QUnit::ToPermBasisAllMeasure()
{
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        RevertBasis1Qb(i);
    }
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        shards[i].ClearInvertPhase();
        RevertBasis2Qb(i, ONLY_INVERT);
        shards[i].DumpMultiBit();
    }
}

#if ENABLE_ALU
void QUnit::CINC(bitCapInt toMod, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CINC range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(
        controls, qubitCount, "QUnit::CINC parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    bitCapInt _perm = pow2(controls.size());
    bi_decrement(&_perm, 1U);
    if (TrimControls(controls, controlVec, &_perm)) {
        return;
    }

    if (controlVec.empty()) {
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

    // Make sure the flag bits are entangled in the same QU.
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
        bi_increment(&toMod, 1U);
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl signMask = pow2Ocl(length - 1U);
    const bitCapIntOcl inOutInt = (bitCapIntOcl)GetCachedPermutation(start, length);
    const bitCapIntOcl inInt = (bitCapIntOcl)toMod;

    bool isOverflow;
    bitCapInt outInt;
    if (isAdd) {
        isOverflow = (overflowIndex != (bitLenInt)(-1)) && isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        outInt = inOutInt + toMod;
    } else {
        isOverflow = (overflowIndex != (bitLenInt)(-1)) && isOverflowSub(inOutInt, inInt, signMask, lengthPower);
        outInt = (inOutInt + lengthPower) - toMod;
    }

    const bool carryOut = bi_compare(outInt, lengthPower) >= 0;
    if (carryOut) {
        bi_and_ip(&outInt, lengthPower - ONE_BCI);
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
    std::vector<bitLenInt> controls)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::INT range is out-of-bounds!");
    }

    if (hasCarry && carryIndex >= qubitCount) {
        throw std::invalid_argument("QUnit::INT carryIndex parameter must be within allocated qubit bounds!");
    }

    if (controls.size()) {
        ThrowIfQbIdArrayIsBad(
            controls, qubitCount, "QUnit::INT parameter controls array values must be within allocated qubit bounds!");
    }

    // Keep the bits separate, if cheap to do so:
    bi_and_ip(&toMod, pow2Mask(length));
    if (bi_compare_0(toMod) == 0) {
        return;
    }

    if (!hasCarry && CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    // All cached classical control bits have been removed from controlVec.
    const bitLenInt controlLen = controls.size();
    std::vector<bitLenInt> allBits(controlLen + 1U);
    std::copy(controls.begin(), controls.end(), allBits.begin());
    std::sort(allBits.begin(), allBits.begin() + controlLen);

    std::vector<bitLenInt*> ebits(allBits.size());
    for (size_t i = 0; i < ebits.size(); ++i) {
        ebits[i] = &allBits[i];
    }

    // Try ripple addition, to avoid entanglement.
    const bitLenInt origLength = length;
    bool carry = false;
    bitLenInt i = 0U;
    while (i < origLength) {
        bool toAdd = bi_and_1(toMod) != 0U;

        if (toAdd == carry) {
            bi_rshift_ip(&toMod, 1U);
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
                MCInvert(controls, ONE_CMPLX, ONE_CMPLX, start);
            }
            carry = (total > 1);

            bi_rshift_ip(&toMod, 1U);
            ++start;
            --length;
            ++i;
        } else {
            // The carry-in is classical.
            if (carry) {
                carry = false;
                bi_increment(&toMod, 1U);
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
                bi_lshift_ip(&bitMask, 1U);

                toAdd = bi_compare_0(toMod & bitMask) != 0U;
                bi_or_ip(&partMod, toMod & bitMask);

                partStart = start + partLength - 1U;
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
                    DirtyShardIndexVector(allBits);
                    QInterfacePtr unit = Entangle(ebits);
                    std::vector<bitLenInt> lControls(controlLen);
                    for (bitLenInt cIndex = 0U; cIndex < controlLen; ++cIndex) {
                        lControls[cIndex] = shards[controls[cIndex]].mapped;
                    }
                    unit->CINC(partMod, shards[start].mapped, partLength, lControls);
                } else {
                    shards[start].unit->INC(partMod, shards[start].mapped, partLength);
                }

                carry = toAdd;
                bi_rshift_ip(&toMod, partLength);
                start += partLength;
                length -= partLength;

                // Break out of the inner loop and return to the flow of the containing loop.
                // (Otherwise, we hit the "continue" calls above.)
                break;
            } while (i < origLength);
        }
    }

    if (!length && (bi_compare_0(toMod) == 0)) {
        // We were able to avoid entangling the carry.
        if (hasCarry && carry) {
            MCInvert(controls, ONE_CMPLX, ONE_CMPLX, carryIndex);
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
            DirtyShardIndexVector(allBits);
            QInterfacePtr unit = Entangle(ebits);
            std::vector<bitLenInt> lControls(controlLen);
            for (bitLenInt cIndex = 0U; cIndex < controlLen; ++cIndex) {
                lControls[cIndex] = shards[controls[cIndex]].mapped;
            }
            unit->CINC(toMod, shards[start].mapped, length, lControls);
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
        bi_increment(&toAdd, 1U);
    }

    INT(toAdd, inOutStart, length, carryIndex, true);
}

/// Subtract integer (without sign, with carry)
void QUnit::DECC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
    } else {
        bi_increment(&toSub, 1U);
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

    bi_and_ip(&toMod, pow2Mask(length));
    if (bi_compare_0(toMod) == 0) {
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

    const bool addendNeg = bi_compare_0(toMod & pow2(length - 1U)) != 0;
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
    if (bi_compare_0(toMul) == 0) {
        SetReg(inOutStart, length, ZERO_BCI);
        SetReg(carryStart, length, ZERO_BCI);
        return;
    } else if (bi_compare_1(toMul) == 0) {
        SetReg(carryStart, length, ZERO_BCI);
        return;
    }

    if (CheckBitsPermutation(inOutStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt res = GetCachedPermutation(inOutStart, length) * toMul;
        SetReg(inOutStart, length, res & lengthMask);
        SetReg(carryStart, length, (res >> length) & lengthMask);
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
    if (bi_compare_1(toDiv) == 0) {
        return;
    }

    if (CheckBitsPermutation(inOutStart, length) && CheckBitsPermutation(carryStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt origRes =
            GetCachedPermutation(inOutStart, length) | (GetCachedPermutation(carryStart, length) << length);
        bitCapInt quo, rem;
        bi_div_mod(origRes, toDiv, &quo, &rem);
        if (bi_compare_0(rem) == 0) {
            SetReg(inOutStart, length, quo & lengthMask);
            SetReg(carryStart, length, (quo >> length) & lengthMask);
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

    if (bi_compare_1(toMod) == 0) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        bitCapInt res;
        bi_div_mod(intPow(toMod, GetCachedPermutation(inStart, length)), modN, NULL, &res);
        SetReg(outStart, length, res);
        return;
    }

    SetReg(outStart, length, ZERO_BCI);

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
    for (size_t i = 0U; i < controlVec.size(); ++i) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    bits[controlVec.size() + 1U] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (size_t i = 0U; i < ebits.size(); ++i) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    if (controlVec.size()) {
        controlsMapped->resize(controlVec.size());
        for (size_t i = 0U; i < controlVec.size(); ++i) {
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

    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(
        toMod, shards[start].mapped, shards[carryStart].mapped, length, controlsMapped);

    DirtyShardRange(start, length);
}

void QUnit::CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt> controlVec)
{
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*std::dynamic_pointer_cast<QAlu>(unit)).*fn)(
        toMod, modN, shards[start].mapped, shards[carryStart].mapped, length, controlsMapped);

    DirtyShardRangePhase(start, length);
}

void QUnit::CMUL(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CMUL inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CMUL carryStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(
        controls, qubitCount, "QUnit::CMUL parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    bitCapInt _perm = pow2(controls.size());
    bi_decrement(&_perm, 1U);
    if (TrimControls(controls, controlVec, &_perm)) {
        return;
    }

    if (controlVec.empty()) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QAlu::CMUL, toMod, start, carryStart, length, controlVec);
}

void QUnit::CDIV(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CDIV inOutStart range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CDIV carryStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(
        controls, qubitCount, "QUnit::CDIV parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    bitCapInt _perm = pow2(controls.size());
    bi_decrement(&_perm, 1U);
    if (TrimControls(controls, controlVec, &_perm)) {
        return;
    }

    if (controlVec.empty()) {
        DIV(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QAlu::CDIV, toMod, start, carryStart, length, controlVec);
}

void QUnit::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (controls.empty()) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, ZERO_BCI);

    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QUnit::CPOWModNOut inStart range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount,
        "QUnit::CPOWModNOut parameter controls array values must be within allocated qubit bounds!");

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    bitCapInt _perm = pow2(controls.size());
    bi_decrement(&_perm, 1U);
    if (TrimControls(controls, controlVec, &_perm)) {
        return;
    }

    CMULModx(&QAlu::CPOWModNOut, toMod, modN, inStart, outStart, length, controlVec);
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(indexStart, indexLength);
    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt value = ZERO_BCI;
    for (bitCapIntOcl j = 0U; j < valueBytes; ++j) {
        bi_or_ip(&value, values[indexInt * valueBytes + j] << (8U * j));
    }

    return value;
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(start, length);
    const bitLenInt bytes = (length + 7U) / 8U;
    bitCapInt value = ZERO_BCI;
    for (bitCapIntOcl j = 0U; j < bytes; ++j) {
        bi_or_ip(&value, values[indexInt * bytes + j] << (8U * j));
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
        return ZERO_BCI;
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
        return ZERO_BCI;
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
        return ZERO_BCI;
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

double QUnit::GetUnitaryFidelity()
{
    double fidelity = exp(logFidelity);

    std::vector<QInterfacePtr> units;
    for (size_t i = 0U; i < shards.size(); ++i) {
        QInterfacePtr toFind = shards[i].unit;
        if (toFind && (find(units.begin(), units.end(), toFind) == units.end())) {
            units.push_back(toFind);
            fidelity *= toFind->GetUnitaryFidelity();
        }
    }
    return fidelity;
}

bool QUnit::ParallelUnitApply(ParallelUnitFn fn, real1_f param1, real1_f param2, real1_f param3, int64_t param4)
{
    std::vector<QInterfacePtr> units;
    for (size_t i = 0U; i < shards.size(); ++i) {
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
        if (bi_compare(GetCachedPermutation((bitLenInt)0U, qubitCount),
                toCompare->GetCachedPermutation((bitLenInt)0U, qubitCount)) == 0) {
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

    QUnitPtr copyPtr = std::make_shared<QUnit>(engines, qubitCount, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);

    copyPtr->SetReactiveSeparate(isReactiveSeparate);
    copyPtr->SetTInjection(useTGadget);
    copyPtr->SetNcrp(roundingThreshold);
    copyPtr->logFidelity = logFidelity;

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
    const std::vector<bitLenInt> controls{ control };

    const complex polarDiff = phaseShard->cmplxDiff;
    const complex polarSame = phaseShard->cmplxSame;

    freezeBasis2Qb = true;
    if (phaseShard->isInvert) {
        if (isAnti) {
            MACInvert(controls, polarSame, polarDiff, target);
        } else {
            MCInvert(controls, polarDiff, polarSame, target);
        }
    } else {
        if (isAnti) {
            MACPhase(controls, polarSame, polarDiff, target);
        } else {
            MCPhase(controls, polarDiff, polarSame, target);
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

    for (const auto& phaseShard : controlsShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        const QEngineShardPtr& partner = phaseShard.first;

        if (buffer->isInvert) {
            continue;
        }

        const complex& polarDiff = buffer->cmplxDiff;
        const complex& polarSame = buffer->cmplxSame;

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddAntiPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    controlsShards = shard.antiControlsShards;

    for (const auto& phaseShard : controlsShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        const QEngineShardPtr& partner = phaseShard.first;

        if (buffer->isInvert) {
            continue;
        }

        const complex& polarDiff = buffer->cmplxDiff;
        const complex& polarSame = buffer->cmplxSame;

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

    for (const auto& phaseShard : targetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;

        const complex& polarDiff = buffer->cmplxDiff;
        const complex& polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard.first;

        if (IS_SAME(polarDiff, polarSame)) {
            continue;
        }

        if (buffer->isInvert && IS_OPPOSITE(polarDiff, polarSame)) {
            continue;
        }

        const bitLenInt control = FindShardIndex(partner);
        shard.RemoveControl(partner);
        ApplyBuffer(buffer, control, bitIndex, false);
    }

    targetOfShards = shard.antiTargetOfShards;

    for (const auto& phaseShard : targetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;

        const complex& polarDiff = buffer->cmplxDiff;
        const complex& polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard.first;

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
