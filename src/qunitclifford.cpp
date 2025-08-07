//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QUnitClifford maintains explicit separability of qubits as an optimization on a
// QStabilizer. See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qunitclifford.hpp"

#define IS_0_R1(r) (abs(r) <= REAL1_EPSILON)
#define IS_1_R1(r) (abs(r) <= REAL1_EPSILON)

namespace Qrack {

QUnitClifford::QUnitClifford(bitLenInt n, const bitCapInt& perm, qrack_rand_gen_ptr rgp, const complex& phaseFac,
    bool doNorm, bool randomGlobalPhase, bool ignored2, int64_t ignored3, bool useHardwareRNG, bool ignored4,
    real1_f ignored5, std::vector<int64_t> ignored6, bitLenInt ignored7, real1_f ignored8)
    : QInterface(n, rgp, doNorm, useHardwareRNG, randomGlobalPhase, REAL1_EPSILON)
    , phaseOffset(ONE_CMPLX)
    , isReactiveSeparate(true)
{
    SetPermutation(perm, phaseFac);
}

QInterfacePtr QUnitClifford::CloneBody(QUnitCliffordPtr copyPtr)
{
    std::map<QStabilizerPtr, QStabilizerPtr> dupeEngines;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        copyPtr->shards[i].mapped = shards[i].mapped;

        QStabilizerPtr unit = shards[i].unit;
        if (dupeEngines.find(unit) == dupeEngines.end()) {
            dupeEngines[unit] = std::dynamic_pointer_cast<QStabilizer>(unit->Clone());
        }

        copyPtr->shards[i].unit = dupeEngines[unit];
    }

    return copyPtr;
}

real1_f QUnitClifford::ExpVarBitsFactorized(
    bool isExp, const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset)
{
    if (perms.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QUnitClifford::ExpectationBitsFactorized() must supply at least twice as many 'perms' as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QUnitClifford::ExpectationBitsAll parameter qubits vector values must be within allocated qubit bounds!");

    std::map<QStabilizerPtr, std::vector<bitLenInt>> qubitMap;
    std::map<QStabilizerPtr, std::vector<bitCapInt>> permMap;
    for (size_t i = 0U; i < bits.size(); ++i) {
        const CliffordShard& shard = shards[bits[i]];
        qubitMap[shard.unit].push_back(shard.mapped);
        permMap[shard.unit].push_back(perms[i << 1U]);
        permMap[shard.unit].push_back(perms[(i << 1U) | 1U]);
    }

    real1_f expectation = ZERO_R1;
    if (isExp) {
        for (const auto& p : qubitMap) {
            expectation += p.first->ExpectationBitsFactorized(p.second, permMap[p.first], offset);
        }
    } else {
        for (const auto& p : qubitMap) {
            expectation += p.first->VarianceBitsFactorized(p.second, permMap[p.first], offset);
        }
    }

    return expectation;
}

real1_f QUnitClifford::ExpVarFloatsFactorized(
    bool isExp, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
{
    if (weights.size() < (bits.size() << 1U)) {
        throw std::invalid_argument(
            "QUnitClifford::ExpectationFloatsFactorized() must supply at least twice as many weights as bits!");
    }

    ThrowIfQbIdArrayIsBad(bits, qubitCount,
        "QUnitClifford::ExpectationFloatsFactorized parameter qubits vector values must be within allocated qubit "
        "bounds!");

    std::map<QStabilizerPtr, std::vector<bitLenInt>> qubitMap;
    std::map<QStabilizerPtr, std::vector<real1_f>> weightMap;
    for (size_t i = 0U; i < bits.size(); ++i) {
        const CliffordShard& shard = shards[bits[i]];
        qubitMap[shard.unit].push_back(shard.mapped);
        weightMap[shard.unit].push_back(weights[i << 1U]);
        weightMap[shard.unit].push_back(weights[(i << 1U) | 1U]);
    }

    real1_f expectation = ZERO_R1;
    if (isExp) {
        for (const auto& p : qubitMap) {
            expectation += p.first->ExpectationFloatsFactorized(p.second, weightMap[p.first]);
        }
    } else {
        for (const auto& p : qubitMap) {
            expectation += p.first->VarianceFloatsFactorized(p.second, weightMap[p.first]);
        }
    }

    return expectation;
}

real1_f QUnitClifford::ProbPermRdm(const bitCapInt& perm, bitLenInt ancillaeStart)
{
    if (ancillaeStart > qubitCount) {
        throw std::invalid_argument("QUnitClifford::ProbPermRdm() ancillaeStart is out-of-bounds!");
    }

    std::map<QStabilizerPtr, bitLenInt> ancillaMap;
    for (size_t i = 0U; i < qubitCount; ++i) {
        const CliffordShard& shard = shards[i];
        if (ancillaMap.find(shard.unit) == ancillaMap.end()) {
            OrderContiguous(shard.unit);
            ancillaMap[shard.unit] = shard.unit->GetQubitCount();
        }
    }

    std::map<QStabilizerPtr, bitCapInt> permMap;
    for (size_t i = 0U; i < ancillaeStart; ++i) {
        if (bi_compare_0(perm & pow2(i)) == 0) {
            continue;
        }
        const CliffordShard& shard = shards[i];
        bi_or_ip(&(permMap[shard.unit]), pow2(shard.mapped));
    }
    for (size_t i = ancillaeStart; i < qubitCount; ++i) {
        const CliffordShard& shard = shards[i];
        if (ancillaMap[shard.unit] > shard.mapped) {
            ancillaMap[shard.unit] = shard.mapped;
        }
    }

    real1 prob = ONE_R1;
    for (const auto& p : ancillaMap) {
        prob *= (real1)p.first->ProbPermRdm(permMap[p.first], p.second);
    }

    return (real1_f)prob;
}

real1_f QUnitClifford::ProbMask(const bitCapInt& mask, const bitCapInt& perm)
{
    bitCapInt v = mask; // count the number of bits set in v
    std::vector<bitLenInt> bits;
    while (bi_compare_0(v) != 0) {
        bitCapInt oldV = v;
        bi_and_ip(&v, v - ONE_BCI); // clear the least significant bit set
        bits.push_back(log2((v ^ oldV) & oldV));
    }

    std::map<QStabilizerPtr, bitCapInt> maskMap;
    std::map<QStabilizerPtr, bitCapInt> permMap;
    for (size_t i = 0U; i < bits.size(); ++i) {
        const CliffordShard& shard = shards[bits[i]];
        bi_or_ip(&(maskMap[shard.unit]), pow2(shard.mapped));
        if (bi_compare_0(pow2(bits[i]) & perm) != 0) {
            bi_or_ip(&(permMap[shard.unit]), pow2(shard.mapped));
        }
    }

    real1 expectation = ZERO_R1;
    for (const auto& p : maskMap) {
        expectation += (real1)p.first->ProbMask(p.second, permMap[p.first]);
    }

    return (real1_f)expectation;
}

void QUnitClifford::SetPermutation(const bitCapInt& perm, const complex& phaseFac)
{
    Dump();

    shards.clear();

    if (phaseFac != CMPLX_DEFAULT_ARG) {
        phaseOffset = phaseFac;
    } else if (randGlobalPhase) {
        phaseOffset = std::polar(ONE_R1, (real1)(2 * PI_R1 * Rand()));
    } else {
        phaseOffset = ONE_CMPLX;
    }

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        shards.emplace_back(0U, MakeStabilizer(1U, bi_and_1(perm >> i), ONE_CMPLX));
    }
}

void QUnitClifford::Detach(bitLenInt start, bitLenInt length, QUnitCliffordPtr dest)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QUnitClifford::Detach range is out-of-bounds!");
    }

    // Move "emulated" bits immediately into the destination, which is initialized.
    // Find a set of shard "units" to order contiguously. Also count how many bits to decompose are in each subunit.
    std::map<QStabilizerPtr, bitLenInt> subunits;
    for (bitLenInt i = 0U; i < length; ++i) {
        CliffordShard& shard = shards[start + i];
        ++(subunits[shard.unit]);
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
    std::map<QStabilizerPtr, bitLenInt> decomposedUnits;
    for (bitLenInt i = 0U; i < length; ++i) {
        CliffordShard& shard = shards[start + i];
        QStabilizerPtr unit = shard.unit;

        if (decomposedUnits.find(unit) == decomposedUnits.end()) {
            decomposedUnits[unit] = start + i;
            bitLenInt subLen = subunits[unit];
            bitLenInt origLen = unit->GetQubitCount();
            if (subLen != origLen) {
                if (dest) {
                    QStabilizerPtr nUnit = MakeStabilizer(subLen, ZERO_BCI);
                    shard.unit->Decompose(shard.mapped, nUnit);
                    shard.unit = nUnit;
                } else {
                    shard.unit->Dispose(shard.mapped, subLen);
                    if (!randGlobalPhase) {
                        phaseOffset *= shard.unit->GetPhaseOffset();
                        shard.unit->ResetPhaseOffset();
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
        const auto subunit = subunits.find(shard.unit);
        if (subunit != subunits.end() &&
            shard.mapped >= (shards[decomposedUnits[shard.unit]].mapped + subunit->second)) {
            shard.mapped -= subunit->second;
        }
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

QStabilizerPtr QUnitClifford::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    std::vector<QStabilizerPtr> units;
    units.reserve((int)(last - first));

    QStabilizerPtr unit1 = shards[**first].unit;
    std::map<QStabilizerPtr, bool> found;

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
            QStabilizerPtr consumed = units[1U];
            bitLenInt offset = unit1->ComposeNoClone(consumed);
            units.erase(units.begin() + 1U);

            for (auto&& shard : shards) {
                if (shard.unit == consumed) {
                    shard.mapped += offset;
                    shard.unit = unit1;
                }
            }
        }

        std::vector<QStabilizerPtr> nUnits;
        std::map<QStabilizerPtr, bitLenInt> offsets;
        std::map<QStabilizerPtr, QStabilizerPtr> offsetPartners;

        for (size_t i = 0U; i < units.size(); i += 2U) {
            QStabilizerPtr retained = units[i];
            QStabilizerPtr consumed = units[i + 1U];
            nUnits.push_back(retained);
            offsets[consumed] = retained->ComposeNoClone(consumed);
            offsetPartners[consumed] = retained;
        }

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto&& shard : shards) {
            const auto search = offsets.find(shard.unit);
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

void QUnitClifford::OrderContiguous(QStabilizerPtr unit)
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
void QUnitClifford::SortUnit(QStabilizerPtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high)
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
    QSortEntry pivot = bits[(low + high) >> 1U];

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

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)

/// Convert the state to ket notation (warning: could be huge!)
void QUnitClifford::GetQuantumState(complex* stateVec)
{
    QUnitCliffordPtr thisCopy = std::dynamic_pointer_cast<QUnitClifford>(Clone());
    thisCopy->shards[0U].unit->NormalizeState(ONE_R1_F, FP_NORM_EPSILON, std::arg(phaseOffset));
    thisCopy->EntangleAll()->GetQuantumState(stateVec);
}

/// Convert the state to ket notation (warning: could be huge!)
void QUnitClifford::GetQuantumState(QInterfacePtr eng)
{
    QUnitCliffordPtr thisCopy = std::dynamic_pointer_cast<QUnitClifford>(Clone());
    thisCopy->shards[0U].unit->NormalizeState(ONE_R1_F, FP_NORM_EPSILON, std::arg(phaseOffset));
    thisCopy->EntangleAll()->GetQuantumState(eng);
}

/// Convert the state to ket notation (warning: could be huge!)
std::map<bitCapInt, complex> QUnitClifford::GetQuantumState()
{
    QUnitCliffordPtr thisCopy = std::dynamic_pointer_cast<QUnitClifford>(Clone());
    thisCopy->shards[0U].unit->NormalizeState(ONE_R1_F, FP_NORM_EPSILON, std::arg(phaseOffset));
    return thisCopy->EntangleAll()->GetQuantumState();
}

/// Convert the state to Schmidt-decomposed ket notation
QUnitStateVectorPtr QUnitClifford::GetDecomposedQuantumState()
{
    std::map<QInterfacePtr, bitLenInt> qis;
    std::map<bitLenInt, IdOffset> idMap;
    std::vector<std::map<bitCapInt, complex>> amps;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        const auto& shard = shards[i];
        if (qis.find(shard.unit) == qis.end()) {
            qis[shard.unit] = qis.size();
            amps.push_back(shard.unit->GetQuantumState());
        }
        idMap[i] = IdOffset(shard.mapped, qis[shard.unit]);
    }
    return std::make_shared<QUnitStateVector>(phaseOffset, idMap, amps);
}

/// Get all probabilities corresponding to ket notation
void QUnitClifford::GetProbs(real1* outputProbs)
{
    QUnitCliffordPtr thisCopy = std::dynamic_pointer_cast<QUnitClifford>(Clone());
    thisCopy->EntangleAll();
    thisCopy->shards[0U].unit->GetProbs(outputProbs);
}

/// Convert the state to ket notation (warning: could be huge!)
complex QUnitClifford::GetAmplitude(const bitCapInt& perm)
{
    if (bi_compare(perm, maxQPower) >= 0) {
        throw std::invalid_argument("QUnitClifford::GetAmplitudeOrProb argument out-of-bounds!");
    }

    std::map<QStabilizerPtr, bitCapInt> perms;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        CliffordShard& shard = shards[i];
        if (perms.find(shard.unit) == perms.end()) {
            perms[shard.unit] = ZERO_BCI;
        }
        if (bi_and_1(perm >> i)) {
            bi_or_ip(&(perms[shard.unit]), pow2(shard.mapped));
        }
    }

    complex result(phaseOffset);
    for (auto&& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
        if (norm(result) <= REAL1_EPSILON) {
            break;
        }
    }

    return result;
}

/// Convert the state to ket notation (warning: could be huge!)
std::vector<complex> QUnitClifford::GetAmplitudes(std::vector<bitCapInt> perms)
{
    std::map<QStabilizerPtr, std::set<bitCapInt>> permsMap;
    for (const auto& perm : perms) {
        std::map<QStabilizerPtr, bitCapInt> permMap;
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            CliffordShard& shard = shards[i];
            if (permMap.find(shard.unit) == permMap.end()) {
                permMap[shard.unit] = ZERO_BCI;
            }
            if (bi_and_1(perm >> i)) {
                bi_or_ip(&(permMap[shard.unit]), pow2(shard.mapped));
            }
        }
        for (const auto& p : permMap) {
            permsMap[p.first].insert(p.second);
        }
    }

    std::map<QStabilizerPtr, std::vector<complex>> ampsMap;
    for (const auto& s : permsMap) {
        ampsMap[s.first] = s.first->GetAmplitudes(std::vector<bitCapInt>(s.second.begin(), s.second.end()));
    }

    std::vector<complex> toRet;
    for (const auto& perm : perms) {
        std::map<QStabilizerPtr, bitCapInt> permMap;
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            CliffordShard& shard = shards[i];
            if (permMap.find(shard.unit) == permMap.end()) {
                permMap[shard.unit] = ZERO_BCI;
            }
            if (bi_and_1(perm >> i)) {
                bi_or_ip(&(permMap[shard.unit]), pow2(shard.mapped));
            }
        }
        complex amp = phaseOffset;
        for (const auto& p : permMap) {
            const std::set<bitCapInt>& s = permsMap[p.first];
            const size_t i = std::distance(s.begin(), s.find(p.second));
            amp *= ampsMap[p.first][i];
            if (norm(amp) <= REAL1_EPSILON) {
                break;
            }
        }
        toRet.push_back(amp);
    }

    return toRet;
}

bool QUnitClifford::SeparateBit(bool value, bitLenInt qubit)
{
    CliffordShard& shard = shards[qubit];
    const QStabilizerPtr unit = shard.unit;

    if (unit->GetQubitCount() == 1U) {
        unit->SetBit(0, value);

        return true;
    }

    const bitLenInt mapped = shard.mapped;

    if (!unit->TrySeparate(mapped)) {
        // This conditional coaxes the unit into separable form, so this should never actually happen.
        return false;
    }

    shard.unit = MakeStabilizer(1U, value);
    shard.mapped = 0U;

    unit->Dispose(mapped, 1U);
    if (!randGlobalPhase) {
        phaseOffset *= unit->GetPhaseOffset();
        unit->ResetPhaseOffset();
    }

    // Update the mappings.
    for (auto&& s : shards) {
        if ((unit == s.unit) && (mapped < s.mapped)) {
            --(s.mapped);
        }
    }

    return true;
}

/// Measure qubit t
bool QUnitClifford::ForceM(bitLenInt t, bool res, bool doForce, bool doApply)
{
    if (t >= qubitCount) {
        throw std::invalid_argument("QUnitClifford::ForceM target parameter must be within allocated qubit bounds!");
    }

    const CliffordShard& shard = shards[t];

    const bool result = shard.unit->ForceM(shard.mapped, res, doForce, doApply);
    if (!randGlobalPhase) {
        phaseOffset *= shard.unit->GetPhaseOffset();
        shard.unit->ResetPhaseOffset();
    }

    if (!doApply || !isReactiveSeparate) {
        return result;
    }

    SeparateBit(result, t);

    return result;
}

std::map<bitCapInt, int> QUnitClifford::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
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
        "QUnitClifford::MultiShotMeasureMask parameter qPowers array values must be within allocated qubit bounds!");

    std::map<QStabilizerPtr, std::vector<bitCapInt>> subQPowers;
    std::map<QStabilizerPtr, std::vector<bitCapInt>> subIQPowers;

    for (size_t i = 0U; i < qPowers.size(); ++i) {
        index = qIndices[i];
        CliffordShard& shard = shards[index];

        subQPowers[shard.unit].push_back(pow2(shard.mapped));
        subIQPowers[shard.unit].push_back(iQPowers[index]);
    }

    std::map<bitCapInt, int> combinedResults;
    combinedResults[ZERO_BCI] = (int)shots;

    for (const auto& subQPower : subQPowers) {
        QStabilizerPtr unit = subQPower.first;
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

    return combinedResults;
}

void QUnitClifford::MultiShotMeasureMask(
    const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    if (qPowers.size() != shards.size()) {
        QStabilizerPtr unit = shards[log2(qPowers[0U])].unit;
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
                    unit = nullptr;
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
                return unit->MultiShotMeasureMask(mappedIndices, shots, shotsArray);
            }
        }
    }

    std::map<bitCapInt, int> results = MultiShotMeasureMask(qPowers, shots);

    size_t j = 0U;
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end() && (j < shots)) {
        for (int i = 0; i < it->second; ++i) {
            shotsArray[j] = (bitCapIntOcl)it->first;
            ++j;
        }

        ++it;
    }
}

void QUnitClifford::SetQuantumState(const complex* inputState)
{
    if (qubitCount > 1U) {
        throw std::domain_error("QUnitClifford::SetQuantumState() not generally implemented!");
    }

    SetPermutation(ZERO_BCI);

    const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
    const real1 sqrtProb = sqrt(prob);
    const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
    const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
    const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
    const complex mtrx[4U]{ sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
    Mtrx(mtrx, 0U);
}

real1_f QUnitClifford::SumSqrDiff(QUnitCliffordPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    QUnitCliffordPtr thisCopyShared, thatCopyShared;
    QUnitClifford* thisCopy;
    QUnitClifford* thatCopy;

    if (shards[0U].unit->GetQubitCount() == qubitCount) {
        OrderContiguous(shards[0U].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnitClifford>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    if (toCompare->shards[0U].unit->GetQubitCount() == qubitCount) {
        toCompare->OrderContiguous(toCompare->shards[0U].unit);
        thatCopy = toCompare.get();
    } else {
        thatCopyShared = std::dynamic_pointer_cast<QUnitClifford>(toCompare->Clone());
        thatCopyShared->EntangleAll();
        thatCopy = thatCopyShared.get();
    }

    return thisCopy->shards[0U].unit->SumSqrDiff(thatCopy->shards[0U].unit);
}

bool QUnitClifford::TrySeparate(bitLenInt qubit)
{
    ThrowIfQubitInvalid(qubit, std::string("QUnitClifford::TrySeparate"));

    CliffordShard& shard = shards[qubit];
    QStabilizerPtr unit = shard.unit;
    const bitLenInt qbc = unit->GetQubitCount();

    if (qbc == 1U) {
        return true;
    }

    std::vector<bitLenInt> eqb = unit->EntangledQubits(shard.mapped);
    if (eqb.size() == qbc) {
        return false;
    }

    for (bitLenInt i = 0U; i < eqb.size(); ++i) {
        unit->Swap(i, eqb[i]);
    }

    // Otherwise, this shard can be decomposed.
    QStabilizerPtr sepUnit = std::dynamic_pointer_cast<QStabilizer>(unit->Decompose(0U, eqb.size()));

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        CliffordShard& oShard = shards[i];
        if (unit != oShard.unit) {
            continue;
        }

        const auto it = std::find(eqb.begin(), eqb.end(), oShard.mapped);
        if (it != eqb.end()) {
            oShard.mapped = std::distance(eqb.begin(), it);
            oShard.unit = sepUnit;
            continue;
        }

        if (oShard.mapped < eqb.size()) {
            oShard.mapped = eqb[oShard.mapped];
        }
        oShard.mapped -= eqb.size();
    }

    return true;
}

std::ostream& operator<<(std::ostream& os, const QUnitCliffordPtr s)
{
    const size_t qubitCount = (size_t)s->GetQubitCount();
    os << qubitCount << std::endl;

    std::map<QStabilizerPtr, std::map<bitLenInt, bitLenInt>> indexMap;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        CliffordShard& shard = s->shards[i];
        indexMap[shard.unit][i] = shard.mapped;
    }
    os << indexMap.size() << std::endl;
    for (const auto& m : indexMap) {
        os << m.second.size() << std::endl;
        for (const auto& p : m.second) {
            os << (size_t)p.first << " " << (size_t)p.second << std::endl;
        }
        os << m.first;
    }

    return os;
}
std::istream& operator>>(std::istream& is, const QUnitCliffordPtr s)
{
    size_t n;
    is >> n;
    s->SetQubitCount(n);
    s->SetPermutation(ZERO_BCI);

    size_t sCount;
    is >> sCount;
    for (size_t i = 0U; i < sCount; ++i) {
        size_t mapSize;
        is >> mapSize;
        std::map<bitLenInt, bitLenInt> indices;
        for (size_t j = 0U; j < mapSize; ++j) {
            size_t t, m;
            is >> t;
            is >> m;
            indices[(bitLenInt)t] = (bitLenInt)m;
        }
        QStabilizerPtr sp = std::make_shared<QStabilizer>(
            0U, ZERO_BCI, s->rand_generator, CMPLX_DEFAULT_ARG, false, s->randGlobalPhase, false, -1, s->useRDRAND);
        is >> sp;

        for (const auto& index : indices) {
            s->shards[index.first] = CliffordShard(index.second, sp);
        }
    }

    return is;
}
} // namespace Qrack
