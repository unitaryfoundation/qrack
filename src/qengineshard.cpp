//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QEngineShard is the atomic qubit unit of the QUnit mapper. "PhaseShard" optimizations are basically just a very
// specific "gate fusion" type optimization, where multiple gates are composed into single product gates before
// application to the state vector, to reduce the total number of gates that need to be applied. Rather than handling
// this as a "QFusion" layer optimization, which will typically sit BETWEEN a base QEngine set of "shards" and a QUnit
// that owns them, this particular gate fusion optimization can be avoid representational entanglement in QUnit in the
// first place, which QFusion would not help with. Alternatively, another QFusion would have to be in place ABOVE the
// QUnit layer, (with QEngine "below,") for this to work. Additionally, QFusion is designed to handle more general gate
// fusion, not specifically controlled phase gates, which are entirely commuting among each other and possibly a
// jumping-off point for further general "Fourier basis" optimizations which should probably reside in QUnit, analogous
// to the |+>/|-> basis changes QUnit takes advantage of for "H" gates.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengineshard.hpp"

#include <functional>
#include <unordered_set>

// Hash provided by Elara (OpenAI custom GPT)
namespace std {
    template <>
    struct hash<Qrack::QEngineShard> {
        std::size_t operator()(const Qrack::QEngineShard& s) const {
            return reinterpret_cast<std::size_t>(&s);
        }
    };
}

namespace Qrack {

bool QEngineShard::ClampAmps()
{
    if (isProbDirty) {
        return false;
    }

    if (norm(amp0) <= FP_NORM_EPSILON) {
        amp0 = ZERO_R1;
        amp1 /= abs(amp1);
        isPhaseDirty = false;

        return true;
    }

    if (norm(amp1) <= FP_NORM_EPSILON) {
        amp1 = ZERO_R1;
        amp0 /= abs(amp0);
        isPhaseDirty = false;

        return true;
    }

    return false;
}
void QEngineShard::DumpMultiBit()
{
    auto phaseShard = controlsShards.begin();
    while (phaseShard != controlsShards.end()) {
        RemoveTarget(phaseShard->first);
        phaseShard = controlsShards.begin();
    }
    phaseShard = targetOfShards.begin();
    while (phaseShard != targetOfShards.end()) {
        RemoveControl(phaseShard->first);
        phaseShard = targetOfShards.begin();
    }
    phaseShard = antiControlsShards.begin();
    while (phaseShard != antiControlsShards.end()) {
        RemoveAntiTarget(phaseShard->first);
        phaseShard = antiControlsShards.begin();
    }
    phaseShard = antiTargetOfShards.begin();
    while (phaseShard != antiTargetOfShards.end()) {
        RemoveAntiControl(phaseShard->first);
        phaseShard = antiTargetOfShards.begin();
    }
}

void QEngineShard::RemoveBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
{
    auto phaseShard = localMap.find(p);
    if (phaseShard != localMap.end()) {
        ((*phaseShard->first).*remoteMapGet)().erase(this);
        localMap.erase(phaseShard);
    }
}

void QEngineShard::DumpBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
{
    ((*this).*optimizeFn)();
    auto phaseShard = localMap.begin();
    while (phaseShard != localMap.end()) {
        ((*this).*remoteFn)(phaseShard->first);
        phaseShard = localMap.begin();
    }
}

void QEngineShard::DumpSamePhaseBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
{
    ((*this).*optimizeFn)();

    auto phaseShard = localMap.begin();
    int lcv = 0;

    while (phaseShard != localMap.end()) {
        PhaseShardPtr buffer = phaseShard->second;
        if (!buffer->isInvert && IS_SAME(buffer->cmplxDiff, buffer->cmplxSame)) {
            ((*this).*remoteFn)(phaseShard->first);
        } else {
            ++lcv;
        }
        phaseShard = localMap.begin();
        std::advance(phaseShard, lcv);
    }
}

void QEngineShard::AddBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteFn)
{
    if (p && (localMap.find(p) == localMap.end())) {
        PhaseShardPtr ps = std::make_shared<PhaseShard>();
        localMap[p] = ps;
        ((*p).*remoteFn)()[this] = ps;
    }
}

void QEngineShard::AddAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame,
    AddRemoveFn localFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
{
    ((*this).*localFn)(control);

    PhaseShardPtr targetOfShard = localMap[control];

    complex ncmplxDiff = targetOfShard->cmplxDiff * cmplxDiff;
    ncmplxDiff /= abs(ncmplxDiff);
    complex ncmplxSame = targetOfShard->cmplxSame * cmplxSame;
    ncmplxSame /= abs(ncmplxSame);

    if (!targetOfShard->isInvert && IS_ARG_0(ncmplxDiff) && IS_ARG_0(ncmplxSame)) {
        /* The buffer is equal to the identity operator, and it can be removed. */
        ((*this).*remoteFn)(control);

        return;
    }

    targetOfShard->cmplxDiff = ncmplxDiff;
    targetOfShard->cmplxSame = ncmplxSame;
}

void QEngineShard::OptimizeBuffer(
    ShardToPhaseMap& localMap, GetBufferFn remoteMapGet, AddAnglesFn phaseFn, bool makeThisControl)
{
    ShardToPhaseMap tempLocalMap = localMap;

    for (const auto& phaseShard : tempLocalMap) {
        const PhaseShardPtr& buffer = phaseShard.second;
        const QEngineShardPtr& partner = phaseShard.first;

        if (buffer->isInvert || !IS_ARG_0(buffer->cmplxDiff)) {
            continue;
        }

        ((*phaseShard.first).*remoteMapGet)().erase(this);
        localMap.erase(partner);

        if (makeThisControl) {
            ((*partner).*phaseFn)(this, ONE_CMPLX, buffer->cmplxSame);
        } else {
            ((*this).*phaseFn)(partner, ONE_CMPLX, buffer->cmplxSame);
        }
    }
}

void QEngineShard::OptimizeBothTargets()
{
    ShardToPhaseMap tempLocalMap = targetOfShards;
    for (const auto& phaseShard : tempLocalMap) {
        const PhaseShardPtr& buffer = phaseShard.second;
        const QEngineShardPtr& partner = phaseShard.first;

        if (buffer->isInvert) {
            continue;
        }

        if (IS_ARG_0(buffer->cmplxDiff)) {
            phaseShard.first->GetControlsShards().erase(this);
            targetOfShards.erase(partner);
            partner->AddPhaseAngles(this, ONE_CMPLX, buffer->cmplxSame);
        } else if (IS_ARG_0(buffer->cmplxSame)) {
            phaseShard.first->GetControlsShards().erase(this);
            targetOfShards.erase(partner);
            partner->AddAntiPhaseAngles(this, buffer->cmplxDiff, ONE_CMPLX);
        }
    }

    tempLocalMap = antiTargetOfShards;
    for (const auto& phaseShard : tempLocalMap) {
        const PhaseShardPtr& buffer = phaseShard.second;
        const QEngineShardPtr& partner = phaseShard.first;

        if (buffer->isInvert) {
            continue;
        }

        if (IS_ARG_0(buffer->cmplxDiff)) {
            phaseShard.first->GetAntiControlsShards().erase(this);
            antiTargetOfShards.erase(partner);
            partner->AddAntiPhaseAngles(this, ONE_CMPLX, buffer->cmplxSame);
        } else if (IS_ARG_0(buffer->cmplxSame)) {
            phaseShard.first->GetAntiControlsShards().erase(this);
            antiTargetOfShards.erase(partner);
            partner->AddPhaseAngles(this, buffer->cmplxDiff, ONE_CMPLX);
        }
    }
}

void QEngineShard::CombineBuffers(GetBufferFn targetMapGet, GetBufferFn controlMapGet, AddAnglesFn angleFn)
{
    ShardToPhaseMap tempControls = ((*this).*controlMapGet)();
    ShardToPhaseMap tempTargets = ((*this).*targetMapGet)();

    for (const auto& phaseShard : tempControls) {
        const QEngineShardPtr& partner = phaseShard.first;

        const auto partnerShard = tempTargets.find(partner);
        if (partnerShard == tempTargets.end()) {
            continue;
        }

        const PhaseShardPtr& buffer1 = phaseShard.second;
        const PhaseShardPtr& buffer2 = partnerShard->second;

        if (!buffer1->isInvert && IS_ARG_0(buffer1->cmplxDiff)) {
            ((*partner).*targetMapGet)().erase(this);
            ((*this).*controlMapGet)().erase(partner);
            ((*this).*angleFn)(partner, ONE_CMPLX, buffer1->cmplxSame);
        } else if (!buffer2->isInvert && IS_ARG_0(buffer2->cmplxDiff)) {
            ((*partner).*controlMapGet)().erase(this);
            ((*this).*targetMapGet)().erase(partner);
            ((*partner).*angleFn)(this, ONE_CMPLX, buffer2->cmplxSame);
        }
    }
}

void QEngineShard::SwapTargetAnti(QEngineShardPtr control)
{
    const auto phaseShard = targetOfShards.find(control);
    const auto antiPhaseShard = antiTargetOfShards.find(control);
    if ((antiPhaseShard == antiTargetOfShards.end()) && (phaseShard == targetOfShards.end())) {
        return;
    }
    if (antiPhaseShard == antiTargetOfShards.end()) {
        std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
        antiTargetOfShards[phaseShard->first] = phaseShard->second;
        targetOfShards.erase(phaseShard);
    } else if (phaseShard == targetOfShards.end()) {
        std::swap(antiPhaseShard->second->cmplxDiff, antiPhaseShard->second->cmplxSame);
        targetOfShards[antiPhaseShard->first] = antiPhaseShard->second;
        antiTargetOfShards.erase(antiPhaseShard);
    } else {
        std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
        std::swap(antiPhaseShard->second->cmplxDiff, antiPhaseShard->second->cmplxSame);
        targetOfShards[control].swap(antiTargetOfShards[control]);
    }
}

void QEngineShard::FlipPhaseAnti()
{
    std::unordered_set<QEngineShardPtr> toSwap;
    for (const auto& ctrlPhaseShard : controlsShards) {
        toSwap.insert(ctrlPhaseShard.first);
    }
    for (const auto& ctrlPhaseShard : antiControlsShards) {
        toSwap.insert(ctrlPhaseShard.first);
    }
    std::unordered_set<QEngineShardPtr>::iterator swapShard;
    for (const auto& swapShard : toSwap) {
        swapShard->SwapTargetAnti(this);
    }
    std::swap(controlsShards, antiControlsShards);

    for (const auto& phaseShard : targetOfShards) {
        std::swap(phaseShard.second->cmplxDiff, phaseShard.second->cmplxSame);
    }
    for (const auto& phaseShard : antiTargetOfShards) {
        std::swap(phaseShard.second->cmplxDiff, phaseShard.second->cmplxSame);
    }
}

void QEngineShard::CommutePhase(const complex& topLeft, const complex& bottomRight)
{
    for (const auto& phaseShard : targetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        if (!buffer->isInvert) {
            return;
        }

        buffer->cmplxDiff *= topLeft / bottomRight;
        buffer->cmplxSame *= bottomRight / topLeft;
    }

    for (const auto& phaseShard : antiTargetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        if (!buffer->isInvert) {
            return;
        }

        buffer->cmplxDiff *= bottomRight / topLeft;
        buffer->cmplxSame *= topLeft / bottomRight;
    }
}

void QEngineShard::RemoveIdentityBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
{
    auto phaseShard = localMap.begin();
    bitLenInt i = 0;

    while (phaseShard != localMap.end()) {
        PhaseShardPtr buffer = phaseShard->second;
        if (!buffer->isInvert && IS_ARG_0(buffer->cmplxDiff) && IS_ARG_0(buffer->cmplxSame)) {
            // The buffer is equal to the identity operator, and it can be removed.
            ((*phaseShard->first).*remoteMapGet)().erase(this);
            localMap.erase(phaseShard);
        } else {
            ++i;
        }

        phaseShard = localMap.begin();
        std::advance(phaseShard, i);
    }
}

void QEngineShard::RemovePhaseBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
{
    auto phaseShard = localMap.begin();
    bitLenInt i = 0;

    while (phaseShard != localMap.end()) {
        if (!phaseShard->second->isInvert) {
            // The buffer is equal to the identity operator, and it can be removed.
            ((*phaseShard->first).*remoteMapGet)().erase(this);
            localMap.erase(phaseShard);
        } else {
            ++i;
        }

        phaseShard = localMap.begin();
        std::advance(phaseShard, i);
    }
}

void QEngineShard::CommuteH()
{
    // See QUnit::CommuteH() for which cases cannot be commuted and are flushed.
    for (const auto& phaseShard : targetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        if (abs(buffer->cmplxDiff - buffer->cmplxSame) < 1) {
            if (buffer->isInvert) {
                buffer->isInvert = false;
                buffer->cmplxSame = -buffer->cmplxSame;
            }
        } else {
            if (buffer->isInvert) {
                std::swap(buffer->cmplxDiff, buffer->cmplxSame);
            } else {
                buffer->cmplxSame = -buffer->cmplxSame;
                buffer->isInvert = true;
            }
        }
    }

    RemoveIdentityBuffers(targetOfShards, &QEngineShard::GetControlsShards);

    for (const auto& phaseShard : antiTargetOfShards) {
        const PhaseShardPtr& buffer = phaseShard.second;
        if (abs(buffer->cmplxDiff - buffer->cmplxSame) < 1) {
            if (buffer->isInvert) {
                buffer->isInvert = false;
                buffer->cmplxDiff = -buffer->cmplxDiff;
            }
        } else {
            if (buffer->isInvert) {
                std::swap(buffer->cmplxDiff, buffer->cmplxSame);
            } else {
                buffer->cmplxDiff = -buffer->cmplxDiff;
                buffer->isInvert = true;
            }
        }
    }

    RemoveIdentityBuffers(antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
}

bool QEngineShard::IsInvertControl()
{
    for (const auto& phaseShard : controlsShards) {
        if (phaseShard.second->isInvert) {
            return true;
        }
    }

    for (const auto& phaseShard : antiControlsShards) {
        if (phaseShard.second->isInvert) {
            return true;
        }
    }

    return false;
}

bool QEngineShard::IsInvertTarget()
{
    for (const auto& phaseShard : targetOfShards) {
        if (phaseShard.second->isInvert) {
            return true;
        }
    }

    for (const auto& phaseShard : antiTargetOfShards) {
        if (phaseShard.second->isInvert) {
            return true;
        }
    }

    return false;
}

void QEngineShard::ClearMapInvertPhase(ShardToPhaseMap& shards)
{
    for (const auto& phaseShard : shards) {
        if (phaseShard.second->isInvert) {
            phaseShard.second->cmplxDiff = ONE_CMPLX;
            phaseShard.second->cmplxSame = ONE_CMPLX;
        }
    }
}
} // namespace Qrack
