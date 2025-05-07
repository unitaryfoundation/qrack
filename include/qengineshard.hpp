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

#pragma once

#include "qinterface.hpp"

#define IS_ARG_0(c) IS_SAME(c, ONE_CMPLX)
#define IS_ARG_PI(c) IS_OPPOSITE(c, ONE_CMPLX)

namespace Qrack {

/** Caches controlled gate phase between shards, (as a case of "gate fusion" optimization particularly useful to QUnit)
 */
struct PhaseShard {
    complex cmplxDiff;
    complex cmplxSame;
    bool isInvert;

    PhaseShard()
        : cmplxDiff(ONE_CMPLX)
        , cmplxSame(ONE_CMPLX)
        , isInvert(false)
    {
    }
};

class QEngineShard;
typedef QEngineShard* QEngineShardPtr;
typedef std::shared_ptr<PhaseShard> PhaseShardPtr;
typedef std::map<QEngineShardPtr, PhaseShardPtr> ShardToPhaseMap;

/** Associates a QInterface object with a set of bits. */
class QEngineShard {
protected:
    typedef ShardToPhaseMap& (QEngineShard::*GetBufferFn)();
    typedef void (QEngineShard::*OptimizeFn)();
    typedef void (QEngineShard::*AddRemoveFn)(QEngineShardPtr);
    typedef void (QEngineShard::*AddAnglesFn)(
        QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame);

public:
    QInterfacePtr unit;
    bitLenInt mapped;
    bool isProbDirty;
    bool isPhaseDirty;
    complex amp0;
    complex amp1;
    Pauli pauliBasis;
    // Shards which this shard controls
    ShardToPhaseMap controlsShards;
    // Shards which this shard (anti-)controls
    ShardToPhaseMap antiControlsShards;
    // Shards of which this shard is a target
    ShardToPhaseMap targetOfShards;
    // Shards of which this shard is an (anti-controlled) target
    ShardToPhaseMap antiTargetOfShards;
    // For FindShardIndex
    bool found;

protected:
    // We'd rather not have these getters at all, but we need their function pointers.
    ShardToPhaseMap& GetControlsShards() { return controlsShards; }
    ShardToPhaseMap& GetAntiControlsShards() { return antiControlsShards; }
    ShardToPhaseMap& GetTargetOfShards() { return targetOfShards; }
    ShardToPhaseMap& GetAntiTargetOfShards() { return antiTargetOfShards; }

public:
    QEngineShard()
        : unit{ nullptr }
        , mapped(0)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , pauliBasis(PauliZ)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
    }

    QEngineShard(const bool& set, const complex& rand_phase = ONE_CMPLX)
        : unit{ nullptr }
        , mapped(0)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , pauliBasis(PauliZ)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
        amp0 = set ? ZERO_CMPLX : rand_phase;
        amp1 = set ? rand_phase : ZERO_CMPLX;
    }

    // Dirty state constructor:
    QEngineShard(QInterfacePtr u, const bitLenInt& mapping)
        : unit(u)
        , mapped(mapping)
        , isProbDirty(true)
        , isPhaseDirty(true)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , pauliBasis(PauliZ)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
    }

    void MakeDirty()
    {
        isProbDirty = true;
        isPhaseDirty = true;
    }

    bool ClampAmps();
    void DumpMultiBit();

protected:
    void RemoveBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteMapGet);

public:
    void RemoveControl(QEngineShardPtr p) { RemoveBuffer(p, targetOfShards, &QEngineShard::GetControlsShards); }
    void RemoveTarget(QEngineShardPtr p) { RemoveBuffer(p, controlsShards, &QEngineShard::GetTargetOfShards); }
    void RemoveAntiControl(QEngineShardPtr p)
    {
        RemoveBuffer(p, antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }
    void RemoveAntiTarget(QEngineShardPtr p)
    {
        RemoveBuffer(p, antiControlsShards, &QEngineShard::GetAntiTargetOfShards);
    }

protected:
    void DumpBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn);
    void DumpSamePhaseBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn);

public:
    void DumpControlOf() { DumpBuffer(&QEngineShard::OptimizeTargets, controlsShards, &QEngineShard::RemoveTarget); }
    void DumpAntiControlOf()
    {
        DumpBuffer(&QEngineShard::OptimizeAntiTargets, antiControlsShards, &QEngineShard::RemoveAntiTarget);
    }
    void DumpSamePhaseControlOf()
    {
        DumpSamePhaseBuffer(&QEngineShard::OptimizeTargets, controlsShards, &QEngineShard::RemoveTarget);
    }
    void DumpSamePhaseAntiControlOf()
    {
        DumpSamePhaseBuffer(&QEngineShard::OptimizeAntiTargets, antiControlsShards, &QEngineShard::RemoveAntiTarget);
    }

protected:
    void AddBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteFn);

public:
    void MakePhaseControlledBy(QEngineShardPtr p) { AddBuffer(p, targetOfShards, &QEngineShard::GetControlsShards); }
    void MakePhaseControlOf(QEngineShardPtr p) { AddBuffer(p, controlsShards, &QEngineShard::GetTargetOfShards); }
    void MakePhaseAntiControlledBy(QEngineShardPtr p)
    {
        AddBuffer(p, antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }
    void MakePhaseAntiControlOf(QEngineShardPtr p)
    {
        AddBuffer(p, antiControlsShards, &QEngineShard::GetAntiTargetOfShards);
    }

protected:
    void AddAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame, AddRemoveFn localFn,
        ShardToPhaseMap& localMap, AddRemoveFn remoteFn);

public:
    void AddPhaseAngles(QEngineShardPtr control, const complex& topLeft, const complex& bottomRight)
    {
        AddAngles(control, topLeft, bottomRight, &QEngineShard::MakePhaseControlledBy, targetOfShards,
            &QEngineShard::RemoveControl);
    }
    void AddAntiPhaseAngles(QEngineShardPtr control, const complex& bottomRight, const complex& topLeft)
    {
        AddAngles(control, bottomRight, topLeft, &QEngineShard::MakePhaseAntiControlledBy, antiTargetOfShards,
            &QEngineShard::RemoveAntiControl);
    }
    void AddInversionAngles(QEngineShardPtr control, const complex& topRight, const complex& bottomLeft)
    {
        MakePhaseControlledBy(control);
        targetOfShards[control]->isInvert = !targetOfShards[control]->isInvert;
        std::swap(targetOfShards[control]->cmplxDiff, targetOfShards[control]->cmplxSame);
        AddPhaseAngles(control, topRight, bottomLeft);
    }
    void AddAntiInversionAngles(QEngineShardPtr control, const complex& bottomLeft, const complex& topRight)
    {
        MakePhaseAntiControlledBy(control);
        antiTargetOfShards[control]->isInvert = !antiTargetOfShards[control]->isInvert;
        std::swap(antiTargetOfShards[control]->cmplxDiff, antiTargetOfShards[control]->cmplxSame);
        AddAntiPhaseAngles(control, bottomLeft, topRight);
    }

protected:
    void OptimizeBuffer(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet, AddAnglesFn phaseFn, bool makeThisControl);

public:
    void OptimizeControls()
    {
        OptimizeBuffer(controlsShards, &QEngineShard::GetTargetOfShards, &QEngineShard::AddPhaseAngles, false);
    }
    void OptimizeTargets()
    {
        OptimizeBuffer(targetOfShards, &QEngineShard::GetControlsShards, &QEngineShard::AddPhaseAngles, true);
    }
    void OptimizeAntiControls()
    {
        OptimizeBuffer(
            antiControlsShards, &QEngineShard::GetAntiTargetOfShards, &QEngineShard::AddAntiPhaseAngles, false);
    }
    void OptimizeAntiTargets()
    {
        OptimizeBuffer(
            antiTargetOfShards, &QEngineShard::GetAntiControlsShards, &QEngineShard::AddAntiPhaseAngles, true);
    }

    void OptimizeBothTargets();

protected:
    void CombineBuffers(GetBufferFn targetMapGet, GetBufferFn controlMapGet, AddAnglesFn angleFn);

public:
    /// If this bit is both control and target of another bit, try to combine the operations into one gate.
    void CombineGates()
    {
        CombineBuffers(
            &QEngineShard::GetTargetOfShards, &QEngineShard::GetControlsShards, &QEngineShard::AddPhaseAngles);
        CombineBuffers(&QEngineShard::GetAntiTargetOfShards, &QEngineShard::GetAntiControlsShards,
            &QEngineShard::AddAntiPhaseAngles);
    }

    void SwapTargetAnti(QEngineShardPtr control);
    void FlipPhaseAnti();
    void CommutePhase(const complex& topLeft, const complex& bottomRight);

protected:
    void RemoveIdentityBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet);
    void RemovePhaseBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet);

public:
    void CommuteH();

    void DumpPhaseBuffers()
    {
        RemovePhaseBuffers(targetOfShards, &QEngineShard::GetControlsShards);
        RemovePhaseBuffers(antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
        RemovePhaseBuffers(controlsShards, &QEngineShard::GetTargetOfShards);
        RemovePhaseBuffers(antiControlsShards, &QEngineShard::GetAntiTargetOfShards);
    }

    bool IsInvertControl();
    bool IsInvertTarget();

protected:
    void ClearMapInvertPhase(ShardToPhaseMap& shards);

public:
    void ClearInvertPhase()
    {
        // Upon measurement, buffered phase can sometimes be totally ignored.
        // If we clear phase before applying buffered inversions, we can optimize application as CNOT.
        ClearMapInvertPhase(controlsShards);
        ClearMapInvertPhase(antiControlsShards);
        ClearMapInvertPhase(targetOfShards);
        ClearMapInvertPhase(antiTargetOfShards);
    }

    bitLenInt GetQubitCount() { return unit ? unit->GetQubitCount() : 1U; };
    real1_f Prob()
    {
        if (!isProbDirty || !unit) {
            return norm(amp1);
        }

        return unit->Prob(mapped);
    }
    bool isClifford()
    {
        return (unit && unit->isClifford(mapped)) ||
            (!unit &&
                ((norm(amp0) <= FP_NORM_EPSILON) || (norm(amp1) <= FP_NORM_EPSILON) ||
                    (norm(amp0 - amp1) <= FP_NORM_EPSILON) || (norm(amp0 + amp1) <= FP_NORM_EPSILON) ||
                    (norm(amp0 - I_CMPLX * amp1) <= FP_NORM_EPSILON) ||
                    (norm(amp0 + I_CMPLX * amp1) <= FP_NORM_EPSILON)));
    };
};

class QEngineShardMap {
protected:
    std::vector<QEngineShard> shards;
    std::vector<bitLenInt> swapMap;

public:
    QEngineShardMap()
    {
        // Intentionally left blank
    }

    QEngineShardMap(const bitLenInt& size)
        : shards(size)
        , swapMap(size)
    {
        for (bitLenInt i = 0U; i < size; ++i) {
            swapMap[i] = i;
        }
    }

    typedef std::vector<QEngineShard>::iterator iterator;

    QEngineShard& operator[](const bitLenInt& i)
    {
        if (i >= swapMap.size()) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QEngineShardMap::operator[]!");
        }

        return shards[swapMap[i]];
    }

    iterator begin() { return shards.begin(); }

    iterator end() { return shards.end(); }

    bitLenInt size() { return shards.size(); }

    void push_back(const QEngineShard& shard)
    {
        shards.push_back(shard);
        swapMap.push_back(swapMap.size());
    }

    void insert(bitLenInt start, QEngineShardMap& toInsert)
    {
        bitLenInt oSize = size();

        shards.insert(shards.end(), toInsert.shards.begin(), toInsert.shards.end());
        swapMap.insert(swapMap.begin() + start, toInsert.swapMap.begin(), toInsert.swapMap.end());

        for (bitLenInt lcv = 0U; lcv < toInsert.size(); ++lcv) {
            swapMap[(size_t)start + lcv] += oSize;
        }
    }

    void erase(bitLenInt begin, bitLenInt end)
    {
        for (bitLenInt index = begin; index < end; ++index) {
            bitLenInt offset = swapMap[index];
            shards.erase(shards.begin() + offset);

            for (bitLenInt& item : swapMap) {
                if (item >= offset) {
                    --item;
                }
            }
        }

        swapMap.erase(swapMap.begin() + begin, swapMap.begin() + end);
    }

    void swap(bitLenInt qubit1, bitLenInt qubit2) { std::swap(swapMap[qubit1], swapMap[qubit2]); }

    void reindex()
    {
        std::map<QInterfacePtr, bitLenInt> indices;
        for (QEngineShard& shard : shards) {
            if (!shard.unit) {
                shard.mapped = 0U;
                continue;
            }
            shard.mapped = indices[shard.unit]++;
        }
        for (size_t i = 0U; i < shards.size(); ++i) {
            swapMap[i] = i;
        }
    }
};
} // namespace Qrack
