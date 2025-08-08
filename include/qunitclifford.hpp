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

#pragma once

#include "qstabilizer.hpp"
#include "qunitstatevector.hpp"

namespace Qrack {

class QUnitClifford;
typedef std::shared_ptr<QUnitClifford> QUnitCliffordPtr;

struct CliffordShard {
    bitLenInt mapped;
    QStabilizerPtr unit;

    CliffordShard(bitLenInt m = 0U, QStabilizerPtr u = nullptr)
        : mapped(m)
        , unit(u)
    {
        // Intentionally left blank
    }

    CliffordShard(const CliffordShard& o)
        : mapped(o.mapped)
        , unit(o.unit)
    {
        // Intentionally left blank
    }
};

class QUnitClifford : public QInterface {
protected:
    complex phaseOffset;
    bool isReactiveSeparate;
    std::vector<CliffordShard> shards;

    using QInterface::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QUnitClifford>(orig)); }
    void Copy(QUnitCliffordPtr orig)
    {
        QInterface::Copy(std::dynamic_pointer_cast<QInterface>(orig));
        phaseOffset = orig->phaseOffset;
        shards = orig->shards;
    }

    void CombinePhaseOffsets(QStabilizerPtr unit)
    {
        if (randGlobalPhase) {
            return;
        }

        phaseOffset *= unit->GetPhaseOffset();
        unit->ResetPhaseOffset();
    }

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QStabilizerPtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    void Detach(bitLenInt start, bitLenInt length, QUnitCliffordPtr dest);

    QStabilizerPtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    QStabilizerPtr EntangleAll()
    {
        if (!qubitCount) {
            return MakeStabilizer(0U);
        }
        std::vector<bitLenInt> bits(qubitCount);
        std::vector<bitLenInt*> ebits(qubitCount);
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            bits[i] = i;
            ebits[i] = &bits[i];
        }

        QStabilizerPtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        OrderContiguous(toRet);

        return toRet;
    }

    void OrderContiguous(QStabilizerPtr unit);

    typedef std::function<void(QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx)>
        CGateFn;
    typedef std::function<void(QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx)> GateFn;
    typedef std::function<void(QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t)> SwapGateFn;
    void CGate(bitLenInt control, bitLenInt target, const complex* mtrx, CGateFn cfn, GateFn fn, bool isAnti)
    {
        ThrowIfQubitInvalid(target, "QUnitClifford::CGate");
        const real1_f p = Prob(control);
        if (p < (ONE_R1_F / 4)) {
            if (isAnti) {
                fn(shards[target].unit, target, mtrx);
            }
            return;
        } else if (p > (3 * ONE_R1_F / 4)) {
            if (!isAnti) {
                fn(shards[target].unit, target, mtrx);
            }
            return;
        }

        std::vector<bitLenInt> bits{ control, target };
        std::vector<bitLenInt*> ebits{ &bits[0U], &bits[1U] };
        QStabilizerPtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        cfn(unit, bits[0U], bits[1U], mtrx);
        CombinePhaseOffsets(unit);
        if (!isReactiveSeparate) {
            return;
        }
        TrySeparate(control);
        TrySeparate(target);
    }
    void SwapGate(bitLenInt control, bitLenInt target, SwapGateFn ufn, const complex& phaseFac)
    {
        const real1_f pc = Prob(control);
        const real1_f pt = Prob(target);
        if (((pc < (ONE_R1_F / 4)) && (pt > (3 * ONE_R1_F / 4))) ||
            ((pt < (ONE_R1_F / 4)) && (pc > (3 * ONE_R1_F / 4)))) {
            Swap(control, target);
            return Phase(phaseFac, phaseFac, target);
        }
        std::vector<bitLenInt> bits{ control, target };
        std::vector<bitLenInt*> ebits{ &bits[0U], &bits[1U] };
        QStabilizerPtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        ufn(unit, bits[0U], bits[1U]);
        CombinePhaseOffsets(unit);
        if (!isReactiveSeparate) {
            return;
        }
        TrySeparate(control);
        TrySeparate(target);
    }

    QInterfacePtr CloneBody(QUnitCliffordPtr copyPtr);

    void ThrowIfQubitInvalid(bitLenInt t, std::string methodName)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                methodName + std::string(" target qubit index parameter must be within allocated qubit bounds!"));
        }
    }

    bitLenInt ThrowIfQubitSetInvalid(const std::vector<bitLenInt>& controls, bitLenInt t, std::string methodName)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                methodName + std::string(" target qubit index parameter must be within allocated qubit bounds!"));
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument(methodName + std::string(" can only have one control qubit!"));
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                methodName + std::string(" control qubit index parameter must be within allocated qubit bounds!"));
        }

        return controls[0U];
    }

    real1_f ExpVarBitsFactorized(bool isExp, const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms,
        const bitCapInt& offset = ZERO_BCI);

    real1_f ExpVarFloatsFactorized(bool isExp, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);

public:
    QUnitClifford(bitLenInt n, const bitCapInt& perm = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phasFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool ignored2 = false, int64_t ignored3 = -1, bool useHardwareRNG = true, bool ignored4 = false,
        real1_f ignored5 = REAL1_EPSILON, std::vector<int64_t> ignored6 = {}, bitLenInt ignored7 = 0U,
        real1_f ignored8 = _qrack_qunit_sep_thresh);

    void SetReactiveSeparate(bool isAggSep) { isReactiveSeparate = isAggSep; }
    bool GetReactiveSeparate() { return isReactiveSeparate; }

    ~QUnitClifford() { Dump(); }

    QInterfacePtr Clone()
    {
        QUnitCliffordPtr copyPtr = std::make_shared<QUnitClifford>(
            qubitCount, ZERO_BCI, rand_generator, phaseOffset, doNormalize, randGlobalPhase, false, 0U, useRDRAND);

        return CloneBody(copyPtr);
    }
    QUnitCliffordPtr CloneEmpty()
    {
        return std::make_shared<QUnitClifford>(
            0U, ZERO_BCI, rand_generator, phaseOffset, doNormalize, randGlobalPhase, false, 0U, useRDRAND);
    }

    bool isClifford() { return true; };
    bool isClifford(bitLenInt qubit) { return true; };

    bitLenInt GetQubitCount() { return qubitCount; }

    bitCapInt GetMaxQPower() { return pow2(qubitCount); }

    void SetRandGlobalPhase(bool isRand)
    {
        for (CliffordShard& shard : shards) {
            shard.unit->SetRandGlobalPhase(isRand);
        }
    }

    void ResetPhaseOffset() { phaseOffset = ONE_CMPLX; }
    complex GetPhaseOffset() { return phaseOffset; }

    bitCapInt PermCount()
    {
        std::map<QStabilizerPtr, QStabilizerPtr> engines;
        bitCapInt permCount = ONE_BCI;
        for (CliffordShard& shard : shards) {
            QStabilizerPtr& unit = shard.unit;
            if (engines.find(unit) == engines.end()) {
                const bitCapInt pg = pow2(unit->gaussian());
                // This would be "*", but Schmidt decomposition makes it "+".
                permCount = permCount + pg;
            }
        }

        return permCount;
    }

    void Clear()
    {
        shards = std::vector<CliffordShard>();
        phaseOffset = ONE_CMPLX;
        qubitCount = 0U;
        maxQPower = ONE_BCI;
    }

    real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarBitsFactorized(true, bits, perms, offset);
    }

    real1_f ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpVarFloatsFactorized(true, bits, weights);
    }

    real1_f VarianceBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarBitsFactorized(false, bits, perms, offset);
    }

    real1_f VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpVarFloatsFactorized(false, bits, weights);
    }

    real1_f ProbPermRdm(const bitCapInt& perm, bitLenInt ancillaeStart);

    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation);

    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    QStabilizerPtr MakeStabilizer(
        bitLenInt length = 1U, const bitCapInt& perm = ZERO_BCI, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        QStabilizerPtr toRet = std::make_shared<QStabilizer>(
            length, perm, rand_generator, phaseFac, false, randGlobalPhase, false, -1, useRDRAND);

        return toRet;
    }

    void SetQuantumState(const complex* inputState);
    void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        throw std::domain_error("QUnitClifford::SetAmplitude() not implemented!");
    }

    /// Apply a CNOT gate with control and target
    void CNOT(bitLenInt c, bitLenInt t)
    {
        H(t);
        if (IsSeparableZ(t)) {
            CZ(c, t);
            return H(t);
        }
        H(t);
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
                unit->CNOT(c, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->X(t); }, false);
    }
    /// Apply a CY gate with control and target
    void CY(bitLenInt c, bitLenInt t)
    {
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CY(c, t); },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->Y(t); }, false);
    }
    /// Apply a CZ gate with control and target
    void CZ(bitLenInt c, bitLenInt t)
    {
        const real1_f p = Prob(t);
        if (p > (3 * ONE_R1_F / 4)) {
            return Z(c);
        }
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CZ(c, t); },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->Z(t); }, false);
    }
    /// Apply an (anti-)CNOT gate with control and target
    void AntiCNOT(bitLenInt c, bitLenInt t)
    {
        H(t);
        if (IsSeparableZ(t)) {
            AntiCZ(c, t);
            return H(t);
        }
        H(t);
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
                unit->AntiCNOT(c, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->X(t); }, true);
    }
    /// Apply an (anti-)CY gate with control and target
    void AntiCY(bitLenInt c, bitLenInt t)
    {
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
                unit->AntiCY(c, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->Y(t); }, true);
    }
    /// Apply an (anti-)CZ gate with control and target
    void AntiCZ(bitLenInt c, bitLenInt t)
    {
        const real1_f p = Prob(t);
        if (p > (3 * ONE_R1_F / 4)) {
            return Phase(-ONE_CMPLX, ONE_CMPLX, c);
        }
        CGate(
            c, t, nullptr,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
                unit->AntiCZ(c, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* unused) { unit->Z(t); }, true);
    }
    /// Apply a Hadamard gate to target
    using QInterface::H;
    void H(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::H"));
        CliffordShard& shard = shards[t];
        shard.unit->H(shard.mapped);
    }
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::S"));
        CliffordShard& shard = shards[t];
        shard.unit->S(shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    void IS(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IS"));
        CliffordShard& shard = shards[t];
        shard.unit->IS(shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    void Z(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Z"));
        CliffordShard& shard = shards[t];
        shard.unit->Z(shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    /// Apply an X (or NOT) gate to target
    using QInterface::X;
    void X(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::X"));
        CliffordShard& shard = shards[t];
        shard.unit->X(shard.mapped);
    }
    /// Apply a Pauli Y gate to target
    void Y(bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Y"));
        CliffordShard& shard = shards[t];
        shard.unit->Y(shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    // Swap two bits
    void Swap(bitLenInt qubit1, bitLenInt qubit2)
    {
        ThrowIfQubitInvalid(qubit1, std::string("QUnitClifford::Swap"));
        ThrowIfQubitInvalid(qubit2, std::string("QUnitClifford::Swap"));

        if (qubit1 == qubit2) {
            return;
        }

        // Simply swap the bit mapping.
        std::swap(shards[qubit1], shards[qubit2]);
    }
    // Swap two bits and apply a phase factor of i if they are different
    void ISwap(bitLenInt c, bitLenInt t)
    {
        SwapGate(
            c, t, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t) { unit->ISwap(c, t); }, I_CMPLX);
    }
    // Swap two bits and apply a phase factor of -i if they are different
    void IISwap(bitLenInt c, bitLenInt t)
    {
        SwapGate(
            c, t, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t) { unit->IISwap(c, t); }, -I_CMPLX);
    }

    /// Measure qubit t
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Measure all qubits
    bitCapInt MAll()
    {
        MaxReduce();
        bitCapInt toRet = QInterface::MAll();
        SetPermutation(toRet);
        return toRet;
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);

    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    void GetQuantumState(QInterfacePtr eng);

    /// Convert the state to sparse ket notation
    std::map<bitCapInt, complex> GetQuantumState();

    /// Convert the state to Schmidt-decomposed sparse ket notation
    QUnitStateVectorPtr GetDecomposedQuantumState();

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    complex GetAmplitude(const bitCapInt& perm);

    /// Get a single basis state amplitude
    std::vector<complex> GetAmplitudes(std::vector<bitCapInt> perms);

    /// Returns all qubits entangled with "qubit" (including itself)
    std::vector<bitLenInt> EntangledQubits(const bitLenInt& qubit, const bool& g)
    {
        ThrowIfQubitInvalid(qubit, std::string("QUnitClifford::EntangledQubits"));
        const CliffordShard& shard = shards[qubit];
        QStabilizerPtr unit = shard.unit;
        std::vector<bitLenInt> eqb = unit->EntangledQubits(shard.mapped, g);
        for (bitLenInt i = 0U; i < eqb.size(); ++i) {
            bitLenInt& qb = eqb[i];
            for (bitLenInt j = 0U; j < qubitCount; ++j) {
                const CliffordShard& oShard = shards[j];
                if ((unit == oShard.unit) && (qb == oShard.mapped)) {
                    qb = j;
                    break;
                }
            }
        }

        return eqb;
    }
    /// Returns "true" if target qubit is a Z basis eigenstate
    bool IsSeparableZ(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparableZ"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparableZ(shard.mapped);
    }
    /// Returns "true" if target qubit is an X basis eigenstate
    bool IsSeparableX(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparableX"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparableX(shard.mapped);
    }
    /// Returns "true" if target qubit is a Y basis eigenstate
    bool IsSeparableY(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparableY"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparableY(shard.mapped);
    }
    /**
     * Returns:
     * 0 if target qubit is not separable
     * 1 if target qubit is a Z basis eigenstate
     * 2 if target qubit is an X basis eigenstate
     * 3 if target qubit is a Y basis eigenstate
     */
    uint8_t IsSeparable(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparable"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparable(shard.mapped);
    }

    bool CanDecomposeDispose(const bitLenInt start, const bitLenInt length)
    {
        return std::dynamic_pointer_cast<QUnitClifford>(Clone())->EntangleAll()->CanDecomposeDispose(start, length);
    }

    using QInterface::Compose;
    bitLenInt Compose(QUnitCliffordPtr toCopy) { return Compose(toCopy, qubitCount); }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QUnitClifford>(toCopy)); }
    bitLenInt Compose(QUnitCliffordPtr toCopy, bitLenInt start)
    {
        if (start > qubitCount) {
            throw std::invalid_argument("QUnit::Compose start index is out-of-bounds!");
        }

        /* Create a clone of the quantum state in toCopy. */
        QUnitCliffordPtr clone = std::dynamic_pointer_cast<QUnitClifford>(toCopy->Clone());

        /* Insert the new shards in the middle */
        shards.insert(shards.begin() + start, clone->shards.begin(), clone->shards.end());

        SetQubitCount(qubitCount + toCopy->GetQubitCount());

        return start;
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QUnitClifford>(toCopy), start);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QUnitClifford>(dest));
    }
    void Decompose(bitLenInt start, QUnitCliffordPtr dest) { Detach(start, dest->GetQubitCount(), dest); }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        QUnitCliffordPtr dest = std::make_shared<QUnitClifford>(
            length, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG, doNormalize, randGlobalPhase, false, 0U, useRDRAND);

        Decompose(start, dest);

        return dest;
    }
    void Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm) { Detach(start, length, nullptr); }
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        if (start > qubitCount) {
            throw std::out_of_range("QUnitClifford::Allocate() cannot start past end of register!");
        }

        if (!qubitCount) {
            SetQubitCount(length);
            SetPermutation(ZERO_BCI);
            return 0U;
        }

        QUnitCliffordPtr nQubits = std::make_shared<QUnitClifford>(length, ZERO_BCI, rand_generator, CMPLX_DEFAULT_ARG,
            false, randGlobalPhase, false, -1, !!hardware_rand_generator);
        return Compose(nQubits, start);
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        if (!randGlobalPhase) {
            phaseOffset *= std::polar(ONE_R1, (real1)phaseArg);
        }
    }
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QUnitClifford>(toCompare));
    }
    virtual real1_f SumSqrDiff(QUnitCliffordPtr toCompare);
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QUnitClifford>(toCompare), error_tol);
    }
    bool ApproxCompare(QUnitCliffordPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        if (!toCompare) {
            return false;
        }

        if (this == toCompare.get()) {
            return true;
        }

        return std::dynamic_pointer_cast<QUnitClifford>(Clone())->EntangleAll()->ApproxCompare(
            std::dynamic_pointer_cast<QUnitClifford>(toCompare->Clone())->EntangleAll(), error_tol);
    }

    real1_f Prob(bitLenInt qubit)
    {
        ThrowIfQubitInvalid(qubit, std::string("QUnitClifford::Prob"));
        CliffordShard& shard = shards[qubit];
        return shard.unit->Prob(shard.mapped);
    }

    void Mtrx(const complex* mtrx, bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Mtrx"));
        CliffordShard& shard = shards[t];
        shard.unit->Mtrx(mtrx, shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Phase"));
        CliffordShard& shard = shards[t];
        shard.unit->Phase(topLeft, bottomRight, shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Invert"));
        CliffordShard& shard = shards[t];
        shard.unit->Invert(topRight, bottomLeft, shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt t)
    {
        if (controls.empty()) {
            return Phase(topLeft, bottomRight, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCPhase"));

        if (IS_SAME(topLeft, ONE_CMPLX) && IS_SAME(bottomRight, -ONE_CMPLX)) {
            return CZ(c, t);
        }

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MCPhase({ c }, mtrx[0U], mtrx[3U], t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Phase(mtrx[0U], mtrx[3U], t); },
            false);
    }
    void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt t)
    {
        if (controls.empty()) {
            return Phase(topLeft, bottomRight, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACPhase"));

        if (IS_SAME(topLeft, ONE_CMPLX) && IS_SAME(bottomRight, -ONE_CMPLX)) {
            return AntiCZ(c, t);
        }

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MACPhase({ c }, mtrx[0U], mtrx[3U], t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Phase(mtrx[0U], mtrx[3U], t); },
            true);
    }
    void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt t)
    {
        if (controls.empty()) {
            return Invert(topRight, bottomLeft, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCInvert"));

        if (IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
            return CNOT(c, t);
        }

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MCInvert({ c }, mtrx[1U], mtrx[2U], t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Invert(mtrx[1U], mtrx[2U], t); },
            false);
    }
    void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt t)
    {
        if (controls.empty()) {
            return Invert(topRight, bottomLeft, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACInvert"));

        if (IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
            return AntiCNOT(c, t);
        }

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MACInvert({ c }, mtrx[1U], mtrx[2U], t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Invert(mtrx[1U], mtrx[2U], t); },
            true);
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if ((norm(mtrx[1U]) <= FP_NORM_EPSILON) && (norm(mtrx[2U]) <= FP_NORM_EPSILON)) {
            return MCPhase(controls, mtrx[0U], mtrx[3U], t);
        }
        if ((norm(mtrx[0U]) <= FP_NORM_EPSILON) && (norm(mtrx[3U]) <= FP_NORM_EPSILON)) {
            return MCInvert(controls, mtrx[1U], mtrx[2U], t);
        }

        if (controls.empty()) {
            return Mtrx(mtrx, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCMtrx"));

        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MCMtrx({ c }, mtrx, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Mtrx(mtrx, t); }, false);
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if ((norm(mtrx[1U]) <= FP_NORM_EPSILON) && (norm(mtrx[2U]) <= FP_NORM_EPSILON)) {
            return MACPhase(controls, mtrx[0U], mtrx[3U], t);
        }
        if ((norm(mtrx[0U]) <= FP_NORM_EPSILON) && (norm(mtrx[3U]) <= FP_NORM_EPSILON)) {
            return MACInvert(controls, mtrx[1U], mtrx[2U], t);
        }

        if (controls.empty()) {
            return Mtrx(mtrx, t);
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACMtrx"));

        CGate(
            c, t, mtrx,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
                unit->MACMtrx({ c }, mtrx, t);
            },
            [](QStabilizerPtr unit, const bitLenInt& t, const complex* mtrx) { unit->Mtrx(mtrx, t); }, true);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt c, bitLenInt t)
    {
        ThrowIfQubitInvalid(c, std::string("QUnitClifford::FSim"));
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::FSim"));

        std::vector<bitLenInt> bits{ c, t };
        std::vector<bitLenInt*> ebits{ &bits[0U], &bits[1U] };
        QStabilizerPtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        unit->FSim(theta, phi, c, t);
        CombinePhaseOffsets(unit);
        if (!isReactiveSeparate) {
            return;
        }
        TrySeparate(c);
        TrySeparate(t);
    }

    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f ignored)
    {
        for (const bitLenInt& qubit : qubits) {
            if (!TrySeparate(qubit)) {
                return false;
            }
        }

        return true;
    }
    bool TrySeparate(bitLenInt qubit);
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return TrySeparate(qubit1);
        }

        const bool q1 = TrySeparate(qubit1);
        const bool q2 = TrySeparate(qubit2);

        return q1 && q2;
    }
    std::vector<bitLenInt> MaxReduce(bitLenInt qubit);
    void MaxReduce();
    bool SeparateBit(bool value, bitLenInt qubit);

    friend std::ostream& operator<<(std::ostream& os, const QUnitCliffordPtr s);
    friend std::istream& operator>>(std::istream& is, const QUnitCliffordPtr s);
};
} // namespace Qrack
