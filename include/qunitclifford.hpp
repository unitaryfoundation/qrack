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

namespace Qrack {

class QUnitClifford;
typedef std::shared_ptr<QUnitClifford> QUnitCliffordPtr;

struct CliffordShard {
    bitLenInt mapped;
    QStabilizerPtr unit;

    CliffordShard(bitLenInt m = 0U, QStabilizerPtr u = NULL)
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
    std::vector<CliffordShard> shards;

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
    void CGate(bitLenInt control, bitLenInt target, const complex* mtrx, CGateFn fn)
    {
        std::vector<bitLenInt> bits{ control, target };
        std::vector<bitLenInt*> ebits{ &bits[0U], &bits[1U] };
        QStabilizerPtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        fn(unit, bits[0U], bits[1U], mtrx);
        CombinePhaseOffsets(unit);
        TrySeparate(control);
        TrySeparate(target);
    }

    QInterfacePtr CloneBody(QUnitCliffordPtr copyPtr);

    bool SeparateBit(bool value, bitLenInt qubit);

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

public:
    QUnitClifford(bitLenInt n, bitCapInt perm = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        complex phasFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true, bool ignored2 = false,
        int64_t ignored3 = -1, bool useHardwareRNG = true, bool ignored4 = false, real1_f ignored5 = REAL1_EPSILON,
        std::vector<int64_t> ignored6 = {}, bitLenInt ignored7 = 0U, real1_f ignored8 = FP_NORM_EPSILON_F);

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

    void SetDevice(int64_t dID) {}

    void SetRandGlobalPhase(bool isRand)
    {
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            shards[i].unit->SetRandGlobalPhase(isRand);
        }
    }

    void ResetPhaseOffset() { phaseOffset = ONE_CMPLX; }
    complex GetPhaseOffset() { return phaseOffset; }

    bitCapInt PermCount()
    {
        std::map<QStabilizerPtr, QStabilizerPtr> engines;
        bitCapInt permCount = ONE_BCI;
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            QStabilizerPtr unit = shards[i].unit;
            if (engines.find(unit) == engines.end()) {
                const bitCapInt pg = pow2(unit->gaussian());
                permCount = permCount * pg;
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
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, bitCapInt offset = ZERO_BCI);

    real1_f ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);

    real1_f ProbPermRdm(bitCapInt perm, bitLenInt ancillaeStart);

    real1_f ProbMask(bitCapInt mask, bitCapInt permutation);

    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    QStabilizerPtr MakeStabilizer(
        bitLenInt length = 1U, bitCapInt perm = ZERO_BCI, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        QStabilizerPtr toRet = std::make_shared<QStabilizer>(
            length, perm, rand_generator, phaseFac, false, randGlobalPhase, false, -1, useRDRAND);

        return toRet;
    }

    void SetQuantumState(const complex* inputState);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        throw std::domain_error("QUnitClifford::SetAmplitude() not implemented!");
    }

    /// Apply a CNOT gate with control and target
    void CNOT(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->CNOT(c, t);
        });
    }
    /// Apply a CY gate with control and target
    void CY(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CY(c, t); });
    }
    /// Apply a CZ gate with control and target
    void CZ(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CZ(c, t); });
    }
    /// Apply an (anti-)CNOT gate with control and target
    void AntiCNOT(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCNOT(c, t);
        });
    }
    /// Apply an (anti-)CY gate with control and target
    void AntiCY(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCY(c, t);
        });
    }
    /// Apply an (anti-)CZ gate with control and target
    void AntiCZ(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCZ(c, t);
        });
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
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->ISwap(c, t);
        });
    }
    // Swap two bits and apply a phase factor of -i if they are different
    void IISwap(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->IISwap(c, t);
        });
    }

    /// Measure qubit t
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Measure all qubits
    bitCapInt MAll()
    {
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

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    complex GetAmplitude(bitCapInt perm);

    /// Get a single basis state amplitude
    std::vector<complex> GetAmplitudes(std::vector<bitCapInt> perms);

    /**
     * Returns "true" if target qubit is a Z basis eigenstate
     */
    bool IsSeparableZ(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparableZ"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparableZ(shard.mapped);
    }

    /**
     * Returns "true" if target qubit is an X basis eigenstate
     */
    bool IsSeparableX(const bitLenInt& t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::IsSeparableX"));
        CliffordShard& shard = shards[t];
        return shard.unit->IsSeparableX(shard.mapped);
    }
    /**
     * Returns "true" if target qubit is a Y basis eigenstate
     */
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
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) { Detach(start, length, nullptr); }
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
            false, randGlobalPhase, false, -1, hardware_rand_generator != NULL);
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
    void Phase(complex topLeft, complex bottomRight, bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Phase"));
        CliffordShard& shard = shards[t];
        shard.unit->Phase(topLeft, bottomRight, shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    void Invert(complex topRight, complex bottomLeft, bitLenInt t)
    {
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::Invert"));
        CliffordShard& shard = shards[t];
        shard.unit->Invert(topRight, bottomLeft, shard.mapped);
        CombinePhaseOffsets(shard.unit);
    }
    void MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt t)
    {
        if (!controls.size()) {
            Phase(topLeft, bottomRight, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCPhase"));

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCPhase({ c }, mtrx[0U], mtrx[3U], t);
        });
    }
    void MACPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt t)
    {
        if (!controls.size()) {
            Phase(topLeft, bottomRight, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACPhase"));

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACPhase({ c }, mtrx[0U], mtrx[3U], t);
        });
    }
    void MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt t)
    {
        if (!controls.size()) {
            Invert(topRight, bottomLeft, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCInvert"));

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCInvert({ c }, mtrx[1U], mtrx[2U], t);
        });
    }
    void MACInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt t)
    {
        if (!controls.size()) {
            Invert(topRight, bottomLeft, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACInvert"));

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACInvert({ c }, mtrx[1U], mtrx[2U], t);
        });
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if (!controls.size()) {
            Mtrx(mtrx, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MCMtrx"));

        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCMtrx({ c }, mtrx, t);
        });
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if (!controls.size()) {
            Mtrx(mtrx, t);
            return;
        }

        const bitLenInt c = ThrowIfQubitSetInvalid(controls, t, std::string("QUnitClifford::MACMtrx"));

        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACMtrx({ c }, mtrx, t);
        });
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt c, bitLenInt t)
    {
        ThrowIfQubitInvalid(c, std::string("QUnitClifford::FSim"));
        ThrowIfQubitInvalid(t, std::string("QUnitClifford::FSim"));

        const complex mtrx[4]{ (real1)theta, (real1)phi, ZERO_CMPLX, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->FSim((real1_f)std::real(mtrx[0U]), (real1_f)std::real(mtrx[1U]), c, t);
        });
    }

    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f ignored)
    {
        for (size_t i = 0U; i < qubits.size(); ++i) {
            if (!TrySeparate(qubits[i])) {
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

    friend std::ostream& operator<<(std::ostream& os, const QUnitCliffordPtr s);
    friend std::istream& operator>>(std::istream& is, const QUnitCliffordPtr s);
};
} // namespace Qrack
