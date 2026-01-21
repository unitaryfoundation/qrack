//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qengineshard.hpp"
#include "qparity.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

namespace Qrack {

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

#if ENABLE_ALU
class QUnit : public QAlu, public QParity, public QInterface {
#else
class QUnit : public QParity, public QInterface {
#endif
protected:
    bool freezeBasis2Qb;
    bool useHostRam;
    bool isReactiveSeparate;
    bool useTGadget;
    bool useExactNC;
    bool isBdt;
    bool isCpu;
    bool isSinglePage;
    bool isSparse;
    size_t aceMb;
    bitLenInt aceQubits;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    real1_f roundingThreshold;
    double logFidelity;
    int64_t devID;
    complex phaseFactor;
    QEngineShardMap shards;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;

    QInterfacePtr MakeEngine(bitLenInt length, const bitCapInt& perm);

    virtual void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QUnit>(orig)); }
    virtual void Copy(QUnitPtr orig)
    {
        QInterface::Copy(orig);
        freezeBasis2Qb = orig->freezeBasis2Qb;
        useHostRam = orig->useHostRam;
        isReactiveSeparate = orig->isReactiveSeparate;
        useTGadget = orig->useTGadget;
        useExactNC = orig->useExactNC;
        thresholdQubits = orig->thresholdQubits;
        separabilityThreshold = orig->separabilityThreshold;
        roundingThreshold = orig->roundingThreshold;
        logFidelity = orig->logFidelity;
        devID = orig->devID;
        phaseFactor = orig->phaseFactor;
        shards = orig->shards;
        deviceIDs = orig->deviceIDs;
        engines = orig->engines;
    }

    double PhaseInfidelity(const complex& p) { return (double)clampProb(abs(arg(p)) / PI_R1); }

    // Drafted by Ellie (the OpenAI custom GPT instance), refined by Dan
    // Returns payload factor in [0,3] = negator + phase0 + phase1
    // clamped to range [0,2] as lower bound for general worst-case input
    // m: row-major [u00, u01, u10, u11]
    double PayloadInfidelityFactor3(const complex* m)
    {
        const complex& u00 = m[0];
        const complex& u01 = m[1];
        const complex& u10 = m[2];
        const complex& u11 = m[3];

        // Negator strength: average off-diagonal power
        const real1_f n01 = norm(u01);
        const real1_f n10 = norm(u10);
        const real1_f N = clampProb((n01 + n10) / 2);

        // Phase contributions per "input basis branch", weighted by where amplitude goes
        const real1_f n00 = norm(u00);
        const real1_f n11 = norm(u11);
        const real1_f P0 = clampProb(n00 * PhaseInfidelity(u00) + n10 * PhaseInfidelity(u10));
        const real1_f P1 = clampProb(n11 * PhaseInfidelity(u11) + n01 * PhaseInfidelity(u01));

        return (double)clampProb((N + P0 + P1) / 2) * 2;
    }

    void CheckFidelity()
    {
#if ENABLE_ENV_VARS
        if ((!(bool)getenv("QRACK_DISABLE_QUNIT_FIDELITY_GUARD")) && (logFidelity <= FIDELITY_MIN)) {
#else
        if (logFidelity <= FIDELITY_MIN) {
#endif
            throw std::runtime_error("QUnit fidelity estimate is effectively 0! (This DOES NOT mean your fidelity is "
                                     "necessarily close to 0! Please read the Qrack README, and then, afterward, "
                                     "consider setting environment variable QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1.)");
        }
    }

    void ElideCz(
        const bool& isAnti, const bitLenInt& control, const bitLenInt& target, const real1_f& pth, const real1_f& pc)
    {
        // Act CNOT shadow.
        const bool ptHi = pth > pc;
        const real1_f pHi = ptHi ? pth : pc;
        const real1_f pLo = ptHi ? pc : pth;
        const bool pState = abs(pHi - HALF_R1) >= abs(pLo - HALF_R1);

        logFidelity += (double)log(pState ? pHi : (ONE_R1_F - pLo));
        CheckFidelity();

        if (pState) {
            if (!ptHi) {
                Phase(ONE_CMPLX, -ONE_CMPLX, target);
            } else if (isAnti) {
                // This makes absolutely no detectable difference,
                // according to canonical quantum mechanics.
                // Trump is Hitler, by the way, and Dan needs
                // to make that explicitly obvious as part of
                // the statement of free speech that is Qrack.
                // (I'm saying, you should've already known.)
                Phase(-ONE_CMPLX, ONE_CMPLX, control);
            } else {
                Phase(ONE_CMPLX, -ONE_CMPLX, control);
            }
        }
    }

public:
    QUnit(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool ignored = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devIDs = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QUnit(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool ignored = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devIDs = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh)
        : QUnit({ QINTERFACE_STABILIZER_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, ignored, norm_thresh, devIDs, qubitThreshold, separation_thresh)
    {
    }

    virtual ~QUnit() { Dump(); }

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        QInterface::SetConcurrency(threadsPerEngine);
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3, int64_t threadsPerEngine,
                std::vector<int64_t> unused4) {
                unit->SetConcurrency((uint32_t)threadsPerEngine);
                return true;
            },
            ZERO_R1_F, ZERO_R1_F, ZERO_R1_F, threadsPerEngine);
    }

    virtual void SetTInjection(bool useGadget)
    {
        useTGadget = useGadget;
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3, int64_t useGadget,
                std::vector<int64_t> unused4) {
                unit->SetTInjection((bool)useGadget);
                return true;
            },
            ZERO_R1_F, ZERO_R1_F, ZERO_R1_F, useGadget ? 1U : 0U);
    }

    virtual void SetUseExactNearClifford(bool useExact)
    {
        useExactNC = useExact;
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3, int64_t useExact,
                std::vector<int64_t> unused4) {
                unit->SetUseExactNearClifford((bool)useExact);
                return true;
            },
            ZERO_R1_F, ZERO_R1_F, ZERO_R1_F, useExact ? 1U : 0U);
    }

    virtual void SetReactiveSeparate(bool isAggSep) { isReactiveSeparate = isAggSep; }
    virtual bool GetReactiveSeparate() { return isReactiveSeparate; }

    virtual void SetDevice(int64_t dID);
    virtual void SetDeviceList(std::vector<int64_t> dIDs);
    virtual int64_t GetDevice() { return devID; }
    virtual std::vector<int64_t> GetDeviceList() { return deviceIDs; }

    virtual real1_f ProbRdm(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(qubit) + " out of range in QUnit::ProbRdm!");
        }

        const QEngineShard& shard = shards[qubit];
        if (!shard.unit) {
            return Prob(qubit);
        }

        return shard.unit->ProbRdm(qubit);
    }
    virtual real1_f CProbRdm(bitLenInt control, bitLenInt target)
    {
        AntiCNOT(control, target);
        const real1_f prob = ProbRdm(target);
        AntiCNOT(control, target);

        return prob;
    }
    virtual real1_f ACProbRdm(bitLenInt control, bitLenInt target)
    {
        CNOT(control, target);
        const real1_f prob = ProbRdm(target);
        CNOT(control, target);

        return prob;
    }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState) { GetQuantumStateOrProbs(outputState, nullptr); }
    virtual void GetProbs(real1* outputProbs) { GetQuantumStateOrProbs(nullptr, outputProbs); }
    virtual void GetQuantumStateOrProbs(complex* outputState, real1* outputProbs);
    virtual complex GetAmplitude(const bitCapInt& perm);
    virtual void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        if (!qubitCount) {
            throw std::domain_error("QUnit::SetAmplitude called for 0 qubits!");
        }

        if (bi_compare(perm, maxQPower) >= 0) {
            throw std::invalid_argument("QUnit::SetAmplitude argument out-of-bounds!");
        }

        EntangleAll();
        shards[0U].unit->SetAmplitude(perm, amp);
    }
    virtual void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);
    using QInterface::Compose;
    virtual bitLenInt Compose(QUnitPtr toCopy) { return Compose(toCopy, qubitCount); }
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QUnit>(toCopy)); }
    virtual bitLenInt Compose(QUnitPtr toCopy, bitLenInt start)
    {
        if (start > qubitCount) {
            throw std::invalid_argument("QUnit::Compose start index is out-of-bounds!");
        }

        ToPermBasisAll();

        /* Create a clone of the quantum state in toCopy. */
        QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(toCopy->Clone());

        clone->ToPermBasisAll();

        /* Insert the new shards in the middle */
        shards.insert(start, clone->shards);

        SetQubitCount(qubitCount + toCopy->GetQubitCount());

        return start;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QUnit>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QUnit>(dest));
    }
    virtual void Decompose(bitLenInt start, QUnitPtr dest) { Detach(start, dest->GetQubitCount(), dest); }
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        QUnitPtr dest = std::make_shared<QUnit>(engines, length, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
            separabilityThreshold);

        Decompose(start, dest);

        return dest;
    }
    virtual void Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }
    virtual void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        Detach(start, length, nullptr);
    }
    virtual bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QUnit>(dest), error_tol);
    }
    virtual bool TryDecompose(bitLenInt start, QUnitPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return Detach(start, dest->GetQubitCount(), dest, true, error_tol);
    }
    using QInterface::Allocate;
    virtual bitLenInt Allocate(bitLenInt start, bitLenInt length);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    using QInterface::H;
    virtual void H(bitLenInt target);
    using QInterface::S;
    virtual void S(bitLenInt target);
    using QInterface::IS;
    virtual void IS(bitLenInt target);

    virtual void ZMask(const bitCapInt& mask) { PhaseParity(PI_R1, mask); }
    virtual void PhaseParity(real1 radians, const bitCapInt& mask);

    virtual void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubitIndex);
    virtual void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubitIndex);
    virtual void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        UCPhase(controls, topLeft, bottomRight, target, m);
    }
    virtual void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        UCInvert(controls, topRight, bottomLeft, target, m);
    }
    virtual void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        UCPhase(controls, topLeft, bottomRight, target, ZERO_BCI);
    }
    virtual void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        UCInvert(controls, topRight, bottomLeft, target, ZERO_BCI);
    }
    virtual void UCPhase(const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight,
        bitLenInt target, const bitCapInt& controlPerm);
    virtual void UCInvert(const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft,
        bitLenInt target, const bitCapInt& controlPerm);
    virtual void Mtrx(const complex* mtrx, bitLenInt qubit);
    virtual void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        bitCapInt m = pow2(controls.size());
        bi_decrement(&m, 1U);
        UCMtrx(controls, mtrx, target, m);
    }
    virtual void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        UCMtrx(controls, mtrx, target, ZERO_BCI);
    }
    virtual void UCMtrx(
        const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, const bitCapInt& controlPerm);
    using QInterface::UniformlyControlledSingleBit;
    virtual void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask);
    virtual void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle);
    virtual void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ForceM;
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    using QInterface::ForceMReg;
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, const bitCapInt& result, bool doForce = true, bool doApply = true);
    virtual bitCapInt HighestProbAll();
    virtual bitCapInt MAll();
    virtual std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);
    virtual void MultiShotMeasureMask(
        const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    /** @} */

#if ENABLE_ALU
    using QInterface::M;
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    virtual void X(bitLenInt q) { QInterface::X(q); }

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
    {
        QInterface::DEC(toSub, start, length);
    }
    virtual void DECS(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    virtual void CINC(
        const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CINC(toAdd, inOutStart, length, controls);
    }
    virtual void CDEC(
        const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls);
    }
    virtual void INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
    virtual void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    virtual void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }

    virtual void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
    virtual void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCDECSC(
        const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCDECSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    virtual void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
    virtual void INCDECBCDC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif
    virtual void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    virtual void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    virtual void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);
    virtual void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length);

    /** @} */
#endif

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void SetReg(bitLenInt start, bitLenInt length, const bitCapInt& value);
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2)
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
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2) { EitherISwap(qubit1, qubit2, false); }
    virtual void IISwap(bitLenInt qubit1, bitLenInt qubit2) { EitherISwap(qubit1, qubit2, true); }
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1_f Prob(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument("QUnit::Prob target parameter must be within allocated qubit bounds!");
        }

        ToPermBasisProb(qubit);
        return ProbBase(qubit);
    }
    virtual real1_f ProbAll(const bitCapInt& perm) { return clampProb((real1_f)norm(GetAmplitudeOrProb(perm, true))); }
    virtual real1_f ProbAllRdm(bool roundRz, const bitCapInt& perm)
    {
        if (!qubitCount) {
            throw std::domain_error("QUnit::ProbAllRdm called for 0 qubits!");
        }

        if (shards[0U].unit && (shards[0U].unit->GetQubitCount() == qubitCount)) {
            OrderContiguous(shards[0U].unit);
            return shards[0U].unit->ProbAllRdm(roundRz, perm);
        }

        QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(Clone());
        QInterfacePtr unit = clone->EntangleAll(true);
        clone->OrderContiguous(unit);

        return unit->ProbAllRdm(roundRz, perm);
    }
    virtual real1_f ProbParity(const bitCapInt& mask);
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QUnit>(toCompare));
    }
    virtual real1_f SumSqrDiff(QUnitPtr toCompare);
    virtual real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(true, false, false, bits, perms, std::vector<real1_f>(), offset, false);
    }
    virtual real1_f ExpectationBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(true, true, false, bits, perms, std::vector<real1_f>(), offset, roundRz);
    }
    virtual real1_f VarianceBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(false, false, false, bits, perms, std::vector<real1_f>(), offset, false);
    }
    virtual real1_f VarianceBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(false, true, false, bits, perms, std::vector<real1_f>(), offset, roundRz);
    }
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);
    virtual void Finish();
    virtual bool isFinished();
    virtual void Dump()
    {
        for (QEngineShard& shard : shards) {
            // Setting shard to NULL seems theoretically memory-safe, as previously used here,
            // but it's safer (in the intent of dumping remaining QInterface work)
            // to set shard.unit to nullptr.
            shard.unit = nullptr;
        }
    }
    using QInterface::isClifford;
    virtual bool isClifford(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(qubit) + " out of range in QUnit::isClifford!");
        }

        return shards[qubit].isClifford();
    };

    using QInterface::TrySeparate;
    virtual bool TrySeparate(bitLenInt qubit);
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);
    virtual bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol)
    {
        for (size_t i = 0U; i < qubits.size(); ++i) {
            Swap(qubitCount - (i + 1U), qubits[i]);
        }

        QUnitPtr dest = std::make_shared<QUnit>(engines, qubits.size(), ZERO_BCI, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs,
            thresholdQubits, separabilityThreshold);

        const bool result = TryDecompose(qubitCount - qubits.size(), dest);
        if (result) {
            Compose(dest);
        }

        for (bitLenInt i = qubits.size(); i > 0U; --i) {
            Swap(qubitCount - i, qubits[i - 1U]);
        }

        return result;
    }
    virtual double GetUnitaryFidelity();
    virtual void ResetUnitaryFidelity() { logFidelity = 0.0; }
    virtual void SetSdrp(real1_f sdrp)
    {
        separabilityThreshold = sdrp;
        isReactiveSeparate = (separabilityThreshold > FP_NORM_EPSILON_F);
    };
    virtual void SetNcrp(real1_f ncrp)
    {
        roundingThreshold = ncrp;
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f rp, real1_f unused, real1_f unused2, int64_t unused3,
                std::vector<int64_t> unused4) {
                unit->SetNcrp(rp);
                return true;
            },
            ncrp, ZERO_R1_F, ZERO_R1_F, 0);
    }
    virtual void SetAceMaxQubits(bitLenInt qb) { aceQubits = qb; }
    virtual void SetSparseAceMaxMb(size_t mb) { aceMb = mb; }

    virtual QInterfacePtr Clone();
    virtual QInterfacePtr Copy();

    /** @} */

protected:
    virtual complex GetAmplitudeOrProb(const bitCapInt& perm, bool isProb);

    virtual void XBase(bitLenInt target)
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

    virtual void YBase(bitLenInt target)
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

    virtual void ZBase(bitLenInt target)
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
    virtual real1_f ProbBase(bitLenInt qubit);

    virtual bool TrySeparateClifford(bitLenInt qubit);

    virtual void EitherISwap(bitLenInt qubit1, bitLenInt qubit2, bool isInverse);

#if ENABLE_ALU
    typedef void (QAlu::*INCxFn)(const bitCapInt&, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QAlu::*INCxxFn)(const bitCapInt&, bitLenInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QAlu::*CMULFn)(const bitCapInt&, bitLenInt, bitLenInt, bitLenInt, const std::vector<bitLenInt>&);
    typedef void (QAlu::*CMULModFn)(
        const bitCapInt&, const bitCapInt&, bitLenInt, bitLenInt, bitLenInt, const std::vector<bitLenInt>&);
    void INT(const bitCapInt& toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
        std::vector<bitLenInt> controlVec = std::vector<bitLenInt>());
    void INTS(const bitCapInt& toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex,
        bool hasCarry);
    void INCx(INCxFn fn, const bitCapInt& toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void INCxx(INCxxFn fn, const bitCapInt& toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index,
        bitLenInt flag2Index);
    QInterfacePtr CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt>* controlsMapped);
    std::vector<bitLenInt> CMULEntangle(
        std::vector<bitLenInt> controlVec, bitLenInt start, const bitCapInt& carryStart, bitLenInt length);
    void CMULx(CMULFn fn, const bitCapInt& toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        std::vector<bitLenInt> controlVec);
    void CMULModx(CMULModFn fn, const bitCapInt& toMod, const bitCapInt& modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt> controlVec);
    bool INTCOptimize(const bitCapInt& toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex);
    bool INTSOptimize(const bitCapInt& toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex);
    bool INTSCOptimize(const bitCapInt& toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex,
        bitLenInt overflowIndex);
    bitCapInt GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values);
    bitCapInt GetIndexedEigenstate(bitLenInt start, bitLenInt length, const unsigned char* values);
#endif

    real1_f ExpVarFactorized(bool isExp, bool isRdm, bool isFloat, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const std::vector<real1_f>& weights, const bitCapInt& offset, bool roundRz)
    {
        if (!qubitCount) {
            throw std::domain_error("QUnit::ProbAllRdm called for 0 qubits!");
        }

        if ((isFloat && (weights.size() < bits.size())) || (!isFloat && (perms.size() < bits.size()))) {
            throw std::invalid_argument("QUnit::ExpectationFactorized() must supply at least as many weights as bits!");
        }

        ThrowIfQbIdArrayIsBad(bits, qubitCount,
            "QUnit::ExpectationFactorized parameter qubits vector values must be within allocated qubit bounds!");

        if (shards[0U].unit && (shards[0U].unit->GetQubitCount() == qubitCount)) {
            OrderContiguous(shards[0U].unit);
            return isExp  ? isFloat
                     ? (isRdm ? shards[0U].unit->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                              : shards[0U].unit->ExpectationFloatsFactorized(bits, weights))
                     : (isRdm ? shards[0U].unit->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset)
                              : shards[0U].unit->ExpectationBitsFactorized(bits, perms, offset))
                 : isFloat ? (isRdm ? shards[0U].unit->VarianceFloatsFactorizedRdm(roundRz, bits, weights)
                                    : shards[0U].unit->VarianceFloatsFactorized(bits, weights))
                          : (isRdm ? shards[0U].unit->VarianceBitsFactorizedRdm(roundRz, bits, perms, offset)
                                   : shards[0U].unit->VarianceBitsFactorized(bits, perms, offset));
        }

        QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(Clone());
        QInterfacePtr unit = clone->EntangleAll(true);
        clone->OrderContiguous(unit);

        return isExp  ? isFloat ? (isRdm ? unit->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                                         : unit->ExpectationFloatsFactorized(bits, weights))
                                : (isRdm ? unit->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset)
                                         : unit->ExpectationBitsFactorized(bits, perms, offset))
             : isFloat ? (isRdm ? unit->VarianceFloatsFactorizedRdm(roundRz, bits, weights)
                                : unit->VarianceFloatsFactorized(bits, weights))
                      : (isRdm ? unit->VarianceBitsFactorizedRdm(roundRz, bits, perms, offset)
                               : unit->VarianceBitsFactorized(bits, perms, offset));
    }

    virtual QInterfacePtr Entangle(std::vector<bitLenInt> bits);
    virtual QInterfacePtr Entangle(std::vector<bitLenInt*> bits);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bool isForProb = false);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);
    virtual QInterfacePtr EntangleRange(
        bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3);
    virtual QInterfacePtr EntangleAll(bool isForProb = false)
    {
        QInterfacePtr toRet = EntangleRange(0, qubitCount, isForProb);
        OrderContiguous(toRet);
        return toRet;
    }

    virtual QInterfacePtr CloneBody(QUnitPtr copyPtr, bool isCopy);

    virtual bool CheckBitsPermutation(bitLenInt start, bitLenInt length = 1);
    virtual bitCapInt GetCachedPermutation(bitLenInt start, bitLenInt length);
    virtual bitCapInt GetCachedPermutation(const std::vector<bitLenInt>& bitArray);
    virtual bool CheckBitsPlus(bitLenInt qubitIndex, bitLenInt length);

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    typedef bool (*ParallelUnitFn)(QInterfacePtr unit, real1_f param1, real1_f param2, real1_f param3, int64_t param4,
        std::vector<int64_t> param5);
    bool ParallelUnitApply(ParallelUnitFn fn, real1_f param1 = ZERO_R1_F, real1_f param2 = ZERO_R1_F,
        real1_f param3 = ZERO_R1_F, int64_t param4 = 0, std::vector<int64_t> param5 = {});

    virtual bool SeparateBit(bool value, bitLenInt qubit);

    void OrderContiguous(QInterfacePtr unit);

    virtual bool Detach(
        bitLenInt start, bitLenInt length, QUnitPtr dest, bool isTry = false, real1_f tol = TRYDECOMPOSE_EPSILON);

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    bool TrimControls(const std::vector<bitLenInt>& controls, std::vector<bitLenInt>& controlVec, bitCapInt* perm);

    template <typename CF>
    void ApplyEitherControlled(std::vector<bitLenInt> controlVec, const std::vector<bitLenInt> targets, CF cfn,
        bool isPhase, const bitCapInt& controlPerm, const double payloadInfidelity);

    void ClampShard(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(qubit) + " out of range in QUnit::ClampShard!");
        }

        QEngineShard& shard = shards[qubit];
        if (!shard.ClampAmps() || !shard.unit) {
            return;
        }

        if (IS_NORM_0(shard.amp1)) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp1)));
            SeparateBit(false, qubit);
        } else if (IS_NORM_0(shard.amp0)) {
            logFidelity += (double)log(clampProb(ONE_R1_F - norm(shard.amp0)));
            SeparateBit(true, qubit);
        }

        CheckFidelity();
    }

    void TransformX2x2(const complex* mtrxIn, complex* mtrxOut)
    {
        mtrxOut[0U] = HALF_R1 * (mtrxIn[0U] + mtrxIn[1U] + mtrxIn[2U] + mtrxIn[3U]);
        mtrxOut[1U] = HALF_R1 * (mtrxIn[0U] - mtrxIn[1U] + mtrxIn[2U] - mtrxIn[3U]);
        mtrxOut[2U] = HALF_R1 * (mtrxIn[0U] + mtrxIn[1U] - mtrxIn[2U] - mtrxIn[3U]);
        mtrxOut[3U] = HALF_R1 * (mtrxIn[0U] - mtrxIn[1U] - mtrxIn[2U] + mtrxIn[3U]);
    }

    void TransformXInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut)
    {
        mtrxOut[0U] = HALF_R1 * (topRight + bottomLeft);
        mtrxOut[1U] = HALF_R1 * (-topRight + bottomLeft);
        mtrxOut[2U] = -mtrxOut[1U];
        mtrxOut[3U] = -mtrxOut[0U];
    }

    void TransformY2x2(const complex* mtrxIn, complex* mtrxOut)
    {
        mtrxOut[0U] = HALF_R1 * (mtrxIn[0U] + I_CMPLX * (mtrxIn[1U] - mtrxIn[2U]) + mtrxIn[3U]);
        mtrxOut[1U] = HALF_R1 * (mtrxIn[0U] - I_CMPLX * (mtrxIn[1U] + mtrxIn[2U]) - mtrxIn[3U]);
        mtrxOut[2U] = HALF_R1 * (mtrxIn[0U] + I_CMPLX * (mtrxIn[1U] + mtrxIn[2U]) - mtrxIn[3U]);
        mtrxOut[3U] = HALF_R1 * (mtrxIn[0U] - I_CMPLX * (mtrxIn[1U] - mtrxIn[2U]) + mtrxIn[3U]);
    }

    void TransformYInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut)
    {
        mtrxOut[0U] = I_CMPLX * HALF_R1 * (topRight - bottomLeft);
        mtrxOut[1U] = I_CMPLX * HALF_R1 * (-topRight - bottomLeft);
        mtrxOut[2U] = -mtrxOut[1U];
        mtrxOut[3U] = -mtrxOut[0U];
    }

    void TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut)
    {
        mtrxOut[0U] = HALF_R1 * (topLeft + bottomRight);
        mtrxOut[1U] = HALF_R1 * (topLeft - bottomRight);
        mtrxOut[2U] = mtrxOut[1U];
        mtrxOut[3U] = mtrxOut[0U];
    }

    void RevertBasisX(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::RevertBasisX!");
        }

        QEngineShard& shard = shards[i];
        if (shard.pauliBasis != PauliX) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        ConvertZToX(i);
    }

    void RevertBasisY(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::RevertBasisY!");
        }

        QEngineShard& shard = shards[i];

        if (shard.pauliBasis != PauliY) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        shard.pauliBasis = PauliX;

        if (shard.unit) {
            shard.unit->SqrtX(shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        QRACK_CONST complex diag = complex(HALF_R1, HALF_R1);
        QRACK_CONST complex cross = complex(HALF_R1, -HALF_R1);
        QRACK_CONST complex mtrx[4U]{ diag, cross, cross, diag };

        const complex Y0 = shard.amp0;
        const complex& Y1 = shard.amp1;
        shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * Y1);
        shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * Y1);
        ClampShard(i);
    }

    void RevertBasis1Qb(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::RevertBasis1Qb!");
        }

        QEngineShard& shard = shards[i];

        if (shard.pauliBasis == PauliY) {
            ConvertYToZ(i);
        } else {
            RevertBasisX(i);
        }
    }

    void RevertBasisToX1Qb(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(i) + " out of range in QUnit::RevertBasisToX1Qb!");
        }

        QEngineShard& shard = shards[i];
        if (shard.pauliBasis == PauliZ) {
            ConvertZToX(i);
        } else if (shard.pauliBasis == PauliY) {
            RevertBasisY(i);
        }
    }

    void RevertBasisToY1Qb(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(i) + " out of range in QUnit::RevertBasisToY1Qb!");
        }

        QEngineShard& shard = shards[i];
        if (shard.pauliBasis == PauliZ) {
            ConvertZToY(i);
        } else if (shard.pauliBasis == PauliX) {
            ConvertXToY(i);
        }
    }

    void ConvertZToX(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::ConvertZToX!");
        }

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

        const complex tempAmp1 = SQRT1_2_R1 * (shard.amp0 - shard.amp1);
        shard.amp0 = SQRT1_2_R1 * (shard.amp0 + shard.amp1);
        shard.amp1 = tempAmp1;
        ClampShard(i);
    }
    void ConvertXToY(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::ConvertXToY!");
        }

        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliY;

        if (shard.unit) {
            shard.unit->ISqrtX(shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        QRACK_CONST complex diag = complex(ONE_R1 / (real1)2, -ONE_R1 / (real1)2);
        QRACK_CONST complex cross = complex(ONE_R1 / (real1)2, ONE_R1 / (real1)2);
        QRACK_CONST complex mtrx[4U]{ diag, cross, cross, diag };

        const complex Y0 = shard.amp0;
        const complex& Y1 = shard.amp1;
        shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * Y1);
        shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * Y1);
        ClampShard(i);
    }
    void ConvertYToZ(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::ConvertYToZ!");
        }

        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliZ;

        if (shard.unit) {
            shard.unit->SH(shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        QRACK_CONST complex row1 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex mtrx[4U]{ row1, row1, complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };

        const complex Y0 = shard.amp0;
        const complex& Y1 = shard.amp1;
        shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * Y1);
        shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * Y1);
        ClampShard(i);
    }
    void ConvertZToY(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(i) + " out of range in QUnit::ConvertZToY!");
        }

        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliY;

        if (shard.unit) {
            shard.unit->HIS(shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        QRACK_CONST complex col1 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex mtrx[4U]{ col1, complex(ZERO_R1, -SQRT1_2_R1), col1, complex(ZERO_R1, SQRT1_2_R1) };

        const complex Y0 = shard.amp0;
        const complex& Y1 = shard.amp1;
        shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * Y1);
        shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * Y1);
        ClampShard(i);
    }
    void ShardAI(bitLenInt qubit, real1_f azimuth, real1_f inclination)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument("Qubit index " + std::to_string(qubit) + " out of range in QUnit::ShardAI!");
        }

        real1 cosineA = (real1)cos(azimuth);
        real1 sineA = (real1)sin(azimuth);
        real1 cosineI = (real1)cos(inclination / 2);
        real1 sineI = (real1)sin(inclination / 2);
        complex expA = complex(cosineA, sineA);
        complex expNegA = complex(cosineA, -sineA);
        complex mtrx[4U]{ cosineI, -expNegA * sineI, expA * sineI, cosineI };

        QEngineShard& shard = shards[qubit];

        const complex Y0 = shard.amp0;
        const complex& Y1 = shard.amp1;
        shard.amp0 = (mtrx[0U] * Y0) + (mtrx[1U] * Y1);
        shard.amp1 = (mtrx[2U] * Y0) + (mtrx[3U] * Y1);
        ClampShard(qubit);
    }

    enum RevertExclusivity { INVERT_AND_PHASE = 0, ONLY_INVERT = 1, ONLY_PHASE = 2 };
    enum RevertControl { CONTROLS_AND_TARGETS = 0, ONLY_CONTROLS = 1, ONLY_TARGETS = 2 };
    enum RevertAnti { CTRL_AND_ANTI = 0, ONLY_CTRL = 1, ONLY_ANTI = 2 };

    void ApplyBuffer(PhaseShardPtr phaseShard, bitLenInt control, bitLenInt target, bool isAnti);
    void ApplyBufferMap(bitLenInt bitIndex, ShardToPhaseMap bufferMap, RevertExclusivity exclusivity, bool isControl,
        bool isAnti, const std::set<bitLenInt>& exceptPartners, bool dumpSkipped);
    void RevertBasis2Qb(bitLenInt i, RevertExclusivity exclusivity = INVERT_AND_PHASE,
        RevertControl controlExclusivity = CONTROLS_AND_TARGETS, RevertAnti antiExclusivity = CTRL_AND_ANTI,
        const std::set<bitLenInt>& exceptControlling = {}, const std::set<bitLenInt>& exceptTargetedBy = {},
        bool dumpSkipped = false, bool skipOptimized = false);

    void Flush0Eigenstate(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(i) + " out of range in QUnit::Flush0Eigenstate!");
        }

        QEngineShard& shard = shards[i];
        shard.DumpControlOf();
        if (randGlobalPhase) {
            shard.DumpSamePhaseAntiControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
    }
    void Flush1Eigenstate(bitLenInt i)
    {
        if (i >= qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(i) + " out of range in QUnit::Flush1Eigenstate!");
        }

        QEngineShard& shard = shards[i];
        shard.DumpAntiControlOf();
        if (randGlobalPhase) {
            shard.DumpSamePhaseControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
    }
    void ToPermBasis(bitLenInt i)
    {
        RevertBasis1Qb(i);
        RevertBasis2Qb(i);
    }
    void ToPermBasis(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0U; i < length; ++i) {
            RevertBasis1Qb(start + i);
        }
        for (bitLenInt i = 0U; i < length; ++i) {
            RevertBasis2Qb(start + i);
        }
    }
    void ToPermBasisProb(bitLenInt qubit)
    {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
    }
    void ToPermBasisProb(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0U; i < length; ++i) {
            RevertBasis1Qb(start + i);
        }
        for (bitLenInt i = 0U; i < length; ++i) {
            RevertBasis2Qb(start + i, ONLY_INVERT, ONLY_TARGETS);
        }
    }
    void ToPermBasisAll() { ToPermBasis(0U, qubitCount); }
    void ToPermBasisProb() { ToPermBasisProb(0U, qubitCount); }
    void ToPermBasisMeasure(bitLenInt qubit)
    {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT);
        RevertBasis2Qb(qubit, ONLY_PHASE, ONLY_CONTROLS);

        shards[qubit].DumpMultiBit();
    }
    void ToPermBasisMeasure(bitLenInt start, bitLenInt length);
    void ToPermBasisAllMeasure();

    void DirtyShardRange(bitLenInt start, bitLenInt length)
    {
        if ((start + length) > qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(start + length) + " out of range in QUnit::DirtyShardRange!");
        }

        for (bitLenInt i = 0U; i < length; ++i) {
            shards[start + i].MakeDirty();
        }
    }

    void DirtyShardRangePhase(bitLenInt start, bitLenInt length)
    {
        if ((start + length) > qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(start + length) + " out of range in QUnit::DirtyShardRangePhase!");
        }

        for (bitLenInt i = 0U; i < length; ++i) {
            shards[start + i].isPhaseDirty = true;
        }
    }

    void DirtyShardIndexVector(std::vector<bitLenInt> bitIndices)
    {
        for (const bitLenInt& bitIndex : bitIndices) {
            if (bitIndex >= qubitCount) {
                throw std::invalid_argument(
                    "Qubit index " + std::to_string(bitIndex) + " out of range in QUnit::DirtyShardRangePhase!");
            }

            shards[bitIndex].MakeDirty();
        }
    }

    void EndEmulation(bitLenInt target)
    {
        if (target >= qubitCount) {
            throw std::invalid_argument(
                "Qubit index " + std::to_string(target) + " out of range in QUnit::EndEmulation!");
        }

        QEngineShard& shard = shards[target];
        if (shard.unit) {
            return;
        }

        if (norm(shard.amp1) <= FP_NORM_EPSILON) {
            shard.unit = MakeEngine(1U, ZERO_BCI);
        } else if (norm(shard.amp0) <= FP_NORM_EPSILON) {
            shard.unit = MakeEngine(1U, ONE_BCI);
        } else {
            const complex bitState[2U]{ shard.amp0, shard.amp1 };
            shard.unit = MakeEngine(1U, ZERO_BCI);
            shard.unit->SetQuantumState(bitState);
        }
    }

    bitLenInt FindShardIndex(QEngineShardPtr shard)
    {
        shard->found = true;
        for (bitLenInt i = 0U; i < shards.size(); ++i) {
            if (shards[i].found) {
                shard->found = false;
                return i;
            }
        }
        shard->found = false;
        return shards.size();
    }

    void CommuteH(bitLenInt bitIndex);

    void OptimizePairBuffers(bitLenInt control, bitLenInt target, bool anti);
};

} // namespace Qrack
