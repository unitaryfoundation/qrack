//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qunitclifford.hpp"

#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

struct QUnitCliffordAmp {
    complex amp;
    QUnitCliffordPtr stabilizer;

    QUnitCliffordAmp(const complex& a, QUnitCliffordPtr s)
        : amp(a)
        , stabilizer(s)
    {
        // Intentionally left blank.
    }
};

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QStabilizer and Qrack::QEngine to maximize
 * performance.
 */
#if ENABLE_ALU
class QStabilizerHybrid : public QAlu, public QParity, public QInterface {
#else
class QStabilizerHybrid : public QParity, public QInterface {
#endif
protected:
    bool useHostRam;
    bool doNormalize;
    bool useTGadget;
    bool isRoundingFlushed;
    bitLenInt thresholdQubits;
    bitLenInt ancillaCount;
    bitLenInt deadAncillaCount;
    bitLenInt maxEngineQubitCount;
    bitLenInt maxAncillaCount;
    bitLenInt origMaxAncillaCount;
    bitLenInt maxStateMapCacheQubitCount;
    real1_f separabilityThreshold;
    real1_f roundingThreshold;
    int64_t devID;
    complex phaseFactor;
    double logFidelity;
    QInterfacePtr engine;
    QUnitCliffordPtr stabilizer;
    QStabilizerHybridPtr rdmClone;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engineTypes;
    std::vector<QInterfaceEngine> cloneEngineTypes;
    std::vector<MpsShardPtr> shards;
    QUnitStateVectorPtr stateMapCache;
    std::default_random_engine prng;

    QUnitCliffordPtr MakeStabilizer(const bitCapInt& perm = ZERO_BCI);
    QInterfacePtr MakeEngine(const bitCapInt& perm = ZERO_BCI);
    QInterfacePtr MakeEngine(const bitCapInt& perm, bitLenInt qbCount);

    void SetQubitCount(bitLenInt qb)
    {
        rdmClone = nullptr;
        QInterface::SetQubitCount(qb);
    }

    void InvertBuffer(bitLenInt qubit);
    void FlushH(bitLenInt qubit);
    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false);
    bool CollapseSeparableShard(bitLenInt qubit);
    bool TrimControls(const std::vector<bitLenInt>& lControls, std::vector<bitLenInt>& output, bool anti = false);
    void CacheEigenstate(bitLenInt target);
    void FlushBuffers();
    void DumpBuffers()
    {
        rdmClone = nullptr;
        for (MpsShardPtr& shard : shards) {
            shard = nullptr;
        }
    }
    bool EitherIsBuffered(bool logical)
    {
        const size_t maxLcv = logical ? (size_t)qubitCount : shards.size();
        for (size_t i = 0U; i < maxLcv; ++i) {
            if (shards[i]) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsBuffered() { return EitherIsBuffered(false); }
    bool IsLogicalBuffered() { return EitherIsBuffered(true); }
    bool EitherIsProbBuffered(bool logical)
    {
        const size_t maxLcv = logical ? (size_t)qubitCount : shards.size();
        for (size_t i = 0U; i < maxLcv; ++i) {
            MpsShardPtr shard = shards[i];
            if (!shard) {
                continue;
            }
            if (shard->IsHPhase() || shard->IsHInvert()) {
                FlushH(i);
            }
            if (shard->IsInvert()) {
                InvertBuffer(i);
            }
            if (!shard->IsPhase()) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsProbBuffered() { return EitherIsProbBuffered(false); }
    bool IsLogicalProbBuffered() { return EitherIsProbBuffered(true); }

    void UpdateRoundingThreshold()
    {
#if ENABLE_ENV_VARS
        if (!isRoundingFlushed && getenv("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD")) {
            roundingThreshold = (real1_f)std::stof(std::string(getenv("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD")));
        }
#endif
        if (maxAncillaCount != (bitLenInt)(-1)) {
            origMaxAncillaCount = maxAncillaCount;
        }
        if ((ONE_R1_F - roundingThreshold) <= FP_NORM_EPSILON) {
            maxAncillaCount = -1;
        } else {
            maxAncillaCount = origMaxAncillaCount;
        }
    }

    std::unique_ptr<complex[]> GetQubitReducedDensityMatrix(bitLenInt qubit)
    {
        // Form the reduced density matrix of the single qubit.
        const real1 z = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->H(qubit);
        const real1 x = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->S(qubit);
        const real1 y = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->IS(qubit);
        stabilizer->H(qubit);

        std::unique_ptr<complex[]> dMtrx(new complex[4]);
        dMtrx[0] = (ONE_CMPLX + z) / complex((real1)2, ZERO_R1);
        dMtrx[1] = x / complex((real1)2, ZERO_R1) - I_CMPLX * (y / complex((real1)2, ZERO_R1));
        dMtrx[2] = x / complex((real1)2, ZERO_R1) + I_CMPLX * (y / complex((real1)2, ZERO_R1));
        dMtrx[3] = (ONE_CMPLX + z) / complex((real1)2, ZERO_R1);
        if (shards[qubit]) {
            const complex adj[4]{ std::conj(shards[qubit]->gate[0]), std::conj(shards[qubit]->gate[2]),
                std::conj(shards[qubit]->gate[1]), std::conj(shards[qubit]->gate[3]) };
            complex out[4];
            mul2x2(dMtrx.get(), adj, out);
            mul2x2(shards[qubit]->gate, out, dMtrx.get());
        }

        return dMtrx;
    }

    template <typename F>
    void CheckShots(unsigned shots, const bitCapInt& m, real1_f partProb, const std::vector<bitCapInt>& qPowers,
        std::vector<real1_f>& rng, F fn)
    {
        for (int64_t shot = rng.size() - 1U; shot >= 0; --shot) {
            if (rng[shot] >= partProb) {
                break;
            }

            bitCapInt sample = ZERO_BCI;
            for (size_t i = 0U; i < qPowers.size(); ++i) {
                if (bi_compare_0(m & qPowers[i]) != 0) {
                    bi_or_ip(&sample, pow2(i));
                }
            }
            fn(sample, (unsigned int)shot);

            rng.erase(rng.begin() + shot);
            if (rng.empty()) {
                break;
            }
        }
    }

    std::vector<real1_f> GenerateShotProbs(unsigned shots)
    {
        std::vector<real1_f> rng;
        rng.reserve(shots);
        for (unsigned shot = 0U; shot < shots; ++shot) {
            rng.push_back(Rand());
        }
        std::sort(rng.begin(), rng.end());
        std::reverse(rng.begin(), rng.end());

        return rng;
    }

    real1_f FractionalRzAngleWithFlush(bitLenInt i, real1_f angle, bool isGateSuppressed = false)
    {
        const real1_f sectorAngle = PI_R1 / 2;
        const real1_f Period = 2 * PI_R1;
        while (angle >= Period) {
            angle -= Period;
        }
        while (angle < 0U) {
            angle += Period;
        }

        const long sector = std::lround((real1_s)(angle / sectorAngle));
        if (!isGateSuppressed) {
            switch (sector) {
            case 1:
                stabilizer->S(i);
                break;
            case 2:
                stabilizer->Z(i);
                break;
            case 3:
                stabilizer->IS(i);
                break;
            case 0:
            default:
                break;
            }
        }

        angle -= (sector * sectorAngle);
        if (angle > PI_R1) {
            angle -= Period;
        }
        if (angle <= -PI_R1) {
            angle += Period;
        }

        return angle;
    }

    void FlushCliffordFromBuffers()
    {
        for (size_t i = 0U; i < qubitCount; ++i) {
            // Flush all buffers as close as possible to Clifford.
            const MpsShardPtr& shard = shards[i];
            if (!shard) {
                continue;
            }
            if (shard->IsHPhase() || shard->IsHInvert()) {
                FlushH(i);
            }
            if (shard->IsInvert()) {
                InvertBuffer(i);
            }
            if (!shard->IsPhase()) {
                // We have a cached non-phase operation.
                continue;
            }
            const real1 angle = (real1)(FractionalRzAngleWithFlush(i, std::arg(shard->gate[3U] / shard->gate[0U])) / 2);
            if ((2 * abs(angle)) <= (FP_NORM_EPSILON * PI_R1)) {
                shards[i] = nullptr;
                continue;
            }
            const real1 angleCos = cos(angle);
            const real1 angleSin = sin(angle);
            shard->gate[0U] = complex(angleCos, -angleSin);
            shard->gate[3U] = complex(angleCos, angleSin);
        }
        RdmCloneFlush();
    }

    QStabilizerHybridPtr RdmCloneHelper()
    {
        if (rdmClone) {
            return rdmClone;
        }

        rdmClone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        rdmClone->RdmCloneFlush(HALF_R1);

        return rdmClone;
    }
    void RdmCloneFlush(real1_f threshold = FP_NORM_EPSILON);

    real1_f ExpVarFactorized(bool isExp, bool isFloat, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const std::vector<real1_f>& weights, const bitCapInt& offset, bool roundRz)
    {
        if (engine) {
            return isExp  ? isFloat ? engine->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                                    : engine->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset)
                 : isFloat ? engine->VarianceFloatsFactorizedRdm(roundRz, bits, weights)
                          : engine->VarianceBitsFactorizedRdm(roundRz, bits, perms, offset);
        }

        if (!roundRz) {
            return isExp  ? isFloat ? stabilizer->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                                    : stabilizer->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset)
                 : isFloat ? stabilizer->VarianceFloatsFactorizedRdm(roundRz, bits, weights)
                          : stabilizer->VarianceBitsFactorizedRdm(roundRz, bits, perms, offset);
        }

        QStabilizerHybridPtr clone = RdmCloneHelper();

        return isExp  ? isFloat ? clone->stabilizer->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                                : clone->stabilizer->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset)
             : isFloat ? clone->stabilizer->VarianceFloatsFactorizedRdm(roundRz, bits, weights)
                      : clone->stabilizer->VarianceBitsFactorizedRdm(roundRz, bits, perms, offset);
    }

    void ClearAncilla(bitLenInt i)
    {
        if (stabilizer->TrySeparate(i)) {
            stabilizer->Dispose(i, 1U);
            shards.erase(shards.begin() + i);
        } else {
            const bitLenInt deadIndex = qubitCount + ancillaCount - 1U;
            stabilizer->SetBit(i, false);
            if (i != deadIndex) {
                stabilizer->Swap(i, deadIndex);
                shards[i].swap(shards[deadIndex]);
            }
            shards.erase(shards.begin() + deadIndex);
            ++deadAncillaCount;
        }
        --ancillaCount;
    }

    real1_f ApproxCompareHelper(
        QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);

    void ISwapHelper(bitLenInt qubit1, bitLenInt qubit2, bool inverse);

    complex GetAmplitudeOrProb(const bitCapInt& perm, bool isProb = false);

    QInterfacePtr CloneBody(bool isCopy);
    using QInterface::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QStabilizerHybrid>(orig)); }
    void Copy(QStabilizerHybridPtr orig)
    {
        QInterface::Copy(std::dynamic_pointer_cast<QInterface>(orig));
        useHostRam = orig->useHostRam;
        doNormalize = orig->doNormalize;
        useTGadget = orig->useTGadget;
        isRoundingFlushed = orig->isRoundingFlushed;
        thresholdQubits = orig->thresholdQubits;
        ancillaCount = orig->ancillaCount;
        deadAncillaCount = orig->deadAncillaCount;
        maxEngineQubitCount = orig->maxEngineQubitCount;
        maxAncillaCount = orig->maxAncillaCount;
        maxStateMapCacheQubitCount = orig->maxStateMapCacheQubitCount;
        separabilityThreshold = orig->separabilityThreshold;
        roundingThreshold = orig->roundingThreshold;
        devID = orig->devID;
        phaseFactor = orig->phaseFactor;
        logFidelity = orig->logFidelity;
        engine = orig->engine;
        stabilizer = orig->stabilizer;
        deviceIDs = orig->deviceIDs;
        engineTypes = orig->engineTypes;
        cloneEngineTypes = orig->cloneEngineTypes;
        shards = orig->shards;
        stateMapCache = orig->stateMapCache;
    }

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool ignored = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QStabilizerHybrid(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool ignored = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh)
        : QStabilizerHybrid({ QINTERFACE_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, ignored, norm_thresh, devList, qubitThreshold, separation_thresh)
    {
    }

    void SetNcrp(real1_f ncrp)
    {
        roundingThreshold = ncrp;
        // Environment variable always overrides:
        UpdateRoundingThreshold();
    };
    void SetTInjection(bool useGadget) { useTGadget = useGadget; }
    bool GetTInjection() { return useTGadget; }
    double GetUnitaryFidelity() { return exp(logFidelity); }
    void ResetUnitaryFidelity() { logFidelity = 0.0; }

    void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    void Dump()
    {
        if (stabilizer) {
            stabilizer->Dump();
        } else {
            engine->Dump();
        }
    }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (engine) {
            SetConcurrency(GetConcurrencyLevel());
        }
    }

    real1_f ProbRdm(bitLenInt qubit)
    {
        if (!ancillaCount || stabilizer->IsSeparable(qubit)) {
            return Prob(qubit);
        }

        std::unique_ptr<complex[]> dMtrx = GetQubitReducedDensityMatrix(qubit);
        QRACK_CONST complex ONE_CMPLX_NEG = complex(-ONE_R1, ZERO_R1);
        QRACK_CONST complex pauliZ[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX_NEG };
        complex pMtrx[4];
        mul2x2(dMtrx.get(), pauliZ, pMtrx);

        return (ONE_R1 - std::real(pMtrx[0]) + std::real(pMtrx[1])) / 2;
    }

    real1_f CProbRdm(bitLenInt control, bitLenInt target)
    {
        AntiCNOT(control, target);
        const real1_f prob = ProbRdm(target);
        AntiCNOT(control, target);

        return prob;
    }

    real1_f ACProbRdm(bitLenInt control, bitLenInt target)
    {
        CNOT(control, target);
        const real1_f prob = ProbRdm(target);
        CNOT(control, target);

        return prob;
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchToEngine();

    bool isClifford() { return !engine; }

    bool isClifford(bitLenInt qubit) { return !engine && !shards[qubit]; };

    bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QInterface::Compose;
    bitLenInt Compose(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, false); };
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy)); }
    bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, true); };
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    bitLenInt ComposeEither(QStabilizerHybridPtr toCopy, bool willDestroy);
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm);
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(const bitCapInt& perm) { return GetAmplitudeOrProb(perm, false); }
    real1_f ProbAll(const bitCapInt& perm) { return (real1_f)norm(GetAmplitudeOrProb(perm, true)); }
    void SetQuantumState(const complex* inputState);
    void SetAmplitude(const bitCapInt& perm, const complex& amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    void Swap(bitLenInt qubit1, bitLenInt qubit2);
    void ISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, false); }
    void IISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, true); }
    void CSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void CSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void AntiCSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void CISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void AntiCISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);

    void XMask(const bitCapInt& mask);
    void YMask(const bitCapInt& mask);
    void ZMask(const bitCapInt& mask);

    real1_f Prob(bitLenInt qubit);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll();

    void Mtrx(const complex* mtrx, bitLenInt target);
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target);
    void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target);
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target);
    void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target);

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(
        const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, const complex* mtrxs);

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);
    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    real1_f ProbParity(const bitCapInt& mask);
    bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->CUniformParityRZ(controls, mask, angle);
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }
    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->PhaseFlipIfLess(greaterPerm, start, length);
    }

    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            return QInterface::INC(toAdd, start, length);
        }

        engine->INC(toAdd, start, length);
    }
    void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            return QInterface::DEC(toSub, start, length);
        }

        engine->DEC(toSub, start, length);
    }
    void DECS(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            return QInterface::DECS(toSub, start, length, overflowIndex);
        }

        engine->DECS(toSub, start, length, overflowIndex);
    }
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (stabilizer) {
            return QInterface::CINC(toAdd, inOutStart, length, controls);
        }

        engine->CINC(toAdd, inOutStart, length, controls);
    }
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            return QInterface::INCS(toAdd, start, length, overflowIndex);
        }

        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (stabilizer) {
            return QInterface::INCDECC(toAdd, start, length, carryIndex);
        }

        engine->INCDECC(toAdd, start, length, carryIndex);
    }
    void INCDECSC(
        const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCBCD(toAdd, start, length);
    }
    void INCDECBCDC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECBCDC(toAdd, start, length, carryIndex);
    }
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedLDA(
            indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedADC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedSBC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->Hash(start, length, values);
    }
#endif

    void PhaseFlip()
    {
        rdmClone = nullptr;
        if (stabilizer) {
            stabilizer->PhaseFlip();
        } else {
            engine->PhaseFlip();
        }
    }
    void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }

    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            return QInterface::SqrtSwap(qubitIndex1, qubitIndex2);
        }

        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            return QInterface::ISqrtSwap(qubitIndex1, qubitIndex2);
        }

        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }

    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), false);
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >=
            ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), true, error_tol);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    real1_f ProbAllRdm(bool roundRz, const bitCapInt& fullRegister);
    real1_f ProbMaskRdm(bool roundRz, const bitCapInt& mask, const bitCapInt& permutation);
    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, offset);
        }

        return engine->ExpectationBitsAll(bits, offset);
    }
    real1_f ExpectationBitsAllRdm(bool roundRz, const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (engine) {
            return engine->ExpectationBitsAllRdm(roundRz, bits, offset);
        }

        if (!roundRz) {
            return stabilizer->ExpectationBitsAll(bits, offset);
        }

        return RdmCloneHelper()->stabilizer->ExpectationBitsAll(bits, offset);
    }
    real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsFactorized(bits, perms, offset);
        }

        return engine->ExpectationBitsFactorized(bits, perms, offset);
    }
    real1_f ExpectationBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(true, false, bits, perms, std::vector<real1_f>(), offset, roundRz);
    }
    real1_f ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        if (stabilizer) {
            return QInterface::ExpectationFloatsFactorized(bits, weights);
        }

        return engine->ExpectationFloatsFactorized(bits, weights);
    }
    real1_f ExpectationFloatsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpVarFactorized(true, true, bits, std::vector<bitCapInt>(), weights, ZERO_BCI, roundRz);
    }
    real1_f VarianceBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (stabilizer) {
            return QInterface::VarianceBitsAll(bits, offset);
        }

        return engine->VarianceBitsAll(bits, offset);
    }
    real1_f VarianceBitsAllRdm(bool roundRz, const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        if (engine) {
            return engine->VarianceBitsAllRdm(roundRz, bits, offset);
        }

        if (!roundRz) {
            return stabilizer->VarianceBitsAll(bits, offset);
        }

        return RdmCloneHelper()->stabilizer->VarianceBitsAll(bits, offset);
    }
    real1_f VarianceBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        if (stabilizer) {
            return QInterface::VarianceBitsFactorized(bits, perms, offset);
        }

        return engine->VarianceBitsFactorized(bits, perms, offset);
    }
    real1_f VarianceBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarFactorized(true, false, bits, perms, std::vector<real1_f>(), offset, roundRz);
    }
    real1_f VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        if (stabilizer) {
            return QInterface::VarianceFloatsFactorized(bits, weights);
        }

        return engine->VarianceFloatsFactorized(bits, weights);
    }
    real1_f VarianceFloatsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpVarFactorized(true, true, bits, std::vector<bitCapInt>(), weights, ZERO_BCI, roundRz);
    }

    bool TrySeparate(bitLenInt qubit);
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);
    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol);

    QInterfacePtr Clone() { return CloneBody(false); }
    QInterfacePtr Copy() { return CloneBody(true); }

    void SetDevice(int64_t dID)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID);
        }
    }

    void SetDeviceList(std::vector<int64_t> dIDs)
    {
        deviceIDs = dIDs;
        if (engine) {
            engine->SetDeviceList(dIDs);
        }
    }
    int64_t GetDevice() { return devID; }
    std::vector<int64_t> GetDeviceList() { return deviceIDs; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }

    friend std::ostream& operator<<(std::ostream& os, const QStabilizerHybridPtr s);
    friend std::istream& operator>>(std::istream& is, const QStabilizerHybridPtr s);
};
} // namespace Qrack
