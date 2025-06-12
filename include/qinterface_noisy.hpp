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

#if ENABLE_ALU
#include "qalu.hpp"
#endif

namespace Qrack {

class QInterfaceNoisy;
typedef std::shared_ptr<QInterfaceNoisy> QInterfaceNoisyPtr;

/**
 * A "Qrack::QInterfaceNoisy" that wraps any other QInterface with a simple noise model
 */
class QInterfaceNoisy : public QInterface {
protected:
    double logFidelity;
    real1_f noiseParam;
    QInterfacePtr engine;
    std::vector<QInterfaceEngine> engines;

    void Apply1QbNoise(bitLenInt qb)
    {
        real1_f n = noiseParam;
#if ENABLE_ENV_VARS
        if (getenv("QRACK_GATE_DEPOLARIZATION")) {
            n = (real1_f)std::stof(std::string(getenv("QRACK_GATE_DEPOLARIZATION")));
        }
#endif
        if (n <= ZERO_R1_F) {
            return;
        }

        engine->DepolarizingChannelWeak1Qb(qb, n);

        if ((n + FP_NORM_EPSILON) >= ONE_R1_F) {
            logFidelity = -1 * std::numeric_limits<float>::infinity();
        } else {
            logFidelity += (double)log(ONE_R1_F - n);
        }
    }

    using QInterface::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QInterfaceNoisy>(orig)); }
    void Copy(QInterfaceNoisyPtr orig)
    {
        QInterface::Copy(orig);
        logFidelity = orig->logFidelity;
        noiseParam = orig->noiseParam;
        engine = orig->engine;
        engines = orig->engines;
    }

public:
    QInterfaceNoisy(bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = _qrack_qunit_sep_thresh)
        : QInterfaceNoisy({ QINTERFACE_TENSOR_NETWORK }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
        // Intentionally left blank;
    }

    QInterfaceNoisy(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = _qrack_qunit_sep_thresh);

    QInterfaceNoisy(QInterfaceNoisy* o)
        : noiseParam(o->noiseParam)
        , engine(o->engine)
        , engines(o->engines)
    {
        engine = o->engine->Clone();
    }

    void SetSdrp(real1_f sdrp) { engine->SetSdrp(sdrp); }
    void SetReactiveSeparate(bool isAggSep) { engine->SetReactiveSeparate(isAggSep); }

    void SetNoiseParameter(real1_f lambda) { noiseParam = lambda; }
    real1_f GetNoiseParameter() { return noiseParam; }

    double GetUnitaryFidelity() { return (double)exp(logFidelity); }
    void ResetUnitaryFidelity() { logFidelity = 0.0; }

    bool isOpenCL() { return engine->isOpenCL(); }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        engine->SetConcurrency(GetConcurrencyLevel());
    }

    real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation)
    {
        return engine->ProbReg(start, length, permutation);
    }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        SetQubitCount(qubitCount + length);
        return engine->Allocate(start, length);
    }
    using QInterface::Compose;
    bitLenInt Compose(QInterfaceNoisyPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->Compose(toCopy->engine);
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy)); }
    bitLenInt Compose(QInterfaceNoisyPtr toCopy, bitLenInt start)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->Compose(toCopy->engine, start);
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QInterfaceNoisyPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->ComposeNoClone(toCopy->engine);
    }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy));
    }
    using QInterface::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QInterfaceNoisy>(dest));
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        QInterfaceNoisyPtr dest = std::make_shared<QInterfaceNoisy>(this);
        engine->Decompose(start, dest->engine);

        return dest;
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QInterfaceNoisy>(dest), error_tol);
    }
    void Decompose(bitLenInt start, QInterfaceNoisyPtr dest)
    {
        engine->Decompose(start, dest->engine);
        SetQubitCount(qubitCount - dest->GetQubitCount());
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        engine->Dispose(start, length);
        SetQubitCount(qubitCount - length);
    }
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm)
    {
        engine->Dispose(start, length, disposedPerm);
        SetQubitCount(qubitCount - length);
    }

    bool TryDecompose(bitLenInt start, QInterfaceNoisyPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        const bitLenInt nQubitCount = qubitCount - dest->GetQubitCount();
        const bool result = engine->TryDecompose(start, dest->engine, error_tol);
        if (result) {
            SetQubitCount(nQubitCount);
        }
        return result;
    }

    void SetQuantumState(const complex* inputState) { engine->SetQuantumState(inputState); }
    void GetQuantumState(complex* outputState) { engine->GetQuantumState(outputState); }
    void GetProbs(real1* outputProbs) { engine->GetProbs(outputProbs); }
    complex GetAmplitude(const bitCapInt& perm) { return engine->GetAmplitude(perm); }
    void SetAmplitude(const bitCapInt& perm, const complex& amp) { engine->SetAmplitude(perm, amp); }
    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG)
    {
        engine->SetPermutation(perm, phaseFac);
    }

    void Mtrx(const complex* mtrx, bitLenInt qubitIndex)
    {
        engine->Mtrx(mtrx, qubitIndex);
        Apply1QbNoise(qubitIndex);
    }
    void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubitIndex)
    {
        engine->Phase(topLeft, bottomRight, qubitIndex);
        Apply1QbNoise(qubitIndex);
    }
    void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubitIndex)
    {
        engine->Invert(topRight, bottomLeft, qubitIndex);
        Apply1QbNoise(qubitIndex);
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MCMtrx(controls, mtrx, target);
        Apply1QbNoise(target);
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MACMtrx(controls, mtrx, target);
        Apply1QbNoise(target);
    }

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt> mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask)
    {
        engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
        Apply1QbNoise(qubitIndex);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }

    void XMask(const bitCapInt& _mask)
    {
        bitCapInt mask = _mask;
        engine->XMask(mask);
        bitCapInt v = mask;
        while (bi_compare_0(mask) != 0) {
            v = v & (v - ONE_BCI);
            Apply1QbNoise(log2(mask ^ v));
            mask = v;
        }
    }
    void PhaseParity(real1_f radians, const bitCapInt& _mask)
    {
        bitCapInt mask = _mask;
        engine->PhaseParity(radians, mask);
        bitCapInt v = mask;
        while (bi_compare_0(mask) != 0) {
            v = v & (v - ONE_BCI);
            Apply1QbNoise(log2(mask ^ v));
            mask = v;
        }
    }

    real1_f CProb(bitLenInt control, bitLenInt target) { return engine->CProb(control, target); }
    real1_f ACProb(bitLenInt control, bitLenInt target) { return engine->ACProb(control, target); }

    void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }
    void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }
    void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSqrtSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSqrtSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }
    void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CISqrtSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCISqrtSwap(controls, qubit1, qubit2);
        Apply1QbNoise(qubit1);
        Apply1QbNoise(qubit2);
        for (const bitLenInt& control : controls) {
            Apply1QbNoise(control);
        }
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->Swap(qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->ISwap(qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->IISwap(qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
        Apply1QbNoise(qubitIndex1);
        Apply1QbNoise(qubitIndex2);
    }

    real1_f Prob(bitLenInt qubitIndex) { return engine->Prob(qubitIndex); }
    real1_f ProbAll(const bitCapInt& fullRegister) { return engine->ProbAll(fullRegister); }
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        return engine->ProbMask(mask, permutation);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QInterfaceNoisy>(toCompare));
    }
    real1_f SumSqrDiff(QInterfaceNoisyPtr toCompare) { return engine->SumSqrDiff(toCompare->engine); }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return engine->ExpectationBitsAll(bits, offset);
    }

    void Finish() { engine->Finish(); }

    bool isFinished() { return engine->isFinished(); }

    void Dump() { engine->Dump(); }

    QInterfacePtr Clone() { return std::make_shared<QInterfaceNoisy>(this); }

    void SetDevice(int64_t dID) { engine->SetDevice(dID); }
    void SetDeviceList(std::vector<int64_t> dIDs) { engine->SetDeviceList(dIDs); }
    int64_t GetDevice() { return engine->GetDevice(); }
    std::vector<int64_t> GetDeviceList() { return engine->GetDeviceList(); }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };
};
} // namespace Qrack
