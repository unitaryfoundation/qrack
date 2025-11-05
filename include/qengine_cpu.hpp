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

#include "qengine.hpp"
#include "statevector.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif

namespace Qrack {

class QEngineCPU;
typedef std::shared_ptr<QEngineCPU> QEngineCPUPtr;

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, const bitCapInt& stride);
template <class BidirectionalIterator>
void rotate(
    BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, const bitCapInt& stride);

/**
 * General purpose QEngineCPU implementation
 */
class QEngineCPU : public QEngine {
protected:
    StateVectorPtr stateVec;
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
    DispatchQueue dispatchQueue;
#endif
    double fidelity;
    bool isSparse;

    using QEngine::Copy;
    void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QEngineCPU>(orig)); }
    void Copy(QEngineCPUPtr orig)
    {
        QEngine::Copy(std::dynamic_pointer_cast<QEngine>(orig));
        stateVec = orig->stateVec;
        fidelity = orig->fidelity;
        isSparse = orig->isSparse;
    }

    StateVectorSparsePtr CastStateVecSparse() { return std::dynamic_pointer_cast<StateVectorSparse>(stateVec); }

public:
    QEngineCPU(bitLenInt qBitCount, const bitCapInt& initState, qrack_rand_gen_ptr rgp = nullptr,
        const complex& phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool ignored = false, int64_t ignored2 = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored3 = {}, bitLenInt ignored4 = 0U,
        real1_f ignored5 = _qrack_qunit_sep_thresh);

    ~QEngineCPU() { Dump(); }

    void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.finish();
#endif
    };

    bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        return dispatchQueue.isFinished();
#else
        return true;
#endif
    }

    void Dump()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dump();
#endif
    }

    real1_f FirstNonzeroPhase()
    {
        if (!stateVec) {
            return ZERO_R1_F;
        }

        return QInterface::FirstNonzeroPhase();
    }

    void ZeroAmplitudes()
    {
        Dump();
        FreeStateVec();
        runningNorm = ZERO_R1;
        fidelity = 1.0;
    }

    void TruncateBySize()
    {
        if (isSparse) {
            fidelity *= CastStateVecSparse()->truncate_to_size(QRACK_SPARSE_MAX_KEYS);
        }
    }

    bool IsZeroAmplitude() { return !stateVec; }
    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length);
    void ShuffleBuffers(QEnginePtr engine);
    void CopyStateVec(QEnginePtr src);

    QEnginePtr CloneEmpty();

    void QueueSetDoNormalize(bool doNorm)
    {
        Dispatch(1U, [this, doNorm] { doNormalize = doNorm; });
    }
    void QueueSetRunningNorm(real1_f runningNrm)
    {
        Dispatch(1U, [this, runningNrm] { runningNorm = runningNrm; });
    }

    void SetQuantumState(const complex* inputState);
    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(const bitCapInt& perm);
    void SetAmplitude(const bitCapInt& perm, const complex& amp);

    using QEngine::Compose;
    bitLenInt Compose(QEngineCPUPtr toCopy);
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy)); }
    std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy);
    bitLenInt Compose(QEngineCPUPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy), start);
    }

    using QEngine::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest);

    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm);

    using QEngine::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    /** @} */

    void XMask(const bitCapInt& mask);
    void PhaseParity(real1_f radians, const bitCapInt& mask);
    void PhaseRootNMask(bitLenInt n, const bitCapInt& mask);

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
#if ENABLE_ALU
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
    void CINC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls);
    void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
#if ENABLE_BCD
    void INCBCD(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
#endif
    void MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void POWModNOut(
        const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void CPOWModNOut(const bitCapInt& base, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);
    void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true);
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);
    void CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length);

    /** @} */
#endif

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);
    using QEngine::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask);
    void UniformParityRZ(const bitCapInt& mask, real1_f angle);
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, const bitCapInt& mask, real1_f angle);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    real1_f Prob(bitLenInt qubitIndex);
    real1_f CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target);
    real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation);
    real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    real1_f ProbParity(const bitCapInt& mask);
    bitCapInt HighestProbAll();
    bitCapInt MAll();
    bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);
    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QEngineCPU>(toCompare)); }
    real1_f SumSqrDiff(QEngineCPUPtr toCompare);
    QInterfacePtr Clone();
    QInterfacePtr Copy();

    /** @} */

protected:
    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength);

    StateVectorPtr AllocStateVec(bitCapIntOcl elemCount)
    {
        if (isSparse) {
            return std::make_shared<StateVectorSparse>(elemCount);
        } else {
            return std::make_shared<StateVectorArray>(elemCount);
        }
    }
    void ResetStateVec(StateVectorPtr sv) { stateVec = sv; }
    void FreeStateVec(complex* sv = nullptr) { stateVec = nullptr; }

    void Dispatch(bitCapIntOcl workItemCount, DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        if ((workItemCount >= pow2Ocl(GetPreferredConcurrencyPower())) && (workItemCount < GetStride())) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr dest);
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG);
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    using QEngine::ApplyM;
    void ApplyM(const bitCapInt& mask, const bitCapInt& result, const complex& nrm);

#if ENABLE_ALU
    void INCDECC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(
        const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCDECBCDC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif

    typedef std::function<bitCapIntOcl(const bitCapIntOcl&, const bitCapIntOcl&)> IOFn;
    void MULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length);
    void CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length, const std::vector<bitLenInt>& controls);

    typedef std::function<bitCapIntOcl(const bitCapIntOcl&)> MFn;
    void ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const bool& inverse = false);
    void CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const std::vector<bitLenInt>& controls, const bool& inverse = false);
#endif
};
} // namespace Qrack
