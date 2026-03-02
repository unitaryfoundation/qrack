//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"
#include "qparity.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

namespace Qrack {

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

/**
 * Abstract QEngine implementation, for all "Schroedinger method" engines
 */
#if ENABLE_ALU
class QEngine : public QAlu, public QParity, public QInterface {
#else
class QEngine : public QParity, public QInterface {
#endif
protected:
    bool useHostRam;
    /// The value stored in runningNorm should always be the total probability implied by the norm of all amplitudes,
    /// summed, at each update. To normalize, we should always multiply by 1/sqrt(runningNorm).
    real1 runningNorm;
    bitCapIntOcl maxQPowerOcl;
    double fidelity;

    inline bool IsPhase(const complex* mtrx) { return IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]); }
    inline bool IsInvert(const complex* mtrx) { return IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]); }

    bool IsIdentity(const complex* mtrx, bool isControlled)
    {
        // If the effect of applying the buffer would be (approximately or exactly) that of applying the identity
        // operator, then we can discard this buffer without applying it.
        if (!IS_NORM_0(mtrx[0U] - mtrx[3U]) || !IsPhase(mtrx)) {
            return false;
        }

        // Now, we know that mtrx[1] and mtrx[2] are 0 and mtrx[0]==mtrx[3].

        // If the global phase offset has been randomized, we assume that global phase offsets are inconsequential, for
        // the user's purposes. If the global phase offset has not been randomized, user code might explicitly depend on
        // the global phase offset.

        if ((isControlled || !randGlobalPhase) && !IS_SAME(ONE_CMPLX, mtrx[0U])) {
            return false;
        }

        // If we haven't returned false by now, we're buffering an identity operator (exactly or up to an arbitrary
        // global phase factor).
        return true;
    }

    void EitherMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, bool isAnti);

    virtual void Copy(QInterfacePtr orig) { Copy(std::dynamic_pointer_cast<QEngine>(orig)); }
    virtual void Copy(QEnginePtr orig)
    {
        QInterface::Copy(orig);
        useHostRam = orig->useHostRam;
        runningNorm = orig->runningNorm;
        maxQPowerOcl = orig->maxQPowerOcl;
    }

public:
    QEngine(bitLenInt qBitCount, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, bool useHardwareRNG = true, real1_f norm_thresh = REAL1_EPSILON)
        : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
        , useHostRam(useHostMem)
        , runningNorm(ONE_R1)
        , maxQPowerOcl(pow2Ocl(qBitCount))
        , fidelity(1.0)
    {
        if (qBitCount > (sizeof(bitCapIntOcl) * bitsInByte)) {
            throw std::invalid_argument(
                "Cannot instantiate a register with greater capacity than native types on emulating system.");
        }
    };

    /** Default constructor, primarily for protected internal use */
    QEngine()
        : useHostRam(false)
        , runningNorm(ONE_R1)
        , maxQPowerOcl(1U)
    {
        // Intentionally left blank
    }

    virtual ~QEngine()
    {
        // Virtual destructor for inheritance
    }

    using QInterface::Copy;

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        maxQPowerOcl = (bitCapIntOcl)maxQPower;
    }

    /** Get in-flight renormalization factor */
    virtual real1_f GetRunningNorm()
    {
        Finish();
        return (real1_f)runningNorm;
    }

    /** Switch to/from host/device state vector bufffer */
    virtual void SwitchHostPtr(bool useHostMem){};
    /** Reset host/device state vector bufffer usage to default */
    virtual void ResetHostPtr() { SwitchHostPtr(useHostRam); }
    /** Set GPU device ID */
    virtual void SetDevice(int64_t dID) {}
    /** Get GPU device ID */
    virtual int64_t GetDevice() { return -1; }

    /** Set all amplitudes to 0, and optionally temporarily deallocate state vector RAM */
    virtual void ZeroAmplitudes() = 0;
    /** Exactly copy the state vector of a different QEngine instance */
    virtual void CopyStateVec(QEnginePtr src) = 0;
    /** Returns "true" only if amplitudes are all totally 0 */
    virtual bool IsZeroAmplitude() = 0;
    /** Copy a "page" of amplitudes from this QEngine's internal state, into `pagePtr`. */
    virtual void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length) = 0;
    /** Copy a "page" of amplitudes from `pagePtr` into this QEngine's internal state. */
    virtual void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length) = 0;
    /** Copy a "page" of amplitudes from another QEngine, pointed to by `pageEnginePtr`, into this QEngine's internal
     * state. */
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length) = 0;
    /** Swap the high half of this engine with the low half of another. This is necessary for gates which cross
     * sub-engine  boundaries. */
    virtual void ShuffleBuffers(QEnginePtr engine) = 0;
    /** Clone this QEngine's settings, with a zeroed state vector */
    virtual QEnginePtr CloneEmpty() = 0;

    /** Add an operation to the (OpenCL) queue, to set the value of `doNormalize`, which controls whether to
     * automatically normalize the state. */
    virtual void QueueSetDoNormalize(bool doNorm) = 0;
    /** Add an operation to the (OpenCL) queue, to set the value of `runningNorm`, which is the normalization constant
     * for the next normalization operation. */
    virtual void QueueSetRunningNorm(real1_f runningNrm) = 0;

    virtual void ZMask(const bitCapInt& mask) { PhaseParity((real1_f)PI_R1, mask); }

    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt ForceM(const std::vector<bitLenInt>& bits, const std::vector<bool>& values, bool doApply = true);
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, const bitCapInt& result, bool doForce = true, bool doApply = true);

    virtual void ApplyM(const bitCapInt& qPower, bool result, const complex& nrm)
    {
        const bitCapInt powerTest = result ? qPower : ZERO_BCI;
        ApplyM(qPower, powerTest, nrm);
    }
    virtual void ApplyM(const bitCapInt& regMask, const bitCapInt& result, const complex& nrm) = 0;

    virtual void Mtrx(const complex* mtrx, bitLenInt qubit);
    virtual void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        EitherMtrx(controls, mtrx, target, false);
    }
    virtual void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        EitherMtrx(controls, mtrx, target, true);
    }
    virtual void UCMtrx(
        const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, const bitCapInt& controlPerm);
    virtual void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

#if ENABLE_ALU
    using QInterface::M;
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    virtual void X(bitLenInt q) { QInterface::X(q); }
    virtual void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
    {
        QInterface::INC(toAdd, start, length);
    }
    virtual void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
    {
        QInterface::DEC(toSub, start, length);
    }
    virtual void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::DECC(toSub, start, length, carryIndex);
    }
    virtual void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::INCS(toAdd, start, length, overflowIndex);
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
#endif

    using QInterface::Swap;
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISwap;
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::IISwap;
    virtual void IISwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::SqrtSwap;
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::FSim;
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual real1_f ProbAll(const bitCapInt& fullRegister)
    {
        if (doNormalize) {
            NormalizeState();
        }

        return clampProb((real1_f)norm(GetAmplitude(fullRegister)));
    }
    virtual real1_f CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target);
    virtual real1_f CProb(bitLenInt control, bitLenInt target) { return CtrlOrAntiProb(true, control, target); }
    virtual real1_f ACProb(bitLenInt control, bitLenInt target) { return CtrlOrAntiProb(false, control, target); }
    virtual real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation) = 0;
    virtual void ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray);
    virtual real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation) = 0;

    virtual real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength) = 0;

    virtual void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        bitCapIntOcl const* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG) = 0;
    virtual void ApplyControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx);
    virtual void ApplyAntiControlled2x2(const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx);

    using QInterface::Decompose;
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        QEnginePtr dest = CloneEmpty();
        dest->SetQubitCount(length);
        Decompose(start, dest);
        return dest;
    }

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);
    virtual void MultiShotMeasureMask(
        const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);
};
} // namespace Qrack
