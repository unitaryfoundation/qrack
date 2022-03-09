//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "mpsshard.hpp"
#include "qbdt_qinterface_node.hpp"
#include "qinterface.hpp"
#include "qparity.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

#define NODE_TO_QINTERFACE(leaf) (std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg)
#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

class QBdt;
typedef std::shared_ptr<QBdt> QBdtPtr;

#if ENABLE_ALU
class QBdt : public QAlu, public QParity, public QInterface {
#else
class QBdt : public QParity, public QInterface {
#endif
protected:
    std::vector<QInterfaceEngine> engines;
    int devID;
    QBdtNodeInterfacePtr root;
    bitLenInt attachedQubitCount;
    bitLenInt bdtQubitCount;
    bitCapInt bdtMaxQPower;
    bool isAttached;

    virtual void SetQubitCount(bitLenInt qb, bitLenInt aqb)
    {
        attachedQubitCount = aqb;
        SetQubitCount(qb);
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        bdtQubitCount = qubitCount - attachedQubitCount;
        bdtMaxQPower = pow2(bdtQubitCount);
    }

    QBdtQInterfaceNodePtr MakeQInterfaceNode(complex scale, bitLenInt qbCount, bitCapInt perm = 0U);

    void FallbackMtrx(const complex* mtrx, bitLenInt target);
    void FallbackMCMtrx(
        const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti);

    QInterfacePtr MakeTempStateVector()
    {
        QInterfacePtr copyPtr = NODE_TO_QINTERFACE(MakeQInterfaceNode(ONE_R1, qubitCount));
        Finish();
        GetQuantumState(copyPtr);

        // If the calling function fully deferences our return, it's automatically freed.
        return copyPtr;
    }
    void SetStateVector()
    {
        if (!bdtQubitCount) {
            return;
        }

        if (isAttached) {
            throw std::domain_error("QBdt::SetStateVector() not yet implemented, after Attach() call!");
        }

        QBdtQInterfaceNodePtr nRoot = MakeQInterfaceNode(ONE_R1, qubitCount);
        GetQuantumState(NODE_TO_QINTERFACE(nRoot));
        root = nRoot;
        SetQubitCount(qubitCount, qubitCount);
    }
    void ResetStateVector()
    {
        if (bdtQubitCount) {
            return;
        }

        QBdtQInterfaceNodePtr oRoot = std::dynamic_pointer_cast<QBdtQInterfaceNode>(root);
        SetQubitCount(qubitCount, 0U);
        SetQuantumState(NODE_TO_QINTERFACE(oRoot));
    }

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);
    template <typename Fn> void ExecuteAsStateVector(Fn operation)
    {
        SetStateVector();
        operation(NODE_TO_QINTERFACE(root));
    }

    template <typename Fn> bitCapInt BitCapIntAsStateVector(Fn operation)
    {
        SetStateVector();
        return operation(NODE_TO_QINTERFACE(root));
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest);

    void ApplyControlledSingle(
        const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti);

    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }

    static bitCapInt RemovePower(bitCapInt perm, bitCapInt power)
    {
        bitCapInt mask = power - ONE_BCI;
        return (perm & mask) | ((perm >> ONE_BCI) & ~mask);
    }

public:
    QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QBdt(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QBdt({ QINTERFACE_STABILIZER_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold,
              separation_thresh)
    {
    }

    virtual bool isBinaryDecisionTree() { return true; };

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        root->Normalize(bdtQubitCount);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QBdt>(toCompare));
    }
    virtual real1_f SumSqrDiff(QBdtPtr toCompare);

    virtual void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG);

    virtual QInterfacePtr Clone();

    virtual void GetQuantumState(complex* state);
    virtual void GetQuantumState(QInterfacePtr eng);
    virtual void SetQuantumState(const complex* state);
    virtual void SetQuantumState(QInterfacePtr eng);
    virtual void GetProbs(real1* outputProbs);

    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->SetAmplitude(perm, amp); });
    }

    using QInterface::Compose;
    virtual bitLenInt Compose(QBdtPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBdt>(toCopy), start);
    }
    virtual bitLenInt Attach(QInterfacePtr toCopy, bitLenInt start)
    {
        if (start == qubitCount) {
            return Attach(toCopy);
        }

        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, qubitCount);
        bitLenInt result = Attach(toCopy, qubitCount);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }
    virtual bitLenInt Attach(QInterfacePtr toCopy);
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QBdt>(dest));
    }
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, NULL); }

    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        DecomposeDispose(start, length, NULL);
    }

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbAll(bitCapInt fullRegister);

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt MAll();

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);

    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubitIndex1, qubitIndex2); });
    }

    virtual real1_f ProbParity(bitCapInt mask)
    {
        QInterfacePtr unit = (!bdtQubitCount) ? NODE_TO_QINTERFACE(root) : MakeTempStateVector();
        return QINTERFACE_TO_QPARITY(unit)->ProbParity(mask);
    }
    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QPARITY(eng)->CUniformParityRZ(controls, controlLen, mask, angle);
        });
    }
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        SetStateVector();
        return QINTERFACE_TO_QPARITY(NODE_TO_QINTERFACE(root))->ForceMParity(mask, result, doForce);
    }

#if ENABLE_ALU
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    virtual void X(bitLenInt q) { QInterface::X(q); }
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { QInterface::INC(toAdd, start, length); }
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::DECC(toSub, start, length, carryIndex);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::INCS(toAdd, start, length, overflowIndex);
    }
    virtual void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls, controlLen);
    }
    virtual void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->PhaseFlipIfLess(greaterPerm, start, length); });
        ResetStateVector();
    }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
        });
        ResetStateVector();
    }
    virtual void INCDECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
        });
    }
    virtual void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCDECSC(toAdd, start, length, carryIndex); });
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCBCD(toAdd, start, length); });
    }
    virtual void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCDECBCDC(toAdd, start, length, carryIndex); });
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->MUL(toMul, inOutStart, carryStart, length); });
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->DIV(toDiv, inOutStart, carryStart, length); });
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->MULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->IMULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->POWModNOut(base, modN, inStart, outStart, length); });
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
        });
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
        });
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedLDA(
                indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedADC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedSBC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->Hash(start, length, values); });
    }
#endif
};
} // namespace Qrack
