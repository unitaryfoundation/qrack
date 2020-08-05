//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <future>

#include "qfactory.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , deviceIDs(devList)
    , thresholdQubitsPerPage(qubitThreshold)
{
    if ((eng != QINTERFACE_CPU) && (eng != QINTERFACE_OPENCL)) {
        throw std::invalid_argument("QPager sub-engine type must be QINTERFACE_CPU or QINTERFACE_OPENCL.");
    }

#if ENABLE_OPENCL
    if ((thresholdQubitsPerPage == 0) && (eng == QINTERFACE_OPENCL)) {
        // Single bit gates act pairwise on amplitudes, so add 1 qubit to the log2 of the preferred concurrency.
        thresholdQubitsPerPage =
            log2(OCLEngine::Instance()->GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;
    }
#endif

    if (thresholdQubitsPerPage == 0) {
        // TODO: Tune for QEngineCPU
        thresholdQubitsPerPage = 18;
    }

    if (deviceIDs.size() == 0) {
        deviceIDs.push_back(-1);
    }

    SetQubitCount(qubitCount);

    if (baseQubitsPerPage > (sizeof(bitCapIntOcl) * bitsInByte)) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }

    bool isPermInPage;
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < basePageCount; i++) {
        isPermInPage = (initState >= pagePerm);
        pagePerm += basePageMaxQPower;
        isPermInPage &= (initState < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(
                baseQubitsPerPage, initState - (pagePerm - basePageMaxQPower), deviceIDs[i % deviceIDs.size()]));
        } else {
            qPages.push_back(MakeEngine(baseQubitsPerPage, 0, deviceIDs[i % deviceIDs.size()]));
            qPages.back()->SetAmplitude(0, ZERO_CMPLX);
        }
    }
}

QEnginePtr QPager::MakeEngine(bitLenInt length, bitCapInt perm, int deviceId)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(
        engine, length, perm, rand_generator, phaseFactor, false, false, useHostRam, deviceId, useRDRAND, isSparse));
}

void QPager::CombineEngines(bitLenInt bit)
{
    bit++;

    if ((qPages.size() == 1U) || (bit < qubitsPerPage())) {
        return;
    }

    if (bit > qubitCount) {
        bit = qubitCount;
    }

    bitCapInt groupCount = pow2(qubitCount - bit);
    bitCapInt groupSize = qPages.size() / groupCount;
    bitCapInt pagePower = pageMaxQPower();
    std::vector<QEnginePtr> nQPages;

    bitCapInt i, j;

    for (i = 0; i < groupCount; i++) {
        nQPages.push_back(MakeEngine(bit, 0, deviceIDs[i % deviceIDs.size()]));
        for (j = 0; j < groupSize; j++) {
            nQPages.back()->SetAmplitudePage(qPages[j + (i * groupSize)], 0, j * pagePower, pagePower);
        }
        nQPages.back()->UpdateRunningNorm();
    }

    qPages = nQPages;
}

void QPager::SeparateEngines()
{
    if (qPages.size() <= basePageCount) {
        return;
    }

    bitCapInt i, j;
    bitCapInt pagesPer = basePageCount / qPages.size();

    std::vector<QEnginePtr> nQPages;
    for (i = 0; i < qPages.size(); i++) {
        for (j = 0; j < pagesPer; j++) {
            nQPages.push_back(MakeEngine(baseQubitsPerPage, 0, deviceIDs[(j + (i * pagesPer)) % deviceIDs.size()]));
            nQPages.back()->SetAmplitudePage(qPages[i], j * basePageMaxQPower, 0, basePageMaxQPower);
            nQPages.back()->UpdateRunningNorm();
        }
    }

    qPages = nQPages;
}

template <typename Qubit1Fn> void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn)
{
    if (target >= qubitsPerPage()) {
        CombineAndOp([fn, target](QEnginePtr engine) { fn(engine, target); }, { target });
        return;
    }

    std::vector<std::future<void>> futures(qPages.size());
    bitCapInt i;
    for (i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        futures[i] = std::async(std::launch::async, [engine, fn, target]() { fn(engine, target); });
    }
    for (i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename Qubit1Fn>
void QPager::MetaControlled(
    bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn, const complex* mtrx)
{
    bitLenInt qpp = qubitsPerPage();
    target -= qpp;
    bitLenInt sqi = qpp - 1U;

    std::vector<bitCapInt> sortedMasks(1U + controls.size());
    bitCapInt targetMask = pow2(target);
    sortedMasks[controls.size()] = targetMask;

    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controls.size(); i++) {
        sortedMasks[i] = pow2(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bool isSpecial, isInvert;
    complex top, bottom;
    if ((mtrx[1] == ZERO_CMPLX) && (mtrx[2] == ZERO_CMPLX)) {
        isSpecial = true;
        isInvert = false;
        top = mtrx[0];
        bottom = mtrx[3];
    } else if ((mtrx[0] == ZERO_CMPLX) && (mtrx[3] == ZERO_CMPLX)) {
        isSpecial = true;
        isInvert = true;
        top = mtrx[1];
        bottom = mtrx[2];
    } else {
        isSpecial = false;
        isInvert = false;
        top = ZERO_CMPLX;
        bottom = ZERO_CMPLX;
    }

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    bitCapInt maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapInt i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async,
            [this, i, fn, sqi, &controlMask, &targetMask, &sortedMasks, &isSpecial, &isInvert, &top, &bottom]() {
                bitCapInt j, k, jLo, jHi;
                jHi = i;
                j = 0;
                for (k = 0; k < (sortedMasks.size()); k++) {
                    jLo = jHi & sortedMasks[k];
                    jHi = (jHi ^ jLo) << ONE_BCI;
                    j |= jLo;
                }
                j |= jHi | controlMask;

                if (isSpecial && isInvert) {
                    std::swap(qPages[j], qPages[j + targetMask]);
                }

                QEnginePtr engine1 = qPages[j];
                QEnginePtr engine2 = qPages[j + targetMask];

                std::future<void> future1, future2;
                if (!isSpecial) {
                    engine1->ShuffleBuffers(engine2);

                    future1 = std::async(std::launch::async, [engine1, fn, sqi]() { fn(engine1, sqi); });
                    future2 = std::async(std::launch::async, [engine2, fn, sqi]() { fn(engine2, sqi); });
                    future1.get();
                    future2.get();

                    engine1->ShuffleBuffers(engine2);
                } else {
                    if (top != ONE_CMPLX) {
                        future1 = std::async(
                            std::launch::async, [engine1, &top]() { engine1->ApplySinglePhase(top, top, 0); });
                    }
                    if (bottom != ONE_CMPLX) {
                        future2 = std::async(
                            std::launch::async, [engine2, &bottom]() { engine2->ApplySinglePhase(bottom, bottom, 0); });
                    }

                    if (top != ONE_CMPLX) {
                        future1.get();
                    }
                    if (bottom != ONE_CMPLX) {
                        future2.get();
                    }
                }
            });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to fit in
// sub-engines).
template <typename Qubit1Fn>
void QPager::SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn)
{
    bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> sortedMasks(controls.size());

    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controls.size(); i++) {
        sortedMasks[i] = pow2(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapInt i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, fn, &controlMask, &target, &sortedMasks]() {
            bitCapInt j, k, jLo, jHi;
            jHi = i;
            j = 0;
            for (k = 0; k < (sortedMasks.size()); k++) {
                jLo = jHi & sortedMasks[k];
                jHi = (jHi ^ jLo) << 1U;
                j |= jLo;
            }
            j |= jHi | controlMask;

            fn(qPages[j], target);
        });
    }
    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

template <typename F> void QPager::CombineAndOp(F fn, std::vector<bitLenInt> bits)
{
    if (qPages.size() == 1U) {
        fn(qPages[0]);
        return;
    }

    bitLenInt highestBit = 0;
    for (bitLenInt i = 0; i < bits.size(); i++) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
    }

    if (highestBit >= qubitsPerPage()) {
        CombineEngines(highestBit);
    }

    std::vector<std::future<void>> futures(qPages.size());
    bitCapInt i;
    for (i = 0; i < qPages.size(); i++) {
        futures[i] = std::async(std::launch::async, [this, fn, i]() { fn(qPages[i]); });
    }
    for (i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }

    if (highestBit >= baseQubitsPerPage) {
        SeparateEngines();
    }
}

template <typename F>
void QPager::CombineAndOpControlled(
    F fn, std::vector<bitLenInt> bits, const bitLenInt* controls, const bitLenInt controlLen)
{
    for (bitLenInt i = 0; i < controlLen; i++) {
        bits.push_back(controls[i]);
    }

    CombineAndOp(fn, bits);
}

bitLenInt QPager::Compose(QPagerPtr toCopy)
{
    CombineEngines();
    toCopy->CombineEngines();
    bitLenInt toRet = qPages[0]->Compose(toCopy->qPages[0]);
    SetQubitCount(qPages[0]->GetQubitCount());
    toCopy->SeparateEngines();
    SeparateEngines();
    return toRet;
}

bitLenInt QPager::Compose(QPagerPtr toCopy, bitLenInt start)
{
    CombineEngines();
    toCopy->CombineEngines();
    bitLenInt toRet = qPages[0]->Compose(toCopy->qPages[0], start);
    SetQubitCount(qPages[0]->GetQubitCount());
    toCopy->SeparateEngines();
    SeparateEngines();
    return toRet;
}

void QPager::Decompose(bitLenInt start, bitLenInt length, QPagerPtr dest)
{
    CombineEngines();
    dest->CombineEngines();
    qPages[0]->Decompose(start, length, dest->qPages[0]);
    SetQubitCount(qPages[0]->GetQubitCount());
    dest->SeparateEngines();
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    CombineEngines();
    qPages[0]->Dispose(start, length);
    SetQubitCount(qPages[0]->GetQubitCount());
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    CombineEngines();
    qPages[0]->Dispose(start, length, disposedPerm);
    SetQubitCount(qPages[0]->GetQubitCount());
    SeparateEngines();
}

void QPager::SetQuantumState(const complex* inputState)
{
    bitCapInt pagePerm = 0;
    bitCapInt pagePower = pageMaxQPower();
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        qPages[i]->SetQuantumState(inputState + pagePerm);
        pagePerm += pagePower;
    }
}

void QPager::GetQuantumState(complex* outputState)
{
    bitCapInt pagePerm = 0;
    bitCapInt pagePower = pageMaxQPower();
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        qPages[i]->GetQuantumState(outputState + pagePerm);
        pagePerm += pagePower;
    }
}

void QPager::GetProbs(real1* outputProbs)
{
    bitCapInt pagePerm = 0;
    bitCapInt pagePower = pageMaxQPower();
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        qPages[i]->GetProbs(outputProbs + pagePerm);
        pagePerm += pagePower;
    }
}

void QPager::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool isPermInPage;
    bitCapInt pagePerm = 0;
    bitCapInt pagePower = pageMaxQPower();
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        isPermInPage = (perm >= pagePerm);
        pagePerm += pagePower;
        isPermInPage &= (perm < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(perm - (pagePerm - pagePower));
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->ApplySingleBit(mtrx, lTarget); });
}

void QPager::ApplySingleEither(const bool& isInvert, complex top, complex bottom, bitLenInt target)
{
    if (target < qubitsPerPage()) {
        if (isInvert) {
            SingleBitGate(target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) {
                engine->ApplySingleInvert(top, bottom, lTarget);
            });
        } else {
            SingleBitGate(target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) {
                engine->ApplySinglePhase(top, bottom, lTarget);
            });
        }
        return;
    }

    bitLenInt qpp = qubitsPerPage();
    target -= qpp;
    bitCapInt targetMask = pow2(target);
    bitCapInt qMask = targetMask - 1U;

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    bitCapInt maxLCV = qPages.size() >> 1U;
    std::vector<std::future<void>> futures(maxLCV);
    bitCapInt i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, &isInvert, &top, &bottom, &targetMask, &qMask]() {
            bitCapInt j = i & qMask;
            j |= (i ^ j) << ONE_BCI;

            if (isInvert) {
                std::swap(qPages[j], qPages[j + targetMask]);
            }

            QEnginePtr engine1 = qPages[j];
            QEnginePtr engine2 = qPages[j + targetMask];

            std::future<void> future1, future2;
            if (top != ONE_CMPLX) {
                future1 = std::async(std::launch::async, [engine1, top]() { engine1->ApplySinglePhase(top, top, 0); });
            }
            if (bottom != ONE_CMPLX) {
                future2 = std::async(
                    std::launch::async, [engine2, bottom]() { engine2->ApplySinglePhase(bottom, bottom, 0); });
            }

            if (top != ONE_CMPLX) {
                future1.get();
            }
            if (bottom != ONE_CMPLX) {
                future2.get();
            }
        });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::ApplyEitherControlledSingleBit(const bool& anti, const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex* mtrx)
{
    if (controlLen == 0) {
        ApplySingleBit(mtrx, target);
        return;
    }

    bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < qpp) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    auto sg = [anti, mtrx, &intraControls](QEnginePtr engine, bitLenInt lTarget) {
        if (intraControls.size()) {
            if (anti) {
                engine->ApplyAntiControlledSingleBit(&(intraControls[0]), intraControls.size(), lTarget, mtrx);
            } else {
                engine->ApplyControlledSingleBit(&(intraControls[0]), intraControls.size(), lTarget, mtrx);
            }
        } else {
            engine->ApplySingleBit(mtrx, lTarget);
        }
    };

    if (metaControls.size() == 0) {
        SingleBitGate(target, sg);
    } else if (target < qpp) {
        SemiMetaControlled(anti, metaControls, target, sg);
    } else {
        MetaControlled(anti, metaControls, target, sg, mtrx);
    }
}

void QPager::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) {
            engine->UniformlyControlledSingleBit(
                controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
        },
        { qubitIndex }, controls, controlLen);
}

void QPager::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    bool isQubit1Meta = qubit1 >= qubitsPerPage();
    bool isQubit2Meta = qubit2 >= qubitsPerPage();
    if (isQubit1Meta && isQubit2Meta) {
        MetaControlledSwap(false, controls, controlLen, qubit1, qubit2, false);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SemiMetaControlledSwap(false, controls, controlLen, qubit1, qubit2, false);
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    bool isQubit1Meta = qubit1 >= qubitsPerPage();
    bool isQubit2Meta = qubit2 >= qubitsPerPage();
    if (isQubit1Meta && isQubit2Meta) {
        MetaControlledSwap(true, controls, controlLen, qubit1, qubit2, false);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SemiMetaControlledSwap(true, controls, controlLen, qubit1, qubit2, false);
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}

bool QPager::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    CombineEngines();
    bool toRet = qPages[0]->ForceM(qubit, result, doForce, doApply);
    SeparateEngines();
    return toRet;
}

void QPager::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp(
        [&](QEnginePtr engine) { engine->INC(toAdd, start, length); }, { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::CINC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CINC(toAdd, start, length, controls, controlLen); },
        { static_cast<bitLenInt>(start + length - 1U) }, controls, controlLen);
}
void QPager::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCS(toAdd, start, length, overflowIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex });
}
void QPager::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCSC(toAdd, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCSC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCD(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCDC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECSC(toSub, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECSC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECBCDC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->MUL(toMul, inOutStart, carryStart, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) });
}
void QPager::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DIV(toDiv, inOutStart, carryStart, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) });
}
void QPager::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->MULModNOut(toMul, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->IMULModNOut(toMul, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->POWModNOut(base, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}

void QPager::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->ZeroPhaseFlip(start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); },
        { static_cast<bitLenInt>(start + length - 1U), flagIndex });
}
void QPager::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->PhaseFlipIfLess(greaterPerm, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::PhaseFlip()
{
    for (bitLenInt i = 0; i < qPages.size(); i++) {
        qPages[i]->PhaseFlip();
    }
}

bitCapInt QPager::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    unsigned char* values, bool resetValue)
{
    CombineAndOp(
        [&](QEnginePtr engine) { engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, true); },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U),
            static_cast<bitLenInt>(valueStart + valueLength - 1U) });

    return 0;
}

bitCapInt QPager::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOp(
        [&](QEnginePtr engine) {
            engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U), static_cast<bitLenInt>(valueStart + valueLength - 1U),
            carryIndex });

    return 0;
}
bitCapInt QPager::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOp(
        [&](QEnginePtr engine) {
            engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U), static_cast<bitLenInt>(valueStart + valueLength - 1U),
            carryIndex });

    return 0;
}
void QPager::Hash(bitLenInt start, bitLenInt length, unsigned char* values)
{
    CombineAndOp([&](QEnginePtr engine) { engine->Hash(start, length, values); },
        { static_cast<bitLenInt>(start + length - 1U) });
}

void QPager::MetaControlledSwap(
    bool anti, const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    bitLenInt qpp = qubitsPerPage();
    qubit1 -= qpp;
    qubit2 -= qpp;
    bitLenInt sqi = qpp - 1U;

    std::vector<bitCapInt> sortedMasks(2U);
    bitCapInt qubit1Mask = pow2(qubit1);
    sortedMasks[0] = qubit1Mask;
    bitCapInt qubit2Mask = pow2(qubit2);
    sortedMasks[1] = qubit2Mask;

    std::vector<bitLenInt> subcontrols;
    bitCapInt controlMask = 0;
    bool isSqiSubcontrol = false;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < sqi) {
            sortedMasks.push_back(pow2(controls[i] - qpp));
            if (!anti) {
                controlMask |= sortedMasks.back();
            }
            sortedMasks.back()--;
        } else if (controls[i] == sqi) {
            isSqiSubcontrol = true;
        } else {
            subcontrols.push_back(controls[i]);
        }
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapInt i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async,
            [this, i, &qubit1Mask, &qubit2Mask, &sortedMasks, &subcontrols, &isIPhaseFac, &sqi, &controlMask,
                &isSqiSubcontrol, &anti]() {
                bitCapInt j, k, jLo, jHi;
                jHi = i;
                j = 0;
                for (k = 0; k < (sortedMasks.size()); k++) {
                    jLo = jHi & sortedMasks[k];
                    jHi = (jHi ^ jLo) << ONE_BCI;
                    j |= jLo;
                }
                j |= jHi;

                QEnginePtr engine1 = qPages[j + qubit1Mask];
                QEnginePtr engine2 = qPages[j + qubit2Mask];

                if (subcontrols.size()) {
                    engine1->ShuffleBuffers(engine2);

                    std::future<void> future1, future2;

                    if (isIPhaseFac) {
                        if (!isSqiSubcontrol || anti) {
                            future1 = std::async(std::launch::async, [engine1, &sqi, &subcontrols]() {
                                engine1->ApplyControlledSingleInvert(
                                    &(subcontrols[0]), subcontrols.size(), sqi, I_CMPLX, I_CMPLX);
                            });
                        }
                        if (!isSqiSubcontrol || !anti) {
                            future2 = std::async(std::launch::async, [engine2, &sqi, &subcontrols]() {
                                engine2->ApplyControlledSingleInvert(
                                    &(subcontrols[0]), subcontrols.size(), sqi, I_CMPLX, I_CMPLX);
                            });
                        }
                    } else {
                        if (!isSqiSubcontrol || anti) {
                            future1 = std::async(std::launch::async, [engine1, &sqi, &subcontrols]() {
                                engine1->ApplyControlledSingleInvert(
                                    &(subcontrols[0]), subcontrols.size(), sqi, ONE_CMPLX, ONE_CMPLX);
                            });
                        }
                        if (!isSqiSubcontrol || !anti) {
                            future2 = std::async(std::launch::async, [engine2, &sqi, &subcontrols]() {
                                engine2->ApplyControlledSingleInvert(
                                    &(subcontrols[0]), subcontrols.size(), sqi, ONE_CMPLX, ONE_CMPLX);
                            });
                        }
                    }

                    if (!isSqiSubcontrol || anti) {
                        future1.get();
                    }
                    if (!isSqiSubcontrol || !anti) {
                        future2.get();
                    }

                    engine1->ShuffleBuffers(engine2);
                } else {
                    std::swap(qPages[j + qubit1Mask], qPages[j + qubit2Mask]);

                    if (isIPhaseFac) {
                        std::future<void> future1, future2;

                        future1 = std::async(
                            std::launch::async, [engine1]() { engine1->ApplySinglePhase(I_CMPLX, I_CMPLX, 0); });
                        future2 = std::async(
                            std::launch::async, [engine2]() { engine2->ApplySinglePhase(I_CMPLX, I_CMPLX, 0); });

                        future1.get();
                        future2.get();
                    }
                }
            });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::SemiMetaControlledSwap(
    bool anti, const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    if (qubit1 > qubit2) {
        std::swap(qubit1, qubit2);
    }

    bitLenInt qpp = qubitsPerPage();
    qubit1 -= qpp;
    bitLenInt sqi = qpp - 1U;

    std::vector<bitCapInt> sortedMasks(1U);
    bitCapInt qubit2Mask = pow2(qubit2);
    sortedMasks[0] = qubit2Mask;

    std::vector<bitLenInt> subcontrols;
    bitCapInt controlMask = 0;
    bool isSqiSubcontrol = false;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < sqi) {
            sortedMasks.push_back(pow2(controls[i] - qpp));
            if (!anti) {
                controlMask |= sortedMasks.back();
            }
            sortedMasks.back()--;
        } else if (controls[i] == sqi) {
            isSqiSubcontrol = true;
        } else {
            subcontrols.push_back(controls[i]);
        }
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapInt i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async,
            [this, i, &qubit1, &qubit2Mask, &sortedMasks, &subcontrols, &isIPhaseFac, &sqi, &controlMask,
                &isSqiSubcontrol, &anti]() {
                bitCapInt j, k, jLo, jHi;
                jHi = i;
                j = 0;
                for (k = 0; k < (sortedMasks.size()); k++) {
                    jLo = jHi & sortedMasks[k];
                    jHi = (jHi ^ jLo) << ONE_BCI;
                    j |= jLo;
                }
                j |= jHi | controlMask;

                QEnginePtr engine1 = qPages[j];
                QEnginePtr engine2 = qPages[j + qubit2Mask];

                engine1->ShuffleBuffers(engine2);

                std::future<void> future1, future2;
                if (isIPhaseFac) {
                    if (!isSqiSubcontrol || anti) {
                        future1 = std::async(std::launch::async, [engine1, &qubit1, &sqi, &subcontrols]() {
                            engine1->CSwap(&(subcontrols[0]), subcontrols.size(), qubit1, sqi);
                            engine1->ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
                        });
                    }
                    if (!isSqiSubcontrol || !anti) {
                        future2 = std::async(std::launch::async, [engine2, &qubit1, &sqi, &subcontrols]() {
                            engine2->CSwap(&(subcontrols[0]), subcontrols.size(), qubit1, sqi);
                            engine2->ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
                        });
                    }
                } else {
                    if (!isSqiSubcontrol || anti) {
                        future1 = std::async(std::launch::async, [engine1, &qubit1, &sqi, &subcontrols]() {
                            engine1->CSwap(&(subcontrols[0]), subcontrols.size(), qubit1, sqi);
                        });
                    }
                    if (!isSqiSubcontrol || !anti) {
                        future2 = std::async(std::launch::async, [engine2, &qubit1, &sqi, &subcontrols]() {
                            engine2->CSwap(&(subcontrols[0]), subcontrols.size(), qubit1, sqi);
                        });
                    }
                }

                if (!isSqiSubcontrol || anti) {
                    future1.get();
                }
                if (!isSqiSubcontrol || !anti) {
                    future2.get();
                }

                engine1->ShuffleBuffers(engine2);
            });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    bool isQubit1Meta = qubit1 >= qubitsPerPage();
    bool isQubit2Meta = qubit2 >= qubitsPerPage();
    if (isQubit1Meta && isQubit2Meta) {
        MetaControlledSwap(false, NULL, 0, qubit1, qubit2, false);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SemiMetaControlledSwap(false, NULL, 0, qubit1, qubit2, false);
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->Swap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    bool isQubit1Meta = qubit1 >= qubitsPerPage();
    bool isQubit2Meta = qubit2 >= qubitsPerPage();
    if (isQubit1Meta && isQubit2Meta) {
        MetaControlledSwap(false, NULL, 0, qubit1, qubit2, true);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SemiMetaControlledSwap(false, NULL, 0, qubit1, qubit2, true);
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->ISwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->SqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->ISqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::FSim(real1 theta, real1 phi, bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->FSim(theta, phi, qubit1, qubit2); }, { qubit1, qubit2 });
}

real1 QPager::Prob(bitLenInt qubitIndex)
{
    if (qPages.size() == 1U) {
        return qPages[0]->Prob(qubitIndex);
    }

    real1 oneChance = ZERO_R1;
    bitCapInt i;

    if (qubitIndex >= qubitsPerPage()) {
        CombineEngines(qubitIndex);
    }

    std::vector<std::future<real1>> futures(qPages.size());
    for (i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
    }
    for (i = 0; i < qPages.size(); i++) {
        oneChance += futures[i].get();
    }

    if (qubitIndex >= baseQubitsPerPage) {
        SeparateEngines();
    }

    return oneChance;
}
real1 QPager::ProbAll(bitCapInt fullRegister)
{
    bitCapInt subIndex = fullRegister / pageMaxQPower();
    fullRegister -= subIndex * pageMaxQPower();
    return qPages[subIndex]->ProbAll(fullRegister);
}
real1 QPager::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    CombineEngines();
    real1 maskChance = qPages[0]->ProbMask(mask, permutation);
    SeparateEngines();
    return maskChance;
}

bool QPager::ApproxCompare(QInterfacePtr toCompare)
{
    QPagerPtr toComparePager = std::dynamic_pointer_cast<QPager>(toCompare);
    CombineEngines();
    toComparePager->CombineEngines();
    bool toRet = qPages[0]->ApproxCompare(toComparePager->qPages[0]);
    toComparePager->SeparateEngines();
    SeparateEngines();
    return toRet;
}
void QPager::UpdateRunningNorm(real1 norm_thresh)
{
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

QInterfacePtr QPager::Clone()
{
    bitLenInt qpp = qubitsPerPage();

    QPagerPtr clone = std::dynamic_pointer_cast<QPager>(
        CreateQuantumInterface(QINTERFACE_QPAGER, engine, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse));

    clone->CombineEngines(qpp - 1U);

    bitCapInt pagePower = pageMaxQPower();
    for (bitCapInt i = 0; i < qPages.size(); i++) {
        clone->qPages[i]->SetAmplitudePage(qPages[i], 0, 0, pagePower);
        clone->qPages[i]->UpdateRunningNorm();
    }

    clone->SeparateEngines();

    return clone;
}

} // namespace Qrack
