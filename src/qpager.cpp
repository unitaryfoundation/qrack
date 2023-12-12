//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_PTHREAD
#include <future>
#endif
#if ENABLE_ENV_VARS
#include <regex>
#include <string>
#endif

#if ENABLE_CUDA && !ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#else
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#endif

namespace Qrack {

QPager::QPager(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool ignored, bool ignored2, bool useHostMem, int64_t deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList, bitLenInt qubitThreshold,
    real1_f sep_thresh)
    : QEngine(qBitCount, rgp, false, false, useHostMem, useHardwareRNG, norm_thresh)
    , isSparse(useSparseStateVec)
    , useTGadget(true)
    , maxPageSetting(-1)
    , maxPageQubits(-1)
    , thresholdQubitsPerPage(qubitThreshold)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , deviceIDs(devList)
    , engines(eng)
{
    Init();

    if (!qubitCount) {
        return;
    }

    bi_and_ip(&initState, maxQPower + ONE_BCI);
    const bitCapIntOcl initStateOcl = (bitCapIntOcl)initState;
    bitCapIntOcl pagePerm = 0U;
    for (bitCapIntOcl i = 0U; i < basePageCount; ++i) {
        bool isPermInPage = bi_compare(initState, pagePerm) >= 0;
        pagePerm += basePageMaxQPower;
        isPermInPage &= (initStateOcl < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(baseQubitsPerPage, i));
            qPages.back()->SetPermutation(initStateOcl - (pagePerm - basePageMaxQPower));
        } else {
            qPages.push_back(MakeEngine(baseQubitsPerPage, i));
        }
    }
}

QPager::QPager(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool ignored, bool ignored2, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, false, false, useHostMem, useHardwareRNG, norm_thresh)
    , isSparse(useSparseStateVec)
    , maxPageSetting(-1)
    , maxPageQubits(-1)
    , thresholdQubitsPerPage(qubitThreshold)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , deviceIDs(devList)
    , engines(eng)
{
    Init();
    LockEngine(enginePtr);
    SeparateEngines();
}

void QPager::Init()
{
    if (!engines.size()) {
#if ENABLE_OPENCL || ENABLE_CUDA
        engines.push_back(QRACK_GPU_SINGLETON.GetDeviceCount() ? QRACK_GPU_ENGINE : QINTERFACE_CPU);
#else
        engines.push_back(QINTERFACE_CPU);
#endif
    }

    if ((engines[0U] == QINTERFACE_HYBRID) || (engines[0] == QRACK_GPU_ENGINE)) {
#if ENABLE_OPENCL || ENABLE_CUDA
        if (!QRACK_GPU_SINGLETON.GetDeviceCount()) {
            engines[0U] = QINTERFACE_CPU;
        }
#else
        engines[0U] = QINTERFACE_CPU;
#endif
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_PAGE_QB")) {
        maxPageSetting = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGE_QB")));
    }
#endif
    bitLenInt engineLevel = 0U;
    rootEngine = engines[0U];
    while ((engines.size() < engineLevel) && (rootEngine != QINTERFACE_CPU) && (rootEngine != QRACK_GPU_ENGINE) &&
        (rootEngine != QINTERFACE_HYBRID)) {
        ++engineLevel;
        rootEngine = engines[engineLevel];
    }

#if ENABLE_OPENCL || ENABLE_CUDA
    if (rootEngine != QINTERFACE_CPU) {
        maxPageQubits = log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex));
        maxPageQubits = (maxPageSetting < maxPageQubits) ? maxPageSetting : 1U;
    }

    if ((rootEngine != QINTERFACE_CPU) && (rootEngine != QRACK_GPU_ENGINE)) {
        rootEngine = QINTERFACE_HYBRID;
    }

    if (!thresholdQubitsPerPage && ((rootEngine == QRACK_GPU_ENGINE) || (rootEngine == QINTERFACE_HYBRID))) {
        useGpuThreshold = true;

        // Limit at the power of 2 less-than-or-equal-to a full max memory allocation segment, or choose with
        // environment variable.
        thresholdQubitsPerPage = maxPageQubits;
    }
#endif

    if (!thresholdQubitsPerPage) {
        useGpuThreshold = false;

#if ENABLE_ENV_VARS
        const bitLenInt pStridePow =
            (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
        const bitLenInt pStridePow = PSTRIDEPOW;
#endif

#if ENABLE_PTHREAD
        const unsigned numCores = GetConcurrencyLevel();
        thresholdQubitsPerPage = pStridePow + ((numCores == 1U) ? 1U : (log2Ocl(numCores - 1U) + 1U));
#else
        thresholdQubitsPerPage = pStridePow + 1U;
#endif
    }

    SetQubitCount(qubitCount);

    maxQubits = sizeof(bitCapIntOcl) * bitsInByte;
#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_PAGING_QB")) {
        maxQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    }
#endif
    if (qubitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_QPAGER_DEVICES")) {
        std::string devListStr = std::string(getenv("QRACK_QPAGER_DEVICES"));
        deviceIDs.clear();
        if (devListStr.compare("")) {
            std::stringstream devListStr_stream(devListStr);
            // See
            // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
            std::regex re("[.]");
            while (devListStr_stream.good()) {
                std::string term;
                getline(devListStr_stream, term, ',');
                // the '-1' is what makes the regex split (-1 := what was not matched)
                std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
                std::vector<std::string> tokens{ first, last };
                if (tokens.size() == 1U) {
                    deviceIDs.push_back(stoi(term));
                    if (deviceIDs.back() == -2) {
                        deviceIDs.back() = (int)devID;
                    }
                    if (deviceIDs.back() == -1) {
#if ENABLE_OPENCL || ENABLE_CUDA
                        deviceIDs.back() = (int)QRACK_GPU_SINGLETON.GetDefaultDeviceID();
#else
                        deviceIDs.back() = 0;
#endif
                    }
                    continue;
                }
                const unsigned maxI = stoi(tokens[0U]);
                std::vector<int> ids(tokens.size() - 1U);
                for (unsigned i = 1U; i < tokens.size(); ++i) {
                    ids[i - 1U] = stoi(tokens[i]);
                    if (ids[i - 1U] == -2) {
                        ids[i - 1U] = (int)devID;
                    }
                    if (ids[i - 1U] == -1) {
#if ENABLE_OPENCL || ENABLE_CUDA
                        ids[i - 1U] = (int)QRACK_GPU_SINGLETON.GetDefaultDeviceID();
#else
                        ids[i - 1U] = 0;
#endif
                    }
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < ids.size(); ++j) {
                        deviceIDs.push_back(ids[j]);
                    }
                }
            }
        }
    }
    if (getenv("QRACK_QPAGER_DEVICES_HOST_POINTER")) {
        std::string devListStr = std::string(getenv("QRACK_QPAGER_DEVICES_HOST_POINTER"));
        if (devListStr.compare("")) {
            std::stringstream devListStr_stream(devListStr);
            // See
            // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
            std::regex re("[.]");
            while (devListStr_stream.good()) {
                std::string term;
                getline(devListStr_stream, term, ',');
                // the '-1' is what makes the regex split (-1 := what was not matched)
                std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
                std::vector<std::string> tokens{ first, last };
                if (tokens.size() == 1U) {
                    devicesHostPointer.push_back((bool)stoi(term));
                    continue;
                }
                const unsigned maxI = stoi(tokens[0U]);
                std::vector<bool> hps(tokens.size() - 1U);
                for (unsigned i = 1U; i < tokens.size(); ++i) {
                    hps[i - 1U] = (bool)stoi(tokens[i]);
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < hps.size(); ++j) {
                        devicesHostPointer.push_back(hps[j]);
                    }
                }
            }
        }
    } else {
        devicesHostPointer.push_back(useHostRam);
    }
#endif

    if (!deviceIDs.size()) {
        deviceIDs.push_back(devID);
    }
}

QEnginePtr QPager::MakeEngine(bitLenInt length, bitCapIntOcl pageId)
{
    QEnginePtr toRet = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(engines, 0U, ZERO_BCI, rand_generator, phaseFactor, false, false,
            GetPageHostPointer(pageId), GetPageDevice(pageId), useRDRAND, isSparse, (real1_f)amplitudeFloor));
    toRet->SetQubitCount(length);
    toRet->SetConcurrency(GetConcurrencyLevel());
    toRet->SetTInjection(useTGadget);

    return toRet;
}

void QPager::GetSetAmplitudePage(complex* pagePtr, const complex* cPagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    const bitCapIntOcl pageLength = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl perm = 0U;
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        if ((perm + length) < offset) {
            continue;
        }
        if (perm >= (offset + length)) {
            break;
        }
        const bitCapIntOcl partOffset = (perm < offset) ? (offset - perm) : 0U;
        const bitCapIntOcl partLength = (length < pageLength) ? length : pageLength;
        if (cPagePtr) {
            qPages[i]->SetAmplitudePage(cPagePtr, partOffset, partLength);
        } else {
            qPages[i]->GetAmplitudePage(pagePtr, partOffset, partLength);
        }
        perm += pageLength;
    }
}

void QPager::CombineEngines(bitLenInt bit)
{
    if (bit > qubitCount) {
        bit = qubitCount;
    }

    if (bit <= qubitsPerPage()) {
        return;
    }

    const bitCapIntOcl groupCount = pow2Ocl(qubitCount - bit);
    const bitCapIntOcl groupSize = (bitCapIntOcl)(qPages.size() / groupCount);
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    std::vector<QEnginePtr> nQPages;

    for (bitCapIntOcl i = 0U; i < groupCount; ++i) {
        QEnginePtr engine = MakeEngine(bit, i);
        nQPages.push_back(engine);
        for (bitCapIntOcl j = 0U; j < groupSize; ++j) {
            const bitCapIntOcl page = j + (i * groupSize);
            engine->SetAmplitudePage(qPages[page], 0U, j * pagePower, pagePower);
            qPages[page] = NULL;
        }
    }

    qPages = nQPages;
}

void QPager::SeparateEngines(bitLenInt thresholdBits, bool noBaseFloor)
{
    if (!noBaseFloor && (thresholdBits < baseQubitsPerPage)) {
        thresholdBits = baseQubitsPerPage;
    }

    if (thresholdBits >= qubitsPerPage()) {
        return;
    }

    const bitCapIntOcl pagesPer = pow2Ocl(qubitCount - thresholdBits) / qPages.size();
    const bitCapIntOcl pageMaxQPower = pow2Ocl(thresholdBits);
    std::vector<QEnginePtr> nQPages;

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        for (bitCapIntOcl j = 0U; j < pagesPer; ++j) {
            nQPages.push_back(MakeEngine(thresholdBits, j + (i * pagesPer)));
            nQPages.back()->SetAmplitudePage(qPages[i], j * pageMaxQPower, 0U, pageMaxQPower);
        }
        qPages[i] = NULL;
    }

    qPages = nQPages;
}

template <typename Qubit1Fn> void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn, bool isSqiCtrl, bool isAnti)
{
    bitLenInt qpp = qubitsPerPage();

    if (doNormalize) {
        real1_f runningNorm = ZERO_R1;
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->Finish();
            runningNorm += qPages[i]->GetRunningNorm();
        }
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->QueueSetRunningNorm(runningNorm);
            qPages[i]->QueueSetDoNormalize(true);
        }
    }

    if (target < qpp) {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            QEnginePtr engine = qPages[i];
            fn(engine, target);
            if (doNormalize) {
                engine->QueueSetDoNormalize(false);
            }
        }

        return;
    }

    const bitLenInt sqi = qpp - 1U;
    target -= qpp;
    const bitCapIntOcl targetPow = pow2Ocl(target);
    const bitCapIntOcl targetMask = targetPow - 1U;
    const bitCapIntOcl maxLcv = qPages.size() >> 1U;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (maxLcv < numCores) ? maxLcv : numCores;
    std::vector<std::future<void>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < maxLcv; ++i) {
        bitCapIntOcl j = i & targetMask;
        j |= (i ^ j) << 1U;

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j | targetPow];

        const bool doNrm = doNormalize;

#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            futures[iF].get();
        }
        futures[iF] = std::async(std::launch::async, [engine1, engine2, isSqiCtrl, isAnti, sqi, fn, doNrm]() {
#endif
            engine1->ShuffleBuffers(engine2);
            if (!isSqiCtrl || isAnti) {
                fn(engine1, sqi);
            }
            if (!isSqiCtrl || !isAnti) {
                fn(engine2, sqi);
            }
            engine1->ShuffleBuffers(engine2);

            if (doNrm) {
                engine1->QueueSetDoNormalize(false);
                engine2->QueueSetDoNormalize(false);
            }
#if ENABLE_PTHREAD
        });
#endif
    }

#if ENABLE_PTHREAD
    for (size_t i = 0U; i < futures.size(); ++i) {
        futures[i].get();
    }
#endif
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename Qubit1Fn>
void QPager::MetaControlled(bitCapInt controlPerm, const std::vector<bitLenInt>& controls, bitLenInt target,
    Qubit1Fn fn, const complex* mtrx, bool isSqiCtrl, bool isIntraCtrled)
{
    const bitLenInt qpp = qubitsPerPage();
    const bitLenInt sqi = qpp - 1U;
    target -= qpp;

    std::vector<bitCapIntOcl> sortedMasks(1U + controls.size());
    const bitCapIntOcl targetPow = pow2Ocl(target);
    sortedMasks[controls.size()] = targetPow - 1U;

    bitCapIntOcl controlMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (((bitCapIntOcl)controlPerm >> i) & 1U) {
            controlMask |= sortedMasks[i];
        }
        --sortedMasks[i];
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bool isSpecial, isInvert;
    complex top, bottom;
    if (!isSqiCtrl && !isIntraCtrled && IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        isSpecial = true;
        isInvert = false;
        top = mtrx[0U];
        bottom = mtrx[3U];
    } else if (!isSqiCtrl && !isIntraCtrled && IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        isSpecial = true;
        isInvert = true;
        top = mtrx[1U];
        bottom = mtrx[2U];
    } else {
        isSpecial = false;
        isInvert = false;
        top = ZERO_CMPLX;
        bottom = ZERO_CMPLX;
    }

    const bitCapIntOcl maxLcv = (bitCapIntOcl)qPages.size() >> (bitCapIntOcl)sortedMasks.size();
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (maxLcv < numCores) ? maxLcv : numCores;
    std::vector<std::future<void>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < maxLcv; ++i) {
        bitCapIntOcl jHi = i;
        bitCapIntOcl j = 0U;
        for (bitCapIntOcl k = 0U; k < (sortedMasks.size()); ++k) {
            const bitCapIntOcl jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << 1U;
            j |= jLo;
        }
        j |= jHi | controlMask;

        if (isInvert) {
            qPages[j].swap(qPages[j | targetPow]);
        }

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j | targetPow];

        if (isSpecial) {
            if (!IS_NORM_0(ONE_CMPLX - top)) {
                engine1->Phase(top, top, 0U);
            }
            if (!IS_NORM_0(ONE_CMPLX - bottom)) {
                engine2->Phase(bottom, bottom, 0U);
            }

            continue;
        }

        const bool isAnti = !((bitCapIntOcl)controlPerm >> controls.size() & 1U);

#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            futures[iF].get();
        }
        futures[iF] = std::async(std::launch::async, [engine1, engine2, isSqiCtrl, isAnti, sqi, fn]() {
#endif
            engine1->ShuffleBuffers(engine2);
            if (!isSqiCtrl || isAnti) {
                fn(engine1, sqi);
            }
            if (!isSqiCtrl || !isAnti) {
                fn(engine2, sqi);
            }
            engine1->ShuffleBuffers(engine2);
#if ENABLE_PTHREAD
        });
#endif
    }

    if (isSpecial) {
        return;
    }

#if ENABLE_PTHREAD
    for (size_t i = 0U; i < futures.size(); ++i) {
        futures[i].get();
    }
#endif
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to
// fit in sub-engines).
template <typename Qubit1Fn>
void QPager::SemiMetaControlled(bitCapInt controlPerm, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn)
{
    const bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> sortedMasks(controls.size());

    bitCapIntOcl controlMask = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (((bitCapIntOcl)controlPerm >> i) & 1U) {
            controlMask |= sortedMasks[i];
        }
        --sortedMasks[i];
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    const bitCapIntOcl maxLcv = (bitCapIntOcl)qPages.size() >> (bitCapIntOcl)sortedMasks.size();
    for (bitCapIntOcl i = 0U; i < maxLcv; ++i) {
        bitCapIntOcl jHi = i;
        bitCapIntOcl j = 0U;
        for (bitCapIntOcl k = 0U; k < (sortedMasks.size()); ++k) {
            const bitCapIntOcl jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << 1U;
            j |= jLo;
        }
        j |= jHi | controlMask;

        fn(qPages[j], target);
    }
}

template <typename F> void QPager::CombineAndOp(F fn, std::vector<bitLenInt> bits)
{
    bitLenInt highestBit = 0U;
    for (size_t i = 0U; i < bits.size(); ++i) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
    }

    CombineEngines(highestBit + 1U);

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        fn(qPages[i]);
    }
}

template <typename F>
void QPager::CombineAndOpControlled(F fn, std::vector<bitLenInt> bits, const std::vector<bitLenInt>& controls)
{
    for (size_t i = 0U; i < controls.size(); ++i) {
        bits.push_back(controls[i]);
    }

    CombineAndOp(fn, bits);
}

bitLenInt QPager::ComposeEither(QPagerPtr toCopy, bool willDestroy)
{
    const bitLenInt toRet = qubitCount;
    if (!toCopy->qubitCount) {
        return toRet;
    }

    const bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
    if (nQubitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }

    if (nQubitCount <= thresholdQubitsPerPage) {
        CombineEngines();
        toCopy->CombineEngines();
        SetQubitCount(nQubitCount);
        return qPages[0U]->Compose(toCopy->qPages[0U]);
    }

    if (qubitCount < toCopy->qubitCount) {
        QPagerPtr toCopyClone = willDestroy ? toCopy : std::dynamic_pointer_cast<QPager>(toCopy->Clone());
        toCopyClone->Compose(shared_from_this(), 0U);
        qPages = toCopyClone->qPages;
        SetQubitCount(nQubitCount);
        return toRet;
    }

    const bitLenInt qpp = qubitsPerPage();
    const bitCapIntOcl nPagePow = pow2Ocl(thresholdQubitsPerPage);
    const bitCapIntOcl pmqp = pageMaxQPower();
    const bitCapIntOcl oPagePow = toCopy->pageMaxQPower();
    const bitCapIntOcl tcqpp = toCopy->qubitsPerPage();
    const bitCapIntOcl maxI = toCopy->maxQPowerOcl - 1U;
    bitCapIntOcl oOffset = oPagePow;
    std::vector<QEnginePtr> nQPages((maxI + 1U) * qPages.size());
    for (bitCapIntOcl i = 0U; i < maxI; ++i) {
        if (willDestroy && (i == oOffset)) {
            oOffset -= oPagePow;
            toCopy->qPages[oOffset >> tcqpp] = NULL;
            oOffset += oPagePow << 1U;
        }

        const complex amp = toCopy->GetAmplitude(i);

        if (IS_NORM_0(amp)) {
            for (bitCapIntOcl j = 0U; j < qPages.size(); ++j) {
                const bitCapIntOcl page = i * qPages.size() + j;
                nQPages[page] = MakeEngine(qpp, (pmqp * page) / nPagePow);
            }
            continue;
        }

        for (bitCapIntOcl j = 0U; j < qPages.size(); ++j) {
            const bitCapIntOcl page = i * qPages.size() + j;
            nQPages[page] = MakeEngine(qpp, (pmqp * page) / nPagePow);
            if (!qPages[j]->IsZeroAmplitude()) {
                nQPages[page]->SetAmplitudePage(qPages[j], 0U, 0U, (bitCapIntOcl)nQPages[page]->GetMaxQPower());
                nQPages[page]->Phase(amp, amp, 0U);
            }
        }
    }

    const complex amp = toCopy->GetAmplitude(maxI);
    if (willDestroy) {
        toCopy->qPages.back() = NULL;
    }
    if (IS_NORM_0(amp)) {
        for (bitCapIntOcl j = 0U; j < qPages.size(); ++j) {
            const bitCapIntOcl page = maxI * qPages.size() + j;
            nQPages[page] = MakeEngine(qpp, (pmqp * page) / nPagePow);
            qPages[j] = NULL;
        }
    } else {
        for (bitCapIntOcl j = 0U; j < qPages.size(); ++j) {
            const bitCapIntOcl page = maxI * qPages.size() + j;
            nQPages[page] = MakeEngine(qpp, (pmqp * page) / nPagePow);
            if (!qPages[j]->IsZeroAmplitude()) {
                nQPages[page]->SetAmplitudePage(qPages[j], 0U, 0U, (bitCapIntOcl)nQPages[page]->GetMaxQPower());
                nQPages[page]->Phase(amp, amp, 0U);
            }
            qPages[j] = NULL;
        }
    }
    qPages = nQPages;

    SetQubitCount(nQubitCount);

    CombineEngines(thresholdQubitsPerPage);
    SeparateEngines();

    return toRet;
}

QInterfacePtr QPager::Decompose(bitLenInt start, bitLenInt length)
{
    QPagerPtr dest = std::make_shared<QPager>(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor);

    Decompose(start, dest);

    return dest;
}

void QPager::Decompose(bitLenInt start, QPagerPtr dest)
{
    const bitLenInt length = dest->GetQubitCount();
    CombineEngines(length + 1U);

    if ((start + length) > qubitsPerPage()) {
        ROR(start, 0, qubitCount);
        Decompose(0, dest);
        ROL(start, 0, qubitCount);
        return;
    }

    dest->CombineEngines();

    bool isDecomposed = false;
    for (size_t i = 0U; i < qPages.size(); ++i) {
        if (!isDecomposed && !qPages[i]->IsZeroAmplitude()) {
            qPages[i]->Decompose(start, dest->qPages[0U]);
            dest->qPages[0U]->UpdateRunningNorm();
            dest->qPages[0U]->NormalizeState();
            isDecomposed = true;
        } else {
            qPages[i]->Dispose(start, length);
        }
    }

    SetQubitCount(qubitCount - length);

    CombineEngines(thresholdQubitsPerPage);
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    CombineEngines(length + 1U);

    if ((start + length) > qubitsPerPage()) {
        ROR(start, 0, qubitCount);
        Dispose(0, length);
        ROL(start, 0, qubitCount);
        return;
    }

    for (size_t i = 0U; i < qPages.size(); ++i) {
        qPages[i]->Dispose(start, length);
    }

    SetQubitCount(qubitCount - length);

    CombineEngines(thresholdQubitsPerPage);
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    CombineEngines(length + 1U);

    if ((start + length) > qubitsPerPage()) {
        ROR(start, 0, qubitCount);
        Dispose(0, length, disposedPerm);
        ROL(start, 0, qubitCount);
        return;
    }

    for (size_t i = 0U; i < qPages.size(); ++i) {
        qPages[i]->Dispose(start, length, disposedPerm);
    }

    SetQubitCount(qubitCount - length);

    CombineEngines(thresholdQubitsPerPage);
    SeparateEngines();
}

bitLenInt QPager::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QPagerPtr nQubits = std::make_shared<QPager>(engines, length, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, thresholdQubitsPerPage);
    return Compose(nQubits, start);
}

void QPager::SetQuantumState(const complex* inputState)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0U;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
    std::vector<std::future<void>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        QEnginePtr engine = qPages[i];
        const bool doNorm = doNormalize;
#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            futures[iF].get();
        }
        futures[iF] = std::async(std::launch::async, [engine, inputState, pagePerm, doNorm]() {
#endif
            engine->SetQuantumState(inputState + pagePerm);
            if (doNorm) {
                engine->UpdateRunningNorm();
            }
#if ENABLE_PTHREAD
        });
#endif
        pagePerm += pagePower;
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
        futures[i].get();
    }
#endif
}

void QPager::GetQuantumState(complex* outputState)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0U;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
    std::vector<std::future<void>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            futures[iF].get();
        }
        futures[iF] = std::async(
            std::launch::async, [engine, outputState, pagePerm]() { engine->GetQuantumState(outputState + pagePerm); });
#else
        engine->GetQuantumState(outputState + pagePerm);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
        futures[i].get();
    }
#endif
}

void QPager::GetProbs(real1* outputProbs)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0U;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
    std::vector<std::future<void>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            futures[iF].get();
        }
        futures[iF] = std::async(
            std::launch::async, [engine, outputProbs, pagePerm]() { engine->GetProbs(outputProbs + pagePerm); });
#else
        engine->GetProbs(outputProbs + pagePerm);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
        futures[i].get();
    }
#endif
}

void QPager::SetPermutation(bitCapInt perm, complex phaseFac)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    const bitCapIntOcl permOcl = (bitCapIntOcl)perm & (maxQPowerOcl - 1U);
    bitCapIntOcl pagePerm = 0U;
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        bool isPermInPage = (permOcl >= pagePerm);
        pagePerm += pagePower;
        isPermInPage &= (permOcl < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(permOcl - (pagePerm - pagePower), phaseFac);
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::Mtrx(const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        Phase(mtrx[0U], mtrx[3U], target);
        return;
    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        Invert(mtrx[1U], mtrx[2U], target);
        return;
    }

    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->Mtrx(mtrx, lTarget); });
}

void QPager::ApplySingleEither(bool isInvert, complex top, complex bottom, bitLenInt target)
{
    bitLenInt qpp = qubitsPerPage();

    if (target < qpp) {
        if (isInvert) {
            SingleBitGate(
                target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) { engine->Invert(top, bottom, lTarget); });
        } else {
            SingleBitGate(
                target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) { engine->Phase(top, bottom, lTarget); });
        }

        return;
    }

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    target -= qpp;
    const bitCapIntOcl targetPow = pow2Ocl(target);
    const bitCapIntOcl qMask = targetPow - 1U;
    const bitCapIntOcl maxLcv = (bitCapIntOcl)qPages.size() >> 1U;
    for (bitCapIntOcl i = 0U; i < maxLcv; ++i) {
        bitCapIntOcl j = i & qMask;
        j |= (i ^ j) << 1U;

        if (isInvert) {
            qPages[j].swap(qPages[j | targetPow]);
        }

        if (!IS_NORM_0(ONE_CMPLX - top)) {
            qPages[j]->Phase(top, top, 0U);
        }
        if (!IS_NORM_0(ONE_CMPLX - bottom)) {
            qPages[j | targetPow]->Phase(bottom, bottom, 0U);
        }
    }
}

void QPager::ApplyEitherControlledSingleBit(
    bitCapInt controlPerm, const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx)
{
    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
    }

    const bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    bool isSqiCtrl = false;
    bitLenInt sqiIndex = 0U;
    bitCapIntOcl intraCtrlPerm = 0U;
    bitCapIntOcl metaCtrlPerm = 0U;
    const bitCapIntOcl cp = (bitCapIntOcl)controlPerm;
    for (size_t i = 0U; i < controls.size(); ++i) {
        if ((target >= qpp) && (controls[i] == (qpp - 1U))) {
            isSqiCtrl = true;
            sqiIndex = i;
        } else if (controls[i] < qpp) {
            intraCtrlPerm |= ((cp >> i) & 1U) << intraControls.size();
            intraControls.push_back(controls[i]);
        } else {
            metaCtrlPerm |= ((cp >> i) & 1U) << metaControls.size();
            metaControls.push_back(controls[i]);
        }
    }

    const bool isAnti = !((cp >> sqiIndex) & 1U);
    if (isSqiCtrl && !isAnti) {
        intraCtrlPerm |= pow2Ocl(intraControls.size());
        metaCtrlPerm |= pow2Ocl(metaControls.size());
    }

    auto sg = [intraCtrlPerm, mtrx, intraControls](QEnginePtr engine, bitLenInt lTarget) {
        engine->UCMtrx(intraControls, mtrx, lTarget, intraCtrlPerm);
    };

    if (!metaControls.size()) {
        SingleBitGate(target, sg, isSqiCtrl, isAnti);
    } else if (target < qpp) {
        SemiMetaControlled(metaCtrlPerm, metaControls, target, sg);
    } else {
        MetaControlled(metaCtrlPerm, metaControls, target, sg, mtrx, isSqiCtrl, intraControls.size());
    }
}

void QPager::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    CombineAndOp([&](QEnginePtr engine) { engine->UniformParityRZ(mask, angle); }, { log2(mask) });
}

void QPager::CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CUniformParityRZ(controls, mask, angle); }, { log2(mask) }, controls);
}

void QPager::XMask(bitCapInt mask)
{
    const bitCapIntOcl pageMask = pageMaxQPower() - 1U;
    const bitCapIntOcl intraMask = (bitCapIntOcl)mask & pageMask;
    bitCapIntOcl interMask = (bitCapIntOcl)mask ^ intraMask;
    bitCapIntOcl v;
    bitLenInt bit;
    while (interMask) {
        v = interMask & (interMask - 1U);
        bit = log2Ocl(interMask ^ v);
        interMask = v;
        X(bit);
    }

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        qPages[i]->XMask(intraMask);
    }
}

void QPager::PhaseParity(real1_f radians, bitCapInt mask)
{
    const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    const bitCapIntOcl pageMask = pageMaxQPower() - 1U;
    const bitCapIntOcl intraMask = (bitCapIntOcl)mask & pageMask;
    const bitCapIntOcl interMask = ((bitCapIntOcl)mask ^ intraMask) >> qubitsPerPage();
    const complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
    const complex iPhaseFac = ONE_CMPLX / phaseFac;
    bitCapIntOcl v;
    for (bitCapIntOcl i = 0; i < qPages.size(); ++i) {
        QEnginePtr engine = qPages[i];

        v = interMask & i;
        for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        if (intraMask) {
            engine->PhaseParity(v ? -radians : radians, intraMask);
        } else if (v) {
            engine->Phase(phaseFac, phaseFac, 0U);
        } else {
            engine->Phase(iPhaseFac, iPhaseFac, 0U);
        }
    }
}

bool QPager::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (qPages.size() == 1U) {
        return qPages[0U]->ForceM(qubit, result, doForce, doApply);
    }

    real1_f oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            real1_f prob = Rand();
            result = (prob <= oneChance);
        }
    }

    real1_f nrmlzr = result ? oneChance : (ONE_R1 - oneChance);

    if (nrmlzr <= ZERO_R1) {
        throw std::invalid_argument("QPager::ForceM() forced a measurement result with 0 probability");
    }

    if (!doApply || ((ONE_R1 - nrmlzr) <= ZERO_R1)) {
        return result;
    }

    const complex nrm = GetNonunitaryPhase() / (real1)std::sqrt((real1_s)nrmlzr);

    const bitLenInt qpp = qubitsPerPage();
    if (qubit < qpp) {
        const bitCapIntOcl qPower = pow2Ocl(qubit);
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->ApplyM(qPower, result, nrm);
        }
    } else {
        const bitLenInt metaQubit = qubit - qpp;
        const bitCapIntOcl qPower = pow2Ocl(metaQubit);
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            if (!(i & qPower) == !result) {
                qPages[i]->Phase(nrm, nrm, 0U);
                qPages[i]->UpdateRunningNorm();
            } else {
                qPages[i]->ZeroAmplitudes();
            }
        }
    }

    return result;
}

#if ENABLE_ALU
void QPager::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECSC(toAdd, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECSC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
#if ENABLE_BCD
void QPager::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCD(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECBCDC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
#endif
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
void QPager::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CMUL(toMul, inOutStart, carryStart, length, controls); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls);
}
void QPager::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CDIV(toDiv, inOutStart, carryStart, length, controls); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls);
}
void QPager::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls);
}
void QPager::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        IMULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls);
}
void QPager::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CPOWModNOut(base, modN, inStart, outStart, length, controls); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls);
}

bitCapInt QPager::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    const unsigned char* values, bool ignored)
{
    CombineEngines();
    return qPages[0U]->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, ignored);
}

bitCapInt QPager::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
    CombineEngines();
    return qPages[0U]->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}
bitCapInt QPager::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
    CombineEngines();
    return qPages[0U]->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}
void QPager::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    CombineEngines();
    return qPages[0U]->Hash(start, length, values);
}

void QPager::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CombineEngines();
    qPages[0U]->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
}
void QPager::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->PhaseFlipIfLess(greaterPerm, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
#endif

void QPager::MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac, bool isInverse)
{
    const bitLenInt qpp = qubitsPerPage();
    qubit1 -= qpp;
    qubit2 -= qpp;

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    const bitCapIntOcl qubit1Pow = pow2Ocl(qubit1);
    const bitCapIntOcl qubit1Mask = qubit1Pow - 1U;
    const bitCapIntOcl qubit2Pow = pow2Ocl(qubit2);
    const bitCapIntOcl qubit2Mask = qubit2Pow - 1U;

    bitCapIntOcl maxLcv = (bitCapIntOcl)qPages.size() >> 2U;
    for (bitCapIntOcl i = 0U; i < maxLcv; ++i) {
        bitCapIntOcl j = i & qubit1Mask;
        bitCapIntOcl jHi = (i ^ j) << 1U;
        bitCapIntOcl jLo = jHi & qubit2Mask;
        j |= jLo | ((jHi ^ jLo) << 1U);

        qPages[j | qubit1Pow].swap(qPages[j | qubit2Pow]);

        if (!isIPhaseFac) {
            continue;
        }

        if (isInverse) {
            qPages[j | qubit1Pow]->Phase(-I_CMPLX, -I_CMPLX, 0U);
            qPages[j | qubit2Pow]->Phase(-I_CMPLX, -I_CMPLX, 0U);
        } else {
            qPages[j | qubit1Pow]->Phase(I_CMPLX, I_CMPLX, 0U);
            qPages[j | qubit2Pow]->Phase(I_CMPLX, I_CMPLX, 0U);
        }
    }
}

void QPager::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const bool isQubit1Meta = qubit1 >= baseQubitsPerPage;
    const bool isQubit2Meta = qubit2 >= baseQubitsPerPage;
    if (isQubit1Meta && isQubit2Meta) {
        SeparateEngines();
        MetaSwap(qubit1, qubit2, false, false);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SeparateEngines();
        QInterface::Swap(qubit1, qubit2);
        return;
    }

    for (size_t i = 0U; i < qPages.size(); ++i) {
        qPages[i]->Swap(qubit1, qubit2);
    }
}
void QPager::EitherISwap(bitLenInt qubit1, bitLenInt qubit2, bool isInverse)
{
    if (qubit1 == qubit2) {
        return;
    }

    const bool isQubit1Meta = qubit1 >= baseQubitsPerPage;
    const bool isQubit2Meta = qubit2 >= baseQubitsPerPage;
    if (isQubit1Meta && isQubit2Meta) {
        SeparateEngines();
        MetaSwap(qubit1, qubit2, true, isInverse);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SeparateEngines();
        QInterface::Swap(qubit1, qubit2);
    }
    if (isQubit1Meta) {
        qubit1 -= baseQubitsPerPage;
        const complex phaseFac = isInverse ? -I_CMPLX : I_CMPLX;
        for (size_t i = 0U; i < qPages.size(); ++i) {
            if ((i >> qubit1) & 1U) {
                qPages[i]->Phase(phaseFac, ONE_CMPLX, qubit2);
            } else {
                qPages[i]->Phase(ONE_CMPLX, phaseFac, qubit2);
            }
        }
        return;
    }
    if (isQubit2Meta) {
        qubit2 -= baseQubitsPerPage;
        const complex phaseFac = isInverse ? -I_CMPLX : I_CMPLX;
        for (size_t i = 0U; i < qPages.size(); ++i) {
            if ((i >> qubit2) & 1U) {
                qPages[i]->Phase(phaseFac, ONE_CMPLX, qubit1);
            } else {
                qPages[i]->Phase(ONE_CMPLX, phaseFac, qubit1);
            }
        }
        return;
    }

    if (isInverse) {
        for (size_t i = 0U; i < qPages.size(); ++i) {
            qPages[i]->IISwap(qubit1, qubit2);
        }
    } else {
        for (size_t i = 0U; i < qPages.size(); ++i) {
            qPages[i]->ISwap(qubit1, qubit2);
        }
    }
}
void QPager::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const std::vector<bitLenInt> controls{ qubit1 };
    const real1 sinTheta = (real1)sin(theta);

    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
        MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));

    const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
    if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
    if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON) {
        IISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->FSim(theta, phi, qubit1, qubit2); }, { qubit1, qubit2 });
}

real1_f QPager::Prob(bitLenInt qubit)
{
    if (qPages.size() == 1U) {
        return qPages[0U]->Prob(qubit);
    }

    const bitLenInt qpp = qubitsPerPage();
    real1 oneChance = ZERO_R1;
    if (qubit < qpp) {
#if ENABLE_PTHREAD
        const unsigned numCores = GetConcurrencyLevel();
        const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
        std::vector<std::future<real1_f>> futures(fCount);
#endif
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
            const bitCapIntOcl iF = i % fCount;
            if (i != iF) {
                oneChance += futures[iF].get();
            }
            futures[iF] = std::async(std::launch::async, [engine, qubit]() { return engine->Prob(qubit); });
#else
            oneChance += engine->Prob(qubit);
#endif
        }

#if ENABLE_PTHREAD
        for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
            oneChance += futures[i].get();
        }
#endif
    } else {
        const bitCapIntOcl qPower = pow2Ocl(qubit - qpp);
        const bitCapIntOcl qMask = qPower - 1U;
        const bitCapIntOcl fSize = qPages.size() >> 1U;
#if ENABLE_PTHREAD
        const unsigned numCores = GetConcurrencyLevel();
        const bitCapIntOcl fCount = (fSize < numCores) ? fSize : numCores;
        std::vector<std::future<real1_f>> futures(fCount);
#endif
        for (bitCapIntOcl i = 0U; i < fSize; ++i) {
            bitCapIntOcl j = i & qMask;
            j |= ((i ^ j) << 1U) | qPower;

            QEnginePtr engine = qPages[j];
#if ENABLE_PTHREAD
            const bitCapIntOcl iF = i % fCount;
            if (i != iF) {
                oneChance += futures[iF].get();
            }
            futures[iF] = std::async(std::launch::async, [engine]() {
                engine->UpdateRunningNorm();
                return engine->GetRunningNorm();
            });
#else
            engine->UpdateRunningNorm();
            oneChance += engine->GetRunningNorm();
#endif
        }

#if ENABLE_PTHREAD
        for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
            oneChance += futures[i].get();
        }
#endif
    }

    return clampProb((real1_f)oneChance);
}

real1_f QPager::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    CombineEngines(log2(mask) + 1U);

    real1_f maskChance = ZERO_R1_F;
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        maskChance += qPages[i]->ProbMask(mask, permutation);
    }
    return clampProb((real1_f)maskChance);
}

real1_f QPager::ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset)
{
    if (bits.size() != qubitCount) {
        return QInterface::ExpectationBitsAll(bits, offset);
    }

    for (bitCapIntOcl i = 0U; i < bits.size(); ++i) {
        if (bits[i] != i) {
            return QInterface::ExpectationBitsAll(bits, offset);
        }
    }

    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    real1_f expectation = ZERO_R1_F;
    bitCapIntOcl pagePerm = 0U;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
    std::vector<std::future<real1_f>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            expectation += futures[iF].get();
        }
        futures[iF] = std::async(std::launch::async, [engine, bits, pagePerm, offset]() {
            return engine->ExpectationBitsAll(bits, pagePerm + (bitCapIntOcl)offset);
        });
#else
        expectation += engine->ExpectationBitsAll(bits, pagePerm + offset);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
        expectation += futures[i].get();
    }
#endif

    return expectation;
}

void QPager::UpdateRunningNorm(real1_f norm_thresh)
{
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

void QPager::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    real1_f nmlzr;
    if (nrm == REAL1_DEFAULT_ARG) {
        nmlzr = ZERO_R1_F;
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            nmlzr += qPages[i]->GetRunningNorm();
        }
    } else {
        nmlzr = nrm;
    }

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        qPages[i]->NormalizeState(nmlzr, norm_thresh, phaseArg);
    }
}

QInterfacePtr QPager::Clone()
{
    SeparateEngines();

    QPagerPtr clone = std::make_shared<QPager>(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, thresholdQubitsPerPage);

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        clone->qPages[i] = std::dynamic_pointer_cast<QEngine>(qPages[i]->Clone());
    }

    return clone;
}

QEnginePtr QPager::CloneEmpty()
{
    SeparateEngines();

    QPagerPtr clone = std::make_shared<QPager>(engines, qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, thresholdQubitsPerPage);

    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        clone->qPages[i] = qPages[i]->CloneEmpty();
    }

    return clone;
}

real1_f QPager::SumSqrDiff(QPagerPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    SeparateEngines(toCompare->qubitsPerPage());
    toCompare->SeparateEngines(qubitsPerPage());
    CombineEngines(toCompare->qubitsPerPage());
    toCompare->CombineEngines(qubitsPerPage());

    real1_f toRet = ZERO_R1_F;
#if ENABLE_PTHREAD
    const unsigned numCores = GetConcurrencyLevel();
    const bitCapIntOcl fCount = (qPages.size() < numCores) ? qPages.size() : numCores;
    std::vector<std::future<real1_f>> futures(fCount);
#endif
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        QEnginePtr lEngine = qPages[i];
        QEnginePtr rEngine = toCompare->qPages[i];
#if ENABLE_PTHREAD
        const bitCapIntOcl iF = i % fCount;
        if (i != iF) {
            toRet += futures[iF].get();
        }
        futures[iF] = std::async(std::launch::async, [lEngine, rEngine]() { return lEngine->SumSqrDiff(rEngine); });
#else
        toRet += lEngine->SumSqrDiff(rEngine);
#endif
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0U; i < futures.size(); ++i) {
        toRet += futures[i].get();
    }
#endif

    return toRet;
}

} // namespace Qrack
