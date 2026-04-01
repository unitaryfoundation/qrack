//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

#if ENABLE_ENV_VARS
#include <regex>
#include <string>
#endif

namespace Qrack {

QUnitMulti::QUnitMulti(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState,
    qrack_rand_gen_ptr rgp, const complex& phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem,
    int64_t deviceID, bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh, bool useTurbo)
    : QUnit(eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG,
          useSparseStateVec, norm_thresh, devList, qubitThreshold, sep_thresh, useTurbo)
    , isQEngineOCL(false)
    , deviceQbList({ (bitLenInt)-1 })
{
#if ENABLE_ENV_VARS
    isRedistributing = (bool)getenv("QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE");
#else
    isRedistributing = false;
#endif

    for (size_t i = 0U; i < engines.size(); i++) {
        if ((engines[i] == QINTERFACE_CPU) || (engines[i] == QINTERFACE_HYBRID)) {
            break;
        }
        if (engines[i] == QRACK_GPU_ENGINE) {
            isQEngineOCL = true;
            break;
        }
    }
    if (engines.back() == QINTERFACE_QPAGER) {
        isQEngineOCL = true;
    }

    if (qubitThreshold) {
        thresholdQubits = qubitThreshold;
    } else {
        const bitLenInt gpuQubits =
            log2Ocl(QRACK_GPU_SINGLETON.GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;
        const bitLenInt cpuQubits = (GetStride() <= 1U) ? 0U : (log2Ocl(GetStride() - 1U) + 1U);
        thresholdQubits = gpuQubits < cpuQubits ? gpuQubits : cpuQubits;
    }

    std::vector<DeviceContextPtr> deviceContext = QRACK_GPU_SINGLETON.GetDeviceContextPtrVector();
    defaultDeviceID = (deviceID < 0) ? QRACK_GPU_SINGLETON.GetDefaultDeviceID() : (size_t)deviceID;

#if ENABLE_ENV_VARS
    if (devList.empty() && getenv("QRACK_QUNITMULTI_DEVICES")) {
        std::string devListStr = std::string(getenv("QRACK_QUNITMULTI_DEVICES"));
        devList.clear();
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
                    devList.push_back(stoi(term));
                    if (devList.back() == -2) {
                        devList.back() = (int)devID;
                    }
                    if (devList.back() == -1) {
                        devList.back() = (int)QRACK_GPU_SINGLETON.GetDefaultDeviceID();
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
                        ids[i - 1U] = (int)QRACK_GPU_SINGLETON.GetDefaultDeviceID();
                    }
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < ids.size(); ++j) {
                        devList.push_back(ids[j]);
                    }
                }
            }
        }
    }

    if (getenv("QRACK_QUNITMULTI_DEVICES_MAX_QB")) {
        std::string deviceQbListStr = std::string(getenv("QRACK_QUNITMULTI_DEVICES_MAX_QB"));
        deviceQbList.clear();
        if (deviceQbListStr.compare("")) {
            std::stringstream deviceQbListStr_stream(deviceQbListStr);
            // See
            // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
            std::regex re("[.]");
            while (deviceQbListStr_stream.good()) {
                std::string term;
                getline(deviceQbListStr_stream, term, ',');
                // the '-1' is what makes the regex split (-1 := what was not matched)
                std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
                std::vector<std::string> tokens{ first, last };
                if (tokens.size() == 1U) {
                    deviceQbList.push_back((bitLenInt)stoi(term));
                    continue;
                }
                const unsigned maxI = stoi(tokens[0U]);
                std::vector<int> ids(tokens.size() - 1U);
                for (unsigned i = 1U; i < tokens.size(); ++i) {
                    ids[i - 1U] = stoi(tokens[i]);
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < ids.size(); ++j) {
                        deviceQbList.push_back(ids[j]);
                    }
                }
            }
        }
    }
#endif

    const size_t devCount = devList.empty() ? deviceContext.size() : devList.size();
    for (size_t i = 0; i < devCount; ++i) {
        if (devList.size() && (devList[i] >= 0) && (devList[i] > ((int64_t)deviceContext.size()))) {
            throw std::runtime_error("QUnitMulti: Requested device doesn't exist.");
        }
        DeviceInfo deviceInfo;
        deviceInfo.id =
            devList.empty() ? i : ((devList[0U] < 0) ? QRACK_GPU_SINGLETON.GetDefaultDeviceID() : (size_t)devList[i]);
        deviceList.push_back(deviceInfo);
    }
    if (devList.empty()) {
        std::swap(deviceList[0U], deviceList[defaultDeviceID]);
    }

    for (size_t i = 0U; i < deviceList.size(); ++i) {
        deviceList[i].maxSize = deviceContext[deviceList[i].id]->GetMaxAlloc();
    }

    if (devList.empty()) {
        std::sort(deviceList.begin() + 1U, deviceList.end(), std::greater<DeviceInfo>());
    }
}

QInterfacePtr QUnitMulti::MakeEngine(bitLenInt length, const bitCapInt& perm)
{
    size_t deviceId = defaultDeviceID;
    uint64_t sz = QRACK_GPU_SINGLETON.GetActiveAllocSize(deviceId);

    for (size_t i = 0U; i < deviceList.size(); ++i) {
        uint64_t tSz = QRACK_GPU_SINGLETON.GetActiveAllocSize(deviceList[i].id);
        if (sz > tSz) {
            sz = tSz;
            deviceId = deviceList[i].id;
        }
    }

    // Suppress passing device list, since QUnitMulti occupies all devices in the list
    QInterfacePtr toRet = CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, deviceId, useRDRAND, false, (real1_f)amplitudeFloor, std::vector<int64_t>{},
        thresholdQubits, separabilityThreshold);

    return toRet;
}

std::vector<QEngineInfo> QUnitMulti::GetQInfos()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    std::vector<QEngineInfo> qinfos;

    for (auto&& shard : shards) {
        if (shard.unit && (std::find(qips.begin(), qips.end(), shard.unit) == qips.end())) {
            qips.push_back(shard.unit);
            const size_t deviceIndex = std::distance(
                deviceList.begin(), std::find_if(deviceList.begin(), deviceList.end(), [&](DeviceInfo di) {
                    return di.id == (shard.unit->GetDevice() < 0) ? QRACK_GPU_SINGLETON.GetDefaultDeviceID()
                                                                  : (size_t)shard.unit->GetDevice();
                }));
            qinfos.push_back(QEngineInfo(shard.unit, deviceIndex));
        }
    }

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    return qinfos;
}

void QUnitMulti::RedistributeQEngines()
{
    // Only redistribute if the env var flag is set and NOT a null string.
    // No need to redistribute, if there is only 1 device
    if (deviceList.size() <= 1U) {
        return;
    }

    // Get shard sizes and devices
    std::vector<QEngineInfo> qinfos = GetQInfos();

    std::vector<bitCapIntOcl> devSizes(deviceList.size(), 0U);

    for (QEngineInfo& qinfo : qinfos) {
        // We want to proactively set OpenCL devices for the event they cross threshold.
        const bitLenInt qbc = qinfo.unit->GetQubitCount();
        const bitLenInt dqb = deviceQbList[qinfo.deviceIndex % deviceQbList.size()];
        if (!isRedistributing && (qbc <= dqb) && (bi_compare(qinfo.unit->GetMaxQPower(), 2U) > 0) &&
            !qinfo.unit->isClifford() && (isQEngineOCL || (qbc > thresholdQubits))) {
            continue;
        }

        // If the original OpenCL device has equal load to the least, we prefer the original.
        int64_t deviceID = qinfo.unit->GetDevice();
        int64_t devIndex = qinfo.deviceIndex;
        bitCapIntOcl sz = devSizes[devIndex];

        // If the original device has 0 determined load, don't switch the unit.
        if (sz) {
            // If the default OpenCL device has equal load to the least, we prefer the default.
            if ((devSizes[0U] < sz) && (qbc <= deviceQbList[0U])) {
                deviceID = deviceList[0U].id;
                devIndex = 0U;
                sz = devSizes[0U];
            }

            // Find the device with the lowest load.
            for (size_t j = 0U; j < deviceList.size(); ++j) {
                const bitLenInt dq = deviceQbList[j % deviceQbList.size()];
                const bitCapInt mqp = devSizes[j] + qinfo.unit->GetMaxQPower();
                if ((devSizes[j] < sz) && (bi_compare(mqp, deviceList[j].maxSize) <= 0) && (qbc <= dq)) {
                    deviceID = deviceList[j].id;
                    devIndex = j;
                    sz = devSizes[j];
                }
            }

            // Add this unit to the device with the lowest load.
            qinfo.unit->SetDevice(deviceID);
        }

        // Update the size of buffers handles by this device.
        if (bi_compare(deviceList[devIndex].maxSize, devSizes[devIndex] + qinfo.unit->GetMaxQPower()) < 0) {
            throw bad_alloc("QUnitMulti: device allocation limits exceeded.");
        }
        devSizes[devIndex] += (bitCapIntOcl)qinfo.unit->GetMaxQPower();
    }
}
} // namespace Qrack
