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

#include "qunitmulti.hpp"
#include "common/oclengine.hpp"
#include "qfactory.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QUnit(eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG,
          useSparseStateVec, norm_thresh, devList, qubitThreshold, sep_thresh)
{
    std::vector<DeviceContextPtr> deviceContext = OCLEngine::Instance()->GetDeviceContextPtrVector();

    if (devList.size() == 0) {
        defaultDeviceID = (deviceID < 0) ? OCLEngine::Instance()->GetDefaultDeviceID() : (size_t)deviceID;

        for (size_t i = 0; i < deviceContext.size(); i++) {
            DeviceInfo deviceInfo;
            deviceInfo.id = i;
            deviceList.push_back(deviceInfo);
        }

        std::swap(deviceList[0], deviceList[defaultDeviceID]);
    } else {
        defaultDeviceID = (devList[0] < 0) ? OCLEngine::Instance()->GetDefaultDeviceID() : (size_t)devList[0];

        for (size_t i = 0; i < devList.size(); i++) {
            DeviceInfo deviceInfo;
            deviceInfo.id = (devList[0] < 0) ? OCLEngine::Instance()->GetDefaultDeviceID() : (size_t)devList[i];
            deviceList.push_back(deviceInfo);
        }
    }

    for (size_t i = 0; i < deviceList.size(); i++) {
        deviceList[i].maxSize = deviceContext[deviceList[i].id]->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    }

    if (devList.size() == 0) {
        std::sort(deviceList.begin() + 1, deviceList.end(), std::greater<DeviceInfo>());
    }

    devID = 0;
}

QInterfacePtr QUnitMulti::MakeEngine(bitLenInt length, bitCapInt perm)
{
    size_t deviceId = defaultDeviceID;
    uint64_t sz = OCLEngine::Instance()->GetActiveAllocSize(deviceId);

    for (size_t i = 0U; i < deviceList.size(); i++) {
        uint64_t tSz = OCLEngine::Instance()->GetActiveAllocSize(deviceList[i].id);
        if (sz > tSz) {
            sz = tSz;
            deviceId = deviceList[i].id;
        }
    }

    // Suppress passing device list, since QUnitMulti occupies all devices in the list
    QInterfacePtr toRet = CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, deviceId, useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{},
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
            int deviceIndex = std::distance(
                deviceList.begin(), std::find_if(deviceList.begin(), deviceList.end(), [&](DeviceInfo di) {
                    return di.id == (shard.unit->GetDeviceID() < 0) ? OCLEngine::Instance()->GetDefaultDeviceID()
                                                                    : (size_t)shard.unit->GetDeviceID();
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
#if ENABLE_ENV_VARS
    if (!getenv("QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE") || strcmp(getenv("QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE"), "")) {
        return;
    }
#else
    return;
#endif

    // No need to redistribute, if there is only 1 device
    if (deviceList.size() == 1) {
        return;
    }

    // Get shard sizes and devices
    std::vector<QEngineInfo> qinfos = GetQInfos();

    std::vector<bitCapInt> devSizes(deviceList.size());
    std::fill(devSizes.begin(), devSizes.end(), 0U);

    for (size_t i = 0; i < qinfos.size(); i++) {
        // If the engine adds negligible load, we can let any given unit keep its
        // residency on this device.
        // In fact, single qubit units will be handled entirely by the CPU, anyway.
        // So will QHybrid "shards" that are below the GPU transition threshold.
        if (!(qinfos[i].unit) || (qinfos[i].unit->GetMaxQPower() <= 2U) ||
            (qinfos[i].unit->GetQubitCount() < thresholdQubits)) {
            continue;
        }

        // If the original OpenCL device has equal load to the least, we prefer the original.
        size_t deviceID = qinfos[i].unit->GetDeviceID();
        size_t devIndex = qinfos[i].deviceIndex;
        bitCapInt sz = devSizes[devIndex];

        // If the original device has 0 determined load, don't switch the unit.
        if (sz > 0) {
            // If the default OpenCL device has equal load to the least, we prefer the default.
            if (devSizes[0] < sz) {
                deviceID = deviceList[0].id;
                devIndex = 0;
                sz = devSizes[0];
            }

            // Find the device with the lowest load.
            for (size_t j = 0; j < deviceList.size(); j++) {
                if ((devSizes[j] < sz) && ((devSizes[j] + qinfos[i].unit->GetMaxQPower()) <= deviceList[j].maxSize)) {
                    deviceID = deviceList[j].id;
                    devIndex = j;
                    sz = devSizes[j];
                }
            }

            // Add this unit to the device with the lowest load.
            qinfos[i].unit->SetDevice(deviceID);
        }

        // Update the size of buffers handles by this device.
        devSizes[devIndex] += qinfos[i].unit->GetMaxQPower();
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
{
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

QInterfacePtr QUnitMulti::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    for (auto bit = first; bit < last; bit++) {
        EndEmulation(**bit);
    }

    QInterfacePtr unit1 = shards[**first].unit;

    bool isAlreadyEntangled = true;
    // If already fully entangled, just return unit1.
    for (auto bit = first + 1; bit < last; bit++) {
        QInterfacePtr unit = shards[**bit].unit;
        if (unit1 != unit) {
            isAlreadyEntangled = false;
            break;
        }
    }

    if (isAlreadyEntangled) {
        return unit1;
    }

    // This does nothing if the first unit is the default device:
    if (deviceList[0].id !=
        ((unit1->GetDeviceID() < 0) ? OCLEngine::Instance()->GetDefaultDeviceID() : (size_t)unit1->GetDeviceID())) {
        // Check if size exceeds single device capacity:
        bitLenInt qubitCount = 0;
        std::map<QInterfacePtr, bool> found;

        for (auto bit = first; bit < last; bit++) {
            QInterfacePtr unit = shards[**bit].unit;
            if (found.find(unit) == found.end()) {
                found[unit] = true;
                qubitCount += unit->GetQubitCount();
            }
        }

        // If device capacity is exceeded, put on default device:
        if (pow2(qubitCount) > unit1->GetMaxSize()) {
            unit1->SetDevice(deviceList[0].id);
        }
    }

    QInterfacePtr toRet = QUnit::EntangleInCurrentBasis(first, last);
    RedistributeQEngines();

    return toRet;
}

bool QUnitMulti::SeparateBit(bool value, bitLenInt qubit)
{
    bool isClifford = shards[qubit].unit->isClifford();
    bool toRet = QUnit::SeparateBit(value, qubit);
    if (!isClifford && toRet) {
        RedistributeQEngines();
    }
    return toRet;
}

QInterfacePtr QUnitMulti::Clone()
{
    // TODO: Copy buffers instead of flushing?
    for (bitLenInt i = 0; i < qubitCount; i++) {
        RevertBasis2Qb(i);
    }

    QUnitMultiPtr copyPtr = std::make_shared<QUnitMulti>(engines, qubitCount, 0, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, defaultDeviceID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, thresholdQubits, separabilityThreshold);

    Finish();
    copyPtr->Finish();

    return CloneBody(copyPtr);
}

} // namespace Qrack
