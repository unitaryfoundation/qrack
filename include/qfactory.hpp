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

#include "qengine_cpu.hpp"
#include "qpager.hpp"
#include "qstabilizerhybrid.hpp"
#include "qtensornetwork.hpp"

#if ENABLE_OPENCL
#include "qengine_opencl.hpp"
#endif

#if ENABLE_CUDA
#include "common/cudaengine.cuh"
#include "qengine_cuda.hpp"
#endif

#if ENABLE_OPENCL || ENABLE_CUDA
#include "qhybrid.hpp"
#include "qunitmulti.hpp"
#else
#include "qunit.hpp"
#endif

#if ENABLE_QBDT
#include "qbdt.hpp"
#include "qbdthybrid.hpp"
#endif

namespace Qrack {

/** Factory method to create specific engine implementations. */
template <typename... Ts>
QInterfacePtr CreateQuantumInterface(
    QInterfaceEngine engine1, QInterfaceEngine engine2, QInterfaceEngine engine3, Ts... args)
{
    QInterfaceEngine engine = engine1;
    std::vector<QInterfaceEngine> engines{ engine2, engine3 };

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_STABILIZER:
        return std::make_shared<QStabilizer>(args...);
    case QINTERFACE_QUNIT_CLIFFORD:
        return std::make_shared<QUnitClifford>(args...);
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(engines, args...);
    case QINTERFACE_BDT_HYBRID:
        return std::make_shared<QBdtHybrid>(engines, args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
    case QINTERFACE_TENSOR_NETWORK:
        return std::make_shared<QTensorNetwork>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
#if ENABLE_CUDA
    case QINTERFACE_CUDA:
        return std::make_shared<QEngineCUDA>(args...);
#endif
#if ENABLE_OPENCL || ENABLE_CUDA
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(engines, args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine1, QInterfaceEngine engine2, Ts... args)
{
    QInterfaceEngine engine = engine1;
    std::vector<QInterfaceEngine> engines{ engine2 };

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_STABILIZER:
        return std::make_shared<QStabilizer>(args...);
    case QINTERFACE_QUNIT_CLIFFORD:
        return std::make_shared<QUnitClifford>(args...);
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(engines, args...);
    case QINTERFACE_BDT_HYBRID:
        return std::make_shared<QBdtHybrid>(engines, args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
    case QINTERFACE_TENSOR_NETWORK:
        return std::make_shared<QTensorNetwork>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
#if ENABLE_CUDA
    case QINTERFACE_CUDA:
        return std::make_shared<QEngineCUDA>(args...);
#endif
#if ENABLE_OPENCL || ENABLE_CUDA
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(engines, args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts> QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_STABILIZER:
        return std::make_shared<QStabilizer>(args...);
    case QINTERFACE_QUNIT_CLIFFORD:
        return std::make_shared<QUnitClifford>(args...);
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(args...);
    case QINTERFACE_BDT_HYBRID:
        return std::make_shared<QBdtHybrid>(args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(args...);
    case QINTERFACE_TENSOR_NETWORK:
        return std::make_shared<QTensorNetwork>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
#if ENABLE_CUDA
    case QINTERFACE_CUDA:
        return std::make_shared<QEngineCUDA>(args...);
#endif
#if ENABLE_OPENCL || ENABLE_CUDA
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts> QInterfacePtr CreateQuantumInterface(std::vector<QInterfaceEngine> engines, Ts... args)
{
    QInterfaceEngine engine = engines[0];
    engines.erase(engines.begin());

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_STABILIZER:
        return std::make_shared<QStabilizer>(args...);
    case QINTERFACE_QUNIT_CLIFFORD:
        return std::make_shared<QUnitClifford>(args...);
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        if (engines.size()) {
            return std::make_shared<QBdt>(engines, args...);
        }
        return std::make_shared<QBdt>(args...);
    case QINTERFACE_BDT_HYBRID:
        if (engines.size()) {
            return std::make_shared<QBdtHybrid>(engines, args...);
        }
        return std::make_shared<QBdtHybrid>(args...);
#endif
    case QINTERFACE_QPAGER:
        if (engines.size()) {
            return std::make_shared<QPager>(engines, args...);
        }
        return std::make_shared<QPager>(args...);
    case QINTERFACE_STABILIZER_HYBRID:
        if (engines.size()) {
            return std::make_shared<QStabilizerHybrid>(engines, args...);
        }
        return std::make_shared<QStabilizerHybrid>(args...);
    case QINTERFACE_QUNIT:
        if (engines.size()) {
            return std::make_shared<QUnit>(engines, args...);
        }
        return std::make_shared<QUnit>(args...);
    case QINTERFACE_TENSOR_NETWORK:
        if (engines.size()) {
            return std::make_shared<QTensorNetwork>(engines, args...);
        }
        return std::make_shared<QTensorNetwork>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
#if ENABLE_CUDA
    case QINTERFACE_CUDA:
        return std::make_shared<QEngineCUDA>(args...);
#endif
#if ENABLE_OPENCL || ENABLE_CUDA
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        if (engines.size()) {
            return std::make_shared<QUnitMulti>(engines, args...);
        }
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

#if ENABLE_OPENCL
#define DEVICE_COUNT (OCLEngine::Instance().GetDeviceCount())
#elif ENABLE_CUDA
#define DEVICE_COUNT (CUDAEngine::Instance().GetDeviceCount())
#endif
template <typename... Ts>
QInterfacePtr CreateArrangedLayers(bool md, bool sd, bool sh, bool bdt, bool pg, bool tn, bool hy, bool oc, Ts... args)
{
#if ENABLE_OPENCL || ENABLE_CUDA
    bool isOcl = oc && (DEVICE_COUNT > 0);
    bool isOclMulti = oc && md && (DEVICE_COUNT > 1);
#else
    bool isOclMulti = false;
#endif

    // Construct backwards, then reverse:
    std::vector<QInterfaceEngine> simulatorType;

#if ENABLE_OPENCL
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
    }
#elif ENABLE_CUDA
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_CUDA : QINTERFACE_CPU);
    }
#endif

    if (pg && !sh && simulatorType.size()) {
        simulatorType.push_back(QINTERFACE_QPAGER);
    }

#if ENABLE_QBDT
    if (bdt && !hy) {
        simulatorType.push_back(QINTERFACE_BDT);
    }
#endif

    if (sh && (!sd || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

    if (sd) {
        simulatorType.push_back(isOclMulti ? QINTERFACE_QUNIT_MULTI : QINTERFACE_QUNIT);
    }

    if (tn) {
        simulatorType.push_back(QINTERFACE_TENSOR_NETWORK);
    }

    // (...then reverse:)
    std::reverse(simulatorType.begin(), simulatorType.end());

    if (!simulatorType.size()) {
#if ENABLE_OPENCL || ENABLE_CUDA
        if (hy && isOcl) {
#if ENABLE_QBDT
            if (bdt) {
                simulatorType.push_back(QINTERFACE_BDT_HYBRID);
            } else {
                simulatorType.push_back(QINTERFACE_HYBRID);
            }
#else
            simulatorType.push_back(QINTERFACE_HYBRID);
#endif
        } else {
#if ENABLE_OPENCL
            simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
#else
            simulatorType.push_back(isOcl ? QINTERFACE_CUDA : QINTERFACE_CPU);
#endif
        }
#else
        if (bdt) {
            simulatorType.push_back(QINTERFACE_BDT_HYBRID);
        } else {
            simulatorType.push_back(QINTERFACE_CPU);
        }
#endif
    }

    return CreateQuantumInterface(simulatorType, args...);
}

} // namespace Qrack
