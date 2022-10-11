//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
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

#if ENABLE_OPENCL
#include "qengine_opencl.hpp"
#include "qhybrid.hpp"
#include "qunitmulti.hpp"
#else
#include "qunit.hpp"
#endif

#if ENABLE_QBDT
#include "qbdt.hpp"
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
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(engines, args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
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
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(engines, args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
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
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        return std::make_shared<QBdt>(args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
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
#if ENABLE_QBDT
    case QINTERFACE_BDT:
        if (engines.size()) {
            return std::make_shared<QBdt>(engines, args...);
        }
        return std::make_shared<QBdt>(args...);
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
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
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

template <typename... Ts>
QInterfacePtr CreateArrangedLayers(bool md, bool sd, bool sh, bool bdt, bool pg, bool zxf, bool hy, bool oc, Ts... args)
{
#if ENABLE_OPENCL
    bool isOcl = oc && (OCLEngine::Instance().GetDeviceCount() > 0);
    bool isOclMulti = oc && md && (OCLEngine::Instance().GetDeviceCount() > 1);
#else
    bool isOclMulti = false;
#endif

    // Construct backwards, then reverse:
    std::vector<QInterfaceEngine> simulatorType;

#if ENABLE_OPENCL
    if (!hy || !isOcl) {
        simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
    }
#endif

    if (pg && !sh && simulatorType.size()) {
        simulatorType.push_back(QINTERFACE_QPAGER);
    }

    if (bdt) {
        simulatorType.push_back(QINTERFACE_BDT);
    }

    if (sh && (!sd || simulatorType.size())) {
        simulatorType.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

    if (sd) {
        simulatorType.push_back(isOclMulti ? QINTERFACE_QUNIT_MULTI : QINTERFACE_QUNIT);
    }

    // (...then reverse:)
    std::reverse(simulatorType.begin(), simulatorType.end());

    if (!simulatorType.size()) {
#if ENABLE_OPENCL
        if (hy && isOcl) {
            simulatorType.push_back(QINTERFACE_HYBRID);
        } else {
            simulatorType.push_back(isOcl ? QINTERFACE_OPENCL : QINTERFACE_CPU);
        }
#else
        simulatorType.push_back(QINTERFACE_CPU);
#endif
    }

    return CreateQuantumInterface(simulatorType, args...);
}

} // namespace Qrack
