//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "wasm_api.hpp"
#include "qcircuit.hpp"
#include "qneuron.hpp"

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>

#define META_LOCK_GUARD() const std::lock_guard<std::mutex> metaLock(metaOperationMutex);

// SIMULATOR_LOCK_GUARD variants will lock simulatorMutexes[nullptr], if the requested simulator doesn't exist.
// This is CORRECT behavior. This will effectively emplace a mutex for nullptr key.
#if CPP_STD > 13
#define SIMULATOR_LOCK_GUARD(simulator)                                                                                \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator]);                                                    \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock =                                                                                                \
            std::make_unique<const std::lock_guard<std::mutex>>(simulatorMutexes[simulator], std::adopt_lock);         \
    }
#else
#define SIMULATOR_LOCK_GUARD(simulator)                                                                                \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator]);                                                    \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[simulator], std::adopt_lock));                      \
    }
#endif

#define SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                 \
    if (sid > simulators.size()) {                                                                                     \
        throw std::invalid_argument("Invalid argument: simulator ID not found!");                                      \
    }                                                                                                                  \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    SIMULATOR_LOCK_GUARD(simulator.get())                                                                              \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }

#define SIMULATOR_LOCK_GUARD_TYPED(sid, def)                                                                           \
    if (sid > simulators.size()) {                                                                                     \
        throw std::invalid_argument("Invalid argument: simulator ID not found!");                                      \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    SIMULATOR_LOCK_GUARD(simulator.get())                                                                              \
    if (!simulator) {                                                                                                  \
        return def;                                                                                                    \
    }

#define SIMULATOR_LOCK_GUARD_BOOL(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, false)

#define SIMULATOR_LOCK_GUARD_REAL1_F(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, ZERO_R1_F)

#define SIMULATOR_LOCK_GUARD_INT(sid) SIMULATOR_LOCK_GUARD_TYPED(sid, 0U)

#if CPP_STD > 13
#define NEURON_LOCK_GUARD(neuron)                                                                                      \
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock;                                                     \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[neuronSimulators[neuron]], neuronMutexes[neuron.get()]);        \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        neuronLock =                                                                                                   \
            std::make_unique<const std::lock_guard<std::mutex>>(neuronMutexes[neuron.get()], std::adopt_lock);         \
        simulatorLock = std::make_unique<const std::lock_guard<std::mutex>>(                                           \
            simulatorMutexes[neuronSimulators[neuron]], std::adopt_lock);                                              \
    }
#else
#define NEURON_LOCK_GUARD(neuron)                                                                                      \
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock;                                                     \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[neuronSimulators[neuron]], neuronMutexes[neuron.get()]);        \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        neuronLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                               \
            new const std::lock_guard<std::mutex>(neuronMutexes[neuron.get()], std::adopt_lock));                      \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[neuronSimulators[neuron]], std::adopt_lock));       \
    }
#endif

#define NEURON_LOCK_GUARD_VOID(nid)                                                                                    \
    if (nid > neurons.size()) {                                                                                        \
        throw std::invalid_argument("Invalid argument: neuron ID not found!");                                         \
    }                                                                                                                  \
                                                                                                                       \
    QNeuronPtr neuron = neurons[nid];                                                                                  \
    NEURON_LOCK_GUARD(neuron)                                                                                          \
    if (!neuron) {                                                                                                     \
        return;                                                                                                        \
    }

#define NEURON_LOCK_GUARD_TYPED(nid, def)                                                                              \
    if (nid > neurons.size()) {                                                                                        \
        throw std::invalid_argument("Invalid argument: neuron ID not found!");                                         \
    }                                                                                                                  \
                                                                                                                       \
    QNeuronPtr neuron = neurons[nid];                                                                                  \
    NEURON_LOCK_GUARD(neuron)                                                                                          \
    if (!neuron) {                                                                                                     \
        return def;                                                                                                    \
    }

#define NEURON_LOCK_GUARD_REAL1_F(nid) NEURON_LOCK_GUARD_TYPED(nid, ZERO_R1_F)

#define NEURON_LOCK_GUARD_INT(nid) NEURON_LOCK_GUARD_TYPED(nid, 0U)

#define NEURON_LOCK_GUARD_AFN(nid) NEURON_LOCK_GUARD_TYPED(nid, QNeuronActivationFn::Sigmoid)

#if CPP_STD > 13
#define CIRCUIT_LOCK_GUARD(circuit)                                                                                    \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, circuitMutexes[circuit.get()]);                                                  \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        circuitLock =                                                                                                  \
            std::make_unique<const std::lock_guard<std::mutex>>(circuitMutexes[circuit.get()], std::adopt_lock);       \
    }
#else
#define CIRCUIT_LOCK_GUARD(circuit)                                                                                    \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, circuitMutexes[circuit.get()]);                                                  \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        circuitLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                              \
            new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()], std::adopt_lock));                    \
    }
#endif

#define CIRCUIT_LOCK_GUARD_TYPED(cid, def)                                                                             \
    if (cid > circuits.size()) {                                                                                       \
        throw std::invalid_argument("Invalid argument: circuit ID not found!");                                        \
    }                                                                                                                  \
                                                                                                                       \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    CIRCUIT_LOCK_GUARD(circuit)                                                                                        \
    if (!circuit) {                                                                                                    \
        return def;                                                                                                    \
    }

#define CIRCUIT_LOCK_GUARD_VOID(cid)                                                                                   \
    if (cid > circuits.size()) {                                                                                       \
        throw std::invalid_argument("Invalid argument: neuron ID not found!");                                         \
    }                                                                                                                  \
                                                                                                                       \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    CIRCUIT_LOCK_GUARD(circuit)                                                                                        \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }

#define CIRCUIT_LOCK_GUARD_INT(cid) CIRCUIT_LOCK_GUARD_TYPED(cid, 0U)

#if CPP_STD > 13
#define CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)                                                                \
    if (sid > simulators.size()) {                                                                                     \
        throw std::invalid_argument("Invalid argument: simulator ID not found!");                                      \
    }                                                                                                                  \
    if (cid > circuits.size()) {                                                                                       \
        throw std::invalid_argument("Invalid argument: neuron ID not found!");                                         \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator.get()], circuitMutexes[circuit.get()]);               \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock =                                                                                                \
            std::make_unique<const std::lock_guard<std::mutex>>(simulatorMutexes[simulator.get()], std::adopt_lock);   \
        circuitLock =                                                                                                  \
            std::make_unique<const std::lock_guard<std::mutex>>(circuitMutexes[circuit.get()], std::adopt_lock);       \
    }                                                                                                                  \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }
#else
#define CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)                                                                \
    if (sid > simulators.size()) {                                                                                     \
        throw std::invalid_argument("Invalid argument: simulator ID not found!");                                      \
    }                                                                                                                  \
    if (cid > circuits.size()) {                                                                                       \
        throw std::invalid_argument("Invalid argument: neuron ID not found!");                                         \
    }                                                                                                                  \
                                                                                                                       \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    QCircuitPtr circuit = circuits[cid];                                                                               \
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock;                                                  \
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock;                                                    \
    if (true) {                                                                                                        \
        std::lock(metaOperationMutex, simulatorMutexes[simulator.get()], circuitMutexes[circuit.get()]);               \
        const std::lock_guard<std::mutex> metaLock(metaOperationMutex, std::adopt_lock);                               \
        simulatorLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()], std::adopt_lock));                \
        circuitLock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                              \
            new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()], std::adopt_lock));                    \
    }                                                                                                                  \
    if (!simulator) {                                                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    if (!circuit) {                                                                                                    \
        return;                                                                                                        \
    }
#endif

#define QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

qrack_rand_gen_ptr randNumGen = std::make_shared<qrack_rand_gen>(time(0));
std::mutex metaOperationMutex;
std::vector<QInterfacePtr> simulators;
std::vector<std::vector<QInterfaceEngine>> simulatorTypes;
std::vector<bool> simulatorHostPointer;
std::map<QInterface*, std::mutex> simulatorMutexes;
std::vector<bool> simulatorReservations;
std::map<QInterface*, std::map<quid, bitLenInt>> shards;
std::vector<int> neuronErrors;
std::vector<QNeuronPtr> neurons;
std::map<QNeuronPtr, QInterface*> neuronSimulators;
std::map<QNeuron*, std::mutex> neuronMutexes;
std::vector<bool> neuronReservations;
std::vector<QCircuitPtr> circuits;
std::map<QCircuit*, std::mutex> circuitMutexes;
std::vector<bool> circuitReservations;

bitLenInt GetSimShardId(QInterfacePtr simulator, bitLenInt i)
{
    const auto& simShardsIt = shards.find(simulator.get());
    if (simShardsIt == shards.end()) {
        throw std::invalid_argument("Invalid argument: simulator shards not found!");
    }

    const auto& shardIt = simShardsIt->second.find(i);
    if (shardIt == simShardsIt->second.end()) {
        throw std::invalid_argument("Invalid argument: qubit ID not found!");
    }

    return shardIt->second;
}

void FillSimShards(QInterfacePtr simulator)
{
    shards[simulator.get()] = {};
    std::map<quid, bitLenInt>& s = shards[simulator.get()];
    for (quid i = 0U; i < simulator->GetQubitCount(); ++i) {
        s[i] = (bitLenInt)i;
    }
}

void TransformPauliBasis(QInterfacePtr simulator, std::vector<QubitPauliBasis> qb)
{

    for (size_t i = 0U; i < qb.size(); ++i) {
        switch (qb[i].b) {
        case PauliX:
            simulator->H(GetSimShardId(simulator, qb[i].qid));
            break;
        case PauliY:
            simulator->IS(GetSimShardId(simulator, qb[i].qid));
            simulator->H(GetSimShardId(simulator, qb[i].qid));
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void RevertPauliBasis(QInterfacePtr simulator, std::vector<QubitPauliBasis> qb)
{
    for (size_t i = 0U; i < qb.size(); ++i) {
        switch (qb[i].b) {
        case PauliX:
            simulator->H(GetSimShardId(simulator, qb[i].qid));
            break;
        case PauliY:
            simulator->H(GetSimShardId(simulator, qb[i].qid));
            simulator->S(GetSimShardId(simulator, qb[i].qid));
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void removeIdentities(std::vector<QubitPauliBasis>* qb)
{
    size_t i = 0U;
    while (i != qb->size()) {
        if ((*qb)[i].b == PauliI) {
            qb->erase(qb->begin() + i);
        } else {
            ++i;
        }
    }
}

void RHelper(quid sid, real1_f phi, QubitPauliBasis qb)
{
    QInterfacePtr simulator = simulators[sid];

    switch (qb.b) {
    case PauliI: {
        // This is a global phase factor, with no measurable physical effect.
        // However, the underlying QInterface will not execute the gate
        // UNLESS it is specifically "keeping book" for non-measurable phase effects.
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->Phase(phaseFac, phaseFac, GetSimShardId(simulator, qb.qid));
        break;
    }
    case PauliX:
        simulator->RX((real1_f)phi, GetSimShardId(simulator, qb.qid));
        break;
    case PauliY:
        simulator->RY((real1_f)phi, GetSimShardId(simulator, qb.qid));
        break;
    case PauliZ:
        simulator->RZ((real1_f)phi, GetSimShardId(simulator, qb.qid));
        break;
    default:
        break;
    }
}

void MCRHelper(quid sid, real1_f phi, std::vector<bitLenInt> c, QubitPauliBasis qb)
{
    QInterfacePtr simulator = simulators[sid];
    for (size_t i = 0U; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }

    if (qb.b == PauliI) {
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->MCPhase(c, phaseFac, phaseFac, GetSimShardId(simulator, qb.qid));

        return;
    }

    real1 cosine = (real1)cos(phi / 2);
    real1 sine = (real1)sin(phi / 2);
    complex pauliR[4U];

    switch (qb.b) {
    case PauliX:
        pauliR[0U] = complex(cosine, ZERO_R1);
        pauliR[1U] = complex(ZERO_R1, -sine);
        pauliR[2U] = complex(ZERO_R1, -sine);
        pauliR[3U] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(c, pauliR, GetSimShardId(simulator, qb.qid));
        break;
    case PauliY:
        pauliR[0U] = complex(cosine, ZERO_R1);
        pauliR[1U] = complex(-sine, ZERO_R1);
        pauliR[2U] = complex(sine, ZERO_R1);
        pauliR[3U] = complex(cosine, ZERO_R1);
        simulator->MCMtrx(c, pauliR, GetSimShardId(simulator, qb.qid));
        break;
    case PauliZ:
        simulator->MCPhase(c, complex(cosine, -sine), complex(cosine, sine), GetSimShardId(simulator, qb.qid));
        break;
    case PauliI:
    default:
        break;
    }
}

inline size_t make_mask(std::vector<QubitPauliBasis> const& qs)
{
    size_t mask = 0U;
    for (const QubitPauliBasis& q : qs) {
        mask = mask | pow2Ocl(q.qid);
    }
    return mask;
}

std::map<quid, bitLenInt>::iterator FindShardValue(bitLenInt v, std::map<quid, bitLenInt>& simMap)
{
    for (auto it = simMap.begin(); it != simMap.end(); ++it) {
        if (it->second == v) {
            // We have the matching it1, if we break.
            return it;
        }
    }

    return simMap.end();
}

void SwapShardValues(bitLenInt v1, bitLenInt v2, std::map<quid, bitLenInt>& simMap)
{
    auto it1 = FindShardValue(v1, simMap);
    auto it2 = FindShardValue(v2, simMap);
    std::swap(it1->second, it2->second);
}

bitLenInt MapArithmetic(QInterfacePtr simulator, std::vector<bitLenInt> q)
{
    bitLenInt start = GetSimShardId(simulator, q[0U]);
    std::unique_ptr<bitLenInt[]> bitArray(new bitLenInt[q.size()]);
    for (size_t i = 0U; i < q.size(); ++i) {
        bitArray[i] = GetSimShardId(simulator, q[i]);
        if (start > bitArray[i]) {
            start = bitArray[i];
        }
    }
    for (size_t i = 0U; i < q.size(); ++i) {
        simulator->Swap(start + i, bitArray[i]);
        SwapShardValues(start + i, bitArray[i], shards[simulator.get()]);
    }
    return start;
}

struct MapArithmeticResult2 {
    bitLenInt start1;
    bitLenInt start2;

    MapArithmeticResult2(bitLenInt s1, bitLenInt s2)
        : start1(s1)
        , start2(s2)
    {
    }
};

MapArithmeticResult2 MapArithmetic2(QInterfacePtr simulator, std::vector<bitLenInt> q1, std::vector<bitLenInt> q2)
{
    bitLenInt start1 = GetSimShardId(simulator, q1[0U]);
    bitLenInt start2 = GetSimShardId(simulator, q2[0U]);
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[q1.size()]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[q1.size()]);
    for (size_t i = 0U; i < q1.size(); ++i) {
        bitArray1[i] = GetSimShardId(simulator, q1[i]);
        if (start1 > bitArray1[i]) {
            start1 = bitArray1[i];
        }

        bitArray2[i] = GetSimShardId(simulator, q2[i]);
        if (start2 > bitArray2[i]) {
            start2 = bitArray2[i];
        }
    }
    bool isReversed = (start2 < start1);
    if (isReversed) {
        std::swap(start1, start2);
        bitArray1.swap(bitArray2);
    }
    for (size_t i = 0U; i < q1.size(); ++i) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }
    if ((start1 + q1.size()) > start2) {
        start2 = start1 + q1.size();
    }
    for (size_t i = 0U; i < q1.size(); ++i) {
        simulator->Swap(start2 + i, bitArray2[i]);
        SwapShardValues(start2 + i, bitArray2[i], shards[simulator.get()]);
    }
    if (isReversed) {
        std::swap(start1, start2);
    }
    return MapArithmeticResult2(start1, start2);
}

MapArithmeticResult2 MapArithmetic3(QInterfacePtr simulator, std::vector<bitLenInt> q1, std::vector<bitLenInt> q2)
{
    bitLenInt start1 = GetSimShardId(simulator, q1[0U]);
    bitLenInt start2 = GetSimShardId(simulator, q2[0U]);
    std::unique_ptr<bitLenInt[]> bitArray1(new bitLenInt[q1.size()]);
    std::unique_ptr<bitLenInt[]> bitArray2(new bitLenInt[q2.size()]);
    for (size_t i = 0U; i < q1.size(); ++i) {
        bitArray1[i] = GetSimShardId(simulator, q1[i]);
        if (start1 > bitArray1[i]) {
            start1 = bitArray1[i];
        }
    }

    for (size_t i = 0U; i < q2.size(); ++i) {
        bitArray2[i] = GetSimShardId(simulator, q2[i]);
        if (start2 > bitArray2[i]) {
            start2 = bitArray2[i];
        }
    }

    bitLenInt n1 = q1.size();
    bitLenInt n2 = q2.size();
    bool isReversed = (start2 < start1);

    if (isReversed) {
        std::swap(start1, start2);
        std::swap(n1, n2);
        bitArray1.swap(bitArray2);
    }

    for (bitLenInt i = 0U; i < n1; ++i) {
        simulator->Swap(start1 + i, bitArray1[i]);
        SwapShardValues(start1 + i, bitArray1[i], shards[simulator.get()]);
    }

    if ((start1 + n1) > start2) {
        start2 = start1 + n1;
    }

    for (bitLenInt i = 0U; i < n2; ++i) {
        simulator->Swap(start2 + i, bitArray2[i]);
        SwapShardValues(start2 + i, bitArray2[i], shards[simulator.get()]);
    }

    if (isReversed) {
        std::swap(start1, start2);
    }

    return MapArithmeticResult2(start1, start2);
}

#if ENABLE_OPENCL
#define DEVICE_COUNT (OCLEngine::Instance().GetDeviceCount())
#elif ENABLE_CUDA
#define DEVICE_COUNT (CUDAEngine::Instance().GetDeviceCount())
#endif

/**
 * (External API) Initialize a simulator ID with "q" qubits and explicit layer options on/off
 */
quid init_count_type(
    bitLenInt q, bool tn, bool md, bool sd, bool sh, bool bdt, bool pg, bool nw, bool hy, bool oc, bool hp)
{
    META_LOCK_GUARD()

    quid sid = (quid)simulators.size();

    for (size_t i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

#if ENABLE_OPENCL || ENABLE_CUDA
    bool isOcl = oc && (DEVICE_COUNT > 0);
    bool isOclMulti = oc && md && (DEVICE_COUNT > 1);
#else
    bool isOcl = false;
    bool isOclMulti = false;
#endif

    // Construct backwards, then reverse:
    std::vector<QInterfaceEngine> simulatorType;

    if (isOcl) {
        simulatorType.push_back(hy ? QINTERFACE_HYBRID : QINTERFACE_OPENCL);
    } else {
        simulatorType.push_back(QINTERFACE_CPU);
    }

    if (pg && isOcl && !hy) {
        simulatorType.push_back(QINTERFACE_QPAGER);
    }

    if (bdt) {
        simulatorType.push_back(QINTERFACE_BDT);
    }

    if (sh) {
        simulatorType.push_back(QINTERFACE_STABILIZER_HYBRID);
    }

    if (sd) {
        simulatorType.push_back(isOclMulti ? QINTERFACE_QUNIT_MULTI : QINTERFACE_QUNIT);
    }

    if (tn) {
        simulatorType.push_back(QINTERFACE_TENSOR_NETWORK);
    }

    if (nw) {
        simulatorType.push_back(QINTERFACE_NOISY);
    }

    // (...then reverse:)
    std::reverse(simulatorType.begin(), simulatorType.end());

    QInterfacePtr simulator{ nullptr };
    if (q) {
        simulator = CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(hp);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = hp;
    }

    if (!q) {
        return sid;
    }

    FillSimShards(simulator);

    return sid;
}

quid init() { return init_count(0, false); }

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
quid init_count(bitLenInt q, bool hp)
{
    META_LOCK_GUARD()

    quid sid = (quid)simulators.size();

    for (size_t i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    const std::vector<QInterfaceEngine> simulatorType{ QINTERFACE_TENSOR_NETWORK, QINTERFACE_QUNIT, QINTERFACE_HYBRID };

    QInterfacePtr simulator{ nullptr };
    if (q) {
        simulator = CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, hp);
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(hp);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = hp;
    }

    if (!q) {
        return sid;
    }

    FillSimShards(simulator);

    return sid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits as purely a stabilizer simulator.
 */
quid init_count_stabilizer(bitLenInt q)
{
    META_LOCK_GUARD()

    quid sid = (quid)simulators.size();

    for (size_t i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    const std::vector<QInterfaceEngine> simulatorType{ QINTERFACE_STABILIZER };

    QInterfacePtr simulator{ nullptr };
    if (q) {
        simulator =
            CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, false);
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(false);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = false;
    }

    if (!q) {
        return sid;
    }

    FillSimShards(simulator);

    return sid;
}

/**
 * (External API) Initialize a simulator ID that clones simulator ID "sid"
 */
quid init_clone(quid sid)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }
    QInterfacePtr oSimulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[oSimulator.get()]));

    quid nsid = (quid)simulators.size();

    for (size_t i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            nsid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    QInterfacePtr simulator = oSimulator->Clone();

    if (nsid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorTypes[sid]);
        simulatorHostPointer.push_back(simulatorHostPointer[sid]);
    } else {
        simulatorReservations[nsid] = true;
        simulators[nsid] = simulator;
        simulatorTypes[nsid] = simulatorTypes[sid];
        simulatorHostPointer[nsid] = simulatorHostPointer[sid];
    }

    FillSimShards(simulator);

    return nsid;
}

quid init_clone_size(quid sid, bitLenInt n)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        throw std::invalid_argument("Simulator ID not found!");
    }
    QInterfacePtr oSimulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[oSimulator.get()]));

    quid nsid = (quid)simulators.size();

    for (quid i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            nsid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    QInterfacePtr simulator = CreateQuantumInterface(
        simulatorTypes[sid], n, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, simulatorHostPointer[sid]);

    if (nsid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorTypes[sid]);
        simulatorHostPointer.push_back(simulatorHostPointer[sid]);
    } else {
        simulatorReservations[nsid] = true;
        simulators[nsid] = simulator;
        simulatorTypes[nsid] = simulatorTypes[sid];
        simulatorHostPointer[nsid] = simulatorHostPointer[sid];
    }

    FillSimShards(simulator);

    return nsid;
}

/**
 * (External API) Initialize a simulator ID with "q" qubits and implicit default layer options.
 */
quid init_qbdd_count(bitLenInt q)
{
    META_LOCK_GUARD()

    quid sid = (quid)simulators.size();

    for (size_t i = 0U; i < simulators.size(); ++i) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    const std::vector<QInterfaceEngine> simulatorType{ QINTERFACE_TENSOR_NETWORK, QINTERFACE_QUNIT,
        QINTERFACE_STABILIZER_HYBRID, QINTERFACE_BDT };

    QInterfacePtr simulator{ nullptr };
    if (q) {
        simulator = CreateQuantumInterface(simulatorType, q, ZERO_BCI, randNumGen);
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
        simulatorTypes.push_back(simulatorType);
        simulatorHostPointer.push_back(false);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
        simulatorTypes[sid] = simulatorType;
        simulatorHostPointer[sid] = false;
    }

    if (!q) {
        return sid;
    }

    FillSimShards(simulator);

    return sid;
}

/**
 * (External API) Destroy a simulator (ID will not be reused)
 */
void destroy(quid sid)
{
    META_LOCK_GUARD()

    shards.erase(simulators[sid].get());
    simulatorMutexes.erase(simulators[sid].get());
    simulators[sid] = nullptr;
    simulatorReservations[sid] = false;
}

/**
 * (External API) Set RNG seed for simulator ID
 */
void seed(quid sid, unsigned s)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulators[sid]->SetRandomSeed(s);
}

/**
 * (External API) Set concurrency level per QEngine shard
 */
void set_concurrency(quid sid, unsigned p)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulators[sid]->SetConcurrency(p);
}

/**
 * (External API) Set GPU device ID on the simulator.
 */
void set_device(quid sid, int64_t did)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulators[sid]->SetDevice(did);
}

void qstabilizer_out_to_file(quid sid, std::string f)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    if (simulatorTypes[sid][0] != QINTERFACE_STABILIZER_HYBRID) {
        throw std::invalid_argument("Cannot write any simulator but QStabilizerHybrid out to file!");
    }

    std::ofstream ofile;
    ofile.open(f.c_str());
    ofile.precision(36);
    ofile << std::dynamic_pointer_cast<QStabilizerHybrid>(simulators[sid]);
    ofile.close();
}
void qstabilizer_in_from_file(quid sid, std::string f)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)

    if (simulatorTypes[sid][0] != QINTERFACE_STABILIZER_HYBRID) {
        throw std::invalid_argument("Cannot read any simulator but QStabilizerHybrid in from file!");
    }

    std::ifstream ifile;
    ifile.open(f.c_str());
    ifile.precision(36);
    ifile >> std::dynamic_pointer_cast<QStabilizerHybrid>(simulators[sid]);
    ifile.close();

    FillSimShards(simulator);
}

/**
 * Select from a distribution of "p.size()" count of elements according the discrete probabilities in "p."
 */
size_t random_choice(quid sid, std::vector<real1> p)
{
    std::discrete_distribution<size_t> dist(p.begin(), p.end());
    return dist(*randNumGen.get());
}

void _PhaseMask(quid sid, real1_f lambda, bitLenInt p, std::vector<bitLenInt> q, bool isParity)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    bitCapInt mask = ZERO_BCI;
    for (size_t i = 0U; i < q.size(); ++i) {
        bi_or_ip(&mask, pow2(GetSimShardId(simulator, q[i])));
    }
    if (isParity) {
        simulator->PhaseParity(lambda, mask);
    } else {
        simulator->PhaseRootNMask(p, mask);
    }
}

void PhaseParity(quid sid, real1_f lambda, std::vector<bitLenInt> q) { _PhaseMask(sid, lambda, 0U, q, true); }

void PhaseRootN(quid sid, bitLenInt p, std::vector<bitLenInt> q) { _PhaseMask(sid, 0.0, p, q, false); }

real1_f _JointEnsembleProbabilityHelper(QInterfacePtr simulator, std::vector<QubitPauliBasis> q, bool doMeasure)
{

    if (q.empty()) {
        return 0.0;
    }

    removeIdentities(&q);

    if (q.empty()) {
        return 0.0;
    }

    bitCapInt mask = ZERO_BCI;
    for (size_t i = 0U; i < q.size(); ++i) {
        bi_or_ip(&mask, pow2(GetSimShardId(simulator, q[i].qid)));
    }

    return (real1_f)(doMeasure ? (QPARITY(simulator)->MParity(mask) ? ONE_R1 : ZERO_R1)
                               : QPARITY(simulator)->ProbParity(mask));
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
real1_f JointEnsembleProbability(quid sid, std::vector<QubitPauliBasis> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    real1_f jointProb = (real1_f)REAL1_DEFAULT_ARG;

    TransformPauliBasis(simulator, q);
    jointProb = _JointEnsembleProbabilityHelper(simulator, q, false);
    RevertPauliBasis(simulator, q);

    return jointProb;
}

/**
 * (External API) Set the simulator to a computational basis permutation.
 */
void ResetAll(quid sid)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetPermutation(ZERO_BCI);
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
void allocateQubit(quid sid, bitLenInt qid)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }

    QInterfacePtr nQubit = CreateQuantumInterface(
        simulatorTypes[sid], 1U, ZERO_BCI, randNumGen, CMPLX_DEFAULT_ARG, false, true, simulatorHostPointer[sid]);

    if (!simulators[sid]) {
        simulators[sid] = nQubit;
        shards[nQubit.get()] = {};
        shards[nQubit.get()][qid] = 0;

        return;
    }

    QInterfacePtr oSimulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[oSimulator.get()]));

    oSimulator->Compose(nQubit);
    shards[simulators[sid].get()][qid] = (simulators[sid]->GetQubitCount() - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
bool release(quid sid, bitLenInt q)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()]));

    // Check that the qubit is in the |0> state, to within a small tolerance.
    bool toRet = simulator->Prob(GetSimShardId(simulator, q)) < (ONE_R1 / 100);

    if (simulator->GetQubitCount() == 1U) {
        shards.erase(simulator.get());
        simulators[sid] = nullptr;
    } else {
        std::map<quid, bitLenInt>& simShards = shards[simulator.get()];
        bitLenInt oIndex = simShards[q];
        simulator->Dispose(oIndex, 1U);
        for (auto it = simShards.begin(); it != simShards.end(); ++it) {
            if (it->second > oIndex) {
                --(it->second);
            }
        }
        simShards.erase(q);
    }

    return toRet;
}

bitLenInt num_qubits(quid sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    return simulator->GetQubitCount();
}

void SetPermutation(quid sid, bitCapInt p)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetPermutation(p);
}

/**
 * (External API) "X" Gate
 */
void X(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->X(GetSimShardId(simulator, q));
}

/**
 * (External API) "Y" Gate
 */
void Y(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->Y(GetSimShardId(simulator, q));
}

/**
 * (External API) "Z" Gate
 */
void Z(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->Z(GetSimShardId(simulator, q));
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
void H(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->H(GetSimShardId(simulator, q));
}

/**
 * (External API) "S" Gate
 */
void S(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->S(GetSimShardId(simulator, q));
}

/**
 * (External API) Square root of X gate
 */
void SX(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SqrtX(GetSimShardId(simulator, q));
}

/**
 * (External API) Square root of Y gate
 */
void SY(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SqrtY(GetSimShardId(simulator, q));
}

/**
 * (External API) "T" Gate
 */
void T(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->T(GetSimShardId(simulator, q));
}

/**
 * (External API) Inverse "S" Gate
 */
void AdjS(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->IS(GetSimShardId(simulator, q));
}

/**
 * (External API) Inverse square root of X gate
 */
void AdjSX(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->ISqrtX(GetSimShardId(simulator, q));
}

/**
 * (External API) Inverse square root of Y gate
 */
void AdjSY(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->ISqrtY(GetSimShardId(simulator, q));
}

/**
 * (External API) Inverse "T" Gate
 */
void AdjT(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->IT(GetSimShardId(simulator, q));
}

/**
 * (External API) 3-parameter unitary gate
 */
void U(quid sid, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->U(GetSimShardId(simulator, q), (real1_f)theta, (real1_f)phi, (real1_f)lambda);
}

/**
 * (External API) 2x2 complex matrix unitary gate
 */
void Mtrx(quid sid, std::vector<complex> m, bitLenInt q)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "Mtrx() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    complex mtrx[4]{ m[0U], m[1U], m[2U], m[3U] };
    simulator->Mtrx(mtrx, GetSimShardId(simulator, q));
}

#define MAP_CONTROLS_AND_LOCK(sid)                                                                                     \
    SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                     \
    for (size_t i = 0; i < c.size(); ++i) {                                                                            \
        c[i] = GetSimShardId(simulator, c[i]);                                                                         \
    }

/**
 * (External API) Controlled "X" Gate
 */
void MCX(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCInvert(c, ONE_CMPLX, ONE_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled "Y" Gate
 */
void MCY(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCInvert(c, -I_CMPLX, I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled "Z" Gate
 */
void MCZ(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCPhase(c, ONE_CMPLX, -ONE_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled "H" Gate
 */
void MCH(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    const complex hGate[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCMtrx(c, hGate, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled "S" Gate
 */
void MCS(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCPhase(c, ONE_CMPLX, I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled "T" Gate
 */
void MCT(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCPhase(c, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
void MCAdjS(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCPhase(c, ONE_CMPLX, -I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
void MCAdjT(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCPhase(c, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
void MCU(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->CU(c, GetSimShardId(simulator, q), (real1_f)theta, (real1_f)phi, (real1_f)lambda);
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
void MCMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "MCMtrx() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    complex mtrx[4]{ m[0U], m[1U], m[2U], m[3U] };
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MCMtrx(c, mtrx, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "X" Gate
 */
void MACX(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACInvert(c, ONE_CMPLX, ONE_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "Y" Gate
 */
void MACY(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACInvert(c, -I_CMPLX, I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "Z" Gate
 */
void MACZ(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACPhase(c, ONE_CMPLX, -ONE_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "H" Gate
 */
void MACH(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    const complex hGate[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACMtrx(c, hGate, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "S" Gate
 */
void MACS(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACPhase(c, ONE_CMPLX, I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled "T" Gate
 */
void MACT(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACPhase(c, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled Inverse "S" Gate
 */
void MACAdjS(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACPhase(c, ONE_CMPLX, -I_CMPLX, GetSimShardId(simulator, q));
}

/**
 * (External API) "Anti-"Controlled Inverse "T" Gate
 */
void MACAdjT(quid sid, std::vector<bitLenInt> c, bitLenInt q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACPhase(c, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
void MACU(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->AntiCU(c, GetSimShardId(simulator, q), (real1_f)theta, (real1_f)phi, (real1_f)lambda);
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
void MACMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "Mtrx() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    complex mtrx[4]{ m[0U], m[1U], m[2U], m[3U] };
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->MACMtrx(c, mtrx, GetSimShardId(simulator, q));
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate with arbitrary control permutation
 */
void UCMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q, bitCapIntOcl p)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "Mtrx() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    complex mtrx[4]{ m[0U], m[1U], m[2U], m[3U] };
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->UCMtrx(c, mtrx, GetSimShardId(simulator, q), p);
}

void Multiplex1Mtrx(quid sid, std::vector<bitLenInt> c, bitLenInt q, std::vector<complex> m)
{
    std::unique_ptr<complex[]> mtrxs(new complex[m.size()]);
    std::copy(m.begin(), m.end(), mtrxs.get());

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->UniformlyControlledSingleBit(c, GetSimShardId(simulator, q), mtrxs.get());
}

#define MAP_MASK_AND_LOCK(sid)                                                                                         \
    SIMULATOR_LOCK_GUARD_VOID(sid)                                                                                     \
    bitCapInt mask = ZERO_BCI;                                                                                         \
    for (size_t i = 0U; i < q.size(); ++i) {                                                                           \
        bi_or_ip(&mask, pow2(GetSimShardId(simulator, q[i])));                                                         \
    }

/**
 * (External API) Multiple "X" Gate
 */
void MX(quid sid, std::vector<bitLenInt> q)
{
    MAP_MASK_AND_LOCK(sid)
    simulator->XMask(mask);
}

/**
 * (External API) Multiple "Y" Gate
 */
void MY(quid sid, std::vector<bitLenInt> q)
{
    MAP_MASK_AND_LOCK(sid)
    simulator->YMask(mask);
}

/**
 * (External API) Multiple "Z" Gate
 */
void MZ(quid sid, std::vector<bitLenInt> q)
{
    MAP_MASK_AND_LOCK(sid)
    simulator->ZMask(mask);
}

/**
 * (External API) Rotation around Pauli axes
 */
void R(quid sid, real1_f phi, QubitPauliBasis q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    RHelper(sid, phi, q);
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
void MCR(quid sid, real1_f phi, std::vector<bitLenInt> c, QubitPauliBasis q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    MCRHelper(sid, phi, c, q);
}

/**
 * (External API) Exponentiation of Pauli operators
 */
void Exp(quid sid, real1_f phi, std::vector<QubitPauliBasis> q)
{
    if (q.empty()) {
        return;
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)

    QubitPauliBasis someQubit(q.front().qid, PauliI);

    removeIdentities(&q);

    if (q.empty()) {
        RHelper(sid, -2 * phi, someQubit);
    } else if (q.size() == 1U) {
        RHelper(sid, -2 * phi, q.front());
    } else {
        TransformPauliBasis(simulator, q);
        QPARITY(simulator)->UniformParityRZ(make_mask(q), -phi);
        RevertPauliBasis(simulator, q);
    }
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
void MCExp(quid sid, real1_f phi, std::vector<bitLenInt> cs, std::vector<QubitPauliBasis> q)
{
    if (q.empty()) {
        return;
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)

    QubitPauliBasis someQubit(q.front().qid, PauliI);

    removeIdentities(&q);

    if (q.empty()) {
        MCRHelper(sid, -2 * phi, cs, someQubit);
    } else if (q.size() == 1U) {
        MCRHelper(sid, -2 * phi, cs, q.front());
    } else {
        TransformPauliBasis(simulator, q);
        QPARITY(simulator)->CUniformParityRZ(cs, make_mask(q), -phi);
        RevertPauliBasis(simulator, q);
    }
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
bool M(quid sid, bitLenInt q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    return simulator->M(GetSimShardId(simulator, q));
}

/**
 * (External API) PSEUDO-QUANTUM: Post-select bit in |0>/|1> basis
 */
bool ForceM(quid sid, bitLenInt q, bool r)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    return simulator->ForceM(GetSimShardId(simulator, q), r);
}

/**
 * (External API) Measure all bits separately in |0>/|1> basis, and return the result in low-to-high order corresponding
 * with first-to-last in original order of allocation.
 */
bitCapInt MAll(quid sid)
{
    SIMULATOR_LOCK_GUARD_INT(sid)
    return simulators[sid]->MAll();
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
bool Measure(quid sid, std::vector<QubitPauliBasis> q)
{
    SIMULATOR_LOCK_GUARD_INT(sid)

    TransformPauliBasis(simulator, q);
    const bool toRet = HALF_R1 <= _JointEnsembleProbabilityHelper(simulator, q, true);
    RevertPauliBasis(simulator, q);

    return toRet;
}

std::vector<long long unsigned int> MeasureShots(quid sid, std::vector<bitLenInt> q, unsigned s)
{
    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }
    QInterfacePtr simulator = simulators[sid];
    SIMULATOR_LOCK_GUARD(simulator.get())
    if (!simulator) {
        return std::vector<long long unsigned int>();
    }

    std::vector<bitCapInt> qPowers;
    qPowers.reserve(q.size());
    for (size_t i = 0U; i < q.size(); ++i) {
        qPowers.push_back(pow2(GetSimShardId(simulator, q[i])));
    }

    std::unique_ptr<long long unsigned int> m(new long long unsigned int[s]);

    simulator->MultiShotMeasureMask(qPowers, s, m.get());
    std::vector<long long unsigned int> toRet(s);
    std::copy(m.get(), m.get() + s, toRet.begin());

    return toRet;
}

void SWAP(quid sid, bitLenInt qi1, bitLenInt qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->Swap(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void ISWAP(quid sid, bitLenInt qi1, bitLenInt qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->ISwap(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void AdjISWAP(quid sid, bitLenInt qi1, bitLenInt qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->IISwap(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void FSim(quid sid, real1_f theta, real1_f phi, bitLenInt qi1, bitLenInt qi2)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->FSim((real1_f)theta, (real1_f)phi, GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void CSWAP(quid sid, std::vector<bitLenInt> c, bitLenInt qi1, bitLenInt qi2)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->CSwap(c, GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void ACSWAP(quid sid, std::vector<bitLenInt> c, bitLenInt qi1, bitLenInt qi2)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->AntiCSwap(c, GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

void Compose(quid sid1, quid sid2, std::vector<bitLenInt> q)
{
    if (!simulators[sid1] || !simulators[sid2]) {
        return;
    }
    const std::lock_guard<std::mutex> simulatorLock1(simulatorMutexes[simulators[sid1].get()]);
    const std::lock_guard<std::mutex> simulatorLock2(simulatorMutexes[simulators[sid2].get()]);

    if (simulatorTypes[sid1].size() != simulatorTypes[sid2].size()) {
        throw std::invalid_argument("Cannot 'Compose()' simulators of different layer stack types!");
    }

    for (size_t i = 0U; i < simulatorTypes[sid1].size(); ++i) {
        if (simulatorTypes[sid1][i] != simulatorTypes[sid2][i]) {
            throw std::invalid_argument("Cannot 'Compose()' simulators of different layer stack types!");
        }
    }

    const QInterfacePtr simulator1 = simulators[sid1];
    const QInterfacePtr simulator2 = simulators[sid2];
    const bitLenInt oQubitCount = simulator1->GetQubitCount();
    const bitLenInt pQubitCount = simulator2->GetQubitCount();
    simulator1->Compose(simulator2);

    std::map<quid, bitLenInt>& s = shards[simulator1.get()];
    for (bitLenInt i = 0; i < pQubitCount; ++i) {
        s[q[i]] = oQubitCount + i;
    }
}

quid Decompose(quid sid, std::vector<bitLenInt> q)
{
    quid nSid = init_clone_size(sid, q.size());

    SIMULATOR_LOCK_GUARD_INT(sid)

    const bitLenInt nQubitIndex = simulator->GetQubitCount() - q.size();
    for (size_t i = 0U; i < q.size(); ++i) {
        simulator->Swap(GetSimShardId(simulator, q[i]), i + nQubitIndex);
    }
    simulator->Decompose(nQubitIndex, simulators[nSid]);

    for (size_t j = 0U; j < q.size(); ++j) {
        std::map<quid, bitLenInt>& simShards = shards[simulator.get()];
        bitLenInt oIndex = simShards[q[j]];
        for (auto it = simShards.begin(); it != simShards.end(); ++it) {
            if (it->second > oIndex) {
                --(it->second);
            }
        }
        simShards.erase(q[j]);
    }

    return nSid;
}

void Dispose(quid sid, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    const bitLenInt nQubitIndex = simulator->GetQubitCount() - q.size();
    for (size_t i = 0U; i < q.size(); ++i) {
        simulator->Swap(GetSimShardId(simulator, q[i]), i + nQubitIndex);
    }
    simulator->Dispose(nQubitIndex, q.size());
    for (size_t j = 0U; j < q.size(); ++j) {
        std::map<quid, bitLenInt>& simShards = shards[simulator.get()];
        bitLenInt oIndex = simShards[q[j]];
        for (auto it = simShards.begin(); it != simShards.end(); ++it) {
            if (it->second > oIndex) {
                --(it->second);
            }
        }
        simShards.erase(q[j]);
    }
}

void AND(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->AND(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void OR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->OR(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void XOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->XOR(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void NAND(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->NAND(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void NOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->NOR(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void XNOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->XNOR(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2), GetSimShardId(simulator, qo));
}

void CLAND(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLAND(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

void CLOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLOR(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

void CLXOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLXOR(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

void CLNAND(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLNAND(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

void CLNOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLNOR(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

void CLXNOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->CLXNOR(ci, GetSimShardId(simulator, qi), GetSimShardId(simulator, qo));
}

std::vector<real1> ProbAll(quid sid, std::vector<bitLenInt> q)
{
    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }
    QInterfacePtr simulator = simulators[sid];
    SIMULATOR_LOCK_GUARD(simulator.get())
    if (!simulator) {
        return std::vector<real1>();
    }

    for (size_t i = 0U; i < q.size(); ++i) {
        q[i] = GetSimShardId(simulator, q[i]);
    }

    bool isOutProbs = false;
    if (q.size() == simulator->GetQubitCount()) {
        isOutProbs = true;
        for (size_t i = 0U; i < q.size(); ++i) {
            if (q[i] == i) {
                continue;
            }
            isOutProbs = false;
            break;
        }
    }

    std::vector<real1> p(pow2Ocl(q.size()));
    if (isOutProbs) {
        simulator->GetProbs(&(p[0U]));
    } else {
        simulator->ProbBitsAll(q, &(p[0U]));
    }

    return p;
}

real1_f _Prob(quid sid, bitLenInt q, bool isRdm)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)
    return isRdm ? simulator->ProbRdm(GetSimShardId(simulator, q)) : simulator->Prob(GetSimShardId(simulator, q));
}

/**
 * (External API) Get the probability that a qubit is in the |1> state.
 */
real1_f Prob(quid sid, bitLenInt q) { return _Prob(sid, q, false); }

/**
 * (External API) Get the probability that a qubit is in the |1> state, treating all ancillary qubits as post-selected T
 * gate gadgets.
 */
real1_f ProbRdm(quid sid, bitLenInt q) { return _Prob(sid, q, true); }

real1_f _PermutationProb(quid sid, std::vector<QubitIndexState> q, bool isRdm, bool r)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    bitCapInt mask = ZERO_BCI;
    bitCapInt perm = ZERO_BCI;
    for (size_t i = 0U; i < q.size(); ++i) {
        const bitCapInt p = pow2(GetSimShardId(simulator, q[i].qid));
        bi_or_ip(&mask, p);
        if (q[i].val) {
            bi_or_ip(&perm, p);
        }
    }

    return isRdm ? simulator->ProbMaskRdm(r, mask, perm) : simulator->ProbMask(mask, perm);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
real1_f PermutationProb(quid sid, std::vector<QubitIndexState> q) { return _PermutationProb(sid, q, false, false); }

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f PermutationProbRdm(quid sid, std::vector<QubitIndexState> q, bool r)
{
    return _PermutationProb(sid, q, true, r);
}

real1_f _PermutationExpVar(quid sid, std::vector<bitLenInt> q, bool r, bool isRdm, bool isExp)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    for (size_t i = 0U; i < q.size(); ++i) {
        q[i] = GetSimShardId(simulator, q[i]);
    }

    return isExp ? isRdm ? simulator->ExpectationBitsAllRdm(r, q) : simulator->ExpectationBitsAll(q)
        : isRdm ? simulator->VarianceBitsAllRdm(r, q)
                 : simulator->VarianceBitsAll(q);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
real1_f PermutationExpectation(quid sid, std::vector<bitLenInt> q)
{
    return _PermutationExpVar(sid, q, false, false, true);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f PermutationExpectationRdm(quid sid, std::vector<bitLenInt> q, bool r)
{
    return _PermutationExpVar(sid, q, r, true, true);
}

/**
 * (External API) Get the permutation variance, based upon the order of input qubits.
 */
real1_f Variance(quid sid, std::vector<bitLenInt> q) { return _PermutationExpVar(sid, q, false, false, false); }

/**
 * (External API) Get the permutation variance, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f VarianceRdm(quid sid, std::vector<bitLenInt> q, bool r) { return _PermutationExpVar(sid, q, r, true, false); }

real1_f FactorizedExpVar(bool isExp, bool isRdm, quid sid, std::vector<QubitIntegerExpVar> q, bool r)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<bitCapInt> _c;
    _q.reserve(q.size());
    _c.reserve(q.size());
    for (size_t i = 0U; i < q.size(); ++i) {
        _q.push_back(GetSimShardId(simulator, q[i].qid));
        _c.push_back(q[i].val);
    }

    return isExp
        ? isRdm ? simulator->ExpectationBitsFactorizedRdm(r, _q, _c) : simulator->ExpectationBitsFactorized(_q, _c)
        : isRdm ? simulator->VarianceBitsFactorizedRdm(r, _q, _c)
                : simulator->VarianceBitsFactorized(_q, _c);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
real1_f FactorizedExpectation(quid sid, std::vector<QubitIntegerExpVar> q)
{
    return FactorizedExpVar(true, false, sid, q, false);
}

/**
 * (External API) Get the permutation variance, based upon the order of input qubits.
 */
real1_f FactorizedVariance(quid sid, std::vector<QubitIntegerExpVar> q)
{
    return FactorizedExpVar(false, false, sid, q, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f FactorizedExpectationRdm(quid sid, std::vector<QubitIntegerExpVar> q, bool r)
{
    return FactorizedExpVar(true, true, sid, q, r);
}

/**
 * (External API) Get the permutation variance, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f FactorizedVarianceRdm(quid sid, std::vector<QubitIntegerExpVar> q, bool r)
{
    return FactorizedExpVar(false, true, sid, q, r);
}

real1_f FactorizedExpVarFp(bool isExp, bool isRdm, quid sid, std::vector<QubitRealExpVar> q, bool r)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<real1_f> _f;
    _q.reserve(q.size());
    _f.reserve(q.size());
    for (size_t i = 0U; i < q.size(); ++i) {
        _q.push_back(GetSimShardId(simulator, q[i].qid));
        _f.push_back(q[i].val);
    }

    return isExp
        ? isRdm ? simulator->ExpectationFloatsFactorizedRdm(r, _q, _f) : simulator->ExpectationFloatsFactorized(_q, _f)
        : isRdm ? simulator->VarianceFloatsFactorizedRdm(r, _q, _f)
                : simulator->VarianceFloatsFactorized(_q, _f);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
real1_f FactorizedExpectationFp(quid sid, std::vector<QubitRealExpVar> q)
{
    return FactorizedExpVarFp(true, false, sid, q, false);
}

/**
 * (External API) Get the permutation variance, based upon the order of input qubits.
 */
real1_f FactorizedVarianceFp(quid sid, std::vector<QubitRealExpVar> q)
{
    return FactorizedExpVarFp(false, false, sid, q, false);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f FactorizedExpectationFpRdm(quid sid, std::vector<QubitRealExpVar> q, bool r)
{
    return FactorizedExpVarFp(true, true, sid, q, r);
}

/**
 * (External API) Get the permutation variance, based upon the order of input qubits, treating all ancillary
 * qubits as post-selected T gate gadgets.
 */
real1_f FactorizedVarianceFpRdm(quid sid, std::vector<QubitRealExpVar> q, bool r)
{
    return FactorizedExpVarFp(false, true, sid, q, r);
}

real1_f UnitaryExpVar(bool isExp, quid sid, std::vector<QubitU3Basis> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<real1> _b;
    const size_t n = q.size();
    _q.reserve(n);
    _b.reserve(3U * n);
    for (size_t i = 0U; i < n; ++i) {
        const QubitU3Basis& qub = q[i];
        _q.emplace_back(GetSimShardId(simulator, qub.qid));
        const size_t i3 = 3U * i;
        for (size_t j = 0U; j < 3U; ++j) {
            _b.emplace_back(qub.b[i3 + j]);
        }
    }

    return isExp ? simulator->ExpectationUnitaryAll(_q, _b) : simulator->VarianceUnitaryAll(_q, _b);
}

/**
 * (External API) Get the single-qubit (3-parameter) operator expectation value for the array of qubits and bases.
 */
real1_f UnitaryExpectation(quid sid, std::vector<QubitU3Basis> q) { return UnitaryExpVar(true, sid, q); }

/**
 * (External API) Get the single-qubit (3-parameter) operator variance for the array of qubits and bases.
 */
real1_f UnitaryVariance(quid sid, std::vector<QubitU3Basis> q) { return UnitaryExpVar(false, sid, q); }

real1_f MatrixExpVar(bool isExp, quid sid, std::vector<QubitMatrixBasis> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<std::shared_ptr<complex>> _b;
    const size_t n = q.size();
    _q.reserve(n);
    _b.reserve(n);
    for (size_t i = 0U; i < n; ++i) {
        const QubitMatrixBasis& qmb = q[i];
        _q.emplace_back(GetSimShardId(simulator, qmb.qid));
        const size_t i4 = i << 2U;
        _b.emplace_back(new complex[4U], std::default_delete<complex[]>());
        for (size_t j = 0U; j < 4U; ++j) {
            _b[i].get()[j] = qmb.b[i4 + j];
        }
    }

    return isExp ? simulator->ExpectationUnitaryAll(_q, _b) : simulator->VarianceUnitaryAll(_q, _b);
}

/**
 * (External API) Get the single-qubit (2x2) operator expectation value for the array of qubits and bases.
 */
real1_f MatrixExpectation(quid sid, std::vector<QubitMatrixBasis> q) { return MatrixExpVar(true, sid, q); }

/**
 * (External API) Get the single-qubit (2x2) operator variance for the array of qubits and bases.
 */
real1_f MatrixVariance(quid sid, std::vector<QubitMatrixBasis> q) { return MatrixExpVar(false, sid, q); }

real1_f UnitaryExpVarEigenVal(bool isExp, quid sid, std::vector<QubitU3BasisEigenVal> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<real1> _b;
    std::vector<real1> _e;
    const size_t n = q.size();
    _q.reserve(n);
    _b.reserve(3U * n);
    _e.reserve(n << 1U);
    for (size_t i = 0U; i < n; ++i) {
        const QubitU3BasisEigenVal& qubev = q[i];
        _q.emplace_back(GetSimShardId(simulator, qubev.qid));
        const size_t i3 = 3U * i;
        for (size_t j = 0U; j < 3U; ++j) {
            _b.emplace_back(qubev.b[i3 + j]);
        }
        _e.emplace_back(qubev.e[0U]);
        _e.emplace_back(qubev.e[1U]);
    }

    return isExp ? simulator->ExpectationUnitaryAll(_q, _b, _e) : simulator->VarianceUnitaryAll(_q, _b, _e);
}

/**
 * (External API) Get the single-qubit (3-parameter) operator expectation value for the array of qubits and bases.
 */
real1_f UnitaryExpectationEigenVal(quid sid, std::vector<QubitU3BasisEigenVal> q)
{
    return UnitaryExpVarEigenVal(true, sid, q);
}

/**
 * (External API) Get the single-qubit (3-parameter) operator variance for the array of qubits and bases.
 */
real1_f UnitaryVarianceEigenVal(quid sid, std::vector<QubitU3BasisEigenVal> q)
{
    return UnitaryExpVarEigenVal(false, sid, q);
}

real1_f MatrixExpVarEigenVal(bool isExp, quid sid, std::vector<QubitMatrixBasisEigenVal> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<std::shared_ptr<complex>> _b;
    std::vector<real1> _e;
    const size_t n = q.size();
    _q.reserve(n);
    _b.reserve(n);
    _e.reserve(n << 1U);
    for (size_t i = 0U; i < n; ++i) {
        const QubitMatrixBasisEigenVal& qmbev = q[i];
        _q.emplace_back(GetSimShardId(simulator, qmbev.qid));
        const size_t i4 = i << 2U;
        _b.emplace_back(new complex[4U], std::default_delete<complex[]>());
        for (size_t j = 0U; j < 4U; ++j) {
            _b[i].get()[j] = qmbev.b[i4 + j];
        }
        _e.emplace_back(qmbev.e[0U]);
        _e.emplace_back(qmbev.e[1U]);
    }

    return simulator->ExpectationUnitaryAll(_q, _b, _e);
}

/**
 * (External API) Get the single-qubit (2x2) operator expectation value for the array of qubits and bases.
 */
real1_f MatrixExpectationEigenVal(quid sid, std::vector<QubitMatrixBasisEigenVal> q)
{
    return MatrixExpVarEigenVal(true, sid, q);
}

/**
 * (External API) Get the single-qubit (2x2) operator variance for the array of qubits and bases.
 */
real1_f MatrixVarianceEigenVal(quid sid, std::vector<QubitMatrixBasisEigenVal> q)
{
    return MatrixExpVarEigenVal(false, sid, q);
}

real1_f PauliExpVar(bool isExp, quid sid, std::vector<QubitPauliBasis> q)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)

    std::vector<bitLenInt> _q;
    std::vector<Pauli> _b;
    _q.reserve(q.size());
    _b.reserve(q.size());
    for (const QubitPauliBasis& qpb : q) {
        _q.emplace_back(GetSimShardId(simulator, qpb.qid));
        _b.emplace_back((Pauli)qpb.b);
    }

    return isExp ? simulator->ExpectationPauliAll(_q, _b) : simulator->VariancePauliAll(_q, _b);
}

/**
 * (External API) Get the Pauli operator expectation value for the array of qubits and bases.
 */
real1_f PauliExpectation(quid sid, std::vector<QubitPauliBasis> q) { return PauliExpVar(true, sid, q); }

/**
 * (External API) Get the Pauli operator variance for the array of qubits and bases.
 */
real1_f PauliVariance(quid sid, std::vector<QubitPauliBasis> q) { return PauliExpVar(false, sid, q); }

void QFT(quid sid, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
#if QBCAPPOW < 32
    for (size_t i = 0U; i < q.size(); ++i) {
        q[i] = GetSimShardId(simulator, q[i]);
    }
#endif
    simulator->QFTR(q);
}
void IQFT(quid sid, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
#if QBCAPPOW < 32
    for (size_t i = 0U; i < q.size(); ++i) {
        q[i] = GetSimShardId(simulator, q[i]);
    }
#endif
    simulator->IQFTR(q);
}

#if ENABLE_ALU
void ADD(quid sid, bitCapInt a, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->INC(a, MapArithmetic(simulator, q), q.size());
}
void SUB(quid sid, bitCapInt a, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->DEC(a, MapArithmetic(simulator, q), q.size());
}
void ADDS(quid sid, bitCapInt a, bitLenInt s, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->INCS(a, MapArithmetic(simulator, q), q.size(), GetSimShardId(simulator, s));
}
void SUBS(quid sid, bitCapInt a, bitLenInt s, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->DECS(a, MapArithmetic(simulator, q), q.size(), GetSimShardId(simulator, s));
}

void MCADD(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    simulator->CINC(a, MapArithmetic(simulator, q), q.size(), c);
}
void MCSUB(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    simulator->CDEC(a, MapArithmetic(simulator, q), q.size(), c);
}

void MUL(quid sid, bitCapInt a, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MUL() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    QALU(simulator)->MUL(a, starts.start1, starts.start2, q.size());
}
void DIV(quid sid, bitCapInt a, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("DIV() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    QALU(simulator)->DIV(a, starts.start1, starts.start2, q.size());
}
void MULN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MULN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    simulator->MULModNOut(a, m, starts.start1, starts.start2, q.size());
}
void DIVN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("DIVN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    simulator->IMULModNOut(a, m, starts.start1, starts.start2, q.size());
}
void POWN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("POWN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    QALU(simulator)->POWModNOut(a, m, starts.start1, starts.start2, q.size());
}

void MCMUL(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MCMUL() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    QALU(simulator)->CMUL(a, starts.start1, starts.start2, q.size(), c);
}
void MCDIV(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MCDIV() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    QALU(simulator)->CDIV(a, starts.start1, starts.start2, q.size(), c);
}
void MCMULN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MCMULN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    simulator->CMULModNOut(a, m, starts.start1, starts.start2, q.size(), c);
}
void MCDIVN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MCMULN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    simulator->CIMULModNOut(a, m, starts.start1, starts.start2, q.size(), c);
}
void MCPOWN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o)
{
    if (q.size() != o.size()) {
        throw std::invalid_argument("MCPOWN() 'q' and 'o' parameters must have same size!");
    }

    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic2(simulator, q, o);
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    QALU(simulator)->CPOWModNOut(a, m, starts.start1, starts.start2, q.size(), c);
}

#if 0
void LDA(quid sid, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic3(simulator, qi, qv);
    QALU(simulator)->IndexedLDA(starts.start1, qi.size(), starts.start2, qv.size(), t, true);
}
void ADC(quid sid, bitLenInt s, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic3(simulator, qi, qv);
    QALU(simulator)->IndexedADC(starts.start1, qi.size(), starts.start2, qv.size(), GetSimShardId(simulator, s), t);
}
void SBC(quid sid, bitLenInt s, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    const MapArithmeticResult2 starts = MapArithmetic3(simulator, qi, qv);
    QALU(simulator)->IndexedSBC(starts.start1, qi.size(), starts.start2, qv.size(), GetSimShardId(simulator, s), t);
}
void Hash(quid sid, std::vector<bitLenInt> q, std::vector<unsigned char> t)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    QALU(simulator)->Hash(MapArithmetic(simulator, n, q), n, t);
}
#endif
#endif

bool TrySeparate1Qb(quid sid, bitLenInt qi1)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)
    return simulators[sid]->TrySeparate(GetSimShardId(simulator, qi1));
}

bool TrySeparate2Qb(quid sid, bitLenInt qi1, bitLenInt qi2)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)
    return simulators[sid]->TrySeparate(GetSimShardId(simulator, qi1), GetSimShardId(simulator, qi2));
}

bool TrySeparateTol(quid sid, std::vector<bitLenInt> q, real1_f tol)
{
    SIMULATOR_LOCK_GUARD_BOOL(sid)
    return simulator->TrySeparate(q, tol);
}

void Separate(quid sid, std::vector<bitLenInt> q)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    std::vector<bitLenInt> bitArray(q.size());
    for (size_t i = 0U; i < q.size(); ++i) {
        bitArray[i] = GetSimShardId(simulator, q[i]);
    }
    const bitLenInt end = simulator->GetQubitCount() - 1U;
    for (size_t i = 0U; i < q.size(); ++i) {
        simulator->Swap(end - i, bitArray[i]);
    }
    QInterfacePtr partSim = simulator->Decompose(end - q.size(), q.size());
    simulator->UpdateRunningNorm();
    simulator->NormalizeState();
    partSim->UpdateRunningNorm();
    partSim->NormalizeState();
    simulator->Compose(partSim);
    for (size_t i = 0U; i < q.size(); ++i) {
        simulator->Swap(end - i, bitArray[i]);
    }
}

double GetUnitaryFidelity(quid sid)
{
    SIMULATOR_LOCK_GUARD_REAL1_F(sid)
    return simulator->GetUnitaryFidelity();
}

void ResetUnitaryFidelity(quid sid)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->ResetUnitaryFidelity();
}

void SetSdrp(quid sid, double sdrp)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetSdrp(sdrp);
}

void SetNcrp(quid sid, double ncrp)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetNcrp(ncrp);
}

void SetReactiveSeparate(quid sid, bool irs)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetReactiveSeparate(irs);
}

void SetTInjection(quid sid, bool irs)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetTInjection(irs);
}

void SetNoiseParameter(quid sid, double np)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->SetNoiseParameter((real1_f)np);
}

void Normalize(quid sid)
{
    SIMULATOR_LOCK_GUARD_VOID(sid)
    simulator->UpdateRunningNorm();
    simulator->NormalizeState();
}

quid init_qneuron(quid sid, std::vector<bitLenInt> c, bitLenInt q, QNeuronActivationFn f, real1_f a, real1_f tol)
{
    META_LOCK_GUARD()

    if (sid > simulators.size()) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<const std::lock_guard<std::mutex>> simulatorLock(
        new const std::lock_guard<std::mutex>(simulatorMutexes[simulator.get()]));
    if (!simulator) {
        throw std::invalid_argument("Invalid argument: simulator ID not found!");
    }

    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = GetSimShardId(simulator, c[i]);
    }
    quid nid = (quid)neurons.size();

    for (size_t i = 0U; i < neurons.size(); ++i) {
        if (neuronReservations[i] == false) {
            nid = i;
            neuronReservations[i] = true;
            break;
        }
    }

    QNeuronPtr neuron = std::make_shared<QNeuron>(simulator, c, GetSimShardId(simulator, q), f, a, tol);
    neuronSimulators[neuron] = simulator.get();

    if (nid == neurons.size()) {
        neuronReservations.push_back(true);
        neurons.push_back(neuron);
        neuronErrors.push_back(0);
    } else {
        neuronReservations[nid] = true;
        neurons[nid] = neuron;
        neuronErrors[nid] = 0;
    }

    return nid;
}

quid clone_qneuron(quid nid)
{
    META_LOCK_GUARD()

    if (nid > neurons.size()) {
        throw std::invalid_argument("Invalid argument: neuron ID not found!");
    }
    QNeuronPtr neuron = neurons[nid];
    std::unique_ptr<const std::lock_guard<std::mutex>> neuronLock(
        new const std::lock_guard<std::mutex>(neuronMutexes[neuron.get()]));

    quid nnid = (quid)neurons.size();

    for (size_t i = 0U; i < neurons.size(); ++i) {
        if (neuronReservations[i] == false) {
            nnid = i;
            neuronReservations[i] = true;
            break;
        }
    }

    QNeuronPtr nNeuron = std::make_shared<QNeuron>(*neuron);
    neuronSimulators[nNeuron] = neuronSimulators[neuron];

    if (nnid == neurons.size()) {
        neuronReservations.push_back(true);
        neurons.push_back(nNeuron);
        neuronErrors.push_back(0);
    } else {
        neuronReservations[nnid] = true;
        neurons[nnid] = nNeuron;
        neuronErrors[nnid] = 0;
    }

    return nnid;
}

void destroy_qneuron(quid nid)
{
    META_LOCK_GUARD()

    neuronMutexes.erase(neurons[nid].get());
    neurons[nid] = nullptr;
    neuronErrors[nid] = 0;
    neuronReservations[nid] = false;
}

void set_qneuron_angles(quid nid, std::vector<real1> angles)
{
    NEURON_LOCK_GUARD_VOID(nid)
    if (angles.size() != (size_t)neuron->GetInputPower()) {
        throw std::invalid_argument(
            "set_qneuron_angles() 'angles' parameter must have 2^n elements for n input qubits!");
    }
    std::unique_ptr<real1[]> _angles(new real1[angles.size()]);
    std::copy(angles.begin(), angles.end(), _angles.get());
    neuron->SetAngles(_angles.get());
}

std::vector<real1> get_qneuron_angles(quid nid)
{
    if (nid > neurons.size()) {
        throw std::invalid_argument("Invalid argument: neuron ID not found!");
    }

    QNeuronPtr neuron = neurons[nid];
    NEURON_LOCK_GUARD(neuron)
    if (!neuron) {
        return std::vector<real1>();
    }

    const bitCapIntOcl inputPower = (bitCapIntOcl)neuron->GetInputPower();
    std::unique_ptr<real1[]> _angles(new real1[inputPower]);
    neuron->GetAngles(_angles.get());
    std::vector<real1> angles(inputPower);
    std::copy(_angles.get(), _angles.get() + inputPower, angles.begin());
    return angles;
}

void set_qneuron_alpha(quid nid, real1_f alpha)
{
    NEURON_LOCK_GUARD_VOID(nid)
    neuron->SetAlpha(alpha);
}

real1_f get_qneuron_alpha(quid nid)
{
    NEURON_LOCK_GUARD_REAL1_F(nid)
    return neuron->GetAlpha();
}

void set_qneuron_activation_fn(quid nid, QNeuronActivationFn f)
{
    NEURON_LOCK_GUARD_VOID(nid)
    neuron->SetActivationFn(f);
}

QNeuronActivationFn get_qneuron_activation_fn(quid nid)
{
    NEURON_LOCK_GUARD_AFN(nid)
    return neuron->GetActivationFn();
}

real1_f qneuron_predict(quid nid, bool e, bool r)
{
    NEURON_LOCK_GUARD_REAL1_F(nid)
    try {
        return neuron->Predict(e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

real1_f qneuron_unpredict(quid nid, bool e)
{
    NEURON_LOCK_GUARD_REAL1_F(nid)
    try {
        return neuron->Unpredict(e);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

real1_f qneuron_learn_cycle(quid nid, bool e)
{
    NEURON_LOCK_GUARD_REAL1_F(nid)
    try {
        return neuron->LearnCycle(e);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
        return 0.5;
    }
}

void qneuron_learn(quid nid, real1_f eta, bool e, bool r)
{
    NEURON_LOCK_GUARD_VOID(nid)
    try {
        neuron->Learn(eta, e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

void qneuron_learn_permutation(quid nid, real1_f eta, bool e, bool r)
{
    NEURON_LOCK_GUARD_VOID(nid)
    try {
        neuron->LearnPermutation(eta, e, r);
    } catch (const std::exception& ex) {
        neuronErrors[nid] = 1;
        std::cout << ex.what() << std::endl;
    }
}

quid init_qcircuit(bool collapse, bool clifford)
{
    META_LOCK_GUARD()
    quid cid = (quid)circuits.size();

    for (size_t i = 0U; i < circuits.size(); ++i) {
        if (circuitReservations[i] == false) {
            cid = i;
            circuitReservations[i] = true;
            break;
        }
    }

    QCircuitPtr circuit = std::make_shared<QCircuit>(collapse, clifford);

    if (cid == circuits.size()) {
        circuitReservations.push_back(true);
        circuits.push_back(circuit);
    } else {
        circuitReservations[cid] = true;
        circuits[cid] = circuit;
    }

    return cid;
}

quid _init_qcircuit_copy(quid cid, bool isInverse, std::set<bitLenInt> q)
{
    META_LOCK_GUARD()

    if (cid > circuits.size()) {
        throw std::invalid_argument("Invalid argument: circuit ID not found!");
    }
    QCircuitPtr circuit = circuits[cid];
    std::unique_ptr<const std::lock_guard<std::mutex>> circuitLock(
        new const std::lock_guard<std::mutex>(circuitMutexes[circuit.get()]));

    quid ncid = (quid)circuits.size();

    for (size_t i = 0U; i < circuits.size(); ++i) {
        if (circuitReservations[i] == false) {
            ncid = i;
            circuitReservations[i] = true;
            break;
        }
    }

    QCircuitPtr nCircuit = isInverse ? circuit->Inverse() : (q.size() ? circuit->PastLightCone(q) : circuit->Clone());

    if (ncid == circuits.size()) {
        circuitReservations.push_back(true);
        circuits.push_back(nCircuit);
    } else {
        circuitReservations[ncid] = true;
        circuits[ncid] = nCircuit;
    }

    return ncid;
}

quid init_qcircuit_clone(quid cid) { return _init_qcircuit_copy(cid, false, {}); }

quid qcircuit_inverse(quid cid) { return _init_qcircuit_copy(cid, true, {}); }

quid qcircuit_past_light_cone(quid cid, std::set<bitLenInt> q) { return _init_qcircuit_copy(cid, false, q); }

void destroy_qcircuit(quid cid)
{
    META_LOCK_GUARD()

    circuitMutexes.erase(circuits[cid].get());
    circuits[cid] = nullptr;
    circuitReservations[cid] = false;
}

bitLenInt get_qcircuit_qubit_count(quid cid)
{
    CIRCUIT_LOCK_GUARD_INT(cid)
    return circuit->GetQubitCount();
}
void qcircuit_swap(quid cid, bitLenInt q1, bitLenInt q2)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    circuit->Swap(q1, q2);
}
void qcircuit_append_1qb(quid cid, std::vector<real1_f> m, bitLenInt q)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "qcircuit_append_1qb() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    CIRCUIT_LOCK_GUARD_VOID(cid)
    const complex mtrx[4]{ complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };
    circuit->AppendGate(std::make_shared<QCircuitGate>(q, mtrx));
}

void qcircuit_append_mc(quid cid, std::vector<real1_f> m, std::vector<bitLenInt> c, bitLenInt q, bitCapInt p)
{
    if (m.size() != 4) {
        throw std::invalid_argument(
            "qcircuit_append_1qb() 'm' parameter must be 4 complex (row-major) components of 2x2 unitary operator!");
    }

    CIRCUIT_LOCK_GUARD_VOID(cid)

    std::vector<bitLenInt> indices(c.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](bitLenInt a, bitLenInt b) -> bool { return c[a] < c[b]; });

    bitCapInt _p = ZERO_BCI;
    std::set<bitLenInt> ctrls;
    for (size_t i = 0U; i < c.size(); ++i) {
        bi_or_ip(&_p, ((p >> i) & 1U) << indices[i]);
        ctrls.insert(c[i]);
    }

    const complex mtrx[4]{ complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };
    circuit->AppendGate(std::make_shared<QCircuitGate>(q, mtrx, ctrls, _p));
}

void qcircuit_run(quid cid, quid sid)
{
    CIRCUIT_AND_SIMULATOR_LOCK_GUARD_VOID(cid, sid)
    circuit->Run(simulator);
}

void qcircuit_out_to_file(quid cid, std::string f)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    std::ofstream ofile;
    ofile.open(f.c_str());
    ofile.precision(36);
    ofile << circuit;
    ofile.close();
}

void qcircuit_in_from_file(quid cid, std::string f)
{
    CIRCUIT_LOCK_GUARD_VOID(cid)
    std::ifstream ifile;
    ifile.open(f.c_str());
    ifile.precision(36);
    ifile >> circuit;
    ifile.close();
}

std::string qcircuit_out_to_string(quid cid)
{
    CIRCUIT_LOCK_GUARD_TYPED(cid, "")
    std::stringstream ss;
    ss.precision(36);
    ss << circuit;
    return ss.str();
}
} // namespace Qrack
