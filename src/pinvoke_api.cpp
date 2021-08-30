//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "pinvoke_api.hpp"
#include "hamiltonian.hpp"

// for details.

#include <map>
#include <mutex>
#include <vector>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

std::mutex metaOperationMutex;

#define META_LOCK_GUARD() const std::lock_guard<std::mutex> metaLock(metaOperationMutex);
// TODO: By design, Qrack should be able to support per-simulator lock guards, in a multithreaded OCL environment. This
// feature might not yet be fully realized.
#define SIMULATOR_LOCK_GUARD(sid) const std::lock_guard<std::mutex> metaLock(metaOperationMutex);

using namespace Qrack;

qrack_rand_gen_ptr randNumGen = std::make_shared<qrack_rand_gen>(time(0));
std::vector<QInterfacePtr> simulators;
std::vector<bool> simulatorReservations;
std::map<QInterfacePtr, std::map<unsigned, bitLenInt>> shards;
bitLenInt _maxShardQubits = 0;
bitLenInt MaxShardQubits()
{
    if (_maxShardQubits == 0) {
        _maxShardQubits =
            getenv("QRACK_MAX_PAGING_QB") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB"))) : -1;
    }

    return _maxShardQubits;
}

void TransformPauliBasis(QInterfacePtr simulator, unsigned len, int* bases, unsigned* qubitIds)
{
    for (unsigned i = 0; i < len; i++) {
        switch (bases[i]) {
        case PauliX:
            simulator->H(shards[simulator][qubitIds[i]]);
            break;
        case PauliY:
            simulator->IS(shards[simulator][qubitIds[i]]);
            simulator->H(shards[simulator][qubitIds[i]]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void RevertPauliBasis(QInterfacePtr simulator, unsigned len, int* bases, unsigned* qubitIds)
{
    for (unsigned i = 0; i < len; i++) {
        switch (bases[i]) {
        case PauliX:
            simulator->H(shards[simulator][qubitIds[i]]);
            break;
        case PauliY:
            simulator->H(shards[simulator][qubitIds[i]]);
            simulator->S(shards[simulator][qubitIds[i]]);
            break;
        case PauliZ:
        case PauliI:
        default:
            break;
        }
    }
}

void removeIdentities(std::vector<int>* b, std::vector<bitLenInt>* qs)
{
    unsigned i = 0;
    while (i != b->size()) {
        if ((*b)[i] == PauliI) {
            b->erase(b->begin() + i);
            qs->erase(qs->begin() + i);
        } else {
            ++i;
        }
    }
}

void RHelper(unsigned sid, unsigned b, double phi, unsigned q)
{
    QInterfacePtr simulator = simulators[sid];

    switch (b) {
    case PauliI: {
        // This is a global phase factor, with no measurable physical effect.
        // However, the underlying QInterface will not execute the gate
        // UNLESS it is specifically "keeping book" for non-measurable phase effects.
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->ApplySinglePhase(phaseFac, phaseFac, shards[simulator][q]);
        break;
    }
    case PauliX:
        simulator->RX(phi, shards[simulator][q]);
        break;
    case PauliY:
        simulator->RY(phi, shards[simulator][q]);
        break;
    case PauliZ:
        simulator->RZ(phi, shards[simulator][q]);
        break;
    default:
        break;
    }
}

void MCRHelper(unsigned sid, unsigned b, double phi, unsigned n, unsigned* c, unsigned q)
{
    QInterfacePtr simulator = simulators[sid];
    std::unique_ptr<bitLenInt[]> ctrlsArray(new bitLenInt[n]);
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    if (b == PauliI) {
        complex phaseFac = exp(complex(ZERO_R1, (real1)(phi / 4)));
        simulator->ApplyControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], phaseFac, phaseFac);
        return;
    }

    real1 cosine = (real1)cos(phi / 2);
    real1 sine = (real1)sin(phi / 2);
    complex pauliR[4];

    switch (b) {
    case PauliX:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(ZERO_R1, -sine);
        pauliR[2] = complex(ZERO_R1, -sine);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], pauliR);
        break;
    case PauliY:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(-sine, ZERO_R1);
        pauliR[2] = complex(sine, ZERO_R1);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], pauliR);
        break;
    case PauliZ:
        simulator->ApplyControlledSinglePhase(
            ctrlsArray.get(), n, shards[simulator][q], complex(cosine, -sine), complex(cosine, sine));
        break;
    case PauliI:
    default:
        break;
    }
}

inline std::size_t make_mask(std::vector<bitLenInt> const& qs)
{
    std::size_t mask = 0;
    for (std::size_t q : qs)
        mask = mask | pow2Ocl(q);
    return mask;
}

extern "C" {

/**
 * (External API) Initialize a simulator ID with 0 qubits
 */
MICROSOFT_QUANTUM_DECL unsigned init() { return init_count(0); }

/**
 * (External API) Initialize a simulator ID with "q" qubits
 */
MICROSOFT_QUANTUM_DECL unsigned init_count(_In_ unsigned q)
{
    META_LOCK_GUARD()

    unsigned sid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    QInterfacePtr simulator = q ? CreateQuantumInterface({ QINTERFACE_OPTIMAL_MULTI }, q, 0, randNumGen) : NULL;

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
    } else {
        simulatorReservations[sid] = true;
        simulators[sid] = simulator;
    }

    if (!q) {
        return sid;
    }

    shards[simulator] = {};
    for (unsigned i = 0; i < q; i++) {
        shards[simulator][i] = (bitLenInt)i;
    }

    return sid;
}

/**
 * (External API) Initialize a simulator ID that clones simulator ID "sid"
 */
MICROSOFT_QUANTUM_DECL unsigned init_clone(_In_ unsigned sid)
{
    META_LOCK_GUARD()

    unsigned nsid = (unsigned)simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            nsid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    QInterfacePtr simulator = simulators[sid]->Clone();
    if (nsid == simulators.size()) {
        simulatorReservations.push_back(true);
        simulators.push_back(simulator);
    } else {
        simulatorReservations[nsid] = true;
        simulators[nsid] = simulator;
    }

    shards[simulator] = {};
    for (unsigned i = 0; i < simulator->GetQubitCount(); i++) {
        shards[simulator][i] = shards[simulators[sid]][i];
    }

    return nsid;
}

/**
 * (External API) Destroy a simulator (ID will not be reused)
 */
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid)
{
    META_LOCK_GUARD()
    // SIMULATOR_LOCK_GUARD(sid)

    shards.erase(simulators[sid]);
    simulators[sid] = NULL;
    simulatorReservations[sid] = false;
}

/**
 * (External API) Set RNG seed for simulator ID
 */
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] != NULL) {
        simulators[sid]->SetRandomSeed(s);
    }
}

/**
 * (External API) Set concurrency level per QEngine shard
 */
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ unsigned sid, _In_ unsigned p)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] != NULL) {
        simulators[sid]->SetConcurrency(p);
    }
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void DumpIds(_In_ unsigned sid, _In_ IdCallback callback)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    if (!simulator) {
        return;
    }

    std::map<unsigned, bitLenInt>::iterator it;
    for (it = shards[simulator].begin(); it != shards[simulator].end(); it++) {
        callback(it->first);
    }
}

/**
 * (External API) "Dump" all IDs from the selected simulator ID into the callback
 */
MICROSOFT_QUANTUM_DECL void Dump(_In_ unsigned sid, _In_ ProbAmpCallback callback)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitCapIntOcl wfnl = (bitCapIntOcl)simulator->GetMaxQPower();
    std::unique_ptr<complex[]> wfn(new complex[wfnl]);
    simulator->GetQuantumState(wfn.get());
    for (size_t i = 0; i < wfnl; i++) {
        if (!callback(i, real(wfn.get()[i]), imag(wfn.get()[i]))) {
            break;
        }
    }
}

/**
 * (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
 */
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
    std::discrete_distribution<std::size_t> dist(p, p + n);
    return dist(*randNumGen.get());
}

double _JointEnsembleProbabilityHelper(QInterfacePtr simulator, unsigned n, int* b, unsigned* q, bool doMeasure)
{

    if (n == 0) {
        return 0.0;
    }

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    removeIdentities(&bVec, &qVec);
    n = (unsigned)qVec.size();

    if (n == 0) {
        return 0.0;
    }

    bitCapInt mask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)n; i++) {
        bitCapInt bit = pow2(shards[simulator][qVec[i]]);
        mask |= bit;
    }

    return (double)(doMeasure ? (simulator->MParity(mask) ? ONE_R1 : ZERO_R1) : simulator->ProbParity(mask));
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, false);

    RevertPauliBasis(simulator, n, b, q);

    return jointProb;
}

/**
 * (External API) Set the simulator to a computational basis permutation.
 */
MICROSOFT_QUANTUM_DECL void ResetAll(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD(sid)
    if (simulators[sid]) {
        simulators[sid]->SetPermutation(0);
    }
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr nQubit = CreateQuantumInterface({ QINTERFACE_OPTIMAL_MULTI }, 1, 0, randNumGen);
    if (simulators[sid] == NULL) {
        simulators[sid] = nQubit;
        shards[simulators[sid]] = {};
    } else {
        simulators[sid]->Compose(nQubit);
    }
    bitLenInt qubitCount = simulators[sid]->GetQubitCount();
    shards[simulators[sid]][qid] = (qubitCount - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL bool release(_In_ unsigned sid, _In_ unsigned q)
{
    QInterfacePtr simulator = simulators[sid];

    // Check that the qubit is in the |0> state, to within a small tolerance.
    bool toRet = simulator->Prob(shards[simulator][q]) < (ONE_R1 / 100);

    if (simulator->GetQubitCount() == 1U) {
        shards.erase(simulator);
        simulators[sid] = NULL;
    } else {
        SIMULATOR_LOCK_GUARD(sid)
        bitLenInt oIndex = shards[simulator][q];
        simulator->Dispose(oIndex, 1U);
        for (unsigned i = 0; i < shards[simulator].size(); i++) {
            if (shards[simulator][i] > oIndex) {
                shards[simulator][i]--;
            }
        }
        shards[simulator].erase(q);
    }

    return toRet;
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD(sid)

    if (simulators[sid] == NULL) {
        return 0U;
    }

    return (unsigned)simulators[sid]->GetQubitCount();
}

/**
 * (External API) "X" Gate
 */
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->X(shards[simulator][q]);
}

/**
 * (External API) "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Y(shards[simulator][q]);
}

/**
 * (External API) "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Z(shards[simulator][q]);
}

/**
 * (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
 */
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->H(shards[simulator][q]);
}

/**
 * (External API) "S" Gate
 */
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->S(shards[simulator][q]);
}

/**
 * (External API) "T" Gate
 */
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->T(shards[simulator][q]);
}

/**
 * (External API) Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->IS(shards[simulator][q]);
}

/**
 * (External API) Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->IT(shards[simulator][q]);
}

/**
 * (External API) 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void U(
    _In_ unsigned sid, _In_ unsigned q, _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->U(shards[simulator][q], theta, phi, lambda);
}

/**
 * (External API) 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void Mtrx(_In_ unsigned sid, _In_reads_(8) double* m, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    QInterfacePtr simulator = simulators[sid];
    simulator->ApplySingleBit(mtrx, shards[simulator][q]);
}

#define MAP_CONTROLS_AND_LOCK(sid)                                                                                     \
    SIMULATOR_LOCK_GUARD(sid)                                                                                          \
    QInterfacePtr simulator = simulators[sid];                                                                         \
    std::unique_ptr<bitLenInt[]> ctrlsArray(new bitLenInt[n]);                                                         \
    for (unsigned i = 0; i < n; i++) {                                                                                 \
        ctrlsArray.get()[i] = shards[simulator][c[i]];                                                                 \
    }

/**
 * (External API) Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSingleInvert(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, ONE_CMPLX);
}

/**
 * (External API) Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSingleInvert(ctrlsArray.get(), n, shards[simulator][q], -I_CMPLX, I_CMPLX);
}

/**
 * (External API) Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, -ONE_CMPLX);
}

/**
 * (External API) Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    const complex hGate[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], hGate);
}

/**
 * (External API) Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, I_CMPLX);
}

/**
 * (External API) Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSinglePhase(
        ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1));
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, -I_CMPLX);
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSinglePhase(
        ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1));
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->CU(ctrlsArray.get(), n, shards[simulator][q], theta, phi, lambda);
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q)
{
    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], mtrx);
}

/**
 * (External API) "Anti-"Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MACX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSingleInvert(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, ONE_CMPLX);
}

/**
 * (External API) "Anti-"Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MACY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSingleInvert(ctrlsArray.get(), n, shards[simulator][q], -I_CMPLX, I_CMPLX);
}

/**
 * (External API) "Anti-"Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MACZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, -ONE_CMPLX);
}

/**
 * (External API) "Anti-"Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MACH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    const complex hGate[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        complex(-SQRT1_2_R1, ZERO_R1) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], hGate);
}

/**
 * (External API) "Anti-"Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, I_CMPLX);
}

/**
 * (External API) "Anti-"Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSinglePhase(
        ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1));
}

/**
 * (External API) "Anti-"Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSinglePhase(ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, -I_CMPLX);
}

/**
 * (External API) "Anti-"Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MACAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSinglePhase(
        ctrlsArray.get(), n, shards[simulator][q], ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1));
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->AntiCU(ctrlsArray.get(), n, shards[simulator][q], theta, phi, lambda);
}

/**
 * (External API) Controlled 2x2 complex matrix unitary gate
 */
MICROSOFT_QUANTUM_DECL void MACMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q)
{
    complex mtrx[4] = { complex((real1)m[0], (real1)m[1]), complex((real1)m[2], (real1)m[3]),
        complex((real1)m[4], (real1)m[5]), complex((real1)m[6], (real1)m[7]) };

    MAP_CONTROLS_AND_LOCK(sid)
    simulator->ApplyAntiControlledSingleBit(ctrlsArray.get(), n, shards[simulator][q], mtrx);
}

/**
 * (External API) Rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    RHelper(sid, b, phi, q);
}

/**
 * (External API) Controlled rotation around Pauli axes
 */
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    MCRHelper(sid, b, phi, n, c, q);
}

/**
 * (External API) Exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void Exp(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    if (bVec.size() == 0) {
        RHelper(sid, PauliI, -2. * phi, someQubit);
    } else if (bVec.size() == 1) {
        RHelper(sid, bVec.front(), -2. * phi, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];

        TransformPauliBasis(simulator, n, b, q);

        std::size_t mask = make_mask(qVec);
        simulator->UniformParityRZ((bitCapInt)mask, -phi);

        RevertPauliBasis(simulator, n, b, q);
    }
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi,
    _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<int> bVec(b, b + n);
    std::vector<bitLenInt> qVec(q, q + n);

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    if (bVec.size() == 0) {
        MCRHelper(sid, PauliI, -2. * phi, nc, cs, someQubit);
    } else if (bVec.size() == 1) {
        MCRHelper(sid, bVec.front(), -2. * phi, nc, cs, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];
        std::vector<bitLenInt> csVec(cs, cs + nc);

        TransformPauliBasis(simulator, n, b, q);

        std::size_t mask = make_mask(qVec);
        simulator->CUniformParityRZ(&(csVec[0]), csVec.size(), (bitCapInt)mask, -phi);

        RevertPauliBasis(simulator, n, b, q);
    }
}

/**
 * (External API) Measure bit in |0>/|1> basis
 */
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    return simulator->M(shards[simulator][q]) ? 1U : 0U;
}

/**
 * (External API) Measure bits in specified Pauli bases
 */
MICROSOFT_QUANTUM_DECL unsigned Measure(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    std::vector<unsigned> bVec;
    std::vector<unsigned> qVec;

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(simulator, n, b, q, true);

    unsigned toRet = (jointProb < (ONE_R1 / 2)) ? 0U : 1U;

    RevertPauliBasis(simulator, n, b, q);

    return toRet;
}

MICROSOFT_QUANTUM_DECL void SWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->Swap(qi1, qi2);
}

MICROSOFT_QUANTUM_DECL void CSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2)
{
    MAP_CONTROLS_AND_LOCK(sid)
    simulator->CSwap(ctrlsArray.get(), n, qi1, qi2);
}

MICROSOFT_QUANTUM_DECL void AND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->AND(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void OR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->OR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void XOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->XOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void NAND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->NAND(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void NOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->NOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void XNOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->XNOR(qi1, qi2, qo);
}

MICROSOFT_QUANTUM_DECL void CLAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLAND(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLXOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLXOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLNAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLNAND(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLNOR(ci, qi, qo);
}

MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->CLXNOR(ci, qi, qo);
}

/**
 * (External API) Get the probability that a qubit is in the |1> state.
 */
MICROSOFT_QUANTUM_DECL double Prob(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    return simulator->Prob(shards[simulator][q]);
}

/**
 * (External API) Get the permutation expectation value, based upon the order of input qubits.
 */
MICROSOFT_QUANTUM_DECL double PermutationExpectation(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD(sid)

    std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
    std::copy(c, c + n, q.get());

    QInterfacePtr simulator = simulators[sid];
    double result = simulator->ExpectationBitsAll(q.get(), n);

    return result;
}

MICROSOFT_QUANTUM_DECL void QFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
#if (QBCAPPOW >= 16) && (QBCAPPOW < 32)
    simulator->QFTR(c, n);
#else
    std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
    std::copy(c, c + n, q.get());
    simulator->QFTR(q.get(), n);
#endif
}
MICROSOFT_QUANTUM_DECL void IQFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
#if (QBCAPPOW >= 16) && (QBCAPPOW < 32)
    simulator->IQFTR(c, n);
#else
    std::unique_ptr<bitLenInt[]> q(new bitLenInt[n]);
    std::copy(c, c + n, q.get());
    simulator->IQFTR(q.get(), n);
#endif
}

MICROSOFT_QUANTUM_DECL bool TrySeparate1Qb(_In_ unsigned sid, _In_ unsigned qi1)
{
    SIMULATOR_LOCK_GUARD(sid)
    return simulators[sid]->TrySeparate(qi1);
}

MICROSOFT_QUANTUM_DECL bool TrySeparate2Qb(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2)
{
    SIMULATOR_LOCK_GUARD(sid)
    return simulators[sid]->TrySeparate(qi1, qi2);
}

MICROSOFT_QUANTUM_DECL bool TrySeparateTol(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ double tol)
{
    SIMULATOR_LOCK_GUARD(sid)

    bitLenInt* qb = new bitLenInt[n];
    std::copy(q, q + n, qb);

    return simulators[sid]->TrySeparate(qb, (bitLenInt)n, (real1_f)tol);
}

MICROSOFT_QUANTUM_DECL void SetReactiveSeparate(_In_ unsigned sid, _In_ bool irs)
{
    SIMULATOR_LOCK_GUARD(sid)
    simulators[sid]->SetReactiveSeparate(irs);
}

#if !(FPPOW < 6 && !ENABLE_COMPLEX_X2)
/**
 * (External API) Simulate a Hamiltonian
 */
MICROSOFT_QUANTUM_DECL void TimeEvolve(_In_ unsigned sid, _In_ double t, _In_ unsigned n,
    _In_reads_(n) _QrackTimeEvolveOpHeader* teos, unsigned mn, _In_reads_(mn) double* mtrx)
{
    bitCapIntOcl mtrxOffset = 0;
    Hamiltonian h(n);
    for (unsigned i = 0; i < n; i++) {
        h[i] = std::make_shared<UniformHamiltonianOp>(teos[i], mtrx + mtrxOffset);
        mtrxOffset += pow2Ocl(teos[i].controlLen) * 8U;
    }

    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->TimeEvolve(h, (real1)t);
}
#endif
}
