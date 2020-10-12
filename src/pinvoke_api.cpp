//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
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

enum Pauli {
    /// Pauli Identity operator. Corresponds to Q# constant "PauliI."
    PauliI = 0U,
    /// Pauli X operator. Corresponds to Q# constant "PauliX."
    PauliX = 1U,
    /// Pauli Y operator. Corresponds to Q# constant "PauliY."
    PauliY = 3U,
    /// Pauli Z operator. Corresponds to Q# constant "PauliZ."
    PauliZ = 2U
};

qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>(time(0));
std::vector<QInterfacePtr> simulators;
std::vector<bool> simulatorReservations;
std::map<QInterfacePtr, std::map<unsigned, bitLenInt>> shards;

void mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx)
{
    for (unsigned i = 0; i < 4; i++) {
        outMtrx[i] = scalar * inMtrx[i];
    }
}

void TransformPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds)
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

void RevertPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds)
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

void removeIdentities(std::vector<unsigned>* b, std::vector<unsigned>* qs)
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
    case PauliI:
        simulator->Exp(phi, shards[simulator][q]);
        break;
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
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    real1 cosine = cos(phi / 2.0);
    real1 sine = sin(phi / 2.0);
    complex pauliR[4];

    switch (b) {
    case PauliI:
        simulator->ApplyControlledSinglePhase(
            ctrlsArray, n, shards[simulator][q], complex(cosine, sine), complex(cosine, sine));
        break;
    case PauliX:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(ZERO_R1, -sine);
        pauliR[2] = complex(ZERO_R1, -sine);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
        break;
    case PauliY:
        pauliR[0] = complex(cosine, ZERO_R1);
        pauliR[1] = complex(-sine, ZERO_R1);
        pauliR[2] = complex(sine, ZERO_R1);
        pauliR[3] = complex(cosine, ZERO_R1);
        simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
        break;
    case PauliZ:
        simulator->ApplyControlledSinglePhase(
            ctrlsArray, n, shards[simulator][q], complex(cosine, -sine), complex(cosine, sine));
        break;
    default:
        break;
    }

    delete[] ctrlsArray;
}

inline bool isDiagonal(std::vector<unsigned> const& b)
{
    for (auto x : b) {
        if (x == PauliX || x == PauliY) {
            return false;
        }
    }
    return true;
}

inline bool poppar(unsigned perm)
{
    // From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
    unsigned int c; // c accumulates the total bits set in v
    for (c = 0; perm; c++) {
        perm &= perm - 1U; // clear the least significant bit set
    }
    return c & 1U;
}

inline std::size_t make_mask(std::vector<unsigned> const& qs)
{
    std::size_t mask = 0;
    for (std::size_t q : qs)
        mask = mask | pow2Ocl(q);
    return mask;
}

// power of square root of -1
inline complex iExp(int power)
{
    int p = ((power % 4) + 8) % 4;
    switch (p) {
    case 0:
        return ONE_CMPLX;
    case 1:
        return I_CMPLX;
    case 2:
        return -ONE_CMPLX;
    case 3:
        return -I_CMPLX;
    }
    return 0;
}

void apply_controlled_exp(std::vector<complex>& wfn, std::vector<unsigned> const& b, double phi,
    std::vector<unsigned> const& cs, std::vector<unsigned> const& qs)
{
    std::size_t cmask = make_mask(cs);

    if (isDiagonal(b)) {
        std::size_t mask = make_mask(qs);
        complex phase = std::exp(complex(0., -phi));

        for (std::intptr_t x = 0; x < static_cast<std::intptr_t>(wfn.size()); x++) {
            if ((x & cmask) == cmask) {
                wfn[x] *= (poppar(x & mask) ? phase : std::conj(phase));
            }
        }
    } else { // see Exp-implementation-details.txt for the explanation of the algorithm below
        std::size_t xy_bits = 0;
        std::size_t yz_bits = 0;
        int y_count = 0;
        for (unsigned i = 0; i < b.size(); ++i) {
            switch (b[i]) {
            case PauliX:
                xy_bits |= (1ull << qs[i]);
                break;
            case PauliY:
                xy_bits |= (1ull << qs[i]);
                yz_bits |= (1ull << qs[i]);
                ++y_count;
                break;
            case PauliZ:
                yz_bits |= (1ull << qs[i]);
                break;
            case PauliI:
                break;
            }
        }

        real1 alpha = (real1)std::cos(phi);
        complex beta = (real1)std::sin(phi) * iExp(3 * y_count + 1);
        complex gamma = (real1)std::sin(phi) * iExp(y_count + 1);

        for (std::intptr_t x = 0; x < static_cast<std::intptr_t>(wfn.size()); x++) {
            std::intptr_t t = x ^ xy_bits;
            if (x < t && ((x & cmask) == cmask)) {
                auto parity = poppar(x & yz_bits);
                auto a = wfn[x];
                auto b = wfn[t];
                wfn[x] = alpha * a + (parity ? -beta : beta) * b;
                wfn[t] = alpha * b + (parity ? -gamma : gamma) * a;
            }
        }
    }
}

extern "C" {

/**
 * (External API) Initialize a simulator ID with 0 qubits
 */
MICROSOFT_QUANTUM_DECL unsigned init() { return init_count(0); }

MICROSOFT_QUANTUM_DECL unsigned init_count(_In_ unsigned q)
{
    META_LOCK_GUARD()

    unsigned sid = simulators.size();

    for (unsigned i = 0; i < simulators.size(); i++) {
        if (simulatorReservations[i] == false) {
            sid = i;
            simulatorReservations[i] = true;
            break;
        }
    }

    if (sid == simulators.size()) {
        simulatorReservations.push_back(true);
    }

    QInterfacePtr simulator = q ? CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, q, 0, rng) : NULL;
    if (sid == simulators.size()) {
        simulators.push_back(simulator);
    } else {
        simulators[sid] = simulator;
    }

    shards[simulator] = {};
    for (unsigned i = 0; i < q; i++) {
        shards[simulator][i] = (bitLenInt)i;
    }

    return sid;
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
    complex* wfn = new complex[wfnl];
    simulator->GetQuantumState(wfn);
    for (size_t i = 0; i < wfnl; i++) {
        if (!callback(i, real(wfn[i]), imag(wfn[i]))) {
            break;
        }
    }
    delete[] wfn;
}

/**
 * (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
 */
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
    std::discrete_distribution<std::size_t> dist(p, p + n);
    return dist(*rng.get());
}

double _JointEnsembleProbabilityHelper(unsigned n, unsigned* b, unsigned* q, QInterfacePtr simulator,
    std::vector<unsigned>* bVec, std::vector<unsigned>* qVec, std::vector<bitCapInt>* qSortedPowers)
{

    if (n == 0) {
        return 0.0;
    }

    bitCapInt mask = 0;

    bVec->resize(n);
    qVec->resize(n);

    std::copy(b, b + n, bVec->begin());
    std::copy(q, q + n, qVec->begin());

    removeIdentities(bVec, qVec);
    n = qVec->size();
    qSortedPowers->resize(n);

    if (n == 0) {
        return 0.0;
    }

    for (bitLenInt i = 0; i < n; i++) {
        bitCapInt bit = pow2(shards[simulator][(*qVec)[i]]);
        (*qSortedPowers)[i] = bit;
        mask |= bit;
    }

    std::sort(qSortedPowers->begin(), qSortedPowers->end());

    bitCapInt pow2n = pow2(n);
    double jointProb = 0;
    bitCapInt perm;
    bool isOdd;

    for (bitCapInt i = 0; i < pow2n; i++) {
        perm = 0U;
        isOdd = false;
        for (bitLenInt j = 0; j < n; j++) {
            if (i & pow2(j)) {
                perm |= (*qSortedPowers)[j];
                isOdd = !isOdd;
            }
        }
        if (isOdd) {
            jointProb += simulator->ProbMask(mask, perm);
        }
    }

    return jointProb;
}

/**
 * (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
 */
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    std::vector<unsigned> bVec;
    std::vector<unsigned> qVec;
    std::vector<bitCapInt> qSortedPowers;

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(n, b, q, simulator, &bVec, &qVec, &qSortedPowers);

    RevertPauliBasis(simulator, n, b, q);

    return jointProb;
}

/**
 * (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr nQubit = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, 1, 0, rng);
    if (simulators[sid] == NULL) {
        simulators[sid] = nQubit;
    } else {
        simulators[sid]->Compose(nQubit);
    }
    shards[simulators[sid]][qid] = (simulators[sid]->GetQubitCount() - 1U);
}

/**
 * (External API) Release 1 qubit with the given qubit ID, under the simulator ID
 */
MICROSOFT_QUANTUM_DECL void release(_In_ unsigned sid, _In_ unsigned q)
{
    QInterfacePtr simulator = simulators[sid];

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
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
    SIMULATOR_LOCK_GUARD(sid)

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
    simulator->IS(shards[simulator][q]);
}

/**
 * (External API) "T" Gate
 */
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->IT(shards[simulator][q]);
}

/**
 * (External API) Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->S(shards[simulator][q]);
}

/**
 * (External API) Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    simulator->T(shards[simulator][q]);
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
 * (External API) Controlled "X" Gate
 */
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, ONE_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "Y" Gate
 */
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], -I_CMPLX, I_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "Z" Gate
 */
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, -ONE_CMPLX);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "H" Gate
 */
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    const complex hGate[4] = { complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
        complex(-M_SQRT1_2, ZERO_R1) };

    simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], hGate);

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 2));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 4));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled Inverse "S" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 2));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled Inverse "T" Gate
 */
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 4));

    delete[] ctrlsArray;
}

/**
 * (External API) Controlled 3-parameter unitary gate
 */
MICROSOFT_QUANTUM_DECL void MCU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];
    bitLenInt* ctrlsArray = new bitLenInt[n];
    for (unsigned i = 0; i < n; i++) {
        ctrlsArray[i] = shards[simulator][c[i]];
    }

    simulator->CU(ctrlsArray, n, shards[simulator][q], theta, phi, lambda);
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
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<unsigned> bVec(n);
    std::vector<unsigned> qVec(n);

    std::copy(b, b + n, bVec.begin());
    std::copy(q, q + n, qVec.begin());

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    if (bVec.size() == 0) {
        RHelper(sid, PauliI, -2. * phi, someQubit);
    } else if (bVec.size() == 1) {
        RHelper(sid, bVec.front(), -2. * phi, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];
        std::vector<complex> wfn((bitCapIntOcl)simulator->GetMaxQPower());
        simulator->GetQuantumState(&(wfn[0]));

        std::vector<unsigned> bVec(n);
        std::copy(b, b + n, bVec.begin());

        std::vector<unsigned> csVec;

        std::vector<unsigned> qVec(n);
        std::copy(q, q + n, qVec.begin());

        apply_controlled_exp(wfn, bVec, phi, csVec, qVec);

        simulator->SetQuantumState(&(wfn[0]));
    }
}

/**
 * (External API) Controlled exponentiation of Pauli operators
 */
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi,
    _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q)
{
    if (n == 0) {
        return;
    }

    SIMULATOR_LOCK_GUARD(sid)

    std::vector<unsigned> bVec(n);
    std::vector<unsigned> qVec(n);

    std::copy(b, b + n, bVec.begin());
    std::copy(q, q + n, qVec.begin());

    unsigned someQubit = qVec.front();

    removeIdentities(&bVec, &qVec);

    if (bVec.size() == 0) {
        MCRHelper(sid, PauliI, -2. * phi, nc, cs, someQubit);
    } else if (bVec.size() == 1) {
        MCRHelper(sid, bVec.front(), -2. * phi, nc, cs, qVec.front());
    } else {
        QInterfacePtr simulator = simulators[sid];
        std::vector<complex> wfn((bitCapIntOcl)simulator->GetMaxQPower());
        simulator->GetQuantumState(&(wfn[0]));

        std::vector<unsigned> csVec(nc);
        std::copy(cs, cs + nc, csVec.begin());

        apply_controlled_exp(wfn, bVec, phi, csVec, qVec);

        simulator->SetQuantumState(&(wfn[0]));
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
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
    SIMULATOR_LOCK_GUARD(sid)

    QInterfacePtr simulator = simulators[sid];

    std::vector<unsigned> bVec;
    std::vector<unsigned> qVec;
    std::vector<bitCapInt> qSortedPowers;

    TransformPauliBasis(simulator, n, b, q);

    double jointProb = _JointEnsembleProbabilityHelper(n, b, q, simulator, &bVec, &qVec, &qSortedPowers);

    unsigned toRet = jointProb < simulator->Rand() ? 0U : 1U;
    bitCapInt len = qVec.size();
    bitCapInt maxQPower = simulator->GetMaxQPower();
    bool isOdd;

    if (jointProb != 0.0 && jointProb != 1.0) {
        complex* nStateVec = new complex[(bitCapIntOcl)simulator->GetMaxQPower()]();
        simulator->GetQuantumState(nStateVec);
        real1 nrmlzr = 0.0;
        for (bitCapIntOcl i = 0; i < maxQPower; i++) {
            isOdd = false;
            for (bitLenInt j = 0; j < len; j++) {
                if (i & qSortedPowers[j]) {
                    isOdd = !isOdd;
                }
            }
            if (isOdd == toRet) {
                nrmlzr += norm(nStateVec[i]);
            } else {
                nStateVec[i] = ZERO_CMPLX;
            }
        }
        simulator->SetQuantumState(nStateVec);
        delete[] nStateVec;
        simulator->NormalizeState(nrmlzr);
    }

    RevertPauliBasis(simulator, n, b, q);

    return toRet;
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

#if !ENABLE_PURE32
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
