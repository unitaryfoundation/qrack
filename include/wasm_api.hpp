// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// (Extensively modified and adapted by Daniel Strano in unitaryfund/qrack)

#pragma once

#include "common/pauli.hpp"
#include "common/qneuron_activation_function.hpp"
#include "common/qrack_types.hpp"

#include <set>
#include <string>
#include <vector>

/**
 * GLOSSARY:
 *     bitLenInt - "bit-length integer" - unsigned integer ID of qubit position in register
 *     bitCapInt - "bit-capacity integer" - unsigned integer single-permutation value of a qubit register (typically
 * "big integer") real1 - "real number (1-dimensional)" - floating-point real-valued number complex - "complex number" -
 * floating-point complex-valued number (with two real1 component dimensions) quid - "quantum (simulator) unique
 * identifier" - unsigned integer that indexes and IDs running simulators, circuits, and neurons
 */

namespace Qrack {

typedef uint64_t quid;

struct QubitIndexState {
    bitLenInt qid;
    bool val;
    QubitIndexState(bitLenInt q, bool v)
        : qid(q)
        , val(v)
    {
        // Intentionally left blank
    }
};

struct QubitIntegerExpVar {
    bitLenInt qid;
    bitCapInt val;
    QubitIntegerExpVar(bitLenInt q, bitCapInt v)
        : qid(q)
        , val(v)
    {
        // Intentionally left blank
    }
};

struct QubitRealExpVar {
    bitLenInt qid;
    real1_f val;
    QubitRealExpVar(bitLenInt q, real1_f v)
        : qid(q)
        , val(v)
    {
        // Intentionally left blank
    }
};

struct QubitPauliBasis {
    bitLenInt qid;
    Pauli b;
    QubitPauliBasis(bitLenInt q, Pauli basis)
        : qid(q)
        , b(basis)
    {
        // Intentionally left blank
    }
};

struct QubitU3Basis {
    bitLenInt qid;
    real1_f b[3U];
    QubitU3Basis(bitLenInt q, std::vector<real1_f> basis)
        : qid(q)
    {
        if (basis.size() != 3U) {
            throw std::invalid_argument("QubitU3Basis must have 3 basis angles!");
        }
        for (int i = 0; i < 3; ++i) {
            b[i] = basis[i];
        }
    }
};

struct QubitMatrixBasis {
    bitLenInt qid;
    complex b[4U];
    QubitMatrixBasis(bitLenInt q, std::vector<complex> basis)
        : qid(q)
    {
        if (basis.size() != 4U) {
            throw std::invalid_argument("QubitMatrixBasis must have 4 matrix components for basis!");
        }
        for (int i = 0; i < 4; ++i) {
            b[i] = basis[i];
        }
    }
};

struct QubitU3BasisEigenVal {
    bitLenInt qid;
    real1_f b[3U];
    real1_f e[2U];
    QubitU3BasisEigenVal(bitLenInt q, std::vector<real1_f> basis, std::vector<real1_f> ex)
        : qid(q)
    {
        if (basis.size() != 3U) {
            throw std::invalid_argument("QubitU3BasisEigenVal must have 3 basis angles!");
        }
        if (ex.size() != 2U) {
            throw std::invalid_argument("QubitU3BasisEigenVal must have 2 eigenvalues!");
        }
        for (int i = 0; i < 3; ++i) {
            b[i] = basis[i];
        }
        e[0U] = ex[0U];
        e[1U] = ex[1U];
    }
};

struct QubitMatrixBasisEigenVal {
    bitLenInt qid;
    complex b[4U];
    real1_f e[2U];
    QubitMatrixBasisEigenVal(bitLenInt q, std::vector<complex> basis, std::vector<real1_f> ex)
        : qid(q)
    {
        if (basis.size() != 4U) {
            throw std::invalid_argument("QubitMatrixBasisEigenVal must have 4 matrix components for basis!");
        }
        if (ex.size() != 2U) {
            throw std::invalid_argument("QubitMatrixBasisEigenVal must have 2 eigenvalues!");
        }
        for (int i = 0; i < 4; ++i) {
            b[i] = basis[i];
        }
        e[0U] = ex[0U];
        e[1U] = ex[1U];
    }
};

/**
 * Options for simulator type in initialization (any set of options theoretically functions together):
 *     tn - "Tensor network" layer - JIT local circuit simplification, light-cone optimization
 *     md - "Multi-device" (TURN OFF IN SERIAL BUILDS) - distribute Schmidt-decomposed or factorized subsytems to
 * different OpenCL devices. sd - "Schmidt decomposition" - use state-factorization optimizations sh - "Stabilizer
 * hybrid" - use ("hybrid") stabilizer and near-Clifford optimizations (does not work well with "tn," "tensor network")
 *     bdt - "Binary decision tree" - use "quantum binary decision diagram" (QBDD) optimization. This can optionally
 * "hybridize" with state vector, for speed. pg - "Pager" (TURN OFF IN SERIAL BUILDS) - split large simulations into a
 * power-of-2 of smaller simulation "pages," for multi-device distribution hy - "(State vector) Hybrid" (TURN OFF IN
 * SERIAL BUILD) - for state vector, "hybridize" CPU/GPU/multi-GPU simulation qubit width thresholds, for speed. oc -
 * "OpenCL" (TURN OFF IN SERIAL BUILD) - use OpenCL acceleration (in general) hp - "Host pointer" (TURN OFF IN SERIAL
 * BUILD) - allocate OpenCL state vectors on "host" instead of "device" (useful for certain accelerators, like Intel HD)
 */
quid init_count_type(
    bitLenInt q, bool tn, bool md, bool sd, bool sh, bool bdt, bool pg, bool nw, bool hy, bool oc, bool hp);

// Utility

/**
 * "Default optimal" (BQP-complete-targeted) simulator type initialization (with "direct memory" option)
 */
quid init_count(bitLenInt q, bool dm);

/**
 * Initialize a simulator ID with "q" qubits as purely a stabilizer simulator.
 */
quid init_count_stabilizer(bitLenInt q);

/**
 * "Quasi-default constructor" (for an empty simulator)
 */
quid init();

/**
 * "Clone" simulator (no-clone theorem does not apply to classical simulation)
 */
quid init_clone(quid sid);

/**
 * "Default optimal" (BQP-complete-targeted) simulator type initialization (with "direct memory" option)
 */
quid init_qbdd_count(bitLenInt q);

/**
 * "Destroy" or release simulator allocation
 */
void destroy(quid sid);

/**
 * "Seed" random number generator (if pseudo-random Mersenne twister is in use)
 */
void seed(quid sid, unsigned s);

/**
 * Set CPU concurrency (if build isn't serial)
 */
void set_concurrency(quid sid, unsigned p);

/**
 * Allocate new qubit with ID
 */
void allocateQubit(quid sid, bitLenInt qid);
/**
 * Release qubit ID
 */
bool release(quid sid, bitLenInt q);
/**
 * Total count of qubits in simulator instance
 */
bitLenInt num_qubits(quid sid);
/**
 * Set bit string permutation eigenstate of simulator instance
 */
void SetPermutation(quid sid, bitCapInt p);

/**
 * Output stabilizer simulation tableau to file (or raise exception for "get_error()" if incompatible simulator type)
 */
void qstabilizer_out_to_file(quid sid, std::string f);
/**
 * Initialize stabilizer simulation from a tableau file (or raise exception for "get_error()" if incompatible simulator
 * type)
 */
void qstabilizer_in_from_file(quid sid, std::string f);

/**
 * Get the probabilities of all permutations of the requested subset of qubits.
 */
std::vector<real1> ProbAll(quid sid, std::vector<bitLenInt> q);
/**
 * Z-basis expectation value of qubit
 */
real1_f Prob(quid sid, bitLenInt q);
/**
 * "Reduced density matrix" Z-basis expectation value of qubit
 */
real1_f ProbRdm(quid sid, bitLenInt q);
/**
 * Probability of specified (single) permutation of any arbitrary group of qubits
 */
real1_f PermutationProb(quid sid, std::vector<QubitIndexState> q);
/**
 * "Reduced density matrix" probability of specified (single) permutation of any arbitrary group of qubits
 */
real1_f PermutationProbRdm(quid sid, std::vector<QubitIndexState> q, bool r);
/**
 * Expectation value for bit-string integer equivalent of specified arbitrary group of qubits
 */
real1_f PermutationExpectation(quid sid, std::vector<bitLenInt> q);
/**
 * "Reduced density matrix" expectation value for bit-string integer equivalent of specified arbitrary group of qubits
 */
real1_f PermutationExpectationRdm(quid sid, std::vector<bitLenInt> q, bool r);
/**
 * Expectation value for bit-string integer from group of qubits with per-qubit integer expectation value
 */
real1_f FactorizedExpectation(quid sid, std::vector<QubitIntegerExpVar> q);
/**
 * "Reduced density matrix" Expectation value for bit-string integer from group of qubits with per-qubit integer
 * expectation value
 */
real1_f FactorizedExpectationRdm(quid sid, std::vector<QubitIntegerExpVar> q, bool r);
/**
 * Expectation value for bit-string integer from group of qubits with per-qubit real1 expectation value
 */
real1_f FactorizedExpectationFp(quid sid, std::vector<QubitRealExpVar> q);
/**
 * "Reduced density matrix" Expectation value for bit-string integer from group of qubits with per-qubit real1
 * expectation value
 */
real1_f FactorizedExpectationFpRdm(quid sid, std::vector<QubitRealExpVar> q, bool r);
/**
 * Get the single-qubit (3-parameter) operator expectation value for the array of qubits and bases.
 */
real1_f UnitaryExpectation(quid sid, std::vector<QubitU3Basis> q);
/**
 * Get the single-qubit (2x2) operator expectation value for the array of qubits and bases.
 */
real1_f MatrixExpectation(quid sid, std::vector<QubitMatrixBasis> q);
/**
 * Get the single-qubit (3-parameter) operator expectation value for the array of qubits and bases.
 */
real1_f UnitaryExpectationEigenVal(quid sid, std::vector<QubitU3BasisEigenVal> q);
/**
 * Get the single-qubit (2x2) operator expectation value for the array of qubits and bases.
 */
real1_f MatrixExpectationEigenVal(quid sid, std::vector<QubitMatrixBasisEigenVal> q);
/**
 * Pauli operator expectation value for the array of qubits and bases.
 */
real1_f PauliExpectation(quid sid, std::vector<QubitPauliBasis> q);
/**
 * Variance for bit-string integer equivalent of specified arbitrary group of qubits
 */
real1_f Variance(quid sid, std::vector<bitLenInt> q);
/**
 * "Reduced density matrix" variance for bit-string integer equivalent of specified arbitrary group of qubits
 */
real1_f VarianceRdm(quid sid, std::vector<bitLenInt> q, bool r);
/**
 * Variance for bit-string integer from group of qubits with per-qubit integer variance
 */
real1_f FactorizedVariance(quid sid, std::vector<QubitIntegerExpVar> q);
/**
 * "Reduced density matrix" variance for bit-string integer from group of qubits with per-qubit integer
 * variance
 */
real1_f FactorizedVarianceRdm(quid sid, std::vector<QubitIntegerExpVar> q, bool r);
/**
 * Variance for bit-string integer from group of qubits with per-qubit real1 variance
 */
real1_f FactorizedVarianceFp(quid sid, std::vector<QubitRealExpVar> q);
/**
 * "Reduced density matrix" variance for bit-string integer from group of qubits with per-qubit real1
 * variance
 */
real1_f FactorizedVarianceFpRdm(quid sid, std::vector<QubitRealExpVar> q, bool r);
/**
 * Get the single-qubit (3-parameter) operator variance for the array of qubits and bases.
 */
real1_f UnitaryVariance(quid sid, std::vector<QubitU3Basis> q);
/**
 * Get the single-qubit (2x2) operator variance for the array of qubits and bases.
 */
real1_f MatrixVariance(quid sid, std::vector<QubitMatrixBasis> q);
/**
 * Get the single-qubit (3-parameter) operator variance for the array of qubits and bases.
 */
real1_f UnitaryVarianceEigenVal(quid sid, std::vector<QubitU3BasisEigenVal> q);
/**
 * Get the single-qubit (2x2) operator variance for the array of qubits and bases.
 */
real1_f MatrixVarianceEigenVal(quid sid, std::vector<QubitMatrixBasisEigenVal> q);
/**
 * Pauli operator variance for the array of qubits and bases.
 */
real1_f PauliVariance(quid sid, std::vector<QubitPauliBasis> q);

/**
 * Select from a distribution of "p.size()" count of elements according to the discrete probabilities in "p."
 */
size_t random_choice(quid sid, std::vector<real1> p);

/**
 * Applies e^(i*angle) phase factor to all combinations of bits with odd parity, based upon permutations of qubits.
 */
void PhaseParity(quid sid, real1_f lambda, std::vector<bitLenInt> q);

/**
 * Applies a -2 * PI_R1 / (2^N) phase rotation to each qubit.
 */
void PhaseRootN(quid sid, bitLenInt p, std::vector<bitLenInt> q);

/**
 * Overall probability of any odd permutation of the masked set of bits
 */
real1_f JointEnsembleProbability(quid sid, std::vector<QubitPauliBasis> q);

// SPAM and non-unitary

/**
 * Measure single qubit (according to Born rules) and return the result
 */
bool M(quid sid, bitLenInt q);
/**
 * (PSEUDO-QUANTUM:) Force measurement result of single qubit (and return the result)
 */
bool ForceM(quid sid, bitLenInt q, bool r);
/**
 * Each in its specified Pauli basis, collapse an ensemble of qubits jointly via measurement.
 */
bool Measure(quid sid, std::vector<QubitPauliBasis> q);
/**
 * Measure all qubits (according to Born rules) and return the result as a bit string (integer).
 */
bitCapInt MAll(quid sid);
/**
 * Repeat (Z-basis) measurement of a set of qubits for a count of "shots" (without collapsing the simulator)
 */
std::vector<long long unsigned int> MeasureShots(quid sid, std::vector<bitLenInt> q, unsigned s);
/**
 * Set simulator to |0> permutation state
 */
void ResetAll(quid sid);

// single-qubit gates
void X(quid sid, bitLenInt q);
void Y(quid sid, bitLenInt q);
void Z(quid sid, bitLenInt q);
void H(quid sid, bitLenInt q);
void S(quid sid, bitLenInt q);
void SX(quid sid, bitLenInt q);
void SY(quid sid, bitLenInt q);
void T(quid sid, bitLenInt q);
void AdjS(quid sid, bitLenInt q);
void AdjSX(quid sid, bitLenInt q);
void AdjSY(quid sid, bitLenInt q);
void AdjT(quid sid, bitLenInt q);
void U(quid sid, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda);
void Mtrx(quid sid, std::vector<complex> m, bitLenInt q);

// multi-controlled single-qubit gates
void MCX(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCY(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCZ(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCH(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCS(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCT(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCAdjS(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCAdjT(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MCU(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda);
void MCMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q);
// multi-("anti"-) controlled single-qubits gates (that activate when all controls are |0>)
void MACX(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACY(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACZ(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACH(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACS(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACT(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACAdjS(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACAdjT(quid sid, std::vector<bitLenInt> c, bitLenInt q);
void MACU(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1_f theta, real1_f phi, real1_f lambda);
void MACMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q);

/**
 * Multi-controlled gate that activates only for the specified permutation of controls, "p"
 */
void UCMtrx(quid sid, std::vector<bitLenInt> c, std::vector<complex> m, bitLenInt q, bitCapIntOcl p);
/**
 * Multi-controlled, single-target multiplexer gate
 */
void Multiplex1Mtrx(quid sid, std::vector<bitLenInt> c, bitLenInt q, std::vector<complex> m);

// coalesced single-qubit gates
void MX(quid sid, std::vector<bitLenInt> q);
void MY(quid sid, std::vector<bitLenInt> q);
void MZ(quid sid, std::vector<bitLenInt> q);

// single-qubit rotations
void R(quid sid, real1_f phi, QubitPauliBasis q);
// multi-controlled single-qubit rotations
void MCR(quid sid, real1_f phi, std::vector<bitLenInt> c, QubitPauliBasis q);

// exponential of Pauli operators
void Exp(quid sid, real1_f phi, std::vector<QubitPauliBasis> q);
// multi-controlled exponential of Pauli operators
void MCExp(quid sid, real1_f phi, std::vector<bitLenInt> c, std::vector<QubitPauliBasis> q);

// swap variants
void SWAP(quid sid, bitLenInt qi1, bitLenInt qi2);
void ISWAP(quid sid, bitLenInt qi1, bitLenInt qi2);
void AdjISWAP(quid sid, bitLenInt qi1, bitLenInt qi2);
void FSim(quid sid, real1_f theta, real1_f phi, bitLenInt qi1, bitLenInt qi2);
void CSWAP(quid sid, std::vector<bitLenInt> c, bitLenInt qi1, bitLenInt qi2);
void ACSWAP(quid sid, std::vector<bitLenInt> c, bitLenInt qi1, bitLenInt qi2);

// Schmidt decomposition
void Compose(quid sid1, quid sid2, std::vector<bitLenInt> q);
quid Decompose(quid sid, std::vector<bitLenInt> q);
void Dispose(quid sid, std::vector<bitLenInt> q);

// Quantum boolean (Toffoli) operations:
// Two qubits input
void AND(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
void OR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
void XOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
void NAND(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
void NOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
void XNOR(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo);
// One qubit, one classical bit input
void CLAND(quid sid, bool ci, bitLenInt qi, bitLenInt qo);
void CLOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo);
void CLXOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo);
void CLNAND(quid sid, bool ci, bitLenInt qi, bitLenInt qo);
void CLNOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo);
void CLXNOR(quid sid, bool ci, bitLenInt qi, bitLenInt qo);

/**
 * Quantum Fourier Transform
 */
void QFT(quid sid, std::vector<bitLenInt> q);
/**
 * (Inverse) Quantum Fourier Transform
 */
void IQFT(quid sid, std::vector<bitLenInt> q);

#if ENABLE_ALU
// Arithmetic logic unit:
// Two's complement
void ADD(quid sid, bitCapInt a, std::vector<bitLenInt> q);
void SUB(quid sid, bitCapInt a, std::vector<bitLenInt> q);
// Overflow
void ADDS(quid sid, bitCapInt a, bitLenInt s, std::vector<bitLenInt> q);
void SUBS(quid sid, bitCapInt a, bitLenInt s, std::vector<bitLenInt> q);
// Controlled
void MCADD(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q);
void MCSUB(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q);
// In-place
void MUL(quid sid, bitCapInt a, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void DIV(quid sid, bitCapInt a, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
// Modulo, out-of-place
void MULN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void DIVN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void POWN(quid sid, bitCapInt a, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
// Controlled in-place
void MCMUL(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void MCDIV(quid sid, bitCapInt a, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
// Controlled modulo, out-of-place
void MCMULN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void MCDIVN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);
void MCPOWN(
    quid sid, bitCapInt a, std::vector<bitLenInt> c, bitCapInt m, std::vector<bitLenInt> q, std::vector<bitLenInt> o);

#if 0
// Amplitude amplification
void LDA(quid sid, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t);
void ADC(quid sid, bitLenInt s, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t);
void SBC(quid sid, bitLenInt s, std::vector<bitLenInt> qi, std::vector<bitLenInt> qv, std::vector<unsigned char> t);
void Hash(quid sid, std::vector<bitLenInt> q, std::vector<unsigned char> t);
#endif
#endif

// Utility functions
/**
 * Try to factorize a single-qubit subsystem out of "bulk" simulator state. (This can improve efficiency but has no
 * logical effect.)
 */
bool TrySeparate1Qb(quid sid, bitLenInt qi1);
/**
 * Try to factorize a two-qubit subsystem out of "bulk" simulator state. (This can improve efficiency but has no logical
 * effect.)
 */
bool TrySeparate2Qb(quid sid, bitLenInt qi1, bitLenInt qi2);
/**
 * Try to factorize a qubit subsystem out of "bulk" simulator state. (This can improve efficiency but has no logical
 * effect.)
 */
bool TrySeparateTol(quid sid, std::vector<bitLenInt> q, real1_f tol);
/**
 * Force (inexact) factorization of qubit subsystem out of "bulk" simulator state.
 */
void Separate(quid sid, std::vector<bitLenInt> q);
/**
 * Report fidelity for "Schmidt decomposition rounding parameter" (SDRP) and "near-Clifford rounding"
 */
double GetUnitaryFidelity(quid sid);
/**
 * Reset fidelity to 1 for "Schmidt decomposition rounding parameter" (SDRP) and "near-Clifford rounding"
 */
void ResetUnitaryFidelity(quid sid);
/**
 * Set "Schmidt decomposition rounding parameter" (SDRP) value (see https://arxiv.org/abs/2304.14969)
 */
void SetSdrp(quid sid, double sdrp);
/**
 * Set "Near-Clifford rounding parameter"
 */
void SetNcrp(quid sid, double sdrp);
/**
 * Turn off/on "reactive separation" feature (for less/more aggressive automatic state factorization)
 */
void SetReactiveSeparate(quid sid, bool irs);
/**
 * Turn off/on "T-injection" feature (for "near-Clifford" simulation with RZ gates)
 */
void SetTInjection(quid sid, bool iti);
/**
 * Set noise parameter (for QInterfaceNoisy)
 */
void SetNoiseParameter(quid sid, double np);
/**
 * Normalize the state (which should never be necessary unless Decompose() is "abused")
 */
void Normalize(quid sid);

/**
 * Initialize a "quantum neuron" that takes a list of qubit "controls" for input and acts on a single "target" output
 * qubit.
 */
quid init_qneuron(quid sid, std::vector<bitLenInt> c, bitLenInt q, QNeuronActivationFn f, real1_f a, real1_f tol);
/**
 * "Clone" a quantum neuron (which is a classical state)
 */
quid clone_qneuron(quid nid);
/**
 * "Destroy" or release simulator allocation
 */
void destroy_qneuron(quid nid);
/**
 * Set the (RY-rotation) angle parameters for each permutation of quantum neuron input qubits
 */
void set_qneuron_angles(quid nid, std::vector<real1> angles);
/**
 * Get the (RY-rotation) angle parameters for each permutation of quantum neuron input qubits
 */
std::vector<real1> get_qneuron_angles(quid nid);
/**
 * Set the "leakage" parameter for "leaky" quantum neuron activation functions
 */
void set_qneuron_alpha(quid nid, real1_f alpha);
/**
 * Get the "leakage" parameter for "leaky" quantum neuron activation functions
 */
real1_f get_qneuron_alpha(quid nid);
/**
 * Set the activation function for a quantum neuron
 */
void set_qneuron_activation_fn(quid nid, QNeuronActivationFn f);
/**
 * Get the activation function for a quantum neuron
 */
QNeuronActivationFn get_qneuron_activation_fn(quid nid);
/**
 * Infer quantum neuron output from inputs (after training)
 */
real1_f qneuron_predict(quid nid, bool e, bool r);
/**
 * Perform the inverse of quantum neuron inference (for "uncomputation")
 */
real1_f qneuron_unpredict(quid nid, bool e);
/**
 * Train a quantum neuron for one epoch, and also uncompute the intermediate side-effects
 */
real1_f qneuron_learn_cycle(quid nid, bool e);
/**
 * Train a quantum neuron for one epoh (without uncomputing any intermediate side-effects)
 */
void qneuron_learn(quid nid, real1_f eta, bool e, bool r);
/**
 * Train a quantum neuron for one epoch, assuming that the input state is a Z-basis eigenstate.
 */
void qneuron_learn_permutation(quid nid, real1_f eta, bool e, bool r);

// Quantum circuit objects
quid init_qcircuit(bool collapse, bool clifford);
quid init_qcircuit_clone(quid cid);
quid qcircuit_inverse(quid cid);
quid qcircuit_past_light_cone(quid cid, std::set<bitLenInt> q);
void destroy_qcircuit(quid cid);
bitLenInt get_qcircuit_qubit_count(quid cid);
void qcircuit_swap(quid cid, bitLenInt q1, bitLenInt q2);
void qcircuit_append_1qb(quid cid, std::vector<complex> m, bitLenInt q);
void qcircuit_append_mc(quid cid, std::vector<complex> m, std::vector<bitLenInt> c, bitLenInt q, bitCapInt p);
void qcircuit_run(quid cid, quid sid);
void qcircuit_out_to_file(quid cid, std::string f);
void qcircuit_in_from_file(quid cid, std::string f);
std::string qcircuit_out_to_string(quid cid);
} // namespace Qrack
