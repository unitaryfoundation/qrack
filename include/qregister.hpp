//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>
#include <atomic>
#include <ctime>
#include <future>
#include <math.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <thread>
#include <vector>

#include "common/complex16simd.hpp"

#define Complex16 Complex16Simd
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {

class CoherentUnit;
/** Enumerated list of supported engines. */
enum CoherentUnitEngine {
    COHERENT_UNIT_ENGINE_SOFTWARE_SERIAL = 0,
    COHERENT_UNIT_ENGINE_SOFTWARE_PARALLEL,
    COHERENT_UNIT_ENGINE_SOFTWARE = COHERENT_UNIT_ENGINE_SOFTWARE_PARALLEL,

    COHERENT_UNIT_ENGINE_SOFTWARE_SEPARATED,

    COHERENT_UNIT_ENGINE_OPENCL,

    COHERENT_UNIT_ENGINE_OPENCL_SEPARATED,

    COHERENT_UNIT_ENGINE_MAX
};

/** Create a CoherentUnit leveraging the specified engine. */
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState);

/**
 * A "Qrack::CoherentUnit" is a qubit permutation state vector with methods to
 * operate on it as by gates and register-like instructions. In brief: All
 * directly interacting qubits must be contained in a single CoherentUnit
 * object, by requirement of quantum mechanics, unless a certain collection of
 * bits represents a "separable quantum subsystem." All registers of a virtual
 * chip will usually be contained in a single CoherentUnit, and they are
 * accesible similar to a one-dimensional array of qubits.
 *
 * See README.md for an overview of the algorithms Qrack employs.
 */
class CoherentUnit {
public:
    /**
     * Initialize a coherent unit with qBitCount number of bits, all to |0>
     * state.
     */
    CoherentUnit(bitLenInt qBitCount);

    CoherentUnit(bitLenInt qBitCount, std::shared_ptr<std::default_random_engine> rgp);

    /**
     * Initialize a coherent unit with qBitCount number of bits, all to |0>
     * state, with a specific phase
     *
     * \warning Overall phase is generally arbitrary and unknowable. Setting two CoherentUnit instances to the same
     * phase usually makes sense only if they are initialized at the same time.
     */
    CoherentUnit(bitLenInt qBitCount, Complex16 phaseFac);

    CoherentUnit(bitLenInt qBitCount, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);

    /**
     * Initialize a coherent unit with qBitCount number of bits, to initState
     * unsigned integer permutation state.
     */
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState);

    /**
     * Initialize a coherent unit with qBitCount number of bits, to initState
     * unsigned integer permutation state, with a specific phase.
     *
     * \warning Overall phase is generally arbitrary and unknowable. Setting two CoherentUnit instances to the same
     * phase usually makes sense only if they are initialized at the same time.
     */
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac);

    /**
     * Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with
     * a shared random number generator.
     */
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp);

    /**
     * Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with
     * a shared random number generator, with a specific phase.
     *
     * \warning Overall phase is generally arbitrary and unknowable. Setting two CoherentUnit instances to the same
     * phase usually makes sense only if they are initialized at the same time.
     */
    CoherentUnit(
        bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);

    /**
     * Initialize a cloned register with same exact quantum state as pqs
     *
     * \warning PSEUDO-QUANTUM
     */
    CoherentUnit(const CoherentUnit& pqs);

    /** Destructor of CoherentUnit */
    virtual ~CoherentUnit() {}

    /** Set the random seed (primarily used for testing) */
    virtual void SetRandomSeed(uint32_t seed);

    /** Set the concurrency level */
    void SetConcurrencyLevel(uint32_t cores) { numCores = cores; }

    /** Get the count of bits in this register */
    int GetQubitCount() { return qubitCount; }

    /** Get the 1 << GetQubitCount() */
    int GetMaxQPower() { return maxQPower; }

    /**
     * Output the exact quantum state of this register as a permutation basis
     * array of complex numbers
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void CloneRawState(Complex16* output);

    /** Generate a random double from 0 to 1 */
    double Rand();

    /** Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int */
    void SetPermutation(bitCapInt perm);

    /** Set an arbitrary pure quantum state */
    virtual void SetQuantumState(Complex16* inputState);

    /**
     * Combine (a copy of) another CoherentUnit with this one, after the last
     * bit index of this one.
     *
     * "Cohere" combines the quantum description of state of two independent
     * CoherentUnit objects into one object, containing the full permutation
     * basis of the full object. The "inputState" bits are added after the last
     * qubit index of the CoherentUnit to which we "Cohere." Informally,
     * "Cohere" is equivalent to "just setting another group of qubits down
     * next to the first" without interacting them. Schroedinger's equation can
     * form a description of state for two independent subsystems at once or
     * "separable quantum subsystems" without interacting them. Once the
     * description of state of the independent systems is combined, we can
     * interact them, and we can describe their entanglements to each other, in
     * which case they are no longer independent. A full entangled description
     * of quantum state is not possible for two independent quantum subsystems
     * until we "Cohere" them.
     *
     * "Cohere" multiplies the probabilities of the indepedent permutation
     * states of the two subsystems to find the probabilites of the entire set
     * of combined permutations, by simple combinatorial reasoning. If the
     * probablity of the "left-hand" subsystem being in |00> is 1/4, and the
     * probablity of the "right-hand" subsystem being in |101> is 1/8, than the
     * probability of the combined |00101> permutation state is 1/32, and so on
     * for all permutations of the new combined state.
     *
     * If the programmer doesn't want to "cheat" quantum mechanically, then the
     * original copy of the state which is duplicated into the larger
     * CoherentUnit should be "thrown away" to satisfy "no clone theorem." This
     * is not semantically enforced in Qrack, because optimization of an
     * emulator might be acheived by "cloning" "under-the-hood" while only
     * exposing a quantum mechanically consistent API or instruction set.
     */
    virtual void Cohere(CoherentUnit& toCopy);

    virtual void Cohere(std::vector<std::shared_ptr<CoherentUnit>> toCopy);

    /**
     * Minimally decohere a set of contiguous bits from the full coherent unit,
     * into "destination."
     *
     * Minimally decohere a set of contigious bits from the full coherent unit.
     * The length of this coherent unit is reduced by the length of bits
     * decohered, and the bits removed are output in the destination
     * CoherentUnit pointer. The destination object must be initialized to the
     * correct number of bits, in 0 permutation state. For quantum mechanical
     * accuracy, the bit set removed and the bit set left behind should be
     * quantum mechanically "separable."
     *
     * Like how "Cohere" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Decohere" is like "just moving a few qubits
     * away from the rest." Schroedinger's equation does not require bits to be
     * explicitly interacted in order to describe their permutation basis, and
     * the descriptions of state of <b>separable</b> subsystems, those which
     * are not entangled with other subsystems, are just as easily removed from
     * the description of state.
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *     prob(|(left) 1000>) = prob(|1000 00>) + prob(|1000 10>) + prob(|1000 01>) + prob(|1000 11>).
     * \f$
     *
     * If the subsystems are not "separable," i.e. if they are entangled, this
     * operation is not well-motivated, and its output is not necessarily
     * defined. (The summing of probabilities over permutations of subsytems
     * will be performed as described above, but this is not quantum
     * mechanically meaningful.) To ensure that the subsystem is "separable,"
     * i.e. that it has no entanglements to other subsystems in the
     * CoherentUnit, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Decohere(bitLenInt start, bitLenInt length, CoherentUnit& destination);

    /**
     * Minimally decohere a set of contigious bits from the full coherent unit,
     * throwing these qubits away.
     *
     * Minimally decohere a set of contigious bits from the full coherent unit,
     * discarding these bits. The length of this coherent unit is reduced by
     * the length of bits decohered. For quantum mechanical accuracy, the bit
     * set removed and the bit set left behind should be quantum mechanically
     * "separable."
     *
     * Like how "Cohere" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Dispose" is like "just moving a few qubits away
     * from the rest, and throwing them in the trash." Schroedinger's equation
     * does not require bits to be explicitly interacted in order to describe
     * their permutation basis, and the descriptions of state of
     * <b>separable</b> subsystems, those which are not entangled with other
     * subsystems, are just as easily removed from the description of state.
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *      prob(|(left) 1000>) = prob(|1000 00>) + prob(|1000 10>) + prob(|1000 01>) + prob(|1000 11>).
     * \f$
     *
     * If the subsystems are not "separable," i.e. if they are entangled, this
     * operation is not well-motivated, and its output is not necessarily
     * defined. (The summing of probabilities over permutations of subsytems
     * will be performed as described above, but this is not quantum
     * mechanically meaningful.) To ensure that the subsystem is "separable,"
     * i.e. that it has no entanglements to other subsystems in the
     * CoherentUnit, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Dispose(bitLenInt start, bitLenInt length);

    /*
     * Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     */

    /**
     * Quantum analog of classical "AND" gate
     *
     * Measures the outputBit, then overwrites it with result.
     */
    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate
     *
     * Measures the outputBit, then overwrites it with result.
     */
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate
     *
     * Measures the outputBit, then overwrites it with result.
     */
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     *  Quantum analog of classical "AND" gate. Takes one qubit input and one
     *  classical bit input. Measures the outputBit, then overwrites it with
     *  result.
     */
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate. Takes one qubit input and one
     * classical bit input. Measures the outputBit, then overwrites it with
     * result.
     */
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate. Takes one qubit input and one
     * classical bit input. Measures the outputBit, then overwrites it with
     * result.
     */
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Doubly-controlled NOT gate
     *
     * If both controls are set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /**
     * Anti doubly-controlled NOT gate
     *
     * If both controls are set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /**
     * Controlled NOT gate
     *
     * If the control is set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CNOT(bitLenInt control, bitLenInt target);

    /**
     * Anti controlled NOT gate
     *
     * If the control is set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);

    /**
     * Hadamard gate
     *
     * Applies a Hadamard gate on qubit at "qubitIndex."
     */
    virtual void H(bitLenInt qubitIndex);

    /**
     * Measurement gate
     *
     * Measures the qubit at "qubitIndex" and returns either "true" or "false."
     * (This "gate" breaks unitarity.)
     *
     * All physical evolution of a quantum state should be "unitary," except
     * measurement. Measurement of a qubit "collapses" the quantum state into
     * either only permutation states consistent with a |0> state for the bit,
     * or else only permutation states consistent with a |1> state for the bit.
     * Measurement also effectively multiplies the overall quantum state vector
     * of the system by a random phase factor, equiprobable over all possible
     * phase angles.
     *
     * Effectively, when a bit measurement is emulated, Qrack calculates the
     * norm of all permutation state components, to find their respective
     * probabilities. The probabilities of all states in which the measured
     * bit is "0" can be summed to give the probability of the bit being "0,"
     * and separately the probabilities of all states in which the measured
     * bit is "1" can be summed to give the probability of the bit being "1."
     * To simulate measurement, a random float between 0 and 1 is compared to
     * the sum of the probability of all permutation states in which the bit
     * is equal to "1". Depending on whether the random float is higher or
     * lower than the probability, the qubit is determined to be either |0> or
     * |1>, (up to phase). If the bit is determined to be |1>, then all
     * permutation eigenstates in which the bit would be equal to |0> have
     * their probability set to zero, and vice versa if the bit is determined
     * to be |0>. Then, all remaining permutation states with nonzero
     * probability are linearly rescaled so that the total probability of all
     * permutation states is again "normalized" to exactly 100% or 1, (within
     * double precision rounding error). Physically, the act of measurement
     * should introduce an overall random phase factor on the state vector,
     * which is emulated by generating another constantly distributed random
     * float to select a phase angle between 0 and 2 * Pi.
     *
     * Measurement breaks unitary evolution of state. All quantum gates except
     * measurement should generally act as a unitary matrix on a permutation
     * state vector. (Note that Boolean comparison convenience methods in Qrack
     * such as "AND," "OR," and "XOR" employ the measurement operation in the
     * act of first clearing output bits before filling them with the result of
     * comparison, and these convenience methods therefore break unitary
     * evolution of state, but in a physically realistic way. Comparable
     * unitary operations would be performed with a combination of X and CCNOT
     * gates, also called "Toffoli" gates, but the output bits would have to be
     * assumed to be in a known fixed state, like all |0>, ahead of time to
     * produce unitary logical comparison operations.)
     */
    virtual bool M(bitLenInt qubitIndex);

    /**
     * X gate
     *
     * Applies the Pauli "X" operator to the qubit at "qubitIndex." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void X(bitLenInt qubitIndex);

    /**
     * Y gate
     *
     * Applies the Pauli "Y" operator to the qubit at "qubitIndex." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void Y(bitLenInt qubitIndex);

    /**
     * Z gate
     *
     * Applies the Pauli "Z" operator to the qubit at "qubitIndex." The Pauli
     * "Z" operator reverses the phase of |1> and leaves |0> unchanged.
     */
    virtual void Z(bitLenInt qubitIndex);

    /**
     * Controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CY(bitLenInt control, bitLenInt target);

    /**
     * Controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CZ(bitLenInt control, bitLenInt target);

    /*
     * Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     */

    /**
     * Phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    virtual void RT(double radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction phase shift gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denominator, bitLenInt qubitIndex);

    /**
     * X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    virtual void RX(double radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction X axis rotation gate
     *
     * Rotates \f$ e^{i*{\pi * numerator} / denominator} \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denominator, bitLenInt qubitIndex);

    /**
     * Controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(double radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /**
     * Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(double radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around Pauli Y
     * axis.
     */
    virtual void RYDyad(int numerator, int denominator, bitLenInt qubitIndex);

    /**
     * Controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(double radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli Y axis.
     */
    virtual void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /**
     * Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(double radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around Pauli Z
     * axis.
     */
    virtual void RZDyad(int numerator, int denominator, bitLenInt qubitIndex);

    /**
     * Controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(double radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli Z axis.
     */
    virtual void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /**
     * Controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */

    virtual void CRT(double radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{i*{\pi *
     * numerator} / denominator} \f$ around |1> state.
     */
    virtual void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /*
     * Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     */

    /** Bitwise Hadamard */
    void H(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli X (or logical "NOT") operator */
    virtual void X(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Y operator */
    void Y(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Z operator */
    void Z(bitLenInt start, bitLenInt length);

    /** Bitwise controlled-not */
    void CNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /**
     * Bitwise "AND"
     *
     * "AND" registers at "inputStart1" and "inputStart2," of "length" bits,
     * placing the result in "outputStart".
     */
    void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /**
     * Classical bitwise "AND"
     *
     * "AND" registers at "inputStart1" and the classic bits of "classicalInput," of "length" bits,
     * placing the result in "outputStart".
     */
    void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "OR" */
    void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "OR" */
    void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "XOR" */
    void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "XOR" */
    void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /**
     * Bitwise phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    void RT(double radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction phase shift gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around |1>
     * state.
     */
    void RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);

    /**
     * Bitwise X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    void RX(double radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction X axis rotation gate
     *
     * Rotates \f$ e^{i*{\pi * numerator} / denominator} \f$ on Pauli x axis.
     */
    void RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    void CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli x axis.
     */
    void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    void RY(double radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around Pauli Y
     * axis.
     */
    void RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    void CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli Y axis.
     */
    void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    void RZ(double radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / denominator} \f$ around Pauli Z
     * axis.
     */
    void RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    void CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * denominator} \f$ around Pauli Z axis.
     */
    void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */
    void CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{i*{\pi *
     * numerator} / denominator} \f$ around |1> state.
     */
    void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    void CY(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /*
     * Arithmetic and other opcode-like gate implemenations.
     */

    /** Arithmetic shift left, with last 2 bits as sign and carry */
    void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Arithmetic shift right, with last 2 bits as sign and carry */
    void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift left, filling the extra bits with |0> */
    void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift right, filling the extra bits with |0> */
    void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift left - shift bits left, and carry last bits. */
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift right - shift bits right, and carry first bits. */
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Add integer (without sign) */
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /** Add integer (without sign, with carry) */
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Add a classical integer to the register, with sign and without carry. */
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /** Add a classical integer to the register, with sign and with carry. */
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);

    /** Add a classical integer to the register, with sign and with (phase-based) carry. */
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Add classical BCD integer (without sign) */
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /** Add classical BCD integer (without sign, with carry) */
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Subtract classical integer (without sign) */
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);

    /** Subtract classical integer (without sign, with carry) */
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Subtract a classical integer from the register, with sign and without carry. */
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Subtract BCD integer (without sign) */
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /** Subtract BCD integer (without sign, with carry) */
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Quantum Fourier Transform - Apply the quantum Fourier transform to the register. */
    virtual void QFT(bitLenInt start, bitLenInt length);

    /** Reverse the phase of the state where the register equals zero. */
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);

    /** The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation. */
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);

    /** Phase flip always - equivalent to Z X Z X on any bit in the CoherentUnit */
    virtual void PhaseFlip();

    /** Set register bits to given permutation */
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /** Measure permutation state of a register */
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    /** Measure permutation state of an 8 bit register */
    unsigned char MReg8(bitLenInt start);

    /**
     * Set 8 bit register bits by a superposed index-offset-based read from
     * classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits are first cleared,
     * then the separable |input, 00000000> permutation state is mapped to
     * |input, values[input]>, with "values[input]" placed in the "outputStart"
     * register.
     *
     * While a CoherentUnit represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM.
     *
     * The physical motivation for this addressing mode can be explained as
     * follows: say that we have a superconducting quantum interface device
     * (SQUID) based chip. SQUIDs have already been demonstrated passing
     * coherently superposed electrical currents. In a sufficiently
     * quantum-mechanically isolated qubit chip with a classical cache, with
     * both classical RAM and registers likely cryogenically isolated from the
     * environment, SQUIDs could (hopefully) pass coherently superposed
     * electrical currents into the classical RAM cache to load values into a
     * qubit register. The state loaded would be a superposition of the values
     * of all RAM to which coherently superposed electrical currents were
     * passed.
     *
     * In qubit system similar to the MOS 6502, say we have qubit-based
     * "accumulator" and "X index" registers, and say that we start with a
     * superposed X index register. In (classical) X addressing mode, the X
     * index register value acts an offset into RAM from a specified starting
     * address. The X addressing mode of a LoaD Accumulator (LDA) instruction,
     * by the physical mechanism described above, should load the accumulator
     * in quantum parallel with the values of every different address of RAM
     * pointed to in superposition by the X index register. The superposed
     * values in the accumulator are entangled with those in the X index
     * register, by way of whatever values the classical RAM pointed to by X
     * held at the time of the load. (If the RAM at index "36" held an unsigned
     * char value of "27," then the value "36" in the X index register becomes
     * entangled with the value "27" in the accumulator, and so on in quantum
     * parallel for all superposed values of the X index register, at once.) If
     * the X index register or accumulator are then measured, the two registers
     * will both always collapse into a random but valid key-value pair of X
     * index offset and value at that classical RAM address.
     *
     * Note that a "superposed store operation in classical RAM" is not
     * possible by analagous reasoning. Classical RAM would become entangled
     * with both the accumulator and the X register. When the state of the
     * registers was collapsed, we would find that only one "store" operation
     * to a single memory address had actually been carried out, consistent
     * with the address offset in the collapsed X register and the byte value
     * in the collapsed accumulator. It would not be possible by this model to
     * write in quantum parallel to more than one address of classical memory
     * at a time.
     */
    virtual unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);

    /**
     * Add to entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * inputStart" is the start index of 8 qubits that act as an index into the
     * 256 byte "values" array. The "outputStart" bits would usually already be
     * entangled with the "inputStart" bits via a SuperposeReg8() operation.
     * With the "inputStart" bits being a "key" and the "outputStart" bits
     * being a value, the permutation state |key, value> is mapped to |key,
     * value + values[key]>. This is similar to classical parallel addition of
     * two arrays.  However, when either of the registers are measured, both
     * registers will collapse into one random VALID key-value pair, with any
     * addition or subtraction done to the "value." See SuperposeReg8() for
     * context.
     *
     * While a CoherentUnit represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "AdcSuperposReg8" and "SbcSuperposeReg8" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * SuperposeReg8().
     */
    virtual unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

    /**
     * Subtract from an entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a SuperposeReg8()
     * operation.  With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value - values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See CoherentUnit::SuperposeReg8 for context.
     *
     * While a CoherentUnit represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "AdcSuperposReg8" and "SbcSuperposeReg8" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * SuperposeReg8().
     */
    virtual unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

    /**
     * Direct measure of bit probability to be in |1> state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual double Prob(bitLenInt qubitIndex);

    /**
     * Direct measure of full register probability to be in permutation state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual double ProbAll(bitCapInt fullRegister);

    /**
     * Direct measure of all bit probabilities in register to be in |1> state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void ProbArray(double* probArray);

    /**
     * Set individual bit to pure |0> (false) or |1> (true) state
     *
     * To set a bit, the bit is first measured. If the result of measurement
     * matches "value," the bit is considered set.  If the result of
     * measurement is the opposite of "value," an X gate is applied to the bit.
     * The state ends up entirely in the "value" state, with a random phase
     * factor.
     */
    virtual void SetBit(bitLenInt qubitIndex1, bool value);

    /** Swap values of two bits in register */
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /** Bitwise swap */
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

protected:
    /// Constructor for SeparatedUnit
    CoherentUnit();

    const bitCapInt ParStride = 8;
    uint32_t randomSeed;
    double runningNorm;
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    std::unique_ptr<Complex16[]> stateVec;
    std::vector<std::unique_ptr<Complex16[]>> gateQueue;
    std::vector<bool> isQueued;

    std::shared_ptr<std::default_random_engine> rand_generator_ptr;
    std::uniform_real_distribution<double> rand_distribution;

    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm);
    void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm);
    void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx);
    void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx);
    void Carry(bitLenInt integerStart, bitLenInt integerLength, bitLenInt carryBit);
    void NormalizeState();
    void Reverse(bitLenInt first, bitLenInt last);
    void UpdateRunningNorm();
    void Mul2x2(const Complex16* leftIn, Complex16* rightOut);
    void FlushQueue(bitLenInt index);
    void FlushQueue(bitLenInt start, bitLenInt length);
    void ResetQueue(bitLenInt index);
    void ResetQueue(bitLenInt start, bitLenInt length);
    bool CheckQueued(bitLenInt start, bitLenInt length);

public:
    /*
     * Parallelization routines for spreading work across multiple cores.
     */

    /** Called once per value between begin and end. */
    typedef std::function<void(const bitCapInt, const int)> ParallelFunc;
    typedef std::function<bitCapInt(const bitCapInt, const int)> IncrementFunc;

    /**
     * Iterate through the permutations a maximum of end-begin times, allowing
     * the caller to control the incrementation offset through 'inc'.
     */
    void par_for_inc(const bitCapInt begin, const bitCapInt end, IncrementFunc, ParallelFunc fn);

    /** Call fn once for every numerical value between begin and end. */
    void par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn);

    /**
     * Skip over the skipPower bits.
     *
     * For example, if skipPower is 2, it will count:
     *   0000, 0001, 0100, 0101, 1000, 1001, 1100, 1101.
     *     ^     ^     ^     ^     ^     ^     ^     ^ - The second bit is
     *                                                   untouched.
     */
    void par_for_skip(const bitCapInt begin, const bitCapInt end, const bitCapInt skipPower,
        const bitLenInt skipBitCount, ParallelFunc fn);

    /** Skip over the bits listed in maskArray in the same fashion as par_for_skip. */
    void par_for_mask(
        const bitCapInt, const bitCapInt, const bitCapInt* maskArray, const bitLenInt maskLen, ParallelFunc fn);

    /** Calculate the normal for the array. */
    double par_norm(const bitCapInt maxQPower, const Complex16* stateArray);

protected:
    int32_t numCores;

}; // namespace Qrack

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

} // namespace Qrack
