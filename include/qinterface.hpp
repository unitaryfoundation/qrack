//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <ctime>
#include <map>
#include <math.h>
#include <memory>
#include <random>
#include <vector>
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

#include "config.h"

#if ENABLE_COMPLEX8
#include <complex>
#define complex std::complex<float>
#define real1 float
#define min_norm 1e-9
#define polar(A, B) std::polar(A, B)
#else
#include "common/complex16simd.hpp"
#define complex Complex16Simd
#define real1 double
#define min_norm 1e-15
#endif

// The state vector must be an aligned piece of RAM, to be used by OpenCL.
// We align to an ALIGN_SIZE byte boundary.
#define ALIGN_SIZE 64

namespace Qrack {

class QInterface;
typedef std::shared_ptr<QInterface> QInterfacePtr;

/**
 * Enumerated list of supported engines.
 *
 * Use QINTERFACE_OPTIMAL for the best supported engine.
 */
enum QInterfaceEngine {

    /**
     * Create a QEngineCPU leveraging only local CPU and memory resources.
     */
    QINTERFACE_CPU = 0,
    /**
     * Create a QEngineOCL, derived from QEngineCPU, leveraging OpenCL hardware
     * to increase the speed of certain calculations.
     */
    QINTERFACE_OPENCL,

    /**
     * Create a QEngineOCLMUlti, composed from multiple QEngineOCLs, using OpenCL
     * in parallel across 2^N devices, for N an integer >= 0.
     */
    QINTERFACE_OPENCL_MULTI,

    /**
     * Create a QUnit, which utilizes other QInterface classes to minimize the
     * amount of work that's needed for any given operation based on the
     * entanglement of the bits involved.
     *
     * This, combined with QINTERFACE_OPTIMAL, is the recommended object to use
     * as a library consumer.
     */
    QINTERFACE_QUNIT,

    QINTERFACE_FIRST = QINTERFACE_CPU,
#if ENABLE_OPENCL
    QINTERFACE_OPTIMAL = QINTERFACE_OPENCL,
#else
    QINTERFACE_OPTIMAL = QINTERFACE_CPU,
#endif

    QINTERFACE_MAX
};

/**
 * A "Qrack::QInterface" is an abstract interface exposing qubit permutation
 * state vector with methods to operate on it as by gates and register-like
 * instructions.
 *
 * See README.md for an overview of the algorithms Qrack employs.
 */
class QInterface {
protected:
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    real1 runningNorm;
    bool doNormalize;

    uint32_t randomSeed;
    std::shared_ptr<std::default_random_engine> rand_generator;
    std::uniform_real_distribution<real1> rand_distribution;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
    }

    /** Generate a random real1 from 0 to 1 */
    virtual real1 Rand() { return rand_distribution(*rand_generator); }
    virtual void SetRandomSeed(uint32_t seed) { rand_generator->seed(seed); }

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm) = 0;
    virtual void ApplyControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyDoublyControlled2x2(
        bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyDoublyAntiControlled2x2(
        bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyM(bitCapInt qPower, bool result, complex nrm) = 0;
    virtual void NormalizeState(real1 nrm = -999.0) = 0;

public:
    QInterface(bitLenInt n, std::shared_ptr<std::default_random_engine> rgp = nullptr, bool doNorm = true)
        : doNormalize(doNorm)
        , rand_distribution(0.0, 1.0)
    {
        SetQubitCount(n);

        if (rgp == NULL) {
            rand_generator = std::make_shared<std::default_random_engine>();
            randomSeed = std::time(0);
            SetRandomSeed(randomSeed);
        } else {
            rand_generator = rgp;
        }
    }

    /** Destructor of QInterface */
    virtual ~QInterface(){};

    /** Get the count of bits in this register */
    int GetQubitCount() { return qubitCount; }

    /** Get the maximum number of basis states, namely \f$ n^2 \f$ for \f$ n \f$ qubits*/
    int GetMaxQPower() { return maxQPower; }

    /** Set an arbitrary pure quantum state */
    virtual void SetQuantumState(complex* inputState) = 0;

    /** Set to a specific permutation */
    virtual void SetPermutation(bitCapInt perm) = 0;

    /**
     * Combine another QInterface with this one, after the last bit index of
     * this one.
     *
     * "Cohere" combines the quantum description of state of two independent
     * QInterface objects into one object, containing the full permutation
     * basis of the full object. The "inputState" bits are added after the last
     * qubit index of the QInterface to which we "Cohere." Informally,
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
     * QInterface should be "thrown away" to satisfy "no clone theorem." This
     * is not semantically enforced in Qrack, because optimization of an
     * emulator might be acheived by "cloning" "under-the-hood" while only
     * exposing a quantum mechanically consistent API or instruction set.
     *
     * Returns the quantum bit offset that the QInterface was appended at, such
     * that bit 5 in toCopy is equal to offset+5 in this object.
     */
    virtual bitLenInt Cohere(QInterfacePtr toCopy) = 0;
    virtual std::map<QInterfacePtr, bitLenInt> Cohere(std::vector<QInterfacePtr> toCopy);

    /**
     * Minimally decohere a set of contiguous bits from the full coherent unit,
     * into "destination."
     *
     * Minimally decohere a set of contigious bits from the full coherent unit.
     * The length of this coherent unit is reduced by the length of bits
     * decohered, and the bits removed are output in the destination
     * QInterface pointer. The destination object must be initialized to the
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
     * QInterface, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest) = 0;

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
     * QInterface, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Dispose(bitLenInt start, bitLenInt length) = 0;

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    /**
     * Apply an arbitrary single bit unitary transformation.
     *
     * If float rounding from the application of the matrix might change the state vector norm, "doCalcNorm" should be
     * set to true.
     */
    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);

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

    /** @} */

    /**
     * \defgroup LogicGates Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     *
     * @{
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

    /** @} */

    /**
     * \defgroup RotGates Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     *
     * @{
     */

    /**
     * Phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    virtual void RT(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction phase shift gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction X axis rotation gate
     *
     * Rotates \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * (Identity) Exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$, exponentiation of the identity operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_x / 2^denomPower} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_y / 2^denomPower} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_z / 2^denomPower} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around Pauli Y
     * axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli Y axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around Pauli Z
     * axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli Z axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */

    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{i*{\pi *
     * numerator} / 2^denomPower} \f$ around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup RegGates Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     *
     * @{
     */

    /** Bitwise Hadamard */
    virtual void H(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli X (or logical "NOT") operator */
    virtual void X(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Y operator */
    virtual void Y(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Z operator */
    virtual void Z(bitLenInt start, bitLenInt length);

    /** Bitwise controlled-not */
    virtual void CNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise "anti-"controlled-not */
    virtual void AntiCNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise doubly controlled-not */
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise doubly "anti-"controlled-not */
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /**
     * Bitwise "AND"
     *
     * "AND" registers at "inputStart1" and "inputStart2," of "length" bits,
     * placing the result in "outputStart".
     */
    virtual void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /**
     * Classical bitwise "AND"
     *
     * "AND" registers at "inputStart1" and the classic bits of "classicalInput," of "length" bits,
     * placing the result in "outputStart".
     */
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "OR" */
    virtual void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "OR" */
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "XOR" */
    virtual void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "XOR" */
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /**
     * Bitwise phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    virtual void RT(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction phase shift gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction X axis rotation gate
     *
     * Rotates \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around Pauli Y
     * axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli Y axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ e^{i*{\pi * numerator} / 2^denomPower} \f$ around Pauli Z
     * axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{i*{\pi * numerator} /
     * 2^denomPower} \f$ around Pauli Z axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{i*{\pi *
     * numerator} / 2^denomPower} \f$ around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise (identity) exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$, exponentiation of the identity operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_x / 2^denomPower} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_y / 2^denomPower} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i * \pi * numerator * \sigma_z / 2^denomPower} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * \todo Many of these have performance that can be improved in QUnit via
     *       implementations with more intelligently chosen Cohere/Decompose
     *       patterns.
     * @{
     */

    /** Arithmetic shift left, with last 2 bits as sign and carry */
    virtual void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Arithmetic shift right, with last 2 bits as sign and carry */
    virtual void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift left, filling the extra bits with |0> */
    virtual void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift right, filling the extra bits with |0> */
    virtual void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift left - shift bits left, and carry last bits. */
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift right - shift bits right, and carry first bits. */
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Add integer (without sign) */
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;

    /** Add integer (without sign, with carry) */
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Add a classical integer to the register, with sign and without carry. */
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;

    /** Add a classical integer to the register, with sign and with carry. */
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;

    /** Add a classical integer to the register, with sign and with (phase-based) carry. */
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Add classical BCD integer (without sign) */
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;

    /** Add classical BCD integer (without sign, with carry) */
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract classical integer (without sign) */
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) = 0;

    /** Subtract classical integer (without sign, with carry) */
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract a classical integer from the register, with sign and without carry. */
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract BCD integer (without sign) */
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;

    /** Subtract BCD integer (without sign, with carry) */
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    /** Quantum Fourier Transform - Apply the quantum Fourier transform to the register. */
    virtual void QFT(bitLenInt start, bitLenInt length);

    /** Reverse the phase of the state where the register equals zero. */
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) = 0;

    /** The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation. */
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) = 0;

    /** Phase flip always - equivalent to Z X Z X on any bit in the QInterface */
    virtual void PhaseFlip() = 0;

    /** Set register bits to given permutation */
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /** Measure permutation state of a register */
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

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
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
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
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values) = 0;

    /**
     * Add to entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * inputStart" is the start index of 8 qubits that act as an index into the
     * 256 byte "values" array. The "outputStart" bits would usually already be
     * entangled with the "inputStart" bits via a IndexedLDA() operation.
     * With the "inputStart" bits being a "key" and the "outputStart" bits
     * being a value, the permutation state |key, value> is mapped to |key,
     * value + values[key]>. This is similar to classical parallel addition of
     * two arrays.  However, when either of the registers are measured, both
     * registers will collapse into one random VALID key-value pair, with any
     * addition or subtraction done to the "value." See IndexedLDA() for
     * context.
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;

    /**
     * Subtract from an entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a IndexedLDA()
     * operation.  With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value - values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See QInterface::IndexedLDA for context.
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;

    /** Swap values of two bits in register */
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /** Bitwise swap */
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Reverse all of the bits in a sequence. */
    virtual void Reverse(bitLenInt first, bitLenInt last)
    {
        while ((first < last) && (first < (last - 1))) {
            last--;
            Swap(first, last);
            first++;
        }
    }

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    /**
     * Direct copy of raw state vector to produce a clone
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void CopyState(QInterfacePtr orig) = 0;

    /**
     * Direct measure of bit probability to be in |1> state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 Prob(bitLenInt qubitIndex) = 0;

    /**
     * Direct measure of full register probability to be in permutation state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 ProbAll(bitCapInt fullRegister) = 0;

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

    /**
     * Act as though a measurement was applied, except force the result of the measurement.
     *
     * That is, genuine measurement of a qubit in superposition has a probabilistic result. This method allows the
     * programmer to choose the outcome of the measurement, and proceed as if the measurement randomly resulted in the
     * chosen bit value.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, real1 nrmlzr = 1.0);

    /** @} */
};
} // namespace Qrack
