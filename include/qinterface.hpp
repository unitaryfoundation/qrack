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

#include "common/parallel_for.hpp"
#include "common/pauli.hpp"
#include "common/rdrandwrapper.hpp"
#include "hamiltonian.hpp"

#include <map>
#include <random>

#if ENABLE_UINT128
#include <ostream>
#endif

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
     * Create a QEngineOCL, leveraging OpenCL hardware to increase the speed of certain calculations.
     */
    QINTERFACE_OPENCL,

    /**
     * Create a QEngineCUDA, leveraging CUDA hardware to increase the speed of certain calculations.
     */
    QINTERFACE_CUDA,

    /**
     * Create a QHybrid, switching between QEngineCPU and QEngineOCL as efficient.
     */
    QINTERFACE_HYBRID,

    /**
     * Create a QBinaryDecisionTree, (CPU-based).
     */
    QINTERFACE_BDT,

    /**
     * Create a QBinaryDecisionTree, (CPU-based).
     */
    QINTERFACE_BDT_HYBRID,

    /**
     * Create a QStabilizer, limited to Clifford/Pauli operations, but efficient.
     */
    QINTERFACE_STABILIZER,

    /**
     * Create a QStabilizerHybrid, switching between a QStabilizer and a QHybrid as efficient.
     */
    QINTERFACE_STABILIZER_HYBRID,

    /**
     * Create a QPager, which breaks up the work of a QEngine into equally sized "pages."
     */
    QINTERFACE_QPAGER,

    /**
     * Create a QUnit, which utilizes other QInterface classes to minimize the amount of work that's needed for any
     * given operation based on the entanglement of the bits involved.
     *
     * This, combined with QINTERFACE_OPTIMAL, is the recommended object to use as a library
     * consumer.
     */
    QINTERFACE_QUNIT,

    /**
     * Create a QUnitMulti, which distributes the explicitly separated "shards" of a QUnit across available OpenCL
     * devices.
     */
    QINTERFACE_QUNIT_MULTI,

    /**
     * Clifford-specialized QUnit
     */
    QINTERFACE_QUNIT_CLIFFORD,

    /**
     * Circuit-simplification layer
     */
    QINTERFACE_TENSOR_NETWORK,

    /**
     * Noisy wrapper layer
     */
    QINTERFACE_NOISY,

#if ENABLE_OPENCL || ENABLE_CUDA
    QINTERFACE_OPTIMAL_SCHROEDINGER = QINTERFACE_QPAGER,
#if ENABLE_OPENCL
    QINTERFACE_OPTIMAL_BASE = QINTERFACE_OPENCL,
#else
    QINTERFACE_OPTIMAL_BASE = QINTERFACE_CUDA,
#endif
#else
    QINTERFACE_OPTIMAL_SCHROEDINGER = QINTERFACE_CPU,

    QINTERFACE_OPTIMAL_BASE = QINTERFACE_CPU,
#endif

    QINTERFACE_OPTIMAL = QINTERFACE_QUNIT,

    QINTERFACE_OPTIMAL_MULTI = QINTERFACE_QUNIT_MULTI,

    QINTERFACE_MAX
};

/**
 * A "Qrack::QInterface" is an abstract interface exposing qubit permutation
 * state vector with methods to operate on it as by gates and register-like
 * instructions.
 *
 * See README.md for an overview of the algorithms Qrack employs.
 */
class QInterface : public ParallelFor {
protected:
    bool doNormalize;
    bool randGlobalPhase;
    bool useRDRAND;
    bitLenInt qubitCount;
    uint32_t randomSeed;
    real1 amplitudeFloor;
    bitCapInt maxQPower;
    qrack_rand_gen_ptr rand_generator;
    std::uniform_real_distribution<real1_s> rand_distribution;
    std::shared_ptr<RdRandom> hardware_rand_generator;

    // Compilers have difficulty figuring out types and overloading if the "norm" handle is passed to std::transform. If
    // you need a safe pointer to norm(), try this:
    static inline real1_f normHelper(const complex& c) { return (real1_f)norm(c); }

    static inline real1_f clampProb(real1_f toClamp)
    {
        if (toClamp < ZERO_R1_F) {
            toClamp = ZERO_R1_F;
        }
        if (toClamp > ONE_R1_F) {
            toClamp = ONE_R1_F;
        }
        return toClamp;
    }

    complex GetNonunitaryPhase()
    {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * (real1_f)PI_R1;
            return complex((real1)cos(angle), (real1)sin(angle));
        } else {
            return ONE_CMPLX;
        }
    }

    template <typename Fn> void MACWrapper(const std::vector<bitLenInt>& controls, Fn fn)
    {
        bitCapInt xMask = ZERO_BCI;
        for (const bitLenInt& control : controls) {
            bi_or_ip(&xMask, pow2(control));
        }

        XMask(xMask);
        fn(controls);
        XMask(xMask);
    }

    virtual bitCapInt SampleClone(const std::vector<bitCapInt>& qPowers)
    {
        QInterfacePtr clone = Clone();

        const bitCapInt rawSample = clone->MAll();
        bitCapInt sample = ZERO_BCI;
        for (size_t i = 0U; i < qPowers.size(); ++i) {
            if (bi_compare_0(rawSample & qPowers[i]) != 0) {
                bi_or_ip(&sample, pow2(i));
            }
        }

        return sample;
    }

    virtual real1_f ExpVarUnitaryAll(bool isExp, const std::vector<bitLenInt>& bits,
        const std::vector<std::shared_ptr<complex>>& basisOps, std::vector<real1_f> eigenVals = {});
    virtual real1_f ExpVarUnitaryAll(bool isExp, const std::vector<bitLenInt>& bits,
        const std::vector<real1_f>& basisOps, std::vector<real1_f> eigenVals = {});
    virtual real1_f ExpVarBitsAll(bool isExp, const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        std::vector<bitCapInt> perms;
        perms.reserve(bits.size() << 1U);
        for (size_t i = 0U; i < bits.size(); ++i) {
            perms.push_back(ZERO_BCI);
            perms.push_back(pow2(i));
        }

        return isExp ? ExpectationBitsFactorized(bits, perms, offset) : VarianceBitsFactorized(bits, perms, offset);
    }

    virtual void Copy(QInterfacePtr orig)
    {
        orig->Finish();
        doNormalize = orig->doNormalize;
        randGlobalPhase = orig->randGlobalPhase;
        useRDRAND = orig->useRDRAND;
        qubitCount = orig->qubitCount;
        randomSeed = orig->randomSeed;
        amplitudeFloor = orig->amplitudeFloor;
        maxQPower = orig->maxQPower;
        rand_generator = orig->rand_generator;
        rand_distribution = orig->rand_distribution;
        hardware_rand_generator = orig->hardware_rand_generator;
    }

public:
    QInterface(bitLenInt n, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false, bool useHardwareRNG = true,
        bool randomGlobalPhase = true, real1_f norm_thresh = REAL1_EPSILON);

    /** Default constructor, primarily for protected internal use */
    QInterface()
        : doNormalize(false)
        , randGlobalPhase(true)
        , useRDRAND(true)
        , qubitCount(0U)
        , randomSeed(0)
        , amplitudeFloor(REAL1_EPSILON)
        , maxQPower(ONE_BCI)
        , rand_distribution(0.0, 1.0)
        , hardware_rand_generator{ nullptr }
    {
        // Intentionally left blank
    }

    virtual ~QInterface()
    {
        // Virtual destructor for inheritance
    }

    void SetRandomSeed(uint32_t seed)
    {
        if (!!rand_generator) {
            rand_generator->seed(seed);
        }
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = pow2(qubitCount);
    }

    /** Set the number of threads in parallel for loops, per component QEngine */
    virtual void SetConcurrency(uint32_t threadsPerEngine) { SetConcurrencyLevel(threadsPerEngine); }

    /** Get the count of bits in this register */
    virtual bitLenInt GetQubitCount() { return qubitCount; }

    /** Get the maximum number of basis states, namely \f$ 2^n \f$ for \f$ n \f$ qubits*/
    virtual bitCapInt GetMaxQPower() { return maxQPower; }

    virtual bool GetIsArbitraryGlobalPhase() { return randGlobalPhase; }

    /** Generate a random real number between 0 and 1 */
    real1_f Rand()
    {
        if (!!hardware_rand_generator) {
            return hardware_rand_generator->Next();
        } else {
            return rand_distribution(*rand_generator);
        }
    }

    /** Set an arbitrary pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void SetQuantumState(const complex* inputState) = 0;

    /** Get the pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void GetQuantumState(complex* outputState) = 0;

    /** Get the pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void GetProbs(real1* outputProbs) = 0;

    /** Get the representational amplitude of a full permutation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual complex GetAmplitude(const bitCapInt& perm) = 0;

    /** Sets the representational amplitude of a full permutation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void SetAmplitude(const bitCapInt& perm, const complex& amp) = 0;

    /** Set to a specific permutation of all qubits */
    virtual void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG);

    /**
     * Combine another QInterface with this one, after the last bit index of
     * this one.
     *
     * "Compose" combines the quantum description of state of two independent
     * QInterface objects into one object, containing the full permutation
     * basis of the full object. The "inputState" bits are added after the last
     * qubit index of the QInterface to which we "Compose." Informally,
     * "Compose" is equivalent to "just setting another group of qubits down
     * next to the first" without interacting them. Schroedinger's equation can
     * form a description of state for two independent subsystems at once or
     * "separable quantum subsystems" without interacting them. Once the
     * description of state of the independent systems is combined, we can
     * interact them, and we can describe their entanglements to each other, in
     * which case they are no longer independent. A full entangled description
     * of quantum state is not possible for two independent quantum subsystems
     * until we "Compose" them.
     *
     * "Compose" multiplies the probabilities of the indepedent permutation
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
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(toCopy, qubitCount); }
    /**
     * This is a variant of `Compose()` for a `toCopy` argument that will definitely not be reused once "Composed,"
     * hence more aggressive optimization can be done.
     */
    virtual bitLenInt ComposeNoClone(QInterfacePtr toCopy) { return Compose(toCopy); }
    /**
     * `Compose()` a vector of peer `QInterface` targets, in sequence.
     */
    virtual std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy);
    /**
     * `Compose()` a `QInterface` peer, inserting its qubit into index order at `start` index.
     */
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start);

    /**
     * Minimally decompose a set of contiguous bits from the separably composed unit,
     * into "destination"
     *
     * Minimally decompose a set of contigious bits from the separably composed unit.
     * The length of this separable unit is reduced by the length of bits decomposed, and the bits removed are output in
     * the destination QInterface pointer. The destination object must be initialized to the correct number of bits, in
     * 0 permutation state. For quantum mechanical accuracy, the bit set removed and the bit set left behind should be
     * quantum mechanically "separable."
     *
     * Like how "Compose" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Decompose" is like "just moving a few qubits
     * away from the rest." Schroedinger's equation does not require bits to be
     * explicitly interacted in order to describe their permutation basis, and
     * the descriptions of state of <b>separable</b> subsystems, those which
     * are not entangled with other subsystems, are just as easily removed from
     * the description of state. (This is equivalent to a "Schmidt decomposition.")
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *     P(|1000>|xy>) = P(|1000 00>) + P(|1000 10>) + P(|1000 01>) + P(|1000 11>).
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
    virtual void Decompose(bitLenInt start, QInterfacePtr dest) = 0;

    /**
     * Schmidt decompose a length of qubits.
     */
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length) = 0;

    /**
     * Minimally decompose a set of contiguous bits from the separably composed unit,
     * and discard the separable bits from index "start" for "length."
     *
     * Minimally decompose a set of contigious bits from the separably composed unit.
     * The length of this separable unit is reduced by the length of bits decomposed, and the bits removed are output in
     * the destination QInterface pointer. The destination object must be initialized to the correct number of bits, in
     * 0 permutation state. For quantum mechanical accuracy, the bit set removed and the bit set left behind should be
     * quantum mechanically "separable."
     *
     * Like how "Compose" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Decompose" is like "just moving a few qubits
     * away from the rest." Schroedinger's equation does not require bits to be
     * explicitly interacted in order to describe their permutation basis, and
     * the descriptions of state of <b>separable</b> subsystems, those which
     * are not entangled with other subsystems, are just as easily removed from
     * the description of state. (This is equivalent to a "Schmidt decomposition.")
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *     P(|1000>|xy>) = P(|1000 00>) + P(|1000 10>) + P(|1000 01>) + P(|1000 11>).
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
     * Dispose a a contiguous set of qubits that are already in a permutation eigenstate.
     */
    virtual void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm) = 0;

    /**
     * Allocate new "length" count of |0> state qubits at end of qubit index position
     */
    virtual bitLenInt Allocate(bitLenInt length) { return Allocate(qubitCount, length); }

    /**
     * Allocate new "length" count of |0> state qubits at specified qubit index start position
     */
    virtual bitLenInt Allocate(bitLenInt start, bitLenInt length) = 0;

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    /**
     * Apply an arbitrary single bit unitary transformation.
     */
    virtual void Mtrx(const complex* mtrx, bitLenInt qubit) = 0;

    /**
     * Apply an arbitrary single bit unitary transformation, with arbitrary control bits.
     */
    virtual void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target) = 0;

    /**
     * Apply an arbitrary single bit unitary transformation, with arbitrary (anti-)control bits.
     */
    virtual void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            MACPhase(controls, mtrx[0U], mtrx[3U], target);
        } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
            MACInvert(controls, mtrx[1U], mtrx[2U], target);
        } else {
            MACWrapper(controls, [this, mtrx, target](const std::vector<bitLenInt>& lc) { MCMtrx(lc, mtrx, target); });
        }
    }

    /**
     * Apply an arbitrary single bit unitary transformation, with arbitrary control bits, with arbitary control
     * permutation.
     */
    virtual void UCMtrx(
        const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, const bitCapInt& controlPerm);

    /**
     * Apply a single bit transformation that only effects phase.
     */
    virtual void Phase(const complex& topLeft, const complex& bottomRight, bitLenInt qubit)
    {
        if ((randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft)) && IS_NORM_0(topLeft - bottomRight)) {
            return;
        }

        const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        Mtrx(mtrx, qubit);
    }

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase.
     */
    virtual void Invert(const complex& topRight, const complex& bottomLeft, bitLenInt qubit)
    {
        const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        Mtrx(mtrx, qubit);
    }

    /**
     * Apply a single bit transformation that only effects phase, with arbitrary control bits.
     */
    virtual void MCPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
            return;
        }

        const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        MCMtrx(controls, mtrx, target);
    }

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control
     * bits.
     */
    virtual void MCInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        MCMtrx(controls, mtrx, target);
    }

    /**
     * Apply a single bit transformation that only effects phase, with arbitrary (anti-)control bits.
     */
    virtual void MACPhase(
        const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight, bitLenInt target)
    {
        if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
            return;
        }

        MACWrapper(controls, [this, topLeft, bottomRight, target](const std::vector<bitLenInt>& lc) {
            MCPhase(lc, topLeft, bottomRight, target);
        });
    }

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary
     * (anti-)control bits.
     */
    virtual void MACInvert(
        const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft, bitLenInt target)
    {
        MACWrapper(controls, [this, topRight, bottomLeft, target](const std::vector<bitLenInt>& lc) {
            MCInvert(lc, topRight, bottomLeft, target);
        });
    }

    /**
     * Apply a single bit transformation that only effects phase, with arbitrary control bits, with arbitrary control
     * permutation.
     */
    virtual void UCPhase(const std::vector<bitLenInt>& controls, const complex& topLeft, const complex& bottomRight,
        bitLenInt target, const bitCapInt& perm)
    {
        if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
            return;
        }

        const complex mtrx[4U]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        UCMtrx(controls, mtrx, target, perm);
    }

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control
     * bits, with arbitrary control permutation.
     */
    virtual void UCInvert(const std::vector<bitLenInt>& controls, const complex& topRight, const complex& bottomLeft,
        bitLenInt target, const bitCapInt& perm)
    {
        const complex mtrx[4U]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        UCMtrx(controls, mtrx, target, perm);
    }

    /**
     * Apply a "uniformly controlled" arbitrary single bit unitary transformation. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different unitary 2x2 complex matrix is associated with each permutation of the control bits. The first control
     * bit index in the "controls" array is the least significant bit of the permutation, proceeding to the most
     * significant bit. "mtrxs" is a flat (1-dimensional) array where each subsequent set of 4 components is an
     * arbitrary 2x2 single bit gate associated with the next permutation of the control bits, starting from 0. All
     * combinations of control bits apply one of the 4 component (flat 2x2) matrices. For k control bits, there are
     * therefore 4 * 2^k complex components in "mtrxs," representing 2^k complex matrices of 2x2 components. (The
     * component ordering in each matrix is the same as all other gates with an arbitrary 2x2 applied to a single bit,
     * such as Qrack::ApplySingleBit.)
     */

    virtual void UniformlyControlledSingleBit(
        const std::vector<bitLenInt>& controls, bitLenInt qubit, const complex* mtrxs)
    {
        UniformlyControlledSingleBit(controls, qubit, mtrxs, std::vector<bitCapInt>(), ZERO_BCI);
    }
    virtual void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubit,
        const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask);

    /**
     * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are
     * applied by left-multiplication in low-to-high vector index order on the state vector.
     *
     * \warning Hamiltonian components might not commute.
     *
     * As a general point of linear algebra, where A and B are linear operators, \f{equation}{e^{i (A + B) t} = e^{i A
     * t} e^{i B t} \f} might NOT hold, if the operators A and B do not commute. As a rule of thumb, A will commute
     * with B at least in the case that A and B act on entirely different sets of qubits. However, for defining the
     * intended Hamiltonian, the programmer can be guaranteed that the exponential factors will be applied
     * right-to-left, by left multiplication, in the order \f{equation}{ e^{-i H_{N - 1} t} e^{-i H_{N - 2} t} \ldots
     * e^{-i H_0 t} \left|\psi \rangle\right. .\f} (For example, if A and B are single bit gates acting on the same
     * bit, form their composition into one gate by the intended right-to-left fusion and apply them as a single
     * HamiltonianOp.)
     */
    virtual void TimeEvolve(Hamiltonian h, real1_f timeDiff);

    /**
     * Apply a swap with arbitrary control bits.
     */
    virtual void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Apply a swap with arbitrary (anti) control bits.
     */
    virtual void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Apply a square root of swap with arbitrary control bits.
     */
    virtual void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Apply a square root of swap with arbitrary (anti) control bits.
     */
    virtual void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Apply an inverse square root of swap with arbitrary control bits.
     */
    virtual void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Apply an inverse square root of swap with arbitrary (anti) control bits.
     */
    virtual void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2);

    /**
     * Doubly-controlled NOT gate
     *
     * If both controls are set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MCInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
    }

    /**
     * Anti doubly-controlled NOT gate
     *
     * If both controls are set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MACInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
    }

    /**
     * Controlled NOT gate
     *
     * If the control is set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CNOT(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
    }

    /**
     * Anti controlled NOT gate
     *
     * If the control is set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCNOT(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MACInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
    }

    /**
     * Controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CY(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCInvert(controls, -I_CMPLX, I_CMPLX, target);
    }

    /**
     * Anti controlled Y gate
     *
     * If the control is set to 0, then the Pauli "Y" operator is applied to the target.
     */
    virtual void AntiCY(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MACInvert(controls, -I_CMPLX, I_CMPLX, target);
    }

    /**
     * Doubly-Controlled Y gate
     *
     * If both "control" bits are set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MCInvert(controls, -I_CMPLX, I_CMPLX, target);
    }

    /**
     * Anti doubly-controlled Y gate
     *
     * If both controls are set to 0, apply Pauli Y operation to target bit.
     */
    virtual void AntiCCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MACInvert(controls, -I_CMPLX, I_CMPLX, target);
    }

    /**
     * Controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CZ(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
    }

    /**
     * Anti controlled Z gate
     *
     * If the control is set to 0, then the Pauli "Z" operator is applied to the target.
     */
    virtual void AntiCZ(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MACPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
    }

    /**
     * Doubly-Controlled Z gate
     *
     * If both "control" bits are set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MCPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
    }

    /**
     * Anti doubly-controlled Z gate
     *
     * If both controls are set to 0, apply Pauli Z operation to target bit.
     */
    virtual void AntiCCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control1, control2 };
        MACPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
    }

    /**
     * General unitary gate
     *
     * Applies a gate guaranteed to be unitary, from three angles, as commonly defined, spanning all possible single bit
     * unitary gates, (up to a global phase factor which has no effect on Hermitian operator expectation values).
     */
    virtual void U(bitLenInt target, real1_f theta, real1_f phi, real1_f lambda);

    /**
     * 2-parameter unitary gate
     *
     * Applies a gate guaranteed to be unitary, from two angles, as commonly defined.
     */
    virtual void U2(bitLenInt target, real1_f phi, real1_f lambda) { U(target, (real1_f)(M_PI / 2), phi, lambda); }

    /**
     * Inverse 2-parameter unitary gate
     *
     * Applies the inverse of U2
     */
    virtual void IU2(bitLenInt target, real1_f phi, real1_f lambda)
    {
        U(target, (real1_f)(M_PI / 2), (real1_f)(-lambda - PI_R1), (real1_f)(-phi + PI_R1));
    }

    /**
     * "Azimuth, Inclination" (RY-RZ)
     *
     * Sets the azimuth and inclination from Z-X-Y basis probability measurements.
     */
    virtual void AI(bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * Invert "Azimuth, Inclination" (RY-RZ)
     *
     * (Inverse of) sets the azimuth and inclination from Z-X-Y basis probability measurements.
     */
    virtual void IAI(bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * Controlled "Azimuth, Inclination" (RY-RZ)
     *
     * If the control bit is set, this gate sets the azimuth and inclination from Z-X-Y basis probability measurements.
     */
    virtual void CAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * (Anti-)Controlled "Azimuth, Inclination" (RY-RZ)
     *
     * If the control bit is reset, this gate sets the azimuth and inclination from Z-X-Y basis probability
     * measurements.
     */
    virtual void AntiCAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * Controlled inverse "Azimuth, Inclination" (RY-RZ)
     *
     * (Inverse of...) If the control bit is set, this gate sets the azimuth and inclination from Z-X-Y basis
     * probability measurements.
     */
    virtual void CIAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * (Anti-)Controlled inverse "Azimuth, Inclination" (RY-RZ)
     *
     * (Inverse of...) If the control bit is reset, this gate sets the azimuth and inclination from Z-X-Y basis
     * probability measurements.
     */
    virtual void AntiCIAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination);

    /**
     * Controlled general unitary gate
     *
     * Applies a controlled gate guaranteed to be unitary, from three angles, as commonly defined, spanning all possible
     * single bit unitary gates, (up to a global phase factor which has no effect on Hermitian operator expectation
     * values).
     */
    virtual void CU(
        const std::vector<bitLenInt>& controls, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda);

    /**
     * (Anti-)Controlled general unitary gate
     *
     * Applies an (anti-)controlled gate guaranteed to be unitary, from three angles, as commonly defined, spanning all
     * possible single bit unitary gates, (up to a global phase factor which has no effect on Hermitian operator
     * expectation values).
     */
    virtual void AntiCU(
        const std::vector<bitLenInt>& controls, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda);

    /**
     * Hadamard gate
     *
     * Applies a Hadamard gate on qubit at "qubit."
     */
    virtual void H(bitLenInt qubit)
    {
        QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex C_SQRT1_2_NEG = complex(-SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex mtrx[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG };
        Mtrx(mtrx, qubit);
    }

    /**
     * Square root of Hadamard gate
     *
     * Applies the square root of the Hadamard gate on qubit at "qubit."
     */
    virtual void SqrtH(bitLenInt qubit)
    {
        QRACK_CONST complex m00 =
            complex((real1)((ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)), (real1)((-ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)));
        QRACK_CONST complex m01 = complex((real1)(SQRT1_2_R1 / 2), (real1)(-SQRT1_2_R1 / 2));
        QRACK_CONST complex m10 = m01;
        QRACK_CONST complex m11 =
            complex((real1)((-ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)), (real1)((ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)));
        QRACK_CONST complex mtrx[4]{ m00, m01, m10, m11 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Y-basis transformation gate
     *
     * Converts from Pauli Z basis to Y, (via H then S gates).
     */
    virtual void SH(bitLenInt qubit)
    {
        QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex C_I_SQRT1_2 = complex(ZERO_R1, SQRT1_2_R1);
        QRACK_CONST complex C_I_SQRT1_2_NEG = complex(ZERO_R1, -SQRT1_2_R1);
        QRACK_CONST complex mtrx[4]{ C_SQRT1_2, C_SQRT1_2, C_I_SQRT1_2, C_I_SQRT1_2_NEG };
        Mtrx(mtrx, qubit);
    }

    /**
     * Y-basis (inverse) transformation gate
     *
     * Converts from Pauli Y basis to Z, (via IS then H gates).
     */
    virtual void HIS(bitLenInt qubit)
    {
        QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex C_I_SQRT1_2 = complex(ZERO_R1, SQRT1_2_R1);
        QRACK_CONST complex C_I_SQRT1_2_NEG = complex(ZERO_R1, -SQRT1_2_R1);
        QRACK_CONST complex mtrx[4]{ C_SQRT1_2, C_I_SQRT1_2_NEG, C_SQRT1_2, C_I_SQRT1_2 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Measurement gate
     *
     * Measures the qubit at "qubit" and returns either "true" or "false."
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
    virtual bool M(bitLenInt qubit) { return ForceM(qubit, false, false); };

    /**
     * Act as if is a measurement was applied, except force the (usually random) result
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true) = 0;

    /**
     * S gate
     *
     * Applies a 1/4 phase rotation to the qubit at "qubit."
     */
    virtual void S(bitLenInt qubit) { Phase(ONE_CMPLX, I_CMPLX, qubit); }

    /**
     * Inverse S gate
     *
     * Applies an inverse 1/4 phase rotation to the qubit at "qubit."
     */
    virtual void IS(bitLenInt qubit) { Phase(ONE_CMPLX, -I_CMPLX, qubit); }

    /**
     * T gate
     *
     * Applies a 1/8 phase rotation to the qubit at "qubit."
     */
    virtual void T(bitLenInt qubit) { Phase(ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), qubit); }

    /**
     * Inverse T gate
     *
     * Applies an inverse 1/8 phase rotation to the qubit at "qubit."
     */
    virtual void IT(bitLenInt qubit) { Phase(ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), qubit); }

    /**
     * "PhaseRootN" gate
     *
     * Applies a -2 * PI_R1 / (2^N) phase rotation to the qubit at "qubit."
     */
    virtual void PhaseRootN(bitLenInt n, bitLenInt qubit)
    {
        if (n == 0) {
            return;
        }

        Phase(ONE_CMPLX, pow(-ONE_CMPLX, (real1)(ONE_R1 / pow2Ocl(n - 1U))), qubit);
    }

    /**
     * Masked PhaseRootN gate
     *
     * Applies a -2 * PI_R1 / (2^N) phase rotation to each qubit in the mask.
     */
    virtual void PhaseRootNMask(bitLenInt n, const bitCapInt& mask);

    /**
     * Parity phase gate
     *
     * Applies e^(i*angle) phase factor to all combinations of bits with odd parity, based upon permutations of qubits.
     */
    virtual void PhaseParity(real1_f radians, const bitCapInt& mask);

    /**
     * X gate
     *
     * Applies the Pauli "X" operator to the qubit at "qubit." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void X(bitLenInt qubit) { Invert(ONE_CMPLX, ONE_CMPLX, qubit); }

    /**
     * Masked X gate
     *
     * Applies the Pauli "X" operator to all qubits in the mask. A qubit index "n" is in the mask if (((1 << n) & mask)
     * > 0). The Pauli "X" operator is equivalent to a logical "NOT."
     */
    virtual void XMask(const bitCapInt& mask);

    /**
     * Y gate
     *
     * Applies the Pauli "Y" operator to the qubit at "qubit." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase.
     * effects.
     */
    virtual void Y(bitLenInt qubit) { Invert(-I_CMPLX, I_CMPLX, qubit); }

    /**
     * Masked Y gate
     *
     * Applies the Pauli "Y" operator to all qubits in the mask. A qubit index "n" is in the mask if (((1 << n) & mask)
     * > 0). The Pauli "Y" operator is similar to a logical "NOT" with permutation phase.
     */
    virtual void YMask(const bitCapInt& mask);

    /**
     * Z gate
     *
     * Applies the Pauli "Z" operator to the qubit at "qubit." The Pauli
     * "Z" operator reverses the phase of |1> and leaves |0> unchanged.
     */
    virtual void Z(bitLenInt qubit) { Phase(ONE_CMPLX, -ONE_CMPLX, qubit); }

    /**
     * Masked Z gate
     *
     * Applies the Pauli "Z" operator to all qubits in the mask. A qubit index "n" is in the mask if (((1 << n) & mask)
     * > 0). The Pauli "Z" operator reverses the phase of |1> and leaves |0> unchanged.
     */
    virtual void ZMask(const bitCapInt& mask);

    /**
     * Square root of X gate
     *
     * Applies the square root of the Pauli "X" operator to the qubit at "qubit." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void SqrtX(bitLenInt qubit)
    {
        QRACK_CONST complex ONE_PLUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex ONE_MINUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Inverse square root of X gate
     *
     * Applies the (by convention) inverse square root of the Pauli "X" operator to the qubit at "qubit." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void ISqrtX(bitLenInt qubit)
    {
        QRACK_CONST complex ONE_PLUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex ONE_MINUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Square root of Y gate
     *
     * Applies the square root of the Pauli "Y" operator to the qubit at "qubit." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void SqrtY(bitLenInt qubit)
    {
        QRACK_CONST complex ONE_PLUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex ONE_PLUS_I_DIV_2_NEG = complex((real1)(-ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2_NEG, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Inverse square root of Y gate
     *
     * Applies the (by convention) inverse square root of the Pauli "Y" operator to the qubit at "qubit." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void ISqrtY(bitLenInt qubit)
    {
        QRACK_CONST complex ONE_MINUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex ONE_MINUS_I_DIV_2_NEG = complex((real1)(-ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2_NEG, ONE_MINUS_I_DIV_2 };
        Mtrx(mtrx, qubit);
    }

    /**
     * Square root of W gate
     *
     * Per 2019 Arute, Google's "quantum supremacy" experiment
     */
    virtual void SqrtW(bitLenInt qubit)
    {
        QRACK_CONST complex diag = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex m01 = complex((real1)(-ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex m10 = complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ diag, m01, m10, diag };
        Mtrx(mtrx, qubit);
    }

    /**
     * Inverse square root of W gate
     *
     * Per 2019 Arute, Google's "quantum supremacy" experiment
     */
    virtual void ISqrtW(bitLenInt qubit)
    {
        QRACK_CONST complex diag = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex m01 = complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex m10 = complex((real1)(-ONE_R1 / 2), (real1)(ONE_R1 / 2));
        QRACK_CONST complex mtrx[4]{ diag, m01, m10, diag };
        Mtrx(mtrx, qubit);
    }

    /**
     * Controlled H gate
     *
     * If the "control" bit is set to 1, then the "H" Walsh-Hadamard transform operator is applied
     * to "target."
     */
    virtual void CH(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex C_SQRT1_2_NEG = complex(-SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex mtrx[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG };
        MCMtrx(controls, mtrx, target);
    }

    /**
     * (Anti-)controlled H gate
     *
     * If the "control" bit is set to 1, then the "H" Walsh-Hadamard transform operator is applied
     * to "target."
     */
    virtual void AntiCH(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex C_SQRT1_2_NEG = complex(-SQRT1_2_R1, ZERO_R1);
        QRACK_CONST complex mtrx[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG };
        MACMtrx(controls, mtrx, target);
    }

    /**
     * Controlled S gate
     *
     * If the "control" bit is set to 1, then the S gate is applied
     * to "target."
     */
    virtual void CS(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, I_CMPLX, target);
    }

    /**
     * (Anti-)controlled S gate
     *
     * If the "control" bit is set to 1, then the S gate is applied
     * to "target."
     */
    virtual void AntiCS(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MACPhase(controls, ONE_CMPLX, I_CMPLX, target);
    }

    /**
     * Controlled inverse S gate
     *
     * If the "control" bit is set to 1, then the inverse S gate is applied
     * to "target."
     */
    virtual void CIS(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, -I_CMPLX, target);
    }

    /**
     * (Anti-)controlled inverse S gate
     *
     * If the "control" bit is set to 1, then the inverse S gate is applied
     * to "target."
     */
    virtual void AntiCIS(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MACPhase(controls, ONE_CMPLX, -I_CMPLX, target);
    }

    /**
     * Controlled T gate
     *
     * If the "control" bit is set to 1, then the T gate is applied
     * to "target."
     */
    virtual void CT(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, complex(SQRT1_2_R1, SQRT1_2_R1), target);
    }

    /**
     * Controlled inverse T gate
     *
     * If the "control" bit is set to 1, then the inverse T gate is applied
     * to "target."
     */
    virtual void CIT(bitLenInt control, bitLenInt target)
    {
        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, complex(SQRT1_2_R1, -SQRT1_2_R1), target);
    }

    /**
     * Controlled "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
    {
        if (n == 0) {
            return;
        }

        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (real1)(ONE_R1 / pow2Ocl(n - 1U))), target);
    }

    /**
     * (Anti-)controlled "PhaseRootN" gate
     *
     * If the "control" bit is set to 0, then the "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void AntiCPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
    {
        if (n == 0) {
            return;
        }

        const std::vector<bitLenInt> controls{ control };
        MACPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (real1)(ONE_R1 / pow2Ocl(n - 1U))), target);
    }

    /**
     * Controlled inverse "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the inverse "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
    {
        if (n == 0) {
            return;
        }

        const std::vector<bitLenInt> controls{ control };
        MCPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (real1)(-ONE_R1 / pow2Ocl(n - 1U))), target);
    }

    /**
     * (Anti-)controlled inverse "PhaseRootN" gate
     *
     * If the "control" bit is set to 0, then the inverse "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void AntiCIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
    {
        if (n == 0) {
            return;
        }

        const std::vector<bitLenInt> controls{ control };
        MACPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (real1)(-ONE_R1 / pow2Ocl(n - 1U))), target);
    }

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
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     *  Quantum analog of classical "AND" gate. Takes one qubit input and one
     *  classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "NAND" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void NAND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "NOR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void NOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XNOR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void XNOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     *  Quantum analog of classical "NAND" gate. Takes one qubit input and one
     *  classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLNAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "NOR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLNOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XNOR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLXNOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /** @} */

    /**
     * \defgroup RotGates Rotational gates
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     *
     * @{
     */

    /**
     * Apply a "uniformly controlled" rotation of a bit around the Pauli Y axis. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different rotation angle is associated with each permutation of the control bits. The first control bit index
     * in the "controls" array is the least significant bit of the permutation, proceeding to the most significant bit.
     * "angles" is an array where each subsequent component is rotation angle associated with the next permutation of
     * the control bits, starting from 0. All combinations of control bits apply one of rotation angles. For k control
     * bits, there are therefore 2^k real components in "angles."
     */
    virtual void UniformlyControlledRY(const std::vector<bitLenInt>& controls, bitLenInt qubit, const real1* angles);

    /**
     * Apply a "uniformly controlled" rotation of a bit around the Pauli Z axis. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different rotation angle is associated with each permutation of the control bits. The first control bit index
     * in the "controls" array is the least significant bit of the permutation, proceeding to the most significant bit.
     * "angles" is an array where each subsequent component is rotation angle associated with the next permutation of
     * the control bits, starting from 0. All combinations of control bits apply one of rotation angles. For k control
     * bits, there are therefore 2^k real components in "angles."
     */
    virtual void UniformlyControlledRZ(const std::vector<bitLenInt>& controls, bitLenInt qubit, const real1* angles);

    /**
     * Phase shift gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around |1> state
     */
    virtual void RT(real1_f radians, bitLenInt qubit);

    /**
     * X axis rotation gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1_f radians, bitLenInt qubit);

    /**
     * Y axis rotation gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1_f radians, bitLenInt qubit);

    /**
     * Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1_f radians, bitLenInt qubit);

    /**
     * Controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i \theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1_f radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i \theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1_f radians, bitLenInt control, bitLenInt target);

#if ENABLE_ROT_API
    /**
     * Dyadic fraction phase shift gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Dyadic fraction X axis rotation gate
     *
     * Rotates \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * (Identity) Exponentiation gate
     *
     * Applies \f$ e^{-i \theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1_f radians, bitLenInt qubit);

    /**
     *  Imaginary exponentiation of arbitrary 2x2 gate
     *
     * Applies \f$ e^{-i*Op} \f$, where "Op" is a 2x2 matrix, (with controls on the application of the gate).
     */
    virtual void Exp(
        const std::vector<bitLenInt>& controls, bitLenInt qubit, const complex* matrix2x2, bool antiCtrled = false);

    /**
     * Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator I / 2^{denomPower}\right) \f$, exponentiation of the identity
     * operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1_f radians, bitLenInt qubit);

    /**
     * Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_x / 2^{denomPower}\right) \f$, exponentiation of the Pauli X
     * operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1_f radians, bitLenInt qubit);

    /**
     * Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_y / 2^{denomPower}\right) \f$, exponentiation of the Pauli Y
     * operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1_f radians, bitLenInt qubit);

    /**
     * Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_z / 2^{denomPower}\right) \f$, exponentiation of the Pauli Z
     * operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i \theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1_f radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ \exp\left(i \pi numerator / 2^{denomPower}\right) \f$ around Pauli Z axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt qubit);

    /**
     * Controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i \pi numerator / 2^{denomPower}\right) \f$ around Pauli Z
     * axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i \theta/2}
     * \f$ around |1> state.
     */

    virtual void CRT(real1_f radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$
     * around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);
#endif

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
    virtual void X(bitLenInt start, bitLenInt length) { XMask(bitRegMask(start, length)); }

#if ENABLE_REG_GATES

    /** Bitwise general unitary */
    virtual void U(bitLenInt start, bitLenInt length, real1_f theta, real1_f phi, real1_f lambda);

    /** Bitwise 2-parameter unitary */
    virtual void U2(bitLenInt start, bitLenInt length, real1_f phi, real1_f lambda);

    /** Bitwise Y-basis transformation gate */
    virtual void SH(bitLenInt start, bitLenInt length);

    /** Bitwise inverse Y-basis transformation gate */
    virtual void HIS(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Hadamard */
    virtual void SqrtH(bitLenInt start, bitLenInt length);

    /** Bitwise S operator (1/4 phase rotation) */
    virtual void S(bitLenInt start, bitLenInt length);

    /** Bitwise inverse S operator (1/4 phase rotation) */
    virtual void IS(bitLenInt start, bitLenInt length);

    /** Bitwise T operator (1/8 phase rotation) */
    virtual void T(bitLenInt start, bitLenInt length);

    /** Bitwise inverse T operator (1/8 phase rotation) */
    virtual void IT(bitLenInt start, bitLenInt length);

    /** Bitwise "PhaseRootN" operator (1/(2^N) phase rotation) */
    virtual void PhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length);

    /** Bitwise inverse "PhaseRootN" operator (1/(2^N) phase rotation) */
    virtual void IPhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Y operator */
    virtual void Y(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Pauli X operator */
    virtual void SqrtX(bitLenInt start, bitLenInt length);

    /** Bitwise inverse square root of Pauli X operator */
    virtual void ISqrtX(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Pauli Y operator */
    virtual void SqrtY(bitLenInt start, bitLenInt length);

    /** Bitwise inverse square root of Pauli Y operator */
    virtual void ISqrtY(bitLenInt start, bitLenInt length);

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

    /** Bitwise controlled-Y */
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length);

    /** Bitwise "anti-"controlled-Y */
    virtual void AntiCY(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise doubly controlled-Y */
    virtual void CCY(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise doubly "anti-"controlled-Y */
    virtual void AntiCCY(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise controlled-Z */
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /** Bitwise "anti-"controlled-Z */
    virtual void AntiCZ(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise doubly controlled-Z */
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise doubly "anti-"controlled-Z */
    virtual void AntiCCZ(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise swap */
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Bitwise ISwap */
    virtual void ISwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Bitwise IISwap */
    virtual void IISwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Bitwise square root of swap */
    virtual void SqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Bitwise inverse square root of swap */
    virtual void ISqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Bitwise "fSim" */
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt start1, bitLenInt start2, bitLenInt length);

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
    virtual void CLAND(bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "OR" */
    virtual void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "OR" */
    virtual void CLOR(bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "XOR" */
    virtual void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "XOR" */
    virtual void CLXOR(bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "NAND" */
    virtual void NAND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "NAND" */
    virtual void CLNAND(
        bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "NOR" */
    virtual void NOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "NOR" */
    virtual void CLNOR(bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "XNOR" */
    virtual void XNOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "XNOR" */
    virtual void CLXNOR(
        bitLenInt qInputStart, const bitCapInt& classicalInput, bitLenInt outputStart, bitLenInt length);

#if ENABLE_ROT_API
    /**
     * Bitwise phase shift gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around |1> state
     */
    virtual void RT(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction phase shift gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise X axis rotation gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction X axis rotation gate
     *
     * Rotates \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i \theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Y axis rotation gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i \theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Z axis rotation gate
     *
     * Rotates as \f$ e^{-i \theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i \theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z
     * axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i \theta/2}
     * \f$ around |1> state.
     */
    virtual void CRT(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$
     * around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise (identity) exponentiation gate
     *
     * Applies \f$ e^{-i \theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator I / 2^{denomPower}\right) \f$, exponentiation of the identity
     * operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_x / 2^{denomPower}\right) \f$, exponentiation of the Pauli X
     * operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_y / 2^{denomPower}\right) \f$, exponentiation of the Pauli Y
     * operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i \theta \sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1_f radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ \exp\left(-i \pi numerator \sigma_z / 2^{denomPower}\right) \f$, exponentiation of the Pauli Z
     * operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);
#endif

    /**
     * Bitwise controlled H gate
     *
     * If the "control" bit is set to 1, then the "H" Walsh-Hadamard transform operator is applied
     * to "target."
     */
    virtual void CH(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled S gate
     *
     * If the "control" bit is set to 1, then the S gate is applied
     * to "target."
     */
    virtual void CS(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse S gate
     *
     * If the "control" bit is set to 1, then the inverse S gate is applied
     * to "target."
     */
    virtual void CIS(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled T gate
     *
     * If the "control" bit is set to 1, then the T gate is applied
     * to "target."
     */
    virtual void CT(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse T gate
     *
     * If the "control" bit is set to 1, then the inverse T gate is applied
     * to "target."
     */
    virtual void CIT(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the inverse "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length);

    /** @} */
#endif

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    /** Circular shift left - shift bits left, and carry last bits. */
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift right - shift bits right, and carry first bits. */
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Arithmetic shift left, with last 2 bits as sign and carry */
    virtual void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Arithmetic shift right, with last 2 bits as sign and carry */
    virtual void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift left, filling the extra bits with |0> */
    virtual void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift right, filling the extra bits with |0> */
    virtual void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Add integer (without sign) */
    virtual void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);

    /** Subtract classical integer (without sign) */
    virtual void DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
    {
        const bitCapInt invToSub = pow2(length) - toSub;
        INC(invToSub, start, length);
    }

    /** Common driver method behind INCC and DECC */
    virtual void INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Add integer (without sign, with carry) */
    virtual void INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        const bool hasCarry = M(carryIndex);
        if (hasCarry) {
            X(carryIndex);
            INCDECC(toAdd + 1U, start, length, carryIndex);
        } else {
            INCDECC(toAdd, start, length, carryIndex);
        }
    }

    /** Subtract classical integer (without sign, with carry) */
    virtual void DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        const bool hasCarry = M(carryIndex);
        bitCapInt invToSub = pow2(length) - toSub;
        if (hasCarry) {
            X(carryIndex);
        } else {
            --invToSub;
        }

        INCDECC(invToSub, start, length, carryIndex);
    }

    /** Add integer (without sign, with controls) */
    virtual void CINC(
        const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls);

    /** Subtract classical integer (without sign, with controls) */
    virtual void CDEC(
        const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        const bitCapInt invToSub = pow2(length) - toSub;
        CINC(invToSub, inOutStart, length, controls);
    }

    /** Add a classical integer to the register, with sign and without carry. */
    virtual void INCS(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        const bitCapInt signMask = pow2(length - 1U);
        INC(signMask, start, length);
        INCDECC(toAdd & ~signMask, start, length, overflowIndex);
        if (bi_compare_0(toAdd & signMask) == 0) {
            DEC(signMask, start, length);
        }
    }

    /** Subtract a classical integer from the register, with sign and without carry. */
    virtual void DECS(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        const bitCapInt invToSub = pow2(length) - toSub;
        INCS(invToSub, start, length, overflowIndex);
    }

    /** Multiplication modulo N by integer, (out of place) */
    virtual void MULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);

    /** Inverse of multiplication modulo N by integer, (out of place) */
    virtual void IMULModNOut(
        const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);

    /** Controlled multiplication modulo N by integer, (out of place) */
    virtual void CMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);

    /** Inverse of controlled multiplication modulo N by integer, (out of place) */
    virtual void CIMULModNOut(const bitCapInt& toMul, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
        bitLenInt length, const std::vector<bitLenInt>& controls);

    /**
     * Quantum analog of classical "Full Adder" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer

        // Assume outputBit is in 0 state.
        CCNOT(inputBit1, inputBit2, carryOut);
        CNOT(inputBit1, inputBit2);
        CCNOT(inputBit2, carryInSumOut, carryOut);
        CNOT(inputBit2, carryInSumOut);
        CNOT(inputBit1, inputBit2);
    }

    /**
     * Inverse of FullAdd
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert FullAdd.)
     */
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
        // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
        // (CNOT and CCNOT are self-inverse.)

        // Assume outputBit is in 0 state.
        CNOT(inputBit1, inputBit2);
        CNOT(inputBit2, carryInSumOut);
        CCNOT(inputBit2, carryInSumOut, carryOut);
        CNOT(inputBit1, inputBit2);
        CCNOT(inputBit1, inputBit2, carryOut);
    }

    /**
     * Controlled quantum analog of classical "Full Adder" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void CFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Inverse of CFullAdd
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert CFullAdd.)
     */
    virtual void CIFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Add a quantum integer to a quantum integer, with carry
     *
     * (Assumes the output register is in the 0 state)
     */
    virtual void ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry);

    /**
     * Inverse of ADC
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert ADC.)
     */
    virtual void IADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry);

    /**
     * Add a quantum integer to a quantum integer, with carry and with controls
     *
     * (Assumes the output register is in the 0 state)
     */
    virtual void CADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
        bitLenInt length, bitLenInt carry);

    /**
     * Inverse of CADC
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert CADC.)
     */
    virtual void CIADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
        bitLenInt length, bitLenInt carry);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    /** Quantum Fourier Transform - Apply the quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void QFT(bitLenInt start, bitLenInt length, bool trySeparate = false);

    /** Quantum Fourier Transform (random access) - Apply the quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void QFTR(const std::vector<bitLenInt>& qubits, bool trySeparate = false);

    /** Inverse Quantum Fourier Transform - Apply the inverse quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void IQFT(bitLenInt start, bitLenInt length, bool trySeparate = false);

    /** Inverse Quantum Fourier Transform (random access) - Apply the inverse quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void IQFTR(const std::vector<bitLenInt>& qubits, bool trySeparate = false);

    /** Reverse the phase of the state where the register equals zero. */
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);

    /** Phase flip always - equivalent to Z X Z X on any bit in the QInterface */
    virtual void PhaseFlip() { Phase(-ONE_CMPLX, -ONE_CMPLX, 0); }

    /** Set register bits to given permutation */
    virtual void SetReg(bitLenInt start, bitLenInt length, const bitCapInt& value);

    /** Measure permutation state of a register */
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length) { return ForceMReg(start, length, ZERO_BCI, false); }

    /** Measure permutation state of all coherent bits */
    virtual bitCapInt MAll() { return MReg(0, qubitCount); }

    /**
     * Act as if is a measurement was applied, except force the (usually random) result
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, const bitCapInt& result, bool doForce = true, bool doApply = true);

    /** Measure bits with indices in array, and return a mask of the results */
    virtual bitCapInt M(const std::vector<bitLenInt>& bits) { return ForceM(bits, std::vector<bool>()); }

    /** Measure bits with indices in array, and return a mask of the results */
    virtual bitCapInt ForceM(const std::vector<bitLenInt>& bits, const std::vector<bool>& values, bool doApply = true);

    /** Swap values of two bits in register */
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);

    /** Swap values of two bits in register, and apply phase factor of i if bits are different */
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);

    /** Inverse ISwap - Swap values of two bits in register, and apply phase factor of -i if bits are different */
    virtual void IISwap(bitLenInt qubit1, bitLenInt qubit2);

    /** Square root of Swap gate */
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);

    /** Inverse square root of Swap gate */
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);

    /** The 2-qubit "fSim" gate, (useful in the simulation of particles with fermionic statistics) */
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
    {
        const std::vector<bitLenInt> controls{ qubit1 };
        const real1 sinTheta = (real1)sin(theta);
        const complex expIPhi = exp(complex(ZERO_R1, -(real1)phi));

        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);

        if ((2 * abs(sinTheta)) <= (FP_NORM_EPSILON * PI_R1)) {
            // ISWAP power is 0.
            return;
        }

        const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
        if ((2 * abs(sinThetaDiffNeg)) <= (FP_NORM_EPSILON * PI_R1)) {
            return ISwap(qubit1, qubit2);
        }

        const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
        if ((2 * abs(sinThetaDiffPos)) <= (FP_NORM_EPSILON * PI_R1)) {
            return IISwap(qubit1, qubit2);
        }

        // Decompose root of ISWAP gate:

        // RXX
        H(qubit1);
        H(qubit2);
        CNOT(qubit1, qubit2);
        RZ(2 * theta, qubit2);
        CNOT(qubit1, qubit2);
        H(qubit2);
        H(qubit1);

        // RYY
        S(qubit1);
        S(qubit2);
        H(qubit1);
        H(qubit2);
        CNOT(qubit1, qubit2);
        RZ(2 * theta, qubit2);
        CNOT(qubit1, qubit2);
        H(qubit2);
        H(qubit1);
        IS(qubit2);
        IS(qubit1);
    }

    /** Reverse all of the bits in a sequence. */
    virtual void Reverse(bitLenInt first, bitLenInt last)
    {
        while ((last > 0U) && (first < (last - 1U))) {
            --last;
            Swap(first, last);
            ++first;
        }
    }

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    /**
     * Direct measure of bit probability to be in |1> state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f Prob(bitLenInt qubit) = 0;
    /**
     * Direct measure of bit probability to be in |1> state, if control bit is |1>.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f CProb(bitLenInt control, bitLenInt target)
    {
        AntiCNOT(control, target);
        const real1_f prob = Prob(target);
        AntiCNOT(control, target);

        return prob;
    }
    /**
     * Direct measure of bit probability to be in |1> state, if control bit is |0>.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ACProb(bitLenInt control, bitLenInt target)
    {
        CNOT(control, target);
        const real1_f prob = Prob(target);
        CNOT(control, target);

        return prob;
    }

    /**
     * Direct measure of full permutation probability
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbAll(const bitCapInt& fullRegister)
    {
        return clampProb((real1_f)norm(GetAmplitude(fullRegister)));
    }

    /**
     * Direct measure of register permutation probability
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbReg(bitLenInt start, bitLenInt length, const bitCapInt& permutation);

    /**
     * Direct measure of masked permutation probability
     *
     * "mask" masks the bits to check the probability of. "permutation" sets the 0 or 1 value for each bit in the mask.
     * Bits which are set in the mask can be set to 0 or 1 in the permutation, while reset bits in the mask should be 0
     * in the permutation.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation);

    /**
     * Direct measure of masked permutation probability
     *
     * "mask" masks the bits to check the probability of. The probabilities of all permutations of the masked bits, from
     * left/low to right/high are returned in the "probsArray" argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void ProbMaskAll(const bitCapInt& mask, real1* probsArray);

    /**
     * Direct measure of listed permutation probability
     *
     * The probabilities of all included permutations of bits, with bits valued from low to high as the order of the
     * "bits" array parameter argument, are returned in the "probsArray" parameter.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void ProbBitsAll(const std::vector<bitLenInt>& bits, real1* probsArray);

    /**
     * Direct measure of variance of listed permutation probability
     *
     * The (bit string) variance of all included permutations of bits, with bits valued from low to high as the order of
     * the "bits" array parameter argument, is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarBitsAll(false, bits, offset);
    }

    /**
     * Direct measure of (reduced density matrix) variance of listed permutation probability
     *
     * The (bit string, reduced density matrix) variance of all included permutations of bits, with bits valued from low
     * to high as the order of the "bits" array parameter argument, is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceBitsAllRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return VarianceBitsAll(bits, offset);
    }

    /**
     * Direct measure of variance of listed Pauli tensor product probability
     *
     * The (bit string) variance of all included permutations of bits, with bits valued from low to high as the order of
     * the "bits" array parameter argument,  is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VariancePauliAll(std::vector<bitLenInt> bits, std::vector<Pauli> paulis);

    /**
     * Direct measure of variance of listed (3-parameter) single-qubit tensor product probability
     *
     * The (bit string) variance of all included permutations of bits, with bits valued from low to high as the order of
     * the "bits" array parameter argument,  is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceUnitaryAll(
        const std::vector<bitLenInt>& bits, const std::vector<real1_f>& basisOps, std::vector<real1_f> eigenVals = {})
    {
        return ExpVarUnitaryAll(false, bits, basisOps, eigenVals);
    }

    /**
     * Direct measure of variance of listed (2x2 operator) single-qubit tensor product probability
     *
     * The (bit string) variance of all included permutations of bits, with bits valued from low to high as the order of
     * the "bits" array parameter argument,  is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceUnitaryAll(const std::vector<bitLenInt>& bits,
        const std::vector<std::shared_ptr<complex>>& basisOps, std::vector<real1_f> eigenVals = {})
    {
        return ExpVarUnitaryAll(false, bits, basisOps, eigenVals);
    }

    /**
     * Direct measure of variance of listed bit string probability
     *
     * The (bit string) variance of all included permutations of bits, with bits valued from low to high as the order of
     * the "bits" array parameter argument, is returned.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);

    /**
     * Direct measure of (reduced density matrix) variance of bits, given an array of qubit weights
     *
     * The weight-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "perms" weight
     * entry. If there are stabilizer ancillae, they are traced out of the reduced density matrix, giving an approximate
     * result.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceFloatsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return VarianceFloatsFactorized(bits, weights);
    }

    /**
     * Get expectation value of bits, given an array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "perms" weight
     * entry.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI);

    /**
     * Get (reduced density matrix) expectation value of bits, given an array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "perms" weight
     * entry. If there are stabilizer ancillae, they are traced out of the reduced density matrix, giving an approximate
     * result.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f VarianceBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return VarianceBitsFactorized(bits, perms, offset);
    }

    /**
     * Get permutation expectation value of bits
     *
     * The permutation expectation value of all included bits is returned, with bits valued from low to high as the
     * order of the "bits" array parameter argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpVarBitsAll(true, bits, offset);
    }

    /**
     * Get Pauli tensor product observable
     *
     * The Pauli tensor basis expectation value of all included bits is returned, with bits valued from low to high as
     * the order of the "bits" array parameter argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationPauliAll(std::vector<bitLenInt> bits, std::vector<Pauli> paulis);

    /**
     * Get single-qubit tensor product (arbitrary real) observable
     *
     * The single-qubit tensor basis (arbitrary real) expectation value of all included bits is returned, with bits
     * valued from low to high as the order of the "bits" array parameter argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationUnitaryAll(const std::vector<bitLenInt>& bits,
        const std::vector<std::shared_ptr<complex>>& basisOps, std::vector<real1_f> eigenVals = {})
    {
        return ExpVarUnitaryAll(true, bits, basisOps, eigenVals);
    }

    /**
     * Get single-qubit (3-parameter) tensor product (arbitrary real) observable
     *
     * The single-qubit (3-parameter) tensor basis (arbitrary real) expectation value of all included bits is returned,
     * with bits valued from low to high as the order of the "bits" array parameter argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationUnitaryAll(
        const std::vector<bitLenInt>& bits, const std::vector<real1_f>& basisOps, std::vector<real1_f> eigenVals = {})
    {
        return ExpVarUnitaryAll(true, bits, basisOps, eigenVals);
    }

    /**
     * Get expectation value of bits, given an array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "perms" weight
     * entry.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI);

    /**
     * Get (reduced density matrix) expectation value of bits, given an array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "perms" weight
     * entry. If there are stabilizer ancillae, they are traced out of the reduced density matrix, giving an approximate
     * result.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationBitsFactorizedRdm(bool roundRz, const std::vector<bitLenInt>& bits,
        const std::vector<bitCapInt>& perms, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpectationBitsFactorized(bits, perms, offset);
    }

    /**
     * Get expectation value of bits, given a (floating-point) array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "weights"
     * entry.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationFloatsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights);

    /**
     * Get (reduced density matrix) expectation value of bits, given a (floating-point) array of qubit weights
     *
     * The weighter-per-qubit expectation value of is returned, with each "bits" entry corresponding to a "weights"
     * entry. If there are stabilizer ancillae, they are traced out of the reduced density matrix, giving an approximate
     * result.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationFloatsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpectationFloatsFactorized(bits, weights);
    }

    /**
     * Direct measure of bit probability to be in |1> state, treating all ancillary qubits as post-selected T gate
     * gadgets
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbRdm(bitLenInt qubit) { return Prob(qubit); }
    /**
     * Direct measure of full permutation probability, treating all ancillary qubits as post-selected T gate gadgets
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbAllRdm(bool roundRz, const bitCapInt& fullRegister) { return ProbAll(fullRegister); }
    /**
     * Direct measure of masked permutation probability, treating all ancillary qubits as post-selected T gate gadgets
     *
     * "mask" masks the bits to check the probability of. "permutation" sets the 0 or 1 value for each bit in the mask.
     * Bits which are set in the mask can be set to 0 or 1 in the permutation, while reset bits in the mask should be 0
     * in the permutation.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ProbMaskRdm(bool roundRz, const bitCapInt& mask, const bitCapInt& permutation)
    {
        return ProbMask(mask, permutation);
    }
    /**
     * Get permutation expectation value of bits, treating all ancillary qubits as post-selected T gate gadgets
     *
     * The permutation expectation value of all included bits is returned, with bits valued from low to high as the
     * order of the "bits" array parameter argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f ExpectationBitsAllRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return ExpectationBitsAll(bits, offset);
    }

    /**
     * Statistical measure of masked permutation probability
     *
     * "qPowers" contains powers of 2^n, each representing QInterface bit "n." The order of these values defines a mask
     * for the result bitCapInt, of 2^0 ~ qPowers[0U] to 2^(qPowerCount - 1) ~ qPowers[qPowerCount - 1], in contiguous
     * ascending order. "shots" specifies the number of samples to take as if totally re-preparing the pre-measurement
     * state. This method returns a dictionary with keys, which are the (masked-order) measurement results, and values,
     * which are the number of "shots" that produced that particular measurement result. This method does not "collapse"
     * the state of this QInterface. (The idea is to efficiently simulate a potentially statistically random sample of
     * multiple re-preparations of the state right before measurement, and to collect random measurement resutls,
     * without forcing the user to re-prepare or "clone" the state.)
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);

    /**
     * Statistical measure of masked permutation probability (returned as array)
     *
     * Same `Qrack::MultiShotMeasureMask()`, except the shots are returned as an array.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void MultiShotMeasureMask(
        const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    /**
     * Set individual bit to pure |0> (false) or |1> (true) state
     *
     * To set a bit, the bit is first measured. If the result of measurement
     * matches "value," the bit is considered set.  If the result of
     * measurement is the opposite of "value," an X gate is applied to the bit.
     * The state ends up entirely in the "value" state, with a random phase
     * factor.
     */
    virtual void SetBit(bitLenInt qubit, bool value)
    {
        if (value != M(qubit)) {
            X(qubit);
        }
    }

    /**
     * Compare state vectors approximately, to determine whether this state vector is the same as the target.
     * (If (1 - <\psi_e|\psi_c>) <= `error_tol` between states |\psi_c> and |\psi_e>, they are "the same.")
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return SumSqrDiff(toCompare) <= error_tol;
    }

    /**
     * Calculates (1 - <\psi_e|\psi_c>) between states |\psi_c> and |\psi_e>.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1_f SumSqrDiff(QInterfacePtr toCompare) = 0;

    /**
     * Attempt to `Decompose()` a bit range. If the result can `Compose()` again to the original state vector with
     * (1 - <\psi_e|\psi_c>) <= `error_tol`, return "true" and complete `Decompose()`; otherwise, restore the original
     * state and return "false."
     */
    virtual bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON);

    /**
     * Force a calculation of the norm of the state vector, in order to make it unit length before the next probability
     * or measurement operation. (On an actual quantum computer, the state should never require manual normalization.)
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) = 0;

    /**
     * Apply the normalization factor found by UpdateRunningNorm() or on the fly by a single bit gate. (On an actual
     * quantum computer, the state should never require manual normalization.)
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F) = 0;

    /**
     * If asynchronous work is still running, block until it finishes. Note that this is never necessary to get correct,
     * timely return values. QEngines and other layers will always internally "Finish" when necessary for correct return
     * values. This is primarily for debugging and benchmarking.
     */
    virtual void Finish() {}

    /**
     * Returns "false" if asynchronous work is still running, and "true" if all previously dispatched asynchronous work
     * is done.
     */
    virtual bool isFinished() { return true; }

    /**
     * If asynchronous work is still running, let the simulator know that it can be aborted. Note that this method is
     * typically used internally where appropriate, such that user code typically does not call Dump().
     */
    virtual void Dump() {}

    /**
     * Returns "true" if current state representation is definitely a binary decision tree, "false" if it is definitely
     * not, or "true" if it cannot be determined.
     */
    virtual bool isBinaryDecisionTree() { return false; }

    /**
     * Returns "true" if current state is identifiably within the Clifford set, or "false" if it is not or cannot be
     * determined.
     */
    virtual bool isClifford() { return false; }

    /**
     * Returns "true" if current qubit state is identifiably within the Clifford set, or "false" if it is not or cannot
     * be determined.
     */
    virtual bool isClifford(bitLenInt qubit) { return false; }

    /**
     * Returns "true" if current simulation is OpenCL-based.
     */
    virtual bool isOpenCL() { return false; }

    /**
     *  Qrack::QUnit types maintain explicit separation of representations of qubits, which reduces memory usage and
     * increases gate speed. This method is used to manually attempt internal separation of a QUnit subsytem. We attempt
     * a Decompose() operation, on a state which might not be separable. If the state is not separable, we abort and
     * return false. Otherwise, we complete the operation, add the separated subsystem back in place into the QUnit
     * "shards," and return true.
     *
     * \warning PSEUDO-QUANTUM
     *
     * This should never change the logical/physical state of the QInterface, only possibly its internal representation,
     * for simulation optimization purposes. This is not a truly quantum computational operation, but it also does not
     * lead to nonphysical effects.
     */
    virtual bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol) { return false; }
    /**
     *  Single-qubit TrySeparate()
     */
    virtual bool TrySeparate(bitLenInt qubit) { return false; }
    /**
     *  Two-qubit TrySeparate()
     */
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2) { return false; }
    /**
     * When "Schmidt-decomposition rounding parameter" ("SDRP") is being used, starting from initial 1.0 fidelity, we
     * compound the "unitary fidelity" by successive multiplication by one minus two times the true unitary probability
     * discarded in each single rounding event. Then, an overall square-root is applied to the final fidelity estimate.
     * This is a useful metric on fidelity that requires no heavy calculational overhead to compute, and it tends to be
     * usually only slightly pessimistic compared to random circuit sampling fidelity calculated via the inner product.
     * This calculation is automatically reset to 1.0 by SetPermutation() or MAll() or can be manually reset by
     * "QInterface::ResetUnitaryFidelity()".
     */
    virtual double GetUnitaryFidelity() { return 1.0; }
    /**
     * Reset the internal fidelity calculation tracker to 1.0.
     */
    virtual void ResetUnitaryFidelity() {}
    /**
     * Set the "Schmidt decomposition rounding parameter" value, (between 0 and 1)
     */
    virtual void SetSdrp(real1_f sdrp){};
    /**
     * Set the "Near-clifford rounding parameter" value, (between 0 and 1)
     */
    virtual void SetNcrp(real1_f ncrp){};
    /**
     *  Set reactive separation option (on by default if available)
     *
     *  If reactive separation is available, as in Qrack::QUnit, then turning this option on attempts to
     * more-aggresively recover separability of subsystems. It can either hurt or help performance, though it commonly
     * helps.
     */
    virtual void SetReactiveSeparate(bool isAggSep) {}
    /**
     *  Get reactive separation option
     *
     *  If reactive separation is available, as in Qrack::QUnit, then turning this option on attempts to
     * more-aggresively recover separability of subsystems. It can either hurt or help performance, though it commonly
     * helps.
     */
    virtual bool GetReactiveSeparate() { return false; }
    /**
     *  Set the option to use T-injection gadgets (off by default)
     *
     *  If T-injection gadgets are available, as in Qrack::QStabilizerHybrid, then turning this option on attempts to
     * simulate Clifford+T with polynomial resource gadgets. It can either hurt or help performance, though it commonly
     * helps.
     */
    virtual void SetTInjection(bool useGadget) {}
    /**
     *  Get the option to use T-injection gadgets
     *
     *  If T-injection gadgets are available, as in Qrack::QStabilizerHybrid, then turning this option on attempts to
     * simulate Clifford+T with polynomial resource gadgets. It can either hurt or help performance, though it commonly
     * helps.
     */
    virtual bool GetTInjection() { return false; }
    /**
     *  Set the noise level option (only for a noisy interface)
     *
     *  If this is a QInterfaceNoisy, adjust the noise level per gate.
     */
    virtual void SetNoiseParameter(real1_f lambda) {}
    /**
     *  Get the noise level option (only for a noisy interface)
     *
     *  If this is a QInterfaceNoisy, return the noise level per gate.
     */
    virtual real1_f GetNoiseParameter() { return ZERO_R1_F; }

    /**
     *  Clone this QInterface
     */
    virtual QInterfacePtr Clone() = 0;

    /**
     *  Copy this QInterface
     */
    virtual QInterfacePtr Copy() { return Clone(); }

    /**
     *  Set the device index, if more than one device is available.
     */
    virtual void SetDevice(int64_t dID) = 0;

    /**
     *  Get the device index. ("-1" is default).
     */
    virtual int64_t GetDevice() { return -1; }

    /**
     *  Get maximum number of amplitudes that can be allocated on current device.
     */
    bitCapIntOcl GetMaxSize() { return pow2Ocl(sizeof(bitCapInt) * 8); };

    /**
     *  Get phase of lowest permutation nonzero amplitude.
     */
    virtual real1_f FirstNonzeroPhase()
    {
        complex amp;
        bitCapInt perm = ZERO_BCI;
        do {
            amp = GetAmplitude(perm);
            bi_increment(&perm, 1U);
        } while ((abs(amp) <= REAL1_EPSILON) && (perm < maxQPower));

        return (real1_f)std::arg(amp);
    }

    /**
     *  Simulate a local qubit depolarizing noise channel, under a stochastic "weak simulation condition." Under "weak"
     * condition, sampling and exact state queries are not accurate, but sampling can be achieved via repeated full
     * execution of a noisy circuit, for each hardware-realistic measurement sample.
     */
    virtual void DepolarizingChannelWeak1Qb(bitLenInt qubit, real1_f lambda);

    /** @} */
};
} // namespace Qrack
