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

/* Needed for bitCapInt typedefs. */
#include "qrack_types.hpp"

#include <set>
#include <vector>

namespace Qrack {

class ParallelFor {
private:
    const bitCapIntOcl pStride;
    unsigned numCores;

public:
    ParallelFor();

    void SetConcurrencyLevel(unsigned num) { numCores = num; }
    unsigned GetConcurrencyLevel() { return numCores; }
    bitCapIntOcl GetStride() { return pStride; }
    /*
     * Parallelization routines for spreading work across multiple cores.
     */

    /**
     * Iterate through the permutations a maximum of end-begin times, allowing
     * the caller to control the incrementation offset through 'inc'.
     */
    void par_for_inc(const bitCapIntOcl begin, const bitCapIntOcl itemCount, IncrementFunc, ParallelFunc fn);

    /** Call fn once for every numerical value between begin and end. */
    void par_for(const bitCapIntOcl begin, const bitCapIntOcl end, ParallelFunc fn);

    /**
     * Skip over the skipPower bits.
     *
     * For example, if skipPower is 2, it will count:
     *   0000, 0001, 0100, 0101, 1000, 1001, 1100, 1101.
     *     ^     ^     ^     ^     ^     ^     ^     ^ - The second bit is
     *                                                   untouched.
     */
    void par_for_skip(const bitCapIntOcl begin, const bitCapIntOcl end, const bitCapIntOcl skipPower,
        const bitLenInt skipBitCount, ParallelFunc fn);

    /** Skip over the bits listed in maskArray in the same fashion as par_for_skip. */
    void par_for_mask(
        const bitCapIntOcl, const bitCapIntOcl, const std::vector<bitCapIntOcl>& maskArray, ParallelFunc fn);

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::set<bitCapIntOcl>& sparseSet, ParallelFunc fn);

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::vector<bitCapIntOcl>& sparseSet, ParallelFunc fn);

    /** Iterate over the power set of 2 sparse state vectors. */
    void par_for_sparse_compose(const std::vector<bitCapIntOcl>& lowSet, const std::vector<bitCapIntOcl>& highSet,
        const bitLenInt& highStart, ParallelFunc fn);

    /** Calculate the normal for the array, (with flooring). */
    real1_f par_norm(const bitCapIntOcl maxQPower, const StateVectorPtr stateArray, real1_f norm_thresh = ZERO_R1_F);

    /** Calculate the normal for the array, (without flooring.) */
    real1_f par_norm_exact(const bitCapIntOcl maxQPower, const StateVectorPtr stateArray);

    /** Parallel for loop specific to QBdt **/
    void par_for_qbdt(const bitCapInt end, BdtFunc fn);
};

} // namespace Qrack
