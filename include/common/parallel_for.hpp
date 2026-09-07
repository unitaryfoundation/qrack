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

#include "qrack_functions.hpp"

#include <functional>

namespace Qrack {

// Called once per value between begin and end.
typedef std::function<void(const size_t&, const unsigned& cpu)> ParallelFunc;
typedef std::function<void(const bitCapInt&, const unsigned& cpu)> ParallelFuncSparse;
typedef std::function<size_t(const size_t&)> IncrementFunc;
typedef std::function<bitCapInt(const size_t&)> IncrementFuncSparse;

class ParallelFor {
private:
    const size_t pStride;
    bitLenInt dispatchThreshold;
    unsigned numCores;

public:
    ParallelFor();

    void SetConcurrencyLevel(unsigned num)
    {
        if (!num) {
            num = 1U;
        }
        if (numCores == num) {
            return;
        }
        numCores = num;
        const bitLenInt pStridePow = log2Ocl(pStride);
        const bitLenInt minStridePow = (bitLenInt)pow2Ocl(log2Ocl(numCores - 1U));
        dispatchThreshold = (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
    }
    unsigned GetConcurrencyLevel() { return numCores; }
    size_t GetStride() { return pStride; }
    bitLenInt GetPreferredConcurrencyPower() { return dispatchThreshold; }
    /*
     * Parallelization routines for spreading work across multiple cores.
     */

    /**
     * Iterate through the permutations a maximum of end-begin times, allowing
     * the caller to control the incrementation offset through 'inc'.
     */
    void par_for_inc(const size_t begin, const size_t itemCount, IncrementFunc, ParallelFunc fn);
    void par_for_inc_sparse(const size_t begin, const size_t itemCount, IncrementFuncSparse, ParallelFuncSparse fn);

    /** Call fn once for every numerical value between begin and end. */
    void par_for(const size_t begin, const size_t end, ParallelFunc fn);

    /**
     * Skip over the skipPower bits.
     *
     * For example, if skipPower is 2, it will count:
     *   0000, 0001, 0100, 0101, 1000, 1001, 1100, 1101.
     *     ^     ^     ^     ^     ^     ^     ^     ^ - The second bit is
     *                                                   untouched.
     */
    void par_for_skip(
        const size_t begin, const size_t end, const size_t skipPower, const bitLenInt skipBitCount, ParallelFunc fn);

    /** Skip over the bits listed in maskArray in the same fashion as par_for_skip. */
    void par_for_mask(const size_t, const size_t, const std::vector<size_t>& maskArray, ParallelFunc fn);

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::set<bitCapInt>& sparseSet, ParallelFuncSparse fn);

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::vector<bitCapInt>& sparseSet, ParallelFuncSparse fn);

    /** Iterate over the power set of 2 sparse state vectors. */
    void par_for_sparse_compose(const std::vector<bitCapInt>& lowSet, const std::vector<bitCapInt>& highSet,
        const bitLenInt& highStart, ParallelFuncSparse fn);

    /** Calculate the normal for the array, (with flooring). */
    real1_f par_norm(const size_t maxQPower, const StateVectorPtr stateArray, real1_f norm_thresh = ZERO_R1_F);

    /** Calculate the normal for the array, (without flooring.) */
    real1_f par_norm_exact(const size_t maxQPower, const StateVectorPtr stateArray);
};

} // namespace Qrack
