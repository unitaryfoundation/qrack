//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
//
// Bounty #1008: Generalize expectation value and variance to all model moments
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "common/qrack_functions.hpp"

namespace Qrack {

/**
 * @defgroup Moment Generalized statistical moments for quantum measurements
 * 
 * This extension generalizes ExpectationBitsAll and VarianceBitsAll into a unified
 * Moment(n) interface. The nth moment E[X^n] provides:
 * - Moment(1) = Expectation (mean)
 * - Moment(2) = Second moment (E[X^2], used for variance)
 * - Moment(3) = Third moment (skewness-related)
 * - Moment(4) = Fourth moment (kurtosis-related)
 * 
 * Variance = Moment(2) - Moment(1)^2
 * 
 * Reference: https://github.com/unitaryfoundation/qrack/issues/1008
 */

class QInterface;
typedef std::shared_ptr<QInterface> QInterfacePtr;

typedef std::vector<Pauli> PauliString;

c
[truncated]
 real1_f MomentPauliString(
        const PauliString& paulis,
        unsigned n,
        const bitCapInt& offset = ZERO_BCI);

    /**
     * Compute expectation value or variance using moment interface
     * 
     * This provides a unified interface that replaces separate
     * ExpectationBitsAll and VarianceBitsAll methods.
     * 
     * @param bits Qubit indices to measure
     * @param isExpectation If true, returns Moment(1), else variance
     * @param offset Permutation offset
     * @return Either expectation (n=1) or variance (computed from moments)
     */
    virtual real1_f ExpectationOrVariance(
        const std::vector<bitLenInt>& bits,
        bool isExpectation,
        const bitCapInt& offset = ZERO_BCI);

    /**
     * Unified moment interface that dispatches to specialized methods
     * 
     * This is the main entry point for all moment calculations.
     * It handles small n directly and delegates larger n appropriately.
     * 
     * @param bits Qubit indices
     * @param n Moment order (1 for expectation, 2 for variance base, etc.)
     * @param isRdm Whether to use reduced density matrix method
     * @param offset Permutation offset
     * @return The nth moment of the measurement distribution
     */
    virtual real1_f MomentAll(
        const std::vector<bitLenInt>& bits,
        unsigned n,
        bool isRdm,
        const bitCapInt& offset = ZERO_BCI);

    /** @} */
};

} // namespace Qrack
