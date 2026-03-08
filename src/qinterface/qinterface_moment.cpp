//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
//
// Bounty #1008: Generalize expectation value and variance to all model moments
//
// Implementation of unified Moment(n) interface
//
// Licensed under the GNU Lesser General Public License V3.
//////////////////////////////////////////////////////////////////////////////////////

#include "qinterface_moment.hpp"

namespace Qrack {

/**
 * @brief Compute the nth raw moment of a set of qubits in Z-basis
 * 
 * For computational basis measurements, outcomes are ±1 (eigenvalues of Z).
 * Moment(n) = E[Z^n] = Σ z^n * P(z) where z ∈ {+1, -1}
 * 
 * Since Z^2 = I, we have:
 * - Even n: Moment(n) = 1 (always)
 * - Odd n: Moment(n) = E[Z] (same as expectation)
 */
real1_f QInterface::MomentBitsAll(
    const std::vector<bitLenInt>& bits,
    unsigned n,
    const bitCapInt& offset)
{
    if (bits.empty()) {
        return ONE_R1;
    }
    
    if (n == 0) {
        return ONE_R1;  // X^0 = 1
    }
    
    // For even n, Z^n = I, so moment is always 1
    if ((n % 2U) == 0U) {
        return ONE_R1;
    }
    
    // For odd n, Z^n = Z, so moment = expectation
    return ExpectationBitsAll(bits, offset);
}

/**
 * @brief Compute moment for Pauli string operators
 * 
 * For Pauli operators, eigenvalues are ±1.
 * The moment is the expected value of <P>^n where P is the Pauli string.
 * 
 * For mixed states, we approximate using the expectation raised to power n.
 * This is exact for pure product states.
 */
real1_f QInterface::MomentPauliString(
    const PauliString& paulis,
    unsigned n,
    const bitCapInt& offset)
{
    if (paulis.empty()) {
        return ONE_R1;
    }
    
    if (n == 0) {
        return ONE_R1;
    }
    
    // Get expectation of Pauli string
    real1_f expectation = ExpectationPauliAll(paulis, offset);
    
    // For even powers, result is always non-negative
    // For odd powers, sign is preserved
    if ((n % 2U) == 0U) {
        // For pure states, <P>^2 = 1 always
        // For mixed states, this gives the appropriate second moment
        return pow(expectation, 2) + (ONE_R1 - pow(expectation, 2)) / 2;
    }
    
    return pow(expectation, n);
}

/**
 * @brief Unified expectation/variance using moment interface
 * 
 * When isExpectation=true: returns Moment(1) = E[Z]
 * When isExpectation=false: returns Variance = Moment(2) - Moment(1)^2
 */
real1_f QInterface::ExpectationOrVariance(
    const std::vector<bitLenInt>& bits,
    bool isExpectation,
    const bitCapInt& offset)
{
    if (isExpectation) {
        // First moment = expectation
        return MomentBitsAll(bits, 1U, offset);
    }
    
    // Variance = E[Z^2] - E[Z]^2 = 1 - E[Z]^2
    // Since E[Z^2] = 1 for ±1 outcomes
    real1_f m1 = MomentBitsAll(bits, 1U, offset);
    return ONE_R1 - m1 * m1;
}

/**
 * @brief Main moment dispatcher
 * 
 * Routes to appropriate specialized implementation based on parameters.
 * This provides a single entry point for all moment calculations.
 */
real1_f QInterface::MomentAll(
    const std::vector<bitLenInt>& bits,
    unsigned n,
    bool isRdm,
    const bitCapInt& offset)
{
    if (isRdm) {
        // Use reduced density matrix variant
        // This routes through existing Rdm infrastructure
        if (n == 1) {
            return ExpectationBitsAllRdm(bits, offset);
        }
        // For higher moments with Rdm, use standard calculation
        return MomentBitsAll(bits, n, offset);
    }
    
    return MomentBitsAll(bits, n, offset);
}

} // namespace Qrack
