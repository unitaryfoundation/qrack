//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
//
// Bounty #1003: Dividend-and-remainder unitary division
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "qalu.hpp"

namespace Qrack {

/**
 * @defgroup DivMod Dividend-and-remainder quantum division
 * 
 * Implements quantum division that produces both quotient and remainder.
 * 
 * Algorithm: While dividend >= divisor:
 *   - Subtract divisor from dividend
 *   - Increment quotient
 * Remainder is the final dividend value.
 * 
 * Mathematical: dividend = divisor * quotient + remainder
 * where 0 <= remainder < divisor
 * 
 * Reference: https://github.com/unitaryfoundation/qrack/issues/1003
 */

class QAlu;
typedef std::shared_ptr<QAlu> QAluPtr;

/**
 * Extended QAlu interface with dividend-and-remainder division
 */
class QAluDivMod : public QAlu {
public:
    /**
     * @defgroup DivModMethods Quantum division with quotient and remainder
     * @{
     */

    /**
     * Quantum division with explicit quotient and remainder registers
     * 
     * Computes: dividend / divisor = quotient ... remainder
     * where remainder is stored back in dividend register (or separate)
     * 
     * The algorithm uses controlled quantum subtraction in a loop:
     * 1. Test if dividend >= divisor (via subtraction with carry)
     * 2. If yes: subtract divisor, increment quotient
     * 3. If no: done, remainder = dividend
     * 
     * @param qDividend Register containing dividend (input, becomes remainder)
     * @param qDivisor Register containing divisor (preserved via copy)
     * @param qQuotient Output register for quotient (initialized to |0...0>)
     * @param qRemainder Output register for remainder (separate from dividend)
     * @param length Bit length of all registers
     * @param controls Optional control qubits (for conditional division)
     */
    virtual void DIVMod(
        bitLenInt qDividend,
        bitLenInt qDivisor,
        bitLenInt qQuotient,
        bitLenInt qRemainder,
        bitLenInt length,
        const std::vector<bitLenInt>& controls = {}) = 0;

    /**
     * Controlled quantum division with quotient and remainder
     * 
     * @param toDiv Classical integer divisor
     * @param start Start of dividend register
     * @param quotientStart Start of quotient output register
     * @param remainderStart Start of remainder output register
     * @param length Bit length
     * @param controls Control qubits
     */
    virtual void CDIVMod(
        const bitCapInt& toDiv,
        bitLenInt start,
        bitLenInt quotientStart,
        bitLenInt remainderStart,
        bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;

    /**
     * Inverse of DIVMod - multiplication with remainder handling
     * 
     * Computes: dividend = divisor * quotient + remainder
     * Used for verifying division by reversing it
     * 
     * @param qDividend Result register (output)
     * @param qDivisor Divisor register
     * @param qQuotient Quotient register
     * @param qRemainder Remainder register
     * @param length Bit length
     */
    virtual void IDIVMod(
        bitLenInt qDividend,
        bitLenInt qDivisor,
        bitLenInt qQuotient,
        bitLenInt qRemainder,
        bitLenInt length) = 0;

    /**
     * Convenience wrapper matching existing DIV pattern
     * 
     * Uses internal temporary registers for quotient/remainder.
     * After execution:
     * - start register contains: quotient
     * - carryStart register contains: remainder
     * 
     * @param toDiv Classical integer divisor
     * @param start Dividend register (becomes quotient)
     * @param carryStart Remainder output register
     * @param length Bit length
     */
    virtual void DIVR(
        const bitCapInt& toDiv,
        bitLenInt start,
        bitLenInt carryStart,
        bitLenInt length) = 0;

    /**
     * Controlled version of DIVR
     * 
     * @param toDiv Classical integer divisor
     * @param start Dividend register
     * @param carryStart Remainder output register
     * @param length Bit length
     * @param controls Control qubits
     */
    virtual void CDIVR(
        const bitCapInt& toDiv,
        bitLenInt start,
        bitLenInt carryStart,
        bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;

    /**
     * Division modulo N with quotient and remainder
     * 
     * Computes: (dividend mod N) / divisor in modular arithmetic
     * Used in cryptographic applications like Shor's algorithm.
     * 
     * @param toDiv Divisor
     * @param modN Modulus
     * @param inStart Input register
     * @param quotientStart Quotient output
     * @param remainderStart Remainder output
     * @param length Bit length
     */
    virtual void DIVModNOut(
        const bitCapInt& toDiv,
        const bitCapInt& modN,
        bitLenInt inStart,
        bitLenInt quotientStart,
        bitLenInt remainderStart,
        bitLenInt length) = 0;

    /** @} */
};

} // namespace Qrack
