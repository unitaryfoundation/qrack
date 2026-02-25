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

#include "qalu.hpp"

#if !ENABLE_ALU
#error ALU has not been enabled
#endif

namespace Qrack {

/// Subtract integer (without sign)
void QAlu::DEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INC(invToSub, start, length);
}

/// Subtract integer (without sign, with controls)
void QAlu::CDEC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    CINC(invToSub, start, length, controls);
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QAlu::DECS(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INCS(invToSub, start, length, overflowIndex);
}

void QAlu::INCC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        INCDECC(toAdd + 1U, start, length, carryIndex);
    } else {
        INCDECC(toAdd, start, length, carryIndex);
    }
}

/// Subtract integer (without sign, with carry)
void QAlu::DECC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
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

/**
 * Add an integer to the register, with sign and with carry. If the overflow is set, flip phase on overflow. Because the
 * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
 * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
 * position before the cast.
 */
void QAlu::INCSC(
    const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        INCDECSC(toAdd + 1U, start, length, overflowIndex, carryIndex);
    } else {
        INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QAlu::DECSC(
    const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    bitCapInt invToSub = pow2(length) - toSub;
    if (hasCarry) {
        X(carryIndex);
    } else {
        --invToSub;
    }

    INCDECSC(invToSub, start, length, overflowIndex, carryIndex);
}

/**
 * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is an
 * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QAlu::INCSC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        INCDECSC(toAdd + 1U, start, length, carryIndex);
    } else {
        INCDECSC(toAdd, start, length, carryIndex);
    }
}

/**
 * Subtract an integer from the register, with sign and with carry. If the overflow is set, flip phase on overflow.
 * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is variable.
 * Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QAlu::DECSC(const bitCapInt& toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    bitCapInt invToSub = pow2(length) - toSub;
    if (hasCarry) {
        X(carryIndex);
    } else {
        --invToSub;
    }

    INCDECSC(invToSub, start, length, carryIndex);
}

#if ENABLE_BCD
/// Subtract BCD integer (without sign)
void QAlu::DECBCD(const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length)
{
    const bitCapInt invToSub = intPow(10U, length / 4U) - toSub;
    INCBCD(invToSub, inOutStart, length);
}

/// Add BCD integer (without sign, with carry)
void QAlu::INCBCDC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    bitCapInt cToAdd = toAdd;
    if (hasCarry) {
        X(carryIndex);
        ++cToAdd;
    }

    INCDECBCDC(cToAdd, inOutStart, length, carryIndex);
}

/// Subtract BCD integer (without sign, with carry)
void QAlu::DECBCDC(const bitCapInt& toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    bitCapInt cToSub = toSub;
    if (hasCarry) {
        X(carryIndex);
    } else {
        ++cToSub;
    }

    const bitCapInt maxVal = intPow(10U, length / 4U);
    cToSub %= maxVal;
    const bitCapInt invToSub = maxVal - cToSub;
    INCDECBCDC(invToSub, inOutStart, length, carryIndex);
}
#endif

} // namespace Qrack
