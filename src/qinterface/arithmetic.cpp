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

#include "qinterface.hpp"

#if !ENABLE_ALU
#error ALU has not been enabled
#endif

namespace Qrack {

// Arithmetic:

/** Add integer (without sign) */
void QInterface::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        if (toAdd & 1U) {
            X(start);
        }
        return;
    }

    std::unique_ptr<bitLenInt[]> bits(new bitLenInt[length]);
    for (bitLenInt i = 0; i < length; i++) {
        bits[i] = start + i;
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0; i < length; i++) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0; j < (lengthMin1 - i); j++) {
            MACInvert(&(bits[i]), j + 1U, ONE_CMPLX, ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }
}

/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INC(invToSub, inOutStart, length);
}

void QInterface::INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    std::unique_ptr<bitLenInt[]> bits(new bitLenInt[length + 1U]);
    for (bitLenInt i = 0; i < length; i++) {
        bits[i] = start + i;
    }
    bits[length] = carryIndex;

    for (bitLenInt i = 0; i < length; i++) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0; j < (length - i); j++) {
            const bitLenInt target = start + (((i + j + 1U) == length) ? carryIndex : ((i + j + 1U) % length));
            MACInvert(&(bits[i]), j + 1U, ONE_CMPLX, ONE_CMPLX, target);
        }
    }
}

void QInterface::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECC(toAdd, start, length, carryIndex);
}

/// Subtract integer (without sign, with carry)
void QInterface::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECC(invToSub, start, length, carryIndex);
}

/** Add integer (without sign, with controls) */
void QInterface::CINC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        INC(toAdd, start, length);
        return;
    }

    if (!length) {
        return;
    }

    if (length == 1U) {
        if (toAdd & 1U) {
            MCInvert(controls, controlLen, ONE_CMPLX, ONE_CMPLX, start);
        }
        return;
    }

    for (bitLenInt i = 0; i < controlLen; i++) {
        X(controls[0]);
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0; i < length; i++) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        MACInvert(controls, controlLen, ONE_CMPLX, ONE_CMPLX, start + i);
        for (bitLenInt j = 0; j < (lengthMin1 - i); j++) {
            std::unique_ptr<bitLenInt[]> bits(new bitLenInt[controlLen + length]);
            std::copy(controls, controls + controlLen, bits.get());
            for (bitLenInt k = 0; k < (j + 1U); k++) {
                bits[controlLen + k] = start + i + k;
            }
            MACInvert(bits.get(), controlLen + j + 1U, ONE_CMPLX, ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }

    for (bitLenInt i = 0; i < controlLen; i++) {
        X(controls[0]);
    }
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    CINC(invToSub, inOutStart, length, controls, controlLen);
}

/** Add a classical integer to the register, with sign and without carry. */
void QInterface::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt signMask = pow2(length - 1U);
    INC(signMask, start, length);
    INCDECC(toAdd & ~signMask, start, length, overflowIndex);
    if (!(toAdd & signMask)) {
        DEC(signMask, start, length);
    }
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INCS(invToSub, start, length, overflowIndex);
}

/**
 * Add an integer to the register, with sign and with carry. If the overflow is set, flip phase on overflow. Because the
 * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
 * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
 * position before the cast.
 */
void QInterface::INCSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, overflowIndex, carryIndex);
}

/**
 * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is an
 * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QInterface::INCSC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECSC(
    bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECSC(invToSub, start, length, overflowIndex, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and with carry. If the overflow is set, flip phase on overflow.
 * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is variable.
 * Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QInterface::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECSC(invToSub, start, length, carryIndex);
}

#if ENABLE_BCD
/// Subtract BCD integer (without sign)
void QInterface::DECBCD(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    const bitCapInt invToSub = intPow(10U, length / 4U) - toSub;
    INCBCD(invToSub, inOutStart, length);
}

/// Add BCD integer (without sign, with carry)
void QInterface::INCBCDC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECBCDC(toAdd, inOutStart, length, carryIndex);
}

/// Subtract BCD integer (without sign, with carry)
void QInterface::DECBCDC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    const bitCapInt maxVal = intPow(10U, length / 4U);
    toSub %= maxVal;
    bitCapInt invToSub = maxVal - toSub;
    INCDECBCDC(invToSub, inOutStart, length, carryIndex);
}
#endif

/// Quantum analog of classical "Full Adder" gate
void QInterface::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer

    // Assume outputBit is in 0 state.
    CCNOT(inputBit1, inputBit2, carryOut);
    CNOT(inputBit1, inputBit2);
    CCNOT(inputBit2, carryInSumOut, carryOut);
    CNOT(inputBit2, carryInSumOut);
    CNOT(inputBit1, inputBit2);
}

/// Inverse of FullAdd
void QInterface::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
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

/// Quantum analog of classical "Full Adder" gate
void QInterface::CFullAdd(const bitLenInt* controlBits, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    std::unique_ptr<bitLenInt[]> cBitsU(new bitLenInt[controlLen + 2U]);
    bitLenInt* cBits = cBitsU.get();
    std::copy(controlBits, controlBits + controlLen, cBits);

    // Assume outputBit is in 0 state.
    cBits[controlLen] = inputBit1;
    cBits[controlLen + 1] = inputBit2;
    MCInvert(cBits, controlLen + 2, ONE_CMPLX, ONE_CMPLX, carryOut);

    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controlLen] = inputBit2;
    cBits[controlLen + 1] = carryInSumOut;
    MCInvert(cBits, controlLen + 2, ONE_CMPLX, ONE_CMPLX, carryOut);

    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, carryInSumOut);

    cBits[controlLen] = inputBit1;
    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, inputBit2);
}

/// Inverse of FullAdd
void QInterface::CIFullAdd(const bitLenInt* controlBits, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
    // (CNOT and CCNOT are self-inverse.)

    std::unique_ptr<bitLenInt[]> cBitsU(new bitLenInt[controlLen + 2U]);
    bitLenInt* cBits = cBitsU.get();
    std::copy(controlBits, controlBits + controlLen, cBits);

    // Assume outputBit is in 0 state.
    cBits[controlLen] = inputBit1;
    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controlLen] = inputBit2;
    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, carryInSumOut);

    cBits[controlLen + 1] = carryInSumOut;
    MCInvert(cBits, controlLen + 2, ONE_CMPLX, ONE_CMPLX, carryOut);

    cBits[controlLen] = inputBit1;
    MCInvert(cBits, controlLen + 1, ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controlLen + 1] = inputBit2;
    MCInvert(cBits, controlLen + 2, ONE_CMPLX, ONE_CMPLX, carryOut);
}

void QInterface::ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    FullAdd(input1, input2, carry, output);

    if (length == 1U) {
        Swap(carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    for (bitLenInt i = 1U; i < end; i++) {
        FullAdd(input1 + i, input2 + i, output + i, output + i + 1);
    }
    FullAdd(input1 + end, input2 + end, output + end, carry);
}

void QInterface::IADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        Swap(carry, output);
        IFullAdd(input1, input2, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    IFullAdd(input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0; i--) {
        IFullAdd(input1 + i, input2 + i, output + i, output + i + 1);
    }
    IFullAdd(input1, input2, carry, output);
}

void QInterface::CADC(const bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
    bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    CFullAdd(controls, controlLen, input1, input2, carry, output);

    if (length == 1) {
        CSwap(controls, controlLen, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    for (bitLenInt i = 1; i < end; i++) {
        CFullAdd(controls, controlLen, input1 + i, input2 + i, output + i, output + i + 1);
    }
    CFullAdd(controls, controlLen, input1 + end, input2 + end, output + end, carry);
}

void QInterface::CIADC(const bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
    bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        CSwap(controls, controlLen, carry, output);
        CIFullAdd(controls, controlLen, input1, input2, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    CIFullAdd(controls, controlLen, input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0; i--) {
        CIFullAdd(controls, controlLen, input1 + i, input2 + i, output + i, output + i + 1);
    }
    CIFullAdd(controls, controlLen, input1, input2, carry, output);
}

} // namespace Qrack
