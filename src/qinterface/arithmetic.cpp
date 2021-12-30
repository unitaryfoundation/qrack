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

/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INC(invToSub, inOutStart, length);
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECS(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INCS(invToSub, inOutStart, length, overflowIndex);
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    CINC(invToSub, inOutStart, length, controls, controlLen);
}

#if ENABLE_BCD
/// Subtract BCD integer (without sign)
void QInterface::DECBCD(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    const bitCapInt invToSub = intPow(10U, length / 4U) - toSub;
    INCBCD(invToSub, inOutStart, length);
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
    if (length == 0) {
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
    if (length == 0) {
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
    if (length == 0) {
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
    if (length == 0) {
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
