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
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            MACInvert(&(bits[i]), j + 1U, ONE_CMPLX, ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }
}

/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INC(invToSub, start, length);
}

void QInterface::INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    std::unique_ptr<bitLenInt[]> bits(new bitLenInt[length + 1U]);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }
    bits[length] = carryIndex;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (length - i); ++j) {
            const bitLenInt target = start + (((i + j + 1U) == length) ? carryIndex : ((i + j + 1U) % length));
            MACInvert(&(bits[i]), j + 1U, ONE_CMPLX, ONE_CMPLX, target);
        }
    }
}

void QInterface::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        ++toAdd;
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
        ++toSub;
    }

    const bitCapInt invToSub = pow2(length) - toSub;
    INCDECC(invToSub, start, length, carryIndex);
}

/** Add integer (without sign, with controls) */
void QInterface::CINC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt const* controls, bitLenInt controlLen)
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

    for (bitLenInt i = 0U; i < controlLen; ++i) {
        X(controls[0]);
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (!((toAdd >> i) & 1U)) {
            continue;
        }
        MACInvert(controls, controlLen, ONE_CMPLX, ONE_CMPLX, start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            std::unique_ptr<bitLenInt[]> bits(new bitLenInt[controlLen + length]);
            std::copy(controls, controls + controlLen, bits.get());
            for (bitLenInt k = 0U; k < (j + 1U); ++k) {
                bits[controlLen + k] = start + i + k;
            }
            MACInvert(bits.get(), controlLen + j + 1U, ONE_CMPLX, ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }

    for (bitLenInt i = 0U; i < controlLen; ++i) {
        X(controls[0]);
    }
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt const* controls, bitLenInt controlLen)
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
 * Multiplication modulo N by integer, (out of place)
 */
void QInterface::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    bitLenInt controls[1];
    for (bitLenInt i = 0U; i < length; ++i) {
        controls[0] = inStart + i;
        const bitCapInt partMul = (toMul * pow2(i)) % modN;
        if (!partMul) {
            continue;
        }
        CINC(partMul, outStart, oLength, controls, 1U);
    }

    if (isPow2) {
        return;
    }

    const bitCapInt diffPow = pow2(length) / modN;
    const bitLenInt lDiff = log2(diffPow);
    controls[0] = inStart + length - (lDiff + 1U);
    for (bitCapInt i = 0U; i < diffPow; ++i) {
        DEC(modN, inStart, length);
        X(controls[0]);
        CDEC(modN, outStart, oLength, controls, 1U);
        X(controls[0]);
    }
    for (bitCapInt i = 0U; i < diffPow; ++i) {
        INC(modN, inStart, length);
    }
}

/**
 * Inverse of multiplication modulo N by integer, (out of place)
 */
void QInterface::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    const bitCapInt diffPow = pow2(length) / modN;
    const bitLenInt lDiff = log2(diffPow);
    bitLenInt controls[1] = { (bitLenInt)(inStart + length - (lDiff + 1U)) };

    if (!isPow2) {
        for (bitCapInt i = 0U; i < diffPow; ++i) {
            DEC(modN, inStart, length);
        }
        for (bitCapInt i = 0U; i < diffPow; ++i) {
            X(controls[0]);
            CINC(modN, outStart, oLength, controls, 1U);
            X(controls[0]);
            INC(modN, inStart, length);
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        controls[0] = inStart + i;
        const bitCapInt partMul = (toMul * pow2(i)) % modN;
        if (!partMul) {
            continue;
        }
        CDEC(partMul, outStart, oLength, controls, 1U);
    }
}

/**
 * Controlled multiplication modulo N by integer, (out of place)
 */
void QInterface::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt const* controls, bitLenInt controlLen)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen + 1U]);
    std::copy(controls, controls + controlLen, lControls.get());
    for (bitLenInt i = 0U; i < length; ++i) {
        lControls[controlLen] = inStart + i;
        const bitCapInt partMul = (toMul * pow2(i)) % modN;
        if (!partMul) {
            continue;
        }
        CINC(partMul, outStart, oLength, lControls.get(), controlLen + 1U);
    }

    if (isPow2) {
        return;
    }

    const bitCapInt diffPow = pow2(length) / modN;
    const bitLenInt lDiff = log2(diffPow);
    lControls[controlLen] = inStart + length - (lDiff + 1U);
    for (bitCapInt i = 0U; i < diffPow; ++i) {
        CDEC(modN, inStart, length, lControls.get(), controlLen);
        X(lControls[controlLen]);
        CDEC(modN, outStart, oLength, lControls.get(), controlLen + 1U);
        X(lControls[controlLen]);
    }
    for (bitCapInt i = 0U; i < diffPow; ++i) {
        CINC(modN, inStart, length, lControls.get(), controlLen);
    }
}

/**
 * Inverse of controlled multiplication modulo N by integer, (out of place)
 */
void QInterface::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt const* controls, bitLenInt controlLen)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen + 1U]);
    std::copy(controls, controls + controlLen, lControls.get());
    const bitCapInt diffPow = pow2(length) / modN;
    const bitLenInt lDiff = log2(diffPow);
    lControls[controlLen] = inStart + length - (lDiff + 1U);

    if (!isPow2) {
        for (bitCapInt i = 0U; i < diffPow; ++i) {
            CDEC(modN, inStart, length, lControls.get(), controlLen);
        }
        for (bitCapInt i = 0U; i < diffPow; ++i) {
            X(lControls[controlLen]);
            CINC(modN, outStart, oLength, lControls.get(), controlLen + 1U);
            X(lControls[controlLen]);
            CINC(modN, inStart, length, lControls.get(), controlLen);
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        lControls[controlLen] = inStart + i;
        const bitCapInt partMul = (toMul * pow2(i)) % modN;
        if (!partMul) {
            continue;
        }
        CDEC(partMul, outStart, oLength, lControls.get(), controlLen + 1U);
    }
}

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
void QInterface::CFullAdd(bitLenInt const* controlBits, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    std::unique_ptr<bitLenInt[]> cBitsU(new bitLenInt[controlLen + 2U]);
    bitLenInt* cBits = cBitsU.get();
    std::copy(controlBits, controlBits + controlLen, cBits);

    // Assume outputBit is in 0 state.
    cBits[controlLen] = inputBit1;
    cBits[controlLen + 1U] = inputBit2;
    MCInvert(cBits, controlLen + 2U, ONE_CMPLX, ONE_CMPLX, carryOut);

    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controlLen] = inputBit2;
    cBits[controlLen + 1U] = carryInSumOut;
    MCInvert(cBits, controlLen + 2U, ONE_CMPLX, ONE_CMPLX, carryOut);

    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, carryInSumOut);

    cBits[controlLen] = inputBit1;
    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, inputBit2);
}

/// Inverse of FullAdd
void QInterface::CIFullAdd(bitLenInt const* controlBits, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
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
    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controlLen] = inputBit2;
    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, carryInSumOut);

    cBits[controlLen + 1U] = carryInSumOut;
    MCInvert(cBits, controlLen + 2U, ONE_CMPLX, ONE_CMPLX, carryOut);

    cBits[controlLen] = inputBit1;
    MCInvert(cBits, controlLen + 1U, ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controlLen + 1U] = inputBit2;
    MCInvert(cBits, controlLen + 2U, ONE_CMPLX, ONE_CMPLX, carryOut);
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
    for (bitLenInt i = 1U; i < end; ++i) {
        FullAdd(input1 + i, input2 + i, output + i, output + i + 1U);
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
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        IFullAdd(input1 + i, input2 + i, output + i, output + i + 1U);
    }
    IFullAdd(input1, input2, carry, output);
}

void QInterface::CADC(bitLenInt const* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
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
    for (bitLenInt i = 1; i < end; ++i) {
        CFullAdd(controls, controlLen, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CFullAdd(controls, controlLen, input1 + end, input2 + end, output + end, carry);
}

void QInterface::CIADC(bitLenInt const* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
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
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        CIFullAdd(controls, controlLen, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CIFullAdd(controls, controlLen, input1, input2, carry, output);
}

} // namespace Qrack
