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

#include "qinterface.hpp"

namespace Qrack {

// Arithmetic:

/** Add integer (without sign) */
void QInterface::INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        if (bi_and_1(toAdd) != 0) {
            X(start);
        }

        return;
    }

    std::vector<bitLenInt> bits(length);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            MACInvert(std::vector<bitLenInt>(bits.begin() + i, bits.begin() + i + j + 1U), ONE_CMPLX, ONE_CMPLX,
                start + ((i + j + 1U) % length));
        }
    }
}

void QInterface::INCDECC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    std::vector<bitLenInt> bits(length + 1U);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }
    bits[length] = carryIndex;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (length - i); ++j) {
            const bitLenInt target = start + (((i + j + 1U) == length) ? carryIndex : ((i + j + 1U) % length));
            MACInvert(
                std::vector<bitLenInt>(bits.begin() + i, bits.begin() + i + j + 1U), ONE_CMPLX, ONE_CMPLX, target);
        }
    }
}

/** Add integer (without sign, with controls) */
void QInterface::CINC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        return INC(toAdd, start, length);
    }

    if (!length) {
        return;
    }

    if (length == 1U) {
        if (bi_and_1(toAdd) != 0) {
            MCInvert(controls, ONE_CMPLX, ONE_CMPLX, start);
        }

        return;
    }

    for (const bitLenInt& control : controls) {
        X(control);
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        MACInvert(controls, ONE_CMPLX, ONE_CMPLX, start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            std::vector<bitLenInt> bits(controls.size() + length);
            std::copy(controls.begin(), controls.end(), bits.begin());
            for (bitLenInt k = 0U; k < (j + 1U); ++k) {
                bits[controls.size() + k] = start + i + k;
            }
            MACInvert(std::vector<bitLenInt>(bits.begin(), bits.begin() + controls.size() + j + 1U), ONE_CMPLX,
                ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }

    for (const bitLenInt& control : controls) {
        X(control);
    }
}

/// Quantum analog of classical "Full Adder" gate
void QInterface::CFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    std::vector<bitLenInt> cBits(controls.size() + 2U);
    std::copy(controls.begin(), controls.end(), cBits.begin());

    // Assume outputBit is in 0 state.
    cBits[controls.size()] = inputBit1;
    cBits[controls.size() + 1U] = inputBit2;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controls.size()] = inputBit2;
    cBits[controls.size() + 1U] = carryInSumOut;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
    MCInvert(std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX,
        carryInSumOut);

    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
}

/// Inverse of FullAdd
void QInterface::CIFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
    // (CNOT and CCNOT are self-inverse.)

    std::vector<bitLenInt> cBits(controls.size() + 2U);
    std::copy(controls.begin(), controls.end(), cBits.begin());

    // Assume outputBit is in 0 state.
    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controls.size()] = inputBit2;
    MCInvert(std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX,
        carryInSumOut);

    cBits[controls.size() + 1U] = carryInSumOut;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);

    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controls.size() + 1U] = inputBit2;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
}

void QInterface::ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    FullAdd(input1, input2, carry, output);

    if (length == 1U) {
        return Swap(carry, output);
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

        return IFullAdd(input1, input2, carry, output);
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    IFullAdd(input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        IFullAdd(input1 + i, input2 + i, output + i, output + i + 1U);
    }
    IFullAdd(input1, input2, carry, output);
}

void QInterface::CADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
    bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    CFullAdd(controls, input1, input2, carry, output);

    if (length == 1) {
        return CSwap(controls, carry, output);
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    for (bitLenInt i = 1; i < end; ++i) {
        CFullAdd(controls, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CFullAdd(controls, input1 + end, input2 + end, output + end, carry);
}

void QInterface::CIADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
    bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        CSwap(controls, carry, output);

        return CIFullAdd(controls, input1, input2, carry, output);
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    CIFullAdd(controls, input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        CIFullAdd(controls, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CIFullAdd(controls, input1, input2, carry, output);
}

} // namespace Qrack
