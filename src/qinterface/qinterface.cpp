//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qinterface.hpp"

namespace Qrack {

// Bit-wise apply "anti-"controlled-not to three registers
void QInterface::Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Swap(qubit1 + bit, qubit2 + bit);
    }
}

// Bit-wise apply "anti-"controlled-not to three registers
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        AntiCCNOT(control1 + bit, control2 + bit, target + bit);
    }
}

void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CCNOT(control1 + bit, control2 + bit, target + bit);
    }
}

void QInterface::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        AntiCNOT(control + bit, target + bit);
    }
}

void QInterface::CNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CNOT(control + bit, target + bit);
    }
}

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QInterface::X(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        X(start + bit);
    }
}

// Single register instructions:

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
void QInterface::H(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        H(start + bit);
    }
}

/// Apply Pauli Y matrix to each bit
void QInterface::Y(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Y(start + bit);
    }
}

/// Apply Pauli Z matrix to each bit
void QInterface::Z(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Z(start + bit);
    }
}

/// Apply controlled Pauli Y matrix to each bit
void QInterface::CY(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CY(control + bit, target + bit);
    }
}

/// Apply controlled Pauli Z matrix to each bit
void QInterface::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CZ(control + bit, target + bit);
    }
}

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLAND(qInputStart + i, cBit, outputStart + i);
    }
}

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLXOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// Arithmetic shift left, with last 2 bits as sign and carry
void QInterface::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROL(shift, start, length);
            SetReg(start, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void QInterface::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROR(shift, start, length);
            SetReg(end - shift - 1, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Logical shift left, filling the extra bits with |0>
void QInterface::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROL(shift, start, length);
            SetReg(start, shift, 0);
        }
    }
}

/// Logical shift right, filling the extra bits with |0>
void QInterface::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            SetReg(start, shift, 0);
            ROR(shift, start, length);
        }
    }
}

/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
void QInterface::QFT(bitLenInt start, bitLenInt length)
{
    if (length > 0) {
        bitLenInt end = start + length;
        bitLenInt i, j;
        for (i = start; i < end; i++) {
            H(i);
            for (j = 1; j < (end - i); j++) {
                CRTDyad(1, j, i + j, i);
            }
        }
    }
}

/// Set register bits to given permutation
void QInterface::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (value == 1));
    } else if ((start == 0) && (length == qubitCount)) {
        SetPermutation(value);
    } else {
        bool bitVal;
        bitCapInt regVal = MReg(start, length);
        for (bitLenInt i = 0; i < length; i++) {
            bitVal = regVal & (1 << i);
            if ((bitVal && !(value & (1 << i))) || (!bitVal && (value & (1 << i))))
                X(start + i);
        }
    }
}

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
void QInterface::RT(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RT(radians, start + bit);
    }
}

/**
 * Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) /
 * 2^denomPower) around |1> state.
 *
 * NOTE THAT * DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QInterface::RTDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RT((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RTDyad(numerator, denominator, start + bit);
    }
}

/**
 * Bitwise (identity) exponentiation gate - Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
 */
void QInterface::Exp(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Exp(radians, start + bit);
    }
}

/**
 * Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
 * exponentiation of the identity operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    Exp((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
 * exponentiation of the identity operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpDyad(numerator, denominator, start + bit);
    }
}

/**
 * Bitwise Pauli X exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
 */
void QInterface::ExpX(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpX(radians, start + bit);
    }
}

/**
 * Dyadic fraction Pauli X exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_x / 2^denomPower} \f$,
 * exponentiation of the Pauli X operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    ExpX((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction Pauli X exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_x / 2^denomPower} \f$,
 * exponentiation of the Pauli X operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpXDyad(numerator, denominator, start + bit);
    }
}

/**
 * Bitwise Pauli Y exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
 */
void QInterface::ExpY(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpY(radians, start + bit);
    }
}

/**
 * Dyadic fraction Pauli Y exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_y / 2^denomPower} \f$,
 * exponentiation of the Pauli Y operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    ExpY((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction Pauli Y exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_y / 2^denomPower} \f$,
 * exponentiation of the Pauli Y operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpYDyad(numerator, denominator, start + bit);
    }
}

/**
 * Dyadic fraction Pauli Z exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_z / 2^denomPower} \f$,
 * exponentiation of the Pauli Z operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpZDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    ExpZ((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Bitwise Pauli Z exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
 */
void QInterface::ExpZ(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpZ(radians, start + bit);
    }
}

/**
 * Dyadic fraction Pauli Z exponentiation gate - Applies \f$ e^{-i * \pi * numerator *\sigma_z / 2^denomPower} \f$,
 * exponentiation of the Pauli Z operator
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QInterface::ExpZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpZDyad(numerator, denominator, start + bit);
    }
}

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
void QInterface::RX(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RX(radians, start + bit);
    }
}

/**
 * Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * 2^denomPower) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QInterface::RXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RX((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction x axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli x
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QInterface::RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RXDyad(numerator, denominator, start + bit);
    }
}

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
void QInterface::RY(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RY(radians, start + bit);
    }
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * 2^denomPower) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QInterface::RYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RY((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction y axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QInterface::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RYDyad(numerator, denominator, start + bit);
    }
}

/// z axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli z axis
void QInterface::RZ(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RZ(radians, start + bit);
    }
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QInterface::RZDyad(int numerator, int denomPower, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RZ((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/**
 * Dyadic fraction z axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QInterface::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RZDyad(numerator, denominator, start + bit);
    }
}

/// Controlled "phase shift gate"
void QInterface::CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRT(radians, control + bit, target + bit);
    }
}

/// Controlled dyadic "phase shift gate" - if control bit is true, rotates target bit as e^(i*(M_PI * numerator) /
/// 2^denomPower) around |1> state
void QInterface::CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    // if (target >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRTDyad control bit cannot also be target.");
    CRT((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction "phase shift gate"
void QInterface::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRTDyad(numerator, denominator, control + bit, target + bit);
    }
}

/// Controlled x axis rotation
void QInterface::CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRX(radians, control + bit, target + bit);
    }
}

/**
 * Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI *
 * numerator) / 2^denomPower) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QInterface::CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRXDyad control bit cannot also be target.");
    CRX((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void QInterface::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRXDyad(numerator, denominator, control + bit, target + bit);
    }
}

/// Controlled y axis rotation
void QInterface::CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRY(radians, control + bit, target + bit);
    }
}

/**
 * Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QInterface::CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRYDyad control bit cannot also be target.");
    CRY((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void QInterface::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRYDyad(numerator, denominator, control + bit, target + bit);
    }
}

/// Controlled z axis rotation
void QInterface::CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRZ(radians, control + bit, target + bit);
    }
}

/**
 * Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around Pauli z
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QInterface::CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZDyad control bit cannot also be target.");
    CRZ((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void QInterface::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        CRZDyad(numerator, denominator, control + bit, target + bit);
    }
}

// Bit-wise apply measurement gate to a register
bitCapInt QInterface::MReg(bitLenInt start, bitLenInt length)
{
    bitCapInt result = 0;
    for (bitLenInt bit = 0; bit < length; bit++) {
        result |= M(start + bit) ? (1 << bit) : 0;
    }
    return result;
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QInterface::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        Reverse(start, end);
        Reverse(start, start + shift);
        Reverse(start + shift, end);
    }
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QInterface::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        Reverse(start + shift, end);
        Reverse(start, start + shift);
        Reverse(start, end);
    }
}

std::map<QInterfacePtr, bitLenInt> QInterface::Cohere(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto&& q : toCopy) {
        ret[q] = Cohere(q);
    }

    return ret;
}

} // namespace Qrack
