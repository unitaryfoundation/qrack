//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

template <typename GateFunc> void QInterface::ControlledLoopFixture(bitLenInt length, GateFunc gate)
{
    // For length-wise application of controlled gates, there's no point in having normalization on, up to the last
    // gate. Application of a controlled gate updates the "running norm". The running norm is corrected on the
    // application of a gate that isn't controlled. We just want one running norm update, for the last gate.
    bool wasNormOn = doNormalize;
    doNormalize = false;
    for (bitLenInt bit = 0; bit < (length - 1); bit++) {
        gate(bit);
    }
    doNormalize = wasNormOn;
    gate(length - 1);
}

// Bit-wise apply swap to two registers
void QInterface::Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Swap(qubit1 + bit, qubit2 + bit);
    }
}

// Bit-wise apply square root of swap to two registers
void QInterface::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        SqrtSwap(qubit1 + bit, qubit2 + bit);
    }
}

// Bit-wise apply inverse square root of swap to two registers
void QInterface::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ISqrtSwap(qubit1 + bit, qubit2 + bit);
    }
}

// Bit-wise apply "anti-"controlled-not to three registers
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { AntiCCNOT(control1 + bit, control2 + bit, target + bit); });
}

void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CCNOT(control1 + bit, control2 + bit, target + bit); });
}

void QInterface::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { AntiCNOT(control + bit, target + bit); });
}

void QInterface::CNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CNOT(control + bit, target + bit); });
}

// Apply S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
void QInterface::S(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        S(start + bit);
    }
}

// Apply inverse S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
void QInterface::IS(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        IS(start + bit);
    }
}

// Apply T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
void QInterface::T(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        T(start + bit);
    }
}

// Apply inverse T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
void QInterface::IT(bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        IT(start + bit);
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

/// Apply general unitary gate to each bit in "length," starting from bit index "start"
void QInterface::U(bitLenInt start, bitLenInt length, real1 theta, real1 phi, real1 lambda)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        U(start + bit, theta, phi, lambda);
    }
}

/// Apply 2-parameter unitary gate to each bit in "length," starting from bit index "start"
void QInterface::U2(bitLenInt start, bitLenInt length, real1 phi, real1 lambda)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        U2(start + bit, phi, lambda);
    }
}

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
    ControlledLoopFixture(length, [&](bitLenInt bit) { CY(control + bit, target + bit); });
}

/// Apply controlled Pauli Z matrix to each bit
void QInterface::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CZ(control + bit, target + bit); });
}

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = bitSlice(i, classicalInput);
        CLAND(qInputStart + i, cBit, outputStart + i);
    }
}

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = bitSlice(i, classicalInput);
        CLOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = bitSlice(i, classicalInput);
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

/// Quantum Fourier Transform - Optimized for going from |0>/|1> to |+>/|-> basis
void QInterface::QFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    bitLenInt end = start + (length - 1U);
    bitLenInt i, j;
    for (i = 0; i < length; i++) {
        H(end - i);
        for (j = 0; j < ((length - 1U) - i); j++) {
            CRT((-M_PI * 2) / intPow(2, j + 2), (end - i) - (j + 1U), end - i);
        }

        if (trySeparate) {
            TrySeparate(end - i);
        }
    }
}

/// Inverse Quantum Fourier Transform - Quantum Fourier transform optimized for going from |+>/|-> to |0>/|1> basis
void QInterface::IQFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    bitLenInt i, j;
    for (i = 0; i < length; i++) {
        for (j = 0; j < i; j++) {
            CRT((M_PI * 2) / intPow(2, j + 2), (start + i) - (j + 1U), start + i);
        }
        H(start + i);

        if (trySeparate) {
            TrySeparate(start + i);
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
            bitVal = bitSlice(i, regVal);
            if ((bitVal && !bitSlice(i, value)) || (!bitVal && bitSlice(i, value)))
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

/// Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around |1> state.
void QInterface::RTDyad(int numerator, int denomPower, bitLenInt qubit)
{
    RT((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
void QInterface::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RTDyad(numerator, denominator, start + bit);
    }
}

/// Bitwise (identity) exponentiation gate - Applies exponentiation of the identity operator
void QInterface::Exp(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        Exp(radians, start + bit);
    }
}

/// Dyadic fraction (identity) exponentiation gate - Applies exponentiation of the identity operator
void QInterface::ExpDyad(int numerator, int denomPower, bitLenInt qubit)
{
    Exp((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
void QInterface::ExpDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpDyad(numerator, denominator, start + bit);
    }
}

/// Bitwise Pauli X exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
void QInterface::ExpX(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpX(radians, start + bit);
    }
}

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
void QInterface::ExpXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpX((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
void QInterface::ExpXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpXDyad(numerator, denominator, start + bit);
    }
}

/// Bitwise Pauli Y exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
void QInterface::ExpY(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpY(radians, start + bit);
    }
}

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
void QInterface::ExpYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpY((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
void QInterface::ExpYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        ExpYDyad(numerator, denominator, start + bit);
    }
}

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
void QInterface::ExpZDyad(int numerator, int denomPower, bitLenInt qubit)
{
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

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
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

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::RXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    RX((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x
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

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    RY((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction y axis rotation gate - Rotates each bit around Pauli y axis.
void QInterface::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RYDyad(numerator, denominator, start + bit);
    }
}

/// z axis rotation gate - Rotates each bit around Pauli z axis
void QInterface::RZ(real1 radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RZ(radians, start + bit);
    }
}

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RZDyad(int numerator, int denomPower, bitLenInt qubit)
{
    RZ((-M_PI * numerator * 2) / pow(2, denomPower), qubit);
}

/// Dyadic fraction z axis rotation gate - Rotates each bit around Pauli y axis.
void QInterface::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        RZDyad(numerator, denominator, start + bit);
    }
}

/// Controlled "phase shift gate"
void QInterface::CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRT(radians, control + bit, target + bit); });
}

/// Controlled dyadic "phase shift gate" - if control bit is true, rotates target bit as e^(i*(M_PI * numerator) /
/// 2^denomPower) around |1> state
void QInterface::CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRT((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction "phase shift gate"
void QInterface::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRTDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled x axis rotation
void QInterface::CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRX(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRX((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void QInterface::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRXDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled y axis rotation
void QInterface::CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRY(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRY((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void QInterface::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRYDyad(numerator, denominator, control + bit, target + bit); });
}

/// Controlled z axis rotation
void QInterface::CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRZ(radians, control + bit, target + bit); });
}

/// Controlled dyadic fraction z axis rotation gate - Rotates around Pauli z axis.
void QInterface::CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRZ((-M_PI * numerator * 2) / pow(2, denomPower), control, target);
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void QInterface::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    ControlledLoopFixture(length, [&](bitLenInt bit) { CRZDyad(numerator, denominator, control + bit, target + bit); });
}

// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce)
{
    bitCapInt res = 0;
    bitCapInt power;
    for (bitLenInt bit = 0; bit < length; bit++) {
        power = pow2(bit);
        res |= ForceM(start + bit, !(!(power & result)), doForce) ? power : 0;
    }
    return res;
}

// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values)
{
    bitCapInt result = 0;
    if (values == NULL) {
        for (bitLenInt bit = 0; bit < length; bit++) {
            result |= M(bits[bit]) ? pow2(bits[bit]) : 0;
        }
    } else {
        for (bitLenInt bit = 0; bit < length; bit++) {
            result |= ForceM(bits[bit], values[bit]) ? pow2(bits[bit]) : 0;
        }
    }
    return result;
}

// Returns probability of permutation of the register
real1 QInterface::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    real1 prob = ONE_R1;
    for (bitLenInt i = 0; i < length; i++) {
        if ((permutation >> i) & 1U) {
            prob *= Prob(start + i);
        } else {
            prob *= (ONE_R1 - Prob(start + i));
        }
    }
    return prob;
}

// Returns probability of permutation of the mask
real1 QInterface::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    real1 prob = ZERO_R1;
    for (bitCapInt lcv = 0; lcv < maxQPower; lcv++) {
        if ((lcv & mask) == permutation) {
            prob += ProbAll(lcv);
        }
    }

    return prob;
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

/// "Circular shift right" - shift bits right, and carry first bits.
void QInterface::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) { ROL(length - shift, start, length); }

std::map<QInterfacePtr, bitLenInt> QInterface::Compose(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto&& q : toCopy) {
        ret[q] = Compose(q);
    }

    return ret;
}

bool QInterface::TryDecompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    Finish();

    bool tempDoNorm = doNormalize;
    doNormalize = false;

    QInterfacePtr unitCopy = Clone();

    unitCopy->Decompose(start, length, dest);
    unitCopy->Compose(dest, start);

    bool didSeparate = ApproxCompare(unitCopy);
    if (didSeparate) {
        // The subsystem is separable.
        Dispose(start, length);
    }

    Finish();

    doNormalize = tempDoNorm;

    return didSeparate;
}

} // namespace Qrack
