//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <algorithm>

#include "qengine_cpu.hpp"

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QEngineCPU::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Doubly-controlled not
void QEngineCPU::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }

    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    CCNOT(control1, control2, target, 1);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QEngineCPU::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }
    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    AntiCCNOT(control1, control2, target, 1);
}

/// Controlled not
void QEngineCPU::CNOT(bitLenInt control, bitLenInt target)
{
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }
    CNOT(control, target, 1);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QEngineCPU::AntiCNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }
    AntiCNOT(control, target, 1);
}

/// Hadamard gate
void QEngineCPU::H(bitLenInt qubit)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("operation on bit index greater than total
    // bits.");
    const Complex16 had[4] = { Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
        Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0) };
    ApplySingleBit(qubit, had, true);
}

/// NOT gate, which is also Pauli x matrix
void QEngineCPU::X(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubit, pauliX, false);
}

/// Apply Pauli Y matrix to bit
void QEngineCPU::Y(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubit, pauliY, false);
}

/// Apply Pauli Z matrix to bit
void QEngineCPU::Z(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplySingleBit(qubit, pauliZ, false);
}

/// Apply controlled Pauli Y matrix to bit
void QEngineCPU::CY(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliY, false);
}

/// Apply controlled Pauli Z matrix to bit
void QEngineCPU::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplyControlled2x2(control, target, pauliZ, false);
}

} // namespace Qrack
