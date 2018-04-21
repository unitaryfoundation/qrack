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

#include "qengine_cpu.hpp"

namespace Qrack {

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
{
    while ((first < last) && (first < (last - stride))) {
        last -= stride;
        std::iter_swap(first, last);
        first += stride;
    }
}

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride)
{
    reverse(first, middle, stride);
    reverse(middle, last, stride);
    reverse(first, last, stride);
}

template void rotate<Complex16 *>(Complex16 *first, Complex16 *middle, Complex16 *last, bitCapInt stride);

/// Swap values of two bits in register
void QEngineCPU::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 != qubit2) {
        Swap(qubit1, qubit2, 1);
    }
}

void QEngineCPU::ApplySingleBit(bitLenInt qubit, const Complex16* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QEngineCPU::ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngineCPU::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(0, qPowers[2], mtrx, 2, qPowersSorted, doCalcNorm);
}

} // namespace Qrack
