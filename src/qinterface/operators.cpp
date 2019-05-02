//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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

// Logic Gates:

/// "AND" compare two bits in QInterface, and store result in outputBit
void QInterface::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, 0);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit, length);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

/// "OR" compare two bits in QInterface, and store result in outputBit
void QInterface::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, (1 << length) - 1);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit, length);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

/// "XOR" compare two bits in QInterface, and store result in outputBit
void QInterface::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetReg(outputBit, length, 0);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit, length);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit, length);
    } else {
        SetReg(outputBit, length, 0);
        CNOT(inputBit1, outputBit, length);
        CNOT(inputBit2, outputBit, length);
    }
}

void QInterface::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, false);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

void QInterface::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, true);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

void QInterface::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetBit(outputBit, false);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit);
    } else {
        SetBit(outputBit, false);
        CNOT(inputBit1, outputBit);
        CNOT(inputBit2, outputBit);
    }
}

void QInterface::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    SetBit(outputBit, false);
    if (inputClassicalBit && (inputQBit != outputBit)) {
        CNOT(inputQBit, outputBit);
    }
}

void QInterface::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputClassicalBit) {
        SetBit(outputBit, true);
    } else if (inputQBit != outputBit) {
        SetBit(outputBit, false);
        CNOT(inputQBit, outputBit);
    }
}

void QInterface::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputQBit != outputBit) {
        SetBit(outputBit, inputClassicalBit);
        CNOT(inputQBit, outputBit);
    } else if (inputClassicalBit) {
        X(outputBit);
    }
}

void QInterface::TimeEvolve(Hamiltonian h, real1 timeDiff)
{
    // Exponentiation of an arbitrary serial string of gates, each HamiltonianOp component times timeDiff, e^(-i * H *
    // t) as e^(-i * H_(N - 1) * t) * e^(-i * H_(N - 2) * t) * ... e^(-i * H_0 * t)

    for (bitLenInt i = 0; i < h.size(); i++) {
        HamiltonianOpPtr op = h[i];
        complex* opMtrx = op->matrix.get();
        complex* mtrx;
        
        bitCapInt maxJ = 4;
        if (op->uniform) {
            maxJ *= 1U<<op->controlLen;
        }
        mtrx = new complex[maxJ];

        for (bitCapInt j = 0; j < maxJ; j++) {
            mtrx[j] = opMtrx[j] * (-timeDiff);
        }

        if (op->toggles) {
            for (bitLenInt j = 0; j < op->controlLen; j++) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        if (op->uniform) {
            complex* expMtrx = new complex[maxJ];
            for (bitCapInt j = 0; j < (1U<<op->controlLen); j++) {
                exp2x2(mtrx + (j * 4), expMtrx + (j * 4));
            }
            UniformlyControlledSingleBit(op->controls, op->controlLen, op->targetBit, expMtrx);
            delete[] expMtrx; 
        } else {
            Exp(op->controls, op->controlLen, op->targetBit, mtrx, op->anti);
        }

        if (op->toggles) {
            for (bitLenInt j = 0; j < op->controlLen; j++) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        delete[] mtrx;
    }
}
} // namespace Qrack
