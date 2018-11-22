//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This header defines buffers for Qrack::QFusion.
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <future>

#include "bitbuffer.hpp"

namespace Qrack {

BitBuffer::BitBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, bool isArith)
    : anti(antiCtrl)
    , isArithmetic(isArith)
    , controls(cntrlLen)
{
    if (cntrlLen > 0) {
        std::copy(cntrls, cntrls + cntrlLen, controls.begin());
        std::sort(controls.begin(), controls.end());
    }
}

bool BitBuffer::Combinable(BitBufferPtr toCmp)
{
    if (toCmp == NULL) {
        // If a bit buffer is empty, it's fine to overwrite it.
        return true;
    }

    // Otherwise, we return "false" if we need to flush, and true if we can keep buffering.

    if (anti != toCmp->anti) {
        return false;
    }

    if (isArithmetic != toCmp->isArithmetic) {
        return false;
    }

    if (controls.size() != toCmp->controls.size()) {
        return false;
    }

    for (bitLenInt i = 0; i < controls.size(); i++) {
        if (controls[i] != toCmp->controls[i]) {
            return false;
        }
    }

    return true;
}

GateBuffer::GateBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, const complex* mtrx)
    : BitBuffer(antiCtrl, cntrls, cntrlLen, false)
    , matrix(new complex[4], std::default_delete<complex[]>())
{
    std::copy(mtrx, mtrx + 4, matrix.get());
}

BitBufferPtr GateBuffer::LeftRightCompose(BitBufferPtr rightBuffer)
{
    // If we pass the threshold number of qubits for buffering, we just do 2x2 complex matrix multiplication.
    // We parallelize this, since we can.
    // If a matrix component is very close to zero, we assume it's floating-point-error on a composition that has an
    // exactly 0 component, number theoretically. (If it's not exactly 0 by number theory, it's numerically
    // negligible, and we're safe.)

    BitOp outBuffer(new complex[4], std::default_delete<complex[]>());

    if (rightBuffer != NULL) {
        GateBuffer* rightGate = dynamic_cast<GateBuffer*>(rightBuffer.get());
        BitOp right = rightGate->matrix;

        std::vector<std::future<void>> futures(4);

        futures[0] = std::async(std::launch::async, [&]() {
            outBuffer.get()[0] = (matrix.get()[0] * right.get()[0]) + (matrix.get()[1] * right.get()[2]);
            if (norm(outBuffer.get()[0]) < min_norm) {
                outBuffer.get()[0] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[1] = std::async(std::launch::async, [&]() {
            outBuffer.get()[1] = (matrix.get()[0] * right.get()[1]) + (matrix.get()[1] * right.get()[3]);
            if (norm(outBuffer.get()[1]) < min_norm) {
                outBuffer.get()[1] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[2] = std::async(std::launch::async, [&]() {
            outBuffer.get()[2] = (matrix.get()[2] * right.get()[0]) + (matrix.get()[3] * right.get()[2]);
            if (norm(outBuffer.get()[2]) < min_norm) {
                outBuffer.get()[2] = complex(ZERO_R1, ZERO_R1);
            }
        });
        futures[3] = std::async(std::launch::async, [&]() {
            outBuffer.get()[3] = (matrix.get()[2] * right.get()[1]) + (matrix.get()[3] * right.get()[3]);
            if (norm(outBuffer.get()[3]) < min_norm) {
                outBuffer.get()[3] = complex(ZERO_R1, ZERO_R1);
            }
        });

        for (int i = 0; i < 4; i++) {
            futures[i].get();
        }
    } else {
        std::copy(matrix.get(), matrix.get() + 4, outBuffer.get());
    }

    return std::make_shared<GateBuffer>(this, outBuffer);
}

void GateBuffer::Apply(QInterfacePtr qReg, const bitLenInt& qubitIndex, std::vector<BitBufferPtr>* bitBuffers)
{
    if (controls.size() == 0) {
        qReg->ApplySingleBit(matrix.get(), true, qubitIndex);
    } else {
        bitLenInt* ctrls = new bitLenInt[controls.size()];
        std::copy(controls.begin(), controls.end(), ctrls);

        if (anti) {
            qReg->ApplyAntiControlledSingleBit(ctrls, controls.size(), qubitIndex, matrix.get());
        } else {
            qReg->ApplyControlledSingleBit(ctrls, controls.size(), qubitIndex, matrix.get());
        }

        delete[] ctrls;
    }
    (*bitBuffers)[qubitIndex] = NULL;
}

bool ArithmeticBuffer::Combinable(BitBufferPtr toCmp)
{
    if (toCmp == NULL) {
        return true;
    }

    if (!BitBuffer::Combinable(toCmp)) {
        return false;
    }

    // BitBuffer::Combinable requires either both or neither of the buyers to be arithmetic, which makes this cast safe
    ArithmeticBuffer* toCmpArith = dynamic_cast<ArithmeticBuffer*>(toCmp.get());
    if (start != toCmpArith->start) {
        return false;
    }

    if (length != toCmpArith->length) {
        return false;
    }

    return true;
}

void ArithmeticBuffer::Apply(QInterfacePtr qReg, const bitLenInt& qubitIndex, std::vector<BitBufferPtr>* bitBuffers)
{
    if (controls.size() == 0) {
        if (toAdd > 0) {
            qReg->INC(toAdd, start, length);
        } else if (toAdd < 0) {
            qReg->DEC(-toAdd, start, length);
        }
    } else {
        bitLenInt* ctrls = new bitLenInt[controls.size()];
        std::copy(controls.begin(), controls.end(), ctrls);

        if (toAdd > 0) {
            qReg->CINC(toAdd, start, length, ctrls, controls.size());
        } else if (toAdd < 0) {
            qReg->CDEC(-toAdd, start, length, ctrls, controls.size());
        }

        delete[] ctrls;
    }

    for (bitLenInt i = 0; i < length; i++) {
        (*bitBuffers)[start + i] = NULL;
    }
}

BitBufferPtr ArithmeticBuffer::LeftRightCompose(BitBufferPtr rightBuffer)
{
    if (rightBuffer) {
        ArithmeticBuffer* aBfr = dynamic_cast<ArithmeticBuffer*>(rightBuffer.get());
        return std::make_shared<ArithmeticBuffer>(this, aBfr->toAdd);
    } else {
        return std::make_shared<ArithmeticBuffer>(this, 0);
    }
}
} // namespace Qrack
