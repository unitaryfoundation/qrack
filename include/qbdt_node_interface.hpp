//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qrack_types.hpp"

namespace Qrack {

class QBdtNodeInterface;
typedef std::shared_ptr<QBdtNodeInterface> QBdtNodeInterfacePtr;

class QBdtNodeInterface {
protected:
    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }
    static void _par_for_qbdt(const bitCapInt begin, const bitCapInt end, BdtFunc fn);
    virtual void PushStateVector(
        const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth) = 0;

public:
    complex scale;
    QBdtNodeInterfacePtr branches[2];

    QBdtNodeInterface()
        : scale(ONE_CMPLX)
    {
        branches[0] = NULL;
        branches[1] = NULL;
    }

    QBdtNodeInterface(complex scl)
        : scale(scl)
    {
        branches[0] = NULL;
        branches[1] = NULL;
    }

    QBdtNodeInterface(complex scl, QBdtNodeInterfacePtr* b)
        : scale(scl)
    {
        branches[0] = b[0];
        branches[1] = b[1];
    }

    virtual ~QBdtNodeInterface()
    {
        // Intentionally left blank
    }

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, bitLenInt size)
    {
        if (norm(scale) <= FP_NORM_EPSILON) {
            return;
        }

        if (depth) {
            depth--;
            if (branches[0]) {
                branches[0]->InsertAtDepth(b, depth, size);
                branches[1]->InsertAtDepth(b, depth, size);
            }

            return;
        }

        QBdtNodeInterfacePtr tempBranches[2] = { branches[0], branches[1] };
        branches[0] = b->branches[0];
        branches[1] = b->branches[1];

        if (!size || !tempBranches[0]) {
            return;
        }

        branches[0]->InsertAtDepth(tempBranches[0], size, 0);
        branches[1]->InsertAtDepth(tempBranches[1], size, 0);
    }

    virtual void SetZero()
    {
        scale = ZERO_CMPLX;
        branches[0] = NULL;
        branches[1] = NULL;
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r)
    {
        if (this == r.get()) {
            return true;
        }

        if (norm(scale - r->scale) > FP_NORM_EPSILON) {
            return false;
        }

        if (branches[0] != r->branches[0]) {
            return false;
        }

        branches[0] = r->branches[0];

        if (branches[1] != r->branches[1]) {
            return false;
        }

        branches[1] = r->branches[1];

        return true;
    }

    virtual QBdtNodeInterfacePtr ShallowClone() = 0;

    virtual void PopStateVector(bitLenInt depth = 1U) = 0;

    virtual void Branch(bitLenInt depth = 1U) = 0;

    virtual void Prune(bitLenInt depth = 1U) = 0;

    virtual void Normalize(bitLenInt depth) = 0;

    virtual void Apply2x2(const complex* mtrx, bitLenInt depth) = 0;
};

bool operator==(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
bool operator!=(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
QBdtNodeInterfacePtr operator-(const QBdtNodeInterfacePtr& t);
} // namespace Qrack
