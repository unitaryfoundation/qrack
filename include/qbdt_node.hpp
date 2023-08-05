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

#include "qbdt_node_interface.hpp"

namespace Qrack {

class QBdtNode;
typedef std::shared_ptr<QBdtNode> QBdtNodePtr;

class QBdtNode : public QBdtNodeInterface {
protected:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
        bitLenInt parDepth = 1U);
#else
    virtual void PushStateVector(complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U);
#endif

public:
    QBdtNode()
        : QBdtNodeInterface()
    {
        // Intentionally left blank
    }

    QBdtNode(complex scl)
        : QBdtNodeInterface(scl)
    {
        // Intentionally left blank
    }

    QBdtNode(complex scl, QBdtNodeInterfacePtr* b)
        : QBdtNodeInterface(scl, b)
    {
        // Intentionally left blank
    }

    virtual ~QBdtNode()
    {
        // Virtual destructor for inheritance
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtNode>(scale, branches); }

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U);

    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDeth = 1U);

    virtual void Prune(bitLenInt depth = 1U, bitLenInt parDepth = 1U);

    virtual void Normalize(bitLenInt depth = 1U);

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, bitLenInt depth);
#else
    virtual void Apply2x2(complex const* mtrx, bitLenInt depth);
#endif

    virtual QBdtNodeInterfacePtr PopSpecial(bitLenInt depth = 1U)
    {
        if (!depth) {
            return shared_from_this();
        }

        if (norm(scale) <= _qrack_qbdt_sep_thresh) {
            SetZero();
            return shared_from_this();
        }

        --depth;

        branches[0U] = branches[0U]->PopSpecial(depth);
        branches[1U] = branches[1U]->PopSpecial(depth);

        return shared_from_this();
    }
};

} // namespace Qrack
