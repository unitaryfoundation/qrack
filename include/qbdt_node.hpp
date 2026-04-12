//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
        bitLenInt parDepth = 1U);
#else
    virtual void PushStateVector(const complex mtrx[4U], QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U);
#endif
#else
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth);
#else
    virtual void PushStateVector(
        const complex mtrx[4U], QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth);
#endif
#endif

public:
    QBdtNode()
        : QBdtNodeInterface()
    {
        // Intentionally left blank
    }

    QBdtNode(const complex& scl)
        : QBdtNodeInterface(scl)
    {
        // Intentionally left blank
    }

    QBdtNode(const complex& scl, QBdtNodeInterfacePtr* b)
        : QBdtNodeInterface(scl, b)
    {
        // Intentionally left blank
    }

    virtual ~QBdtNode()
    {
        // Virtual destructor for inheritance
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtNode>(scale, branches); }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);
#else
    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size);
#endif

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U);
#else
    virtual void PopStateVector(bitLenInt depth = 1U);
#endif

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDeth = 1U);
#else
    virtual void Branch(bitLenInt depth = 1U);
#endif

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void Prune(bitLenInt depth = 1U, bitLenInt parDepth = 1U);
#else
    virtual void Prune(bitLenInt depth = 1U);
#endif

    virtual void Normalize(bitLenInt depth = 1U);

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, bitLenInt depth);
#else
    virtual void Apply2x2(const complex* mtrx, bitLenInt depth);
#endif
};

} // namespace Qrack
