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

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

namespace Qrack {

class QBdtNodeInterface;
typedef std::shared_ptr<QBdtNodeInterface> QBdtNodeInterfacePtr;

class QBdtNodeInterface {
protected:
    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }
    static void _par_for_qbdt(const bitCapInt begin, const bitCapInt end, BdtFunc fn);
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b0,
        QBdtNodeInterfacePtr& b1, bitLenInt depth) = 0;
#else
    virtual void PushStateVector(
        const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth) = 0;
#endif

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

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size) = 0;

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size);

    virtual void SetZero()
    {
        scale = ZERO_CMPLX;
        branches[0] = NULL;
        branches[1] = NULL;
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual QBdtNodeInterfacePtr ShallowClone() = 0;

    virtual void PopStateVector(bitLenInt depth = 1U) = 0;

    virtual void Branch(bitLenInt depth = 1U) = 0;

    virtual void Prune(bitLenInt depth = 1U) = 0;

    virtual void Normalize(bitLenInt depth) = 0;

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, bitLenInt depth) = 0;
#else
    virtual void Apply2x2(const complex* mtrx, bitLenInt depth) = 0;
#endif
};

bool operator==(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
bool operator!=(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
QBdtNodeInterfacePtr operator-(const QBdtNodeInterfacePtr& t);
} // namespace Qrack
