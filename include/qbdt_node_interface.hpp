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

#include <mutex>

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
    static void _par_for_qbdt(const bitCapInt end, BdtFunc fn);

public:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2 mtrxCol1, const complex2 mtrxCol2, QBdtNodeInterfacePtr& b0,
        QBdtNodeInterfacePtr& b1, bitLenInt depth, bitLenInt parDepth = 1U)
#else
    virtual void PushStateVector(complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::PushStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    complex scale;
    QBdtNodeInterfacePtr branches[2U];
    std::mutex mtx;

    QBdtNodeInterface()
        : scale(ONE_CMPLX)
    {
        branches[0U] = NULL;
        branches[1U] = NULL;
    }

    QBdtNodeInterface(complex scl)
        : scale(scl)
    {
        branches[0U] = NULL;
        branches[1U] = NULL;
    }

    QBdtNodeInterface(complex scl, QBdtNodeInterfacePtr* b)
        : scale(scl)
    {
        branches[0U] = b[0U];
        branches[1U] = b[1U];
    }

    virtual ~QBdtNodeInterface()
    {
        // Virtual destructor for inheritance
    }

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size)
    {
        throw std::out_of_range("QBdtNodeInterface::InsertAtDepth() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size);

    virtual void SetZero()
    {
        scale = ZERO_CMPLX;
        branches[0U] = NULL;
        branches[1U] = NULL;
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual bool isEqualUnder(QBdtNodeInterfacePtr r);

    virtual QBdtNodeInterfacePtr ShallowClone()
    {
        throw std::out_of_range("QBdtNodeInterface::ShallowClone() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::PopStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Branch(bitLenInt depth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::Branch() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Prune(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::Prune() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Normalize(bitLenInt depth)
    {
        throw std::out_of_range("QBdtNodeInterface::Normalize() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, bitLenInt depth)
#else
    virtual void Apply2x2(complex const* mtrx, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::Apply2x2() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_COMPLEX_X2
    virtual void PushSpecial(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b1)
#else
    virtual void PushSpecial(complex const* mtrx, QBdtNodeInterfacePtr& b1)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::PushSpecial() not implemented! (You probably called "
                                "PushStateVector() past terminal depth.)");
    }
};

bool operator==(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
bool operator!=(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs);
} // namespace Qrack
