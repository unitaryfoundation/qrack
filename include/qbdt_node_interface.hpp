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

#include "common/qrack_functions.hpp"

#include <functional>
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

typedef std::function<bitCapInt(const bitCapInt&)> BdtFunc;

class QBdtNodeInterface {
protected:
    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)(bi_and_1(perm >> bit)); }
    static void _par_for_qbdt(const bitCapInt& end, BdtFunc fn);

public:
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
        bitLenInt parDepth = 1U)
#else
    virtual void PushStateVector(const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U)
#endif
#else
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
#else
    virtual void PushStateVector(
        const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
#endif
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::PushStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    complex scale;
    QBdtNodeInterfacePtr branches[2U];
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::mutex mtx;
#endif

    QBdtNodeInterface()
        : scale(ONE_CMPLX)
        , branches{ nullptr, nullptr }
    {
        // Intentionally left blank
    }

    QBdtNodeInterface(const complex& scl)
        : scale(scl)
        , branches{ nullptr, nullptr }
    {
        // Intentionally left blank
    }

    QBdtNodeInterface(const complex& scl, QBdtNodeInterfacePtr* b)
        : scale(scl)
        , branches{ b[0U], b[1U] }
    {
        // Intentionally left blank
    }

    virtual ~QBdtNodeInterface()
    {
        // Virtual destructor for inheritance
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U)
#else
    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::InsertAtDepth() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(
        bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);
#else
    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size);
#endif

    virtual void SetZero()
    {
        scale = ZERO_CMPLX;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (branches[0U]) {
            QBdtNodeInterfacePtr b0 = branches[0U];
            std::lock_guard<std::mutex> lock(b0->mtx);
            branches[0U] = nullptr;
        }

        if (branches[1U]) {
            QBdtNodeInterfacePtr b1 = branches[1U];
            std::lock_guard<std::mutex> lock(b1->mtx);
            branches[1U] = nullptr;
        }
#else
        branches[0U] = nullptr;
        branches[1U] = nullptr;
#endif
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual bool isEqualUnder(QBdtNodeInterfacePtr r);

    virtual bool isEqualBranch(QBdtNodeInterfacePtr r, const bool& b);

    virtual QBdtNodeInterfacePtr ShallowClone()
    {
        throw std::out_of_range("QBdtNodeInterface::ShallowClone() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
#else
    virtual void PopStateVector(bitLenInt depth = 1U)
#endif
    {
        if (!depth) {
            return;
        }

        throw std::out_of_range("QBdtNodeInterface::PopStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
#else
    virtual void Branch(bitLenInt depth = 1U)
#endif
    {
        if (!depth) {
            return;
        }

        throw std::out_of_range("QBdtNodeInterface::Branch() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    virtual void Prune(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
#else
    virtual void Prune(bitLenInt depth = 1U)
#endif
    {
        if (!depth) {
            return;
        }

        throw std::out_of_range("QBdtNodeInterface::Prune() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Normalize(bitLenInt depth = 1U)
    {
        if (!depth) {
            return;
        }

        throw std::out_of_range("QBdtNodeInterface::Normalize() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, bitLenInt depth)
#else
    virtual void Apply2x2(const complex* mtrx, bitLenInt depth)
#endif
    {
        if (!depth) {
            return;
        }

        throw std::out_of_range("QBdtNodeInterface::Apply2x2() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

#if ENABLE_COMPLEX_X2
    virtual void PushSpecial(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b1)
#else
    virtual void PushSpecial(const complex* mtrx, QBdtNodeInterfacePtr& b1)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::PushSpecial() not implemented! (You probably called "
                                "PushStateVector() past terminal depth.)");
    }
};

bool operator==(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs);
bool operator!=(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs);
} // namespace Qrack
