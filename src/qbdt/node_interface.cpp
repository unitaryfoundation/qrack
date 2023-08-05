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

#include "qbdt_node_interface.hpp"

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#include <future>
#include <thread>
#endif

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_SAME_AMP(a, b) (abs((a) - (b)) <= REAL1_EPSILON)
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#endif

namespace Qrack {

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
const unsigned numThreads = std::thread::hardware_concurrency() << 1U;
#if ENABLE_ENV_VARS
const bitLenInt pStridePow =
    (((bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW)) +
        1U) >>
    1U;
#else
const bitLenInt pStridePow = (PSTRIDEPOW + 1U) >> 1U;
#endif
const bitCapInt pStride = pow2(pStridePow);
#endif

bool operator==(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs)
{
    if (!lhs) {
        return !rhs;
    }

    return lhs->isEqual(rhs);
}

bool operator!=(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs) { return !(lhs == rhs); }

bool QBdtNodeInterface::isEqual(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (!IS_SAME_AMP(scale, r->scale)) {
        return false;
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[0U].get() != r->branches[0U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[0U];
        QBdtNodeInterfacePtr rLeaf = r->branches[0U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[0U] = r->branches[0U];
    }

    if ((!branches[1U]) != (!r->branches[1U])) {
        return false;
    }

    if (branches[1U].get() != r->branches[1U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[1U];
        QBdtNodeInterfacePtr rLeaf = r->branches[1U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[1U] = r->branches[1U];
    }

    return true;
}

bool QBdtNodeInterface::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (IS_NODE_0(scale)) {
        return IS_NODE_0(r->scale);
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[0U].get() != r->branches[0U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[0U];
        QBdtNodeInterfacePtr rLeaf = r->branches[0U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> rLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[0U] = r->branches[0U];
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[1U].get() != r->branches[1U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[1U];
        QBdtNodeInterfacePtr rLeaf = r->branches[1U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[1U] = r->branches[1U];
    }

    return true;
}

QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(
    bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (IS_NODE_0(scale)) {
        return NULL;
    }

    Branch();

    if (depth) {
        --depth;

        QBdtNodeInterfacePtr toRet1, toRet2;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
            ++parDepth;
            std::future<QBdtNodeInterfacePtr> future0 = std::async(
                std::launch::async, [&] { return branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth); });
            toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
            toRet1 = future0.get();
        } else {
            toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
            toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
        }
#else
        toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
        toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
#endif

        return (norm(branches[0U]->scale) > norm(branches[1U]->scale)) ? toRet1 : toRet2;
    }

    QBdtNodeInterfacePtr toRet = ShallowClone();
    toRet->scale /= abs(toRet->scale);

    if (!size) {
        branches[0U] = NULL;
        branches[1U] = NULL;

        return toRet;
    }

    QBdtNodeInterfacePtr temp = toRet->RemoveSeparableAtDepth(size, 0);
    branches[0U] = temp->branches[0U];
    branches[1U] = temp->branches[1U];

    return toRet;
}

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
void QBdtNodeInterface::_par_for_qbdt(const bitCapInt end, BdtFunc fn)
{
    const bitCapInt Stride = pStride;
    unsigned threads = (unsigned)(end / pStride);
    if (threads > numThreads) {
        threads = numThreads;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            j |= fn(j);
        }
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&myMutex, &idx, &end, &Stride, fn) {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                bitCapInt j;
                for (j = 0U; j < maxJ; ++j) {
                    bitCapInt k = j + l;
                    k |= fn(k);
                    j = k - l;
                    if (j >= maxJ) {
                        std::lock_guard<std::mutex> lock(myMutex);
                        idx |= j / Stride;
                        break;
                    }
                }
            }
        }));
    }

    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu].get();
    }
}
#else
void QBdtNodeInterface::_par_for_qbdt(const bitCapInt end, BdtFunc fn)
{
    for (bitCapInt j = 0U; j < end; ++j) {
        j |= fn(j);
    }
}
#endif
} // namespace Qrack
