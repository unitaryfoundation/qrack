//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "common/parallel_for.hpp"
#include "statevector.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#if ENABLE_PTHREAD
#include <atomic>
#include <future>

#define DECLARE_ATOMIC_BITCAPINT() std::atomic<size_t> idx;
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Qrack {

ParallelFor::ParallelFor()
#if ENABLE_ENV_VARS
    : pStride(getenv("QRACK_PSTRIDEPOW") ? pow2Cpu((bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))))
                                         : pow2Cpu((bitLenInt)PSTRIDEPOW))
#else
    : pStride(pow2Cpu((bitLenInt)PSTRIDEPOW))
#endif
#if ENABLE_PTHREAD
    , numCores(std::thread::hardware_concurrency())
#else
    , numCores(1U)
#endif
{
    const bitLenInt pStridePow = log2Ocl(pStride);
    const bitLenInt minStridePow = (numCores > 1U) ? (bitLenInt)pow2Cpu(log2Ocl(numCores - 1U)) : 0U;
    dispatchThreshold = (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
}

void ParallelFor::par_for(const size_t begin, const size_t end, ParallelFunc fn)
{
    par_for_inc(begin, end - begin, [](const size_t& i) { return i; }, fn);
}

void ParallelFor::par_for_set(const std::set<bitCapInt>& sparseSet, ParallelFuncSparse fn)
{
    std::vector<bitCapInt> keys(sparseSet.begin(), sparseSet.end());
    par_for_set(keys, fn);
}

void ParallelFor::par_for_set(const std::vector<bitCapInt>& sparseSet, ParallelFuncSparse fn)
{
    par_for_inc_sparse(0U, sparseSet.size(), [&sparseSet](const size_t& i) { return sparseSet[i]; }, fn);
}

void ParallelFor::par_for_sparse_compose(const std::vector<bitCapInt>& lowSet, const std::vector<bitCapInt>& highSet,
    const bitLenInt& highStart, ParallelFuncSparse fn)
{
    const size_t lowSize = lowSet.size();
    par_for_inc_sparse(
        0U, lowSize * highSet.size(),
        [&lowSize, &highStart, &lowSet, &highSet](const size_t& i) {
            const size_t lowPerm = i % lowSize;
            const size_t highPerm = (i - lowPerm) / lowSize;
            auto it = lowSet.begin();
            std::advance(it, lowPerm);
            bitCapInt perm = *it;
            it = highSet.begin();
            std::advance(it, highPerm);
            perm = perm | ((*it) << highStart);
            return perm;
        },
        fn);
}

void ParallelFor::par_for_skip(
    const size_t begin, const size_t end, const size_t skipMask, const bitLenInt maskWidth, ParallelFunc fn)
{
    /*
     * Add maskWidth bits by shifting the incrementor up that number of
     * bits, filling with 0's.
     *
     * For example, if the skipMask is 0x8, then the lowMask will be 0x7
     * and the high mask will be ~(0x7 + 0x8) ==> ~0xf, shifted by the
     * number of extra bits to add.
     */

    if ((skipMask << maskWidth) >= end) {
        // If we're skipping trailing bits, this is much cheaper:
        return par_for(begin, skipMask, fn);
    }

    const size_t lowMask = skipMask - 1U;
    const size_t highMask = ~lowMask;

    IncrementFunc incFn;
    if (!lowMask) {
        // If we're skipping leading bits, this is much cheaper:
        incFn = [maskWidth](const size_t& i) { return (i << maskWidth); };
    } else {
        incFn = [lowMask, highMask, maskWidth](
                    const size_t& i) { return ((i & lowMask) | ((i & highMask) << maskWidth)); };
    }

    par_for_inc(begin, (end - begin) >> maskWidth, incFn, fn);
}

void ParallelFor::par_for_mask(
    const size_t begin, const size_t end, const std::vector<size_t>& maskArray, ParallelFunc fn)
{
    const bitLenInt maskLen = maskArray.size();
    /* Pre-calculate the masks to simplify the increment function later. */
    std::unique_ptr<size_t[][2]> masks(new size_t[maskLen][2]);

    bool onlyLow = true;
    for (bitLenInt i = 0; i < maskLen; ++i) {
        masks[i][0U] = maskArray[i] - 1U; // low mask
        masks[i][1U] = (~(masks[i][0U] + maskArray[i])); // high mask
        if (maskArray[maskLen - i - 1U] != (end >> (i + 1U))) {
            onlyLow = false;
        }
    }

    IncrementFunc incFn;
    if (onlyLow) {
        par_for(begin, end >> maskLen, fn);
    } else {
        incFn = [&masks, maskLen](const size_t& iConst) {
            /* Push i apart, one mask at a time. */
            size_t i = iConst;
            for (bitLenInt m = 0U; m < maskLen; ++m) {
                i = ((i << 1U) & masks[m][1U]) | (i & masks[m][0U]);
            }
            return i;
        };

        par_for_inc(begin, (end - begin) >> maskLen, incFn, fn);
    }
}

#if ENABLE_PTHREAD
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const size_t begin, const size_t itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const size_t Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }

    if (threads <= 1U) {
        const size_t maxLcv = begin + itemCount;
        for (size_t j = begin; j < maxLcv; ++j) {
            fn(inc(j), 0U);
        }

        return;
    }

    DECLARE_ATOMIC_BITCAPINT();
    idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(cpu, &idx, &begin, &itemCount, &Stride, inc, fn) {
            for (;;) {
                size_t i;
                ATOMIC_INC();
                const size_t l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const size_t maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (size_t j = 0U; j < maxJ; ++j) {
                    fn(inc(begin + j + l), cpu);
                }
            }
        }));
    }

    for (std::future<void>& future : futures) {
        future.get();
    }
}

void ParallelFor::par_for_inc_sparse(
    const size_t begin, const size_t itemCount, IncrementFuncSparse inc, ParallelFuncSparse fn)
{
    const size_t Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }

    if (threads <= 1U) {
        const bitCapInt maxLcv = begin + itemCount;
        for (size_t j = begin; j < maxLcv; ++j) {
            fn(inc(j), 0U);
        }

        return;
    }

    DECLARE_ATOMIC_BITCAPINT();
    idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(cpu, &idx, &begin, &itemCount, &Stride, inc, fn) {
            for (;;) {
                size_t i;
                ATOMIC_INC();
                const size_t l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const size_t maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (size_t j = 0U; j < maxJ; ++j) {
                    fn(inc(begin + j + l), cpu);
                }
            }
        }));
    }

    for (std::future<void>& future : futures) {
        future.get();
    }
}

real1_f ParallelFor::par_norm(const size_t itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    const size_t Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        const real1 nrm_thresh = (real1)norm_thresh;
        for (size_t j = 0U; j < itemCount; ++j) {
            const real1 nrm = norm(stateArray->read(j));
            if (nrm >= nrm_thresh) {
                nrmSqr += nrm;
            }
        }

        return (real1_f)nrmSqr;
    }

    DECLARE_ATOMIC_BITCAPINT();
    idx = 0U;
    std::vector<std::future<real1_f>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&idx, &itemCount, stateArray, &Stride, &norm_thresh) {
            const real1 nrm_thresh = (real1)norm_thresh;
            real1 sqrNorm = ZERO_R1;
            for (;;) {
                size_t i;
                ATOMIC_INC();
                const size_t l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const size_t maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (size_t j = 0U; j < maxJ; ++j) {
                    size_t k = i * Stride + j;
                    const real1 nrm = norm(stateArray->read(k));
                    if (nrm >= nrm_thresh) {
                        sqrNorm += nrm;
                    }
                }
            }
            return (real1_f)sqrNorm;
        }));
    }

    real1_f nrmSqr = ZERO_R1_F;
    for (std::future<real1_f>& future : futures) {
        nrmSqr += future.get();
    }

    return nrmSqr;
}

real1_f ParallelFor::par_norm_exact(const size_t itemCount, const StateVectorPtr stateArray)
{
    const size_t Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        for (size_t j = 0U; j < itemCount; ++j) {
            nrmSqr += norm(stateArray->read(j));
        }

        return (real1_f)nrmSqr;
    }

    DECLARE_ATOMIC_BITCAPINT();
    idx = 0U;
    std::vector<std::future<real1_f>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&idx, &itemCount, &Stride, stateArray) {
            real1 sqrNorm = ZERO_R1;
            for (;;) {
                size_t i;
                ATOMIC_INC();
                const size_t l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const size_t maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (size_t j = 0U; j < maxJ; ++j) {
                    sqrNorm += norm(stateArray->read(i * Stride + j));
                }
            }
            return (real1_f)sqrNorm;
        }));
    }

    real1_f nrmSqr = ZERO_R1_F;
    for (std::future<real1_f>& future : futures) {
        nrmSqr += future.get();
    }

    return nrmSqr;
}
#else
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const size_t begin, const size_t itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const size_t maxLcv = begin + itemCount;
    for (size_t j = begin; j < maxLcv; ++j) {
        fn(inc(j), 0U);
    }
}

void ParallelFor::par_for_inc_sparse(
    const size_t begin, const size_t itemCount, IncrementFuncSparse inc, ParallelFuncSparse fn)
{
    const bitCapInt maxLcv = begin + itemCount;
    for (size_t j = begin; j < maxLcv; ++j) {
        fn(inc(j), 0U);
    }
}

real1_f ParallelFor::par_norm(const size_t itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    real1_f nrmSqr = ZERO_R1;
    for (size_t j = 0U; j < itemCount; ++j) {
        const real1_f nrm = norm(stateArray->read(j));
        if (nrm >= norm_thresh) {
            nrmSqr += nrm;
        }
    }

    return nrmSqr;
}

real1_f ParallelFor::par_norm_exact(const size_t itemCount, const StateVectorPtr stateArray)
{
    real1_f nrmSqr = ZERO_R1;
    for (size_t j = 0U; j < itemCount; ++j) {
        nrmSqr += norm(stateArray->read(j));
    }

    return nrmSqr;
}
#endif
} // namespace Qrack
