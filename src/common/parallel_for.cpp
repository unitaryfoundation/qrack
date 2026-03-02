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

#define DECLARE_ATOMIC_BITCAPINT() std::atomic<bitCapIntOcl> idx;
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Qrack {

ParallelFor::ParallelFor()
#if ENABLE_ENV_VARS
    : pStride(getenv("QRACK_PSTRIDEPOW") ? pow2Ocl((bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))))
                                         : pow2Ocl((bitLenInt)PSTRIDEPOW))
#else
    : pStride(pow2Ocl((bitLenInt)PSTRIDEPOW))
#endif
#if ENABLE_PTHREAD
    , numCores(std::thread::hardware_concurrency())
#else
    , numCores(1U)
#endif
{
    const bitLenInt pStridePow = log2Ocl(pStride);
    const bitLenInt minStridePow = (numCores > 1U) ? (bitLenInt)pow2Ocl(log2Ocl(numCores - 1U)) : 0U;
    dispatchThreshold = (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
}

void ParallelFor::par_for(const bitCapIntOcl begin, const bitCapIntOcl end, ParallelFunc fn)
{
    par_for_inc(
        begin, end - begin, [](const bitCapIntOcl& i) { return i; }, fn);
}

void ParallelFor::par_for_set(const std::set<bitCapIntOcl>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0U, sparseSet.size(),
        [&sparseSet](const bitCapIntOcl& i) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_set(const std::vector<bitCapIntOcl>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0U, sparseSet.size(),
        [&sparseSet](const bitCapIntOcl& i) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_sparse_compose(const std::vector<bitCapIntOcl>& lowSet,
    const std::vector<bitCapIntOcl>& highSet, const bitLenInt& highStart, ParallelFunc fn)
{
    const bitCapIntOcl lowSize = lowSet.size();
    par_for_inc(
        0U, lowSize * highSet.size(),
        [&lowSize, &highStart, &lowSet, &highSet](const bitCapIntOcl& i) {
            const bitCapIntOcl lowPerm = i % lowSize;
            const bitCapIntOcl highPerm = (i - lowPerm) / lowSize;
            auto it = lowSet.begin();
            std::advance(it, lowPerm);
            bitCapIntOcl perm = *it;
            it = highSet.begin();
            std::advance(it, highPerm);
            perm |= (*it) << highStart;
            return perm;
        },
        fn);
}

void ParallelFor::par_for_skip(const bitCapIntOcl begin, const bitCapIntOcl end, const bitCapIntOcl skipMask,
    const bitLenInt maskWidth, ParallelFunc fn)
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

    const bitCapIntOcl lowMask = skipMask - 1U;
    const bitCapIntOcl highMask = ~lowMask;

    IncrementFunc incFn;
    if (!lowMask) {
        // If we're skipping leading bits, this is much cheaper:
        incFn = [maskWidth](const bitCapIntOcl& i) { return (i << maskWidth); };
    } else {
        incFn = [lowMask, highMask, maskWidth](
                    const bitCapIntOcl& i) { return ((i & lowMask) | ((i & highMask) << maskWidth)); };
    }

    par_for_inc(begin, (end - begin) >> maskWidth, incFn, fn);
}

void ParallelFor::par_for_mask(
    const bitCapIntOcl begin, const bitCapIntOcl end, const std::vector<bitCapIntOcl>& maskArray, ParallelFunc fn)
{
    const bitLenInt maskLen = maskArray.size();
    /* Pre-calculate the masks to simplify the increment function later. */
    std::unique_ptr<bitCapIntOcl[][2]> masks(new bitCapIntOcl[maskLen][2]);

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
        incFn = [&masks, maskLen](const bitCapIntOcl& iConst) {
            /* Push i apart, one mask at a time. */
            bitCapIntOcl i = iConst;
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
void ParallelFor::par_for_inc(
    const bitCapIntOcl begin, const bitCapIntOcl itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const bitCapIntOcl Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }

    if (threads <= 1U) {
        const bitCapIntOcl maxLcv = begin + itemCount;
        for (bitCapIntOcl j = begin; j < maxLcv; ++j) {
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
                bitCapIntOcl i;
                ATOMIC_INC();
                const bitCapIntOcl l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const bitCapIntOcl maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (bitCapIntOcl j = 0U; j < maxJ; ++j) {
                    fn(inc(begin + j + l), cpu);
                }
            }
        }));
    }

    for (std::future<void>& future : futures) {
        future.get();
    }
}

real1_f ParallelFor::par_norm(const bitCapIntOcl itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    const bitCapIntOcl Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        const real1 nrm_thresh = (real1)norm_thresh;
        for (bitCapIntOcl j = 0U; j < itemCount; ++j) {
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
                bitCapIntOcl i;
                ATOMIC_INC();
                const bitCapIntOcl l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const bitCapIntOcl maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (bitCapIntOcl j = 0U; j < maxJ; ++j) {
                    bitCapIntOcl k = i * Stride + j;
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

real1_f ParallelFor::par_norm_exact(const bitCapIntOcl itemCount, const StateVectorPtr stateArray)
{
    const bitCapIntOcl Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        for (bitCapIntOcl j = 0U; j < itemCount; ++j) {
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
                bitCapIntOcl i;
                ATOMIC_INC();
                const bitCapIntOcl l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const bitCapIntOcl maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (bitCapIntOcl j = 0U; j < maxJ; ++j) {
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
void ParallelFor::par_for_inc(
    const bitCapIntOcl begin, const bitCapIntOcl itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const bitCapIntOcl maxLcv = begin + itemCount;
    for (bitCapIntOcl j = begin; j < maxLcv; ++j) {
        fn(inc(j), 0U);
    }
}

real1_f ParallelFor::par_norm(const bitCapIntOcl itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    real1_f nrmSqr = ZERO_R1;
    for (bitCapIntOcl j = 0U; j < itemCount; ++j) {
        const real1_f nrm = norm(stateArray->read(j));
        if (nrm >= norm_thresh) {
            nrmSqr += nrm;
        }
    }

    return nrmSqr;
}

real1_f ParallelFor::par_norm_exact(const bitCapIntOcl itemCount, const StateVectorPtr stateArray)
{
    real1_f nrmSqr = ZERO_R1;
    for (bitCapIntOcl j = 0U; j < itemCount; ++j) {
        nrmSqr += norm(stateArray->read(j));
    }

    return nrmSqr;
}
#endif
} // namespace Qrack
