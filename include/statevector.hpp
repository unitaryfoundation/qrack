//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This header defines buffers for Qrack::QFusion.
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/parallel_for.hpp"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <random>

#ifdef ENABLE_PTHREAD
#include <future>
#endif

#include <unordered_map>
#define SparseStateVecMap std::unordered_map<bitCapIntOcl, complex>

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

namespace Qrack {

class StateVectorArray;
class StateVectorSparse;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
class StateVector : public ParallelFor {
protected:
    bitCapIntOcl capacity;

public:
    bool isReadLocked;

    StateVector(bitCapIntOcl cap)
        : capacity(cap)
        , isReadLocked(true)
    {
    }
    virtual ~StateVector()
    {
        // Intentionally left blank.
    }

    virtual complex read(const bitCapIntOcl& i) = 0;
#if ENABLE_COMPLEX_X2
    virtual complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2) = 0;
#endif
    virtual void write(const bitCapIntOcl& i, const complex& c) = 0;
    /// Optimized "write" that is only guaranteed to write if either amplitude is nonzero. (Useful for the result of 2x2
    /// tensor slicing.)
    virtual void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2) = 0;
    virtual void clear() = 0;
    virtual void copy_in(const complex* inArray) = 0;
    virtual void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy_in(StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset,
        const bitCapIntOcl length) = 0;
    virtual void copy_out(complex* outArray) = 0;
    virtual void copy_out(complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy(StateVectorPtr toCopy) = 0;
    virtual void shuffle(StateVectorPtr svp) = 0;
    virtual void get_probs(real1* outArray) = 0;
    virtual bool is_sparse() = 0;
};

class StateVectorArray : public StateVector {
public:
    std::unique_ptr<complex[], void (*)(complex*)> amplitudes;

protected:
#if defined(__APPLE__)
    complex* _aligned_state_vec_alloc(bitCapIntOcl allocSize)
    {
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
        return (complex*)toRet;
    }
#endif

    std::unique_ptr<complex[], void (*)(complex*)> Alloc(bitCapIntOcl elemCount)
    {
#if defined(__ANDROID__)
        return std::unique_ptr<complex[], void (*)(complex*)>(new complex[elemCount], [](complex* c) { delete c; });
#else
        // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
        size_t allocSize = sizeof(complex) * elemCount;
        if (allocSize < QRACK_ALIGN_SIZE) {
            allocSize = QRACK_ALIGN_SIZE;
        }
#if defined(__APPLE__)
        return std::unique_ptr<complex[], void (*)(complex*)>(
            _aligned_state_vec_alloc(allocSize), [](complex* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
        return std::unique_ptr<complex[], void (*)(complex*)>(
            (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE), [](complex* c) { _aligned_free(c); });
#else
        return std::unique_ptr<complex[], void (*)(complex*)>(
            (complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize), [](complex* c) { free(c); });
#endif
#endif
    }

    virtual void Free() { amplitudes = nullptr; }

public:
    StateVectorArray(bitCapIntOcl cap)
        : StateVector(cap)
        , amplitudes(Alloc(capacity))
    {
        // Intentionally left blank.
    }

    virtual ~StateVectorArray() { Free(); }

    complex read(const bitCapIntOcl& i) { return amplitudes.get()[i]; };

#if ENABLE_COMPLEX_X2
    complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2)
    {
        return complex2(amplitudes.get()[i1], amplitudes.get()[i2]);
    }
#endif

    void write(const bitCapIntOcl& i, const complex& c) { amplitudes.get()[i] = c; };

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        amplitudes.get()[i1] = c1;
        amplitudes.get()[i2] = c2;
    };

    void clear()
    {
        par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = ZERO_CMPLX; });
    }

    void copy_in(const complex* copyIn)
    {
        if (copyIn) {
            par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = copyIn[lcv]; });
        } else {
            par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = ZERO_CMPLX; });
        }
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (copyIn) {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + offset] = copyIn[lcv]; });
        } else {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + offset] = ZERO_CMPLX; });
        }
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        if (copyInSv) {
            const complex* copyIn = std::dynamic_pointer_cast<StateVectorArray>(copyInSv)->amplitudes.get() + srcOffset;
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + dstOffset] = copyIn[lcv]; });
        } else {
            par_for(0, length,
                [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv + dstOffset] = ZERO_CMPLX; });
        }
    }

    void copy_out(complex* copyOut)
    {
        par_for(0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { copyOut[lcv] = amplitudes[lcv]; });
    }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        par_for(
            0, length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { copyOut[lcv] = amplitudes[lcv + offset]; });
    }

    void copy(StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorArray>(toCopy)); }

    void copy(StateVectorArrayPtr toCopy)
    {
        par_for(0, capacity,
            [&](const bitCapIntOcl& lcv, const unsigned& cpu) { amplitudes[lcv] = toCopy->amplitudes[lcv]; });
    }

    void shuffle(StateVectorPtr svp) { shuffle(std::dynamic_pointer_cast<StateVectorArray>(svp)); }

    void shuffle(StateVectorArrayPtr svp)
    {
        const bitCapIntOcl offset = capacity >> 1U;
        par_for(0, offset, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const complex tmp = amplitudes[lcv + offset];
            amplitudes[lcv + offset] = svp->amplitudes[lcv];
            svp->amplitudes[lcv] = tmp;
        });
    }

    void get_probs(real1* outArray)
    {
        par_for(
            0, capacity, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { outArray[lcv] = norm(amplitudes[lcv]); });
    }

    bool is_sparse() { return false; }
};

class StateVectorSparse : public StateVector {
protected:
    SparseStateVecMap amplitudes;
    std::mutex mtx;

    complex readUnlocked(const bitCapIntOcl& i)
    {
        auto it = amplitudes.find(i);
        return (it == amplitudes.end()) ? ZERO_CMPLX : it->second;
    }

    complex readLocked(const bitCapIntOcl& i)
    {
        std::lock_guard<std::mutex> lock(mtx);
        return readUnlocked(i);
    }

public:
    StateVectorSparse(bitCapIntOcl cap)
        : StateVector(cap)
        , amplitudes()
    {
    }

    size_t size() { return amplitudes.size(); }

    real1_f truncate_to_size(size_t maxAmps)
    {
        if (amplitudes.size() <= maxAmps) {
            return ONE_R1_F;
        }

        std::set<real1> nrms;
        for (const auto& pair : amplitudes) {
            real1 nrm = norm(pair.second);
            if (nrms.size() < maxAmps) {
                nrms.insert(nrm);
            } else if (*(nrms.begin()) < nrm) {
                nrms.erase(nrms.begin());
                nrms.insert(nrm);
            }
        }

        const real1 limit = *(nrms.begin());
        const real1 fidelity = std::accumulate(nrms.begin(), nrms.end(), ZERO_R1);
        const real1 invFidelity = ONE_R1 / fidelity;
        nrms.clear();

        SparseStateVecMap nAmplitudes;
        for (auto it = amplitudes.begin(); it != amplitudes.end(); ++it) {
            if (norm(it->second) >= limit) {
                nAmplitudes[it->first] = it->second;
            }
        }
        amplitudes.clear();
        amplitudes = nAmplitudes;
        nAmplitudes.clear();

        if (amplitudes.size() > maxAmps) {
            std::vector<bitCapIntOcl> indices;
            for (auto it = amplitudes.begin(); (nAmplitudes.size() > maxAmps) && (it != amplitudes.end()); ++it) {
                if (norm(it->second) == limit) {
                    indices.push_back(it->first);
                }
            }
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            indices.resize(amplitudes.size() - maxAmps);
            for (const bitCapIntOcl& idx : indices) {
                amplitudes.erase(idx);
            }
        }

        for (auto& pair : amplitudes) {
            pair.second *= invFidelity;
        }

        return fidelity;
    }

    complex read(const bitCapIntOcl& i) { return isReadLocked ? readLocked(i) : readUnlocked(i); }

#if ENABLE_COMPLEX_X2
    complex2 read2(const bitCapIntOcl& i1, const bitCapIntOcl& i2)
    {
        if (isReadLocked) {
            return complex2(readLocked(i1), readLocked(i2));
        }
        return complex2(readUnlocked(i1), readUnlocked(i2));
    }
#endif

    void write(const bitCapIntOcl& i, const complex& c)
    {
        const bool isCSet = abs(c) > REAL1_EPSILON;
        if (isCSet) {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes[i] = c;
        } else {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes.erase(i);
        }
    }

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        const bool isC1Set = abs(c1) > REAL1_EPSILON;
        const bool isC2Set = abs(c2) > REAL1_EPSILON;
        if (!isC1Set && !isC2Set) {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes.erase(i1);
            amplitudes.erase(i2);
        } else if (isC1Set && isC2Set) {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes[i1] = c1;
            amplitudes[i2] = c2;
        } else if (isC1Set) {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes.erase(i2);
            amplitudes[i1] = c1;
        } else {
            std::lock_guard<std::mutex> lock(mtx);
            amplitudes.erase(i1);
            amplitudes[i2] = c2;
        }
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        amplitudes.clear();
    }

    void copy_in(const complex* copyIn)
    {
        if (!copyIn) {
            clear();
            return;
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (bitCapIntOcl i = 0U; i < capacity; ++i) {
            if (abs(copyIn[i]) <= REAL1_EPSILON) {
                amplitudes.erase(i);
            } else {
                amplitudes[i] = copyIn[i];
            }
        }
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (!copyIn) {
            std::lock_guard<std::mutex> lock(mtx);
            for (bitCapIntOcl i = 0U; i < length; ++i) {
                amplitudes.erase(i);
            }

            return;
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (bitCapIntOcl i = 0U; i < length; ++i) {
            if (abs(copyIn[i]) <= REAL1_EPSILON) {
                amplitudes.erase(i);
            } else {
                amplitudes[i + offset] = copyIn[i];
            }
        }
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        StateVectorSparsePtr copyIn = std::dynamic_pointer_cast<StateVectorSparse>(copyInSv);

        if (!copyIn) {
            std::lock_guard<std::mutex> lock(mtx);
            for (bitCapIntOcl i = 0U; i < length; ++i) {
                amplitudes.erase(i + srcOffset);
            }

            return;
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (bitCapIntOcl i = 0U; i < length; ++i) {
            complex amp = copyIn->read(i + srcOffset);
            if (abs(amp) <= REAL1_EPSILON) {
                amplitudes.erase(i + srcOffset);
            } else {
                amplitudes[i + dstOffset] = amp;
            }
        }
    }

    void copy_out(complex* copyOut)
    {
        for (bitCapIntOcl i = 0U; i < capacity; ++i) {
            copyOut[i] = read(i);
        }
    }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        for (bitCapIntOcl i = 0U; i < length; ++i) {
            copyOut[i] = read(i + offset);
        }
    }

    void copy(const StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorSparse>(toCopy)); }

    void copy(StateVectorSparsePtr toCopy)
    {
        std::lock_guard<std::mutex> lock(mtx);
        amplitudes = toCopy->amplitudes;
    }

    void shuffle(StateVectorPtr svp) { shuffle(std::dynamic_pointer_cast<StateVectorSparse>(svp)); }

    void shuffle(StateVectorSparsePtr svp)
    {
        const size_t halfCap = (size_t)(capacity >> 1U);
        std::lock_guard<std::mutex> lock(mtx);
        for (bitCapIntOcl i = 0U; i < halfCap; ++i) {
            complex amp = svp->read(i);
            svp->write(i, read(i + halfCap));
            write(i + halfCap, amp);
        }
    }

    void get_probs(real1* outArray)
    {
        for (bitCapIntOcl i = 0U; i < capacity; ++i) {
            outArray[i] = norm(read(i));
        }
    }

    bool is_sparse() { return (amplitudes.size() < (size_t)(capacity >> 1U)); }

    std::vector<bitCapIntOcl> iterable()
    {
        std::vector<std::vector<bitCapIntOcl>> toRet(GetConcurrencyLevel());
        std::vector<std::vector<bitCapIntOcl>>::iterator toRetIt;

        // For lock_guard scope
        if (true) {
            std::lock_guard<std::mutex> lock(mtx);

            par_for(0U, amplitudes.size(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                auto it = amplitudes.begin();
                std::advance(it, lcv);
                toRet[cpu].push_back(it->first);
            });
        }

        for (int64_t i = (int64_t)(toRet.size() - 1U); i >= 0; i--) {
            if (toRet[i].empty()) {
                toRetIt = toRet.begin();
                std::advance(toRetIt, i);
                toRet.erase(toRetIt);
            }
        }

        if (toRet.empty()) {
            return {};
        }

        while (toRet.size() > 1U) {
            // Work odd unit into collapse sequence:
            if (toRet.size() & 1U) {
                toRet[toRet.size() - 2U].insert(
                    toRet[toRet.size() - 2U].end(), toRet[toRet.size() - 1U].begin(), toRet[toRet.size() - 1U].end());
                toRet.pop_back();
            }

            const int64_t combineCount = (int64_t)(toRet.size() >> 1U);
#if ENABLE_PTHREAD
            std::vector<std::future<void>> futures(combineCount);
            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                futures[i] = std::async(std::launch::async, [i, combineCount, &toRet]() {
                    toRet[i].insert(toRet[i].end(), toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                    toRet[i + combineCount].clear();
                });
            }
            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                futures[i].get();
                toRet.pop_back();
            }
#else
            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                toRet[i].insert(toRet[i].end(), toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                toRet.pop_back();
            }
#endif
        }

        return toRet[0U];
    }

    /// Returns empty if iteration should be over full set, otherwise just the iterable elements:
    std::set<bitCapIntOcl> iterable(
        const bitCapIntOcl& setMask, const bitCapIntOcl& filterMask = 0, const bitCapIntOcl& filterValues = 0)
    {
        if (!filterMask && filterValues) {
            return {};
        }

        const bitCapIntOcl unsetMask = ~setMask;

        std::vector<std::set<bitCapIntOcl>> toRet(GetConcurrencyLevel());
        std::vector<std::set<bitCapIntOcl>>::iterator toRetIt;

        // For lock_guard scope
        if (true) {
            std::lock_guard<std::mutex> lock(mtx);

            if (!filterMask && !filterValues) {
                par_for(0U, amplitudes.size(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                    auto it = amplitudes.begin();
                    std::advance(it, lcv);
                    toRet[cpu].insert(it->first & unsetMask);
                });
            } else {
                const bitCapIntOcl unfilterMask = ~filterMask;
                par_for(0U, amplitudes.size(), [&](const bitCapIntOcl lcv, const unsigned& cpu) {
                    auto it = amplitudes.begin();
                    std::advance(it, lcv);
                    if ((it->first & filterMask) == filterValues) {
                        toRet[cpu].insert(it->first & unsetMask & unfilterMask);
                    }
                });
            }
        }

        for (int64_t i = (int64_t)(toRet.size() - 1U); i >= 0; i--) {
            if (toRet[i].empty()) {
                toRetIt = toRet.begin();
                std::advance(toRetIt, i);
                toRet.erase(toRetIt);
            }
        }

        if (toRet.empty()) {
            return {};
        }

        while (toRet.size() > 1U) {
            // Work odd unit into collapse sequence:
            if (toRet.size() & 1U) {
                toRet[toRet.size() - 2U].insert(toRet[toRet.size() - 1U].begin(), toRet[toRet.size() - 1U].end());
                toRet.pop_back();
            }

            const int64_t combineCount = (int64_t)(toRet.size() >> 1U);
#if ENABLE_PTHREAD
            std::vector<std::future<void>> futures(combineCount);
            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                futures[i] = std::async(std::launch::async, [i, combineCount, &toRet]() {
                    toRet[i].insert(toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                    toRet[i + combineCount].clear();
                });
            }

            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                futures[i].get();
                toRet.pop_back();
            }
#else
            for (int64_t i = (combineCount - 1); i >= 0; i--) {
                toRet[i].insert(toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                toRet.pop_back();
            }
#endif
        }

        return toRet[0U];
    }
};

} // namespace Qrack
