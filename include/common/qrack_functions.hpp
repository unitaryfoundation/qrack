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

#pragma once

#include "qrack_types.hpp"

#ifdef __APPLE__
#include "TargetConditionals.h"
#elif ENABLE_INTRINSICS
#include "immintrin.h"
#endif

#include <set>
#include <vector>
#if CPP_STD >= 20
#include <bit>
#endif

#define _bi_compare(left, right)                                                                                       \
    if (left > right) {                                                                                                \
        return 1;                                                                                                      \
    }                                                                                                                  \
    if (left < right) {                                                                                                \
        return -1;                                                                                                     \
    }                                                                                                                  \
                                                                                                                       \
    return 0;

#if (QBCAPPOW < 7) || ((QBCAPPOW < 8) && defined(__SIZEOF_INT128__)) || ((QBCAPPOW > 7) && defined(BOOST_AVAILABLE))
inline void bi_not_ip(bitCapInt* left) { *left = ~(*left); }
inline void bi_and_ip(bitCapInt* left, const bitCapInt& right) { *left &= right; }
inline void bi_or_ip(bitCapInt* left, const bitCapInt& right) { *left |= right; }
inline void bi_xor_ip(bitCapInt* left, const bitCapInt& right) { *left ^= right; }
inline double bi_to_double(const bitCapInt& in) { return (double)in; }

inline void bi_increment(bitCapInt* pBigInt, const bitCapInt& value) { *pBigInt += value; }
inline void bi_decrement(bitCapInt* pBigInt, const bitCapInt& value) { *pBigInt -= value; }

inline void bi_lshift_ip(bitCapInt* left, const size_t& right) { *left <<= right; }
inline void bi_rshift_ip(bitCapInt* left, const size_t& right) { *left >>= right; }

inline int bi_and_1(const bitCapInt& left) { return (bool)(left & 1); }

inline int bi_compare(const bitCapInt& left, const bitCapInt& right) { _bi_compare(left, right) }
inline int bi_compare_0(const bitCapInt& left) { return (int)(bool)left; }
inline int bi_compare_1(const bitCapInt& left) { _bi_compare(left, 1U); }

inline void bi_add_ip(bitCapInt* left, const bitCapInt& right) { *left += right; }
inline void bi_sub_ip(bitCapInt* left, const bitCapInt& right) { *left -= right; }

inline void bi_div_mod(const bitCapInt& left, const bitCapInt& right, bitCapInt* quotient, bitCapInt* rmndr)
{
    if (quotient) {
        *quotient = left / right;
    }
    if (rmndr) {
        *rmndr = left % right;
    }
}
#ifdef __SIZEOF_INT128__
inline void bi_div_mod_small(const bitCapInt& left, uint64_t right, bitCapInt* quotient, uint64_t* rmndr)
{
    if (quotient) {
        *quotient = left / right;
    }
    if (rmndr) {
        *rmndr = (uint64_t)(left % right);
    }
}
#else
inline void bi_div_mod_small(const bitCapInt& left, uint32_t right, bitCapInt* quotient, uint32_t* rmndr)
{
    if (quotient) {
        *quotient = left / right;
    }
    if (rmndr) {
        *rmndr = (uint32_t)(left % right);
    }
}
#endif
#endif

namespace Qrack {

inline bitLenInt log2Ocl(bitCapIntOcl n)
{
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if CPP_STD >= 20
    return std::bit_width(n) - 1U;
#elif ENABLE_INTRINSICS && defined(_WIN32) && !defined(__CYGWIN__)
#if UINTPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - _lzcnt_u32((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - _lzcnt_u64((unsigned long long)n) - 1U);
#endif
#elif ENABLE_INTRINSICS && !defined(__APPLE__)
#if UINTPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
    bitLenInt pow = 0U;
    bitCapIntOcl p = n >> 1U;
    while (p) {
        p >>= 1U;
        ++pow;
    }
    return pow;
#endif
}

inline bitLenInt popCountOcl(bitCapIntOcl n)
{
#if CPP_STD >= 20
    return (bitLenInt)std::popcount(n);
#elif (defined(__GNUC__) || defined(__clang__)) && !TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR
    return __builtin_popcount(n);
#else
    bitLenInt popCount;
    for (popCount = 0U; n; ++popCount) {
        n &= n - 1U;
    }
    return popCount;
#endif
}

#if (QBCAPPOW < 7) || ((QBCAPPOW < 8) && defined(__SIZEOF_INT128__)) || ((QBCAPPOW > 7) && defined(BOOST_AVAILABLE))
inline int bi_log2(const bitCapInt& n) { return log2Ocl((bitCapIntOcl)n); }
#endif
inline bitLenInt log2(bitCapInt n) { return (bitLenInt)bi_log2(n); }

inline bitCapInt pow2(const bitLenInt& p) { return ONE_BCI << p; }
inline bitCapIntOcl pow2Ocl(const bitLenInt& p) { return (bitCapIntOcl)1U << p; }
inline bitCapInt pow2Mask(const bitLenInt& p)
{
    bitCapInt toRet = ONE_BCI << p;
    bi_decrement(&toRet, 1U);
    return toRet;
}
inline bitCapIntOcl pow2MaskOcl(const bitLenInt& p) { return ((bitCapIntOcl)1U << p) - 1U; }
inline bitCapInt bitSlice(const bitLenInt& bit, const bitCapInt& source) { return (ONE_BCI << bit) & source; }
inline bitCapIntOcl bitSliceOcl(const bitLenInt& bit, const bitCapIntOcl& source)
{
    return ((bitCapIntOcl)1U << bit) & source;
}
inline bitCapInt bitRegMask(const bitLenInt& start, const bitLenInt& length)
{
    bitCapInt toRet = ONE_BCI << length;
    bi_decrement(&toRet, 1U);
    bi_lshift_ip(&toRet, start);
    return toRet;
}
inline bitCapIntOcl bitRegMaskOcl(const bitLenInt& start, const bitLenInt& length)
{
    return (((bitCapIntOcl)1U << length) - 1U) << start;
}
// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x)
{
    bitCapInt y = x;
    bi_decrement(&y, 1U);
    bi_and_ip(&y, x);
    return (bi_compare_0(x) != 0) && (bi_compare_0(y) == 0);
}
inline bool isPowerOfTwoOcl(const bitCapIntOcl& x) { return x && !(x & (x - 1U)); }
inline bool isBadBitRange(const bitLenInt& start, const bitLenInt& length, const bitLenInt& qubitCount)
{
    return ((start + length) > qubitCount) || ((bitLenInt)(start + length) < start);
}
inline bool isBadPermRange(const bitCapIntOcl& start, const bitCapIntOcl& length, const bitCapIntOcl& maxQPowerOcl)
{
    return ((start + length) > maxQPowerOcl) || ((bitCapIntOcl)(start + length) < start);
}
inline void ThrowIfQbIdArrayIsBad(
    const std::vector<bitLenInt>& controls, const bitLenInt& qubitCount, std::string message)
{
    std::set<bitLenInt> dupes;
    for (const bitLenInt& control : controls) {
        if (control >= qubitCount) {
            throw std::invalid_argument(message);
        }

        if (dupes.find(control) == dupes.end()) {
            dupes.insert(control);
        } else {
            throw std::invalid_argument(message + " (Found duplicate qubit indices!)");
        }
    }
}

// These are utility functions defined in qinterface/protected.cpp:
unsigned char* cl_alloc(size_t ucharCount);
void cl_free(void* toFree);
void mul2x2(const complex* left, const complex* right, complex* out);
void exp2x2(const complex* matrix2x2, complex* outMatrix2x2);
void log2x2(const complex* matrix2x2, complex* outMatrix2x2);
void inv2x2(const complex* matrix2x2, complex* outMatrix2x2);
bool isOverflowAdd(
    bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower);
bool isOverflowSub(
    bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower);
bitCapInt pushApartBits(const bitCapInt& perm, const std::vector<bitCapInt>& skipPowers);
bitCapInt intPow(const bitCapInt& base, const bitCapInt& power);
bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power);

#if QBCAPPOW > 6
std::ostream& operator<<(std::ostream& os, const bitCapInt& b);
std::istream& operator>>(std::istream& is, bitCapInt& b);
#endif

#if ENABLE_ENV_VARS
const real1_f _qrack_qunit_sep_thresh = getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")
    ? (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")))
    : FP_NORM_EPSILON;
const real1_f _qrack_qbdt_sep_thresh = getenv("QRACK_QBDT_SEPARABILITY_THRESHOLD")
    ? (real1_f)std::stof(std::string(getenv("QRACK_QBDT_SEPARABILITY_THRESHOLD")))
    : FP_NORM_EPSILON;
const bitLenInt QRACK_MAX_CPU_QB_DEFAULT =
    getenv("QRACK_MAX_CPU_QB") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_CPU_QB"))) : -1;
const bitLenInt QRACK_MAX_PAGE_QB_DEFAULT = getenv("QRACK_MAX_PAGE_QB")
    ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGE_QB")))
    : QRACK_MAX_CPU_QB_DEFAULT;
const bitLenInt QRACK_MAX_PAGING_QB_DEFAULT = getenv("QRACK_MAX_PAGING_QB")
    ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")))
    : QRACK_MAX_CPU_QB_DEFAULT;
const bitLenInt QRACK_QRACK_QTENSORNETWORK_THRESHOLD_CPU_QB =
    getenv("QRACK_MAX_CPU_QB") ? QRACK_MAX_CPU_QB_DEFAULT : 32U;
const bitLenInt PSTRIDEPOW_DEFAULT =
    (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
const size_t QRACK_QBDT_MAX_ALLOC_MB_DEFAULT =
    (size_t)(getenv("QRACK_QBDT_MAX_ALLOC_MB") ? std::stoi(std::string(getenv("QRACK_QBDT_MAX_ALLOC_MB"))) : -1);
const size_t QRACK_SPARSE_MAX_ALLOC_MB_DEFAULT =
    (size_t)(getenv("QRACK_SPARSE_MAX_ALLOC_MB") ? std::stoi(std::string(getenv("QRACK_SPARSE_MAX_ALLOC_MB"))) : -1);
#else
const real1_f _qrack_qunit_sep_thresh = FP_NORM_EPSILON;
const real1_f _qrack_qbdt_sep_thresh = FP_NORM_EPSILON;
const bitLenInt QRACK_MAX_CPU_QB_DEFAULT = -1;
const bitLenInt QRACK_MAX_PAGE_QB_DEFAULT = QRACK_MAX_CPU_QB_DEFAULT;
const bitLenInt QRACK_MAX_PAGING_QB_DEFAULT = QRACK_MAX_CPU_QB_DEFAULT;
const bitLenInt QRACK_QRACK_QTENSORNETWORK_THRESHOLD_CPU_QB = 32U;
const bitLenInt PSTRIDEPOW_DEFAULT = PSTRIDEPOW;
const size_t QRACK_QBDT_MAX_ALLOC_MB_DEFAULT = -1;
const size_t QRACK_SPARSE_MAX_ALLOC_MB_DEFAULT = -1;
#endif
const size_t QRACK_QBDT_MAX_ALLOC_BYTES_DEFAULT = QRACK_QBDT_MAX_ALLOC_MB_DEFAULT * 1024U * 1024U;
const size_t QRACK_SPARSE_MAX_ALLOC_BYTES_DEFAULT = QRACK_SPARSE_MAX_ALLOC_MB_DEFAULT * 1024U * 1024U;
const size_t QRACK_SPARSE_MAX_KEYS = (QRACK_SPARSE_MAX_ALLOC_MB_DEFAULT / SPARSE_KEY_BYTES) >> 1U;
} // namespace Qrack
