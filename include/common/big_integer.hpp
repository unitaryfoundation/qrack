//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qimcifa contributors, 2022, 2023. All rights reserved.
//
// This header has been adapted for OpenCL and C, from big_integer.c by Andre Azevedo.
//
// Original file:
//
// big_integer.c
//     Description: "Arbitrary"-precision integer
//     Author: Andre Azevedo <http://github.com/andreazevedo>
//
// The MIT License (MIT)
//
// Copyright (c) 2014 Andre Azevedo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "config.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

using std::size_t;

#ifdef __SIZEOF_INT128__
#define BIG_INTEGER_WORD_BITS 128U
#define BIG_INTEGER_WORD_POWER 7U
#define BIG_INTEGER_WORD unsigned __int128
#define BIG_INTEGER_HALF_WORD uint64_t
#define BIG_INTEGER_HALF_WORD_MASK 0xFFFFFFFFFFFFFFFFULL
#define BIG_INTEGER_HALF_WORD_MASK_NOT 0xFFFFFFFFFFFFFFFF0000000000000000ULL
#else
#define BIG_INTEGER_WORD_BITS 64U
#define BIG_INTEGER_WORD_POWER 6U
#define BIG_INTEGER_WORD uint64_t
#define BIG_INTEGER_HALF_WORD uint32_t
#define BIG_INTEGER_HALF_WORD_MASK 0xFFFFFFFFULL
#define BIG_INTEGER_HALF_WORD_MASK_NOT 0xFFFFFFFF00000000ULL
#endif

// This can be any power of 2 greater than (or equal to) 64:
constexpr size_t BIG_INTEGER_BITS = (1 << QBCAPPOW);
constexpr int BIG_INTEGER_WORD_SIZE = BIG_INTEGER_BITS / BIG_INTEGER_WORD_BITS;

// The rest of the constants need to be consistent with the one above:
constexpr size_t BIG_INTEGER_HALF_WORD_BITS = BIG_INTEGER_WORD_BITS >> 1U;
constexpr int BIG_INTEGER_HALF_WORD_SIZE = BIG_INTEGER_WORD_SIZE << 1U;
constexpr int BIG_INTEGER_MAX_WORD_INDEX = BIG_INTEGER_WORD_SIZE - 1U;

typedef struct BigInteger {
    BIG_INTEGER_WORD bits[BIG_INTEGER_WORD_SIZE];

    inline BigInteger()
    {
        for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
            bits[i] = 0U;
        }
    }

    inline BigInteger(const BigInteger& val)
    {
        for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
            bits[i] = val.bits[i];
        }
    }

    inline BigInteger(const BIG_INTEGER_WORD& val)
    {
        bits[0] = val;
        for (int i = 1; i < BIG_INTEGER_WORD_SIZE; ++i) {
            bits[i] = 0U;
        }
    }

#ifdef __SIZEOF_INT128__
    inline explicit operator unsigned __int128() const { return (unsigned __int128)bits[0U]; }
#endif
    inline explicit operator uint64_t() const { return (uint64_t)bits[0U]; }
    inline explicit operator uint32_t() const { return (uint32_t)bits[0U]; }
} BigInteger;

inline void bi_set_0(BigInteger* p)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        p->bits[i] = 0U;
    }
}

inline BigInteger bi_copy(const BigInteger& in)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = in.bits[i];
    }
    return result;
}

inline void bi_copy_ip(const BigInteger& in, BigInteger* out)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        out->bits[i] = in.bits[i];
    }
}

inline int bi_compare(const BigInteger& left, const BigInteger& right)
{
    for (int i = BIG_INTEGER_MAX_WORD_INDEX; i >= 0; --i) {
        if (left.bits[i] > right.bits[i]) {
            return 1;
        }

        if (left.bits[i] < right.bits[i]) {
            return -1;
        }
    }

    return 0;
}

inline int bi_compare_0(const BigInteger& left)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        if (left.bits[i]) {
            return 1;
        }
    }

    return 0;
}

inline int bi_compare_1(const BigInteger& left)
{
    for (int i = BIG_INTEGER_MAX_WORD_INDEX; i > 0; --i) {
        if (left.bits[i]) {
            return 1;
        }
    }

    if (left.bits[0] > 1U) {
        return 1;
    }

    if (left.bits[0] < 1U) {
        return -1;
    }

    return 0;
}

inline BigInteger operator+(const BigInteger& left, const BigInteger& right)
{
    BigInteger result;
    result.bits[0U] = 0U;
    for (int i = 0; i < BIG_INTEGER_MAX_WORD_INDEX; ++i) {
        result.bits[i] += left.bits[i] + right.bits[i];
        result.bits[i + 1] = (result.bits[i] < left.bits[i]) ? 1 : 0;
    }
    result.bits[BIG_INTEGER_MAX_WORD_INDEX] += right.bits[BIG_INTEGER_MAX_WORD_INDEX];
    return result;
}

inline void bi_add_ip(BigInteger* left, const BigInteger& right)
{
    for (int i = 0; i < BIG_INTEGER_MAX_WORD_INDEX; ++i) {
        BIG_INTEGER_WORD temp = left->bits[i];
        left->bits[i] += right.bits[i];
        int j = i;
        while ((j < BIG_INTEGER_MAX_WORD_INDEX) && (left->bits[j] < temp)) {
            temp = left->bits[++j]++;
        }
    }
    left->bits[BIG_INTEGER_MAX_WORD_INDEX] += right.bits[BIG_INTEGER_MAX_WORD_INDEX];
}

inline BigInteger operator-(const BigInteger& left, const BigInteger& right)
{
    BigInteger result;
    result.bits[0U] = 0U;
    for (int i = 0; i < BIG_INTEGER_MAX_WORD_INDEX; ++i) {
        result.bits[i] += left.bits[i] - right.bits[i];
        result.bits[i + 1] = (result.bits[i] > left.bits[i]) ? -1 : 0;
    }
    result.bits[BIG_INTEGER_MAX_WORD_INDEX] -= right.bits[BIG_INTEGER_MAX_WORD_INDEX];
    return result;
}

inline void bi_sub_ip(BigInteger* left, const BigInteger& right)
{
    for (int i = 0; i < BIG_INTEGER_MAX_WORD_INDEX; ++i) {
        BIG_INTEGER_WORD temp = left->bits[i];
        left->bits[i] -= right.bits[i];
        int j = i;
        while ((j < BIG_INTEGER_MAX_WORD_INDEX) && (left->bits[j] > temp)) {
            temp = left->bits[++j]--;
        }
    }
    left->bits[BIG_INTEGER_MAX_WORD_INDEX] -= right.bits[BIG_INTEGER_MAX_WORD_INDEX];
}

inline void bi_increment(BigInteger* pBigInt, const BIG_INTEGER_WORD& value)
{
    BIG_INTEGER_WORD temp = pBigInt->bits[0];
    pBigInt->bits[0] += value;

    if (temp <= pBigInt->bits[0]) {
        return;
    }

    for (int i = 1; i < BIG_INTEGER_WORD_SIZE; i++) {
        temp = pBigInt->bits[i]++;
        if (temp <= pBigInt->bits[i]) {
            break;
        }
    }
}

inline void bi_decrement(BigInteger* pBigInt, const BIG_INTEGER_WORD& value)
{
    BIG_INTEGER_WORD temp = pBigInt->bits[0];
    pBigInt->bits[0] -= value;

    if (temp >= pBigInt->bits[0]) {
        return;
    }

    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; i++) {
        temp = pBigInt->bits[i]--;
        if (temp >= pBigInt->bits[i]) {
            break;
        }
    }
}

inline BigInteger bi_load(BIG_INTEGER_WORD* a)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = a[i];
    }

    return result;
}

inline BigInteger bi_lshift_word(const BigInteger& left, BIG_INTEGER_WORD rightMult)
{
    if (!rightMult) {
        return left;
    }

    BigInteger result = 0;
    for (int i = rightMult; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = left.bits[i - rightMult];
    }

    return result;
}

inline void bi_lshift_word_ip(BigInteger* left, BIG_INTEGER_WORD rightMult)
{
    rightMult &= 63U;

    if (!rightMult) {
        return;
    }

    for (int i = rightMult; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i] = left->bits[i - rightMult];
    }
    for (BIG_INTEGER_WORD i = 0U; i < rightMult; ++i) {
        left->bits[i] = 0U;
    }
}

inline BigInteger bi_rshift_word(const BigInteger& left, const BIG_INTEGER_WORD& rightMult)
{
    if (!rightMult) {
        return left;
    }

    BigInteger result = 0U;
    for (int i = rightMult; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i - rightMult] = left.bits[i];
    }

    return result;
}

inline void bi_rshift_word_ip(BigInteger* left, const BIG_INTEGER_WORD& rightMult)
{
    if (!rightMult) {
        return;
    }

    for (int i = rightMult; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i - rightMult] = left->bits[i];
    }
    for (BIG_INTEGER_WORD i = 0U; i < rightMult; ++i) {
        left->bits[BIG_INTEGER_MAX_WORD_INDEX - i] = 0U;
    }
}

inline BigInteger operator<<(const BigInteger& left, BIG_INTEGER_WORD right)
{
    const int rShift64 = right >> BIG_INTEGER_WORD_POWER;
    const int rMod = right - (rShift64 << BIG_INTEGER_WORD_POWER);

    BigInteger result = bi_lshift_word(left, rShift64);

    if (!rMod) {
        return result;
    }

    const int rModComp = BIG_INTEGER_WORD_BITS - rMod;
    BIG_INTEGER_WORD carry = 0U;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        right = result.bits[i];
        result.bits[i] = carry | (right << rMod);
        carry = right >> rModComp;
    }

    return result;
}

inline void bi_lshift_ip(BigInteger* left, BIG_INTEGER_WORD right)
{
    const int rShift64 = right >> BIG_INTEGER_WORD_POWER;
    const int rMod = right - (rShift64 << BIG_INTEGER_WORD_POWER);

    bi_lshift_word_ip(left, rShift64);

    if (!rMod) {
        return;
    }

    const int rModComp = BIG_INTEGER_WORD_BITS - rMod;
    BIG_INTEGER_WORD carry = 0U;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        right = left->bits[i];
        left->bits[i] = carry | (right << rMod);
        carry = right >> rModComp;
    }
}

inline BigInteger operator>>(const BigInteger& left, BIG_INTEGER_WORD right)
{
    const int rShift64 = right >> BIG_INTEGER_WORD_POWER;
    const int rMod = right - (rShift64 << BIG_INTEGER_WORD_POWER);

    BigInteger result = bi_rshift_word(left, rShift64);

    if (!rMod) {
        return result;
    }

    const int rModComp = BIG_INTEGER_WORD_BITS - rMod;
    BIG_INTEGER_WORD carry = 0U;
    for (int i = BIG_INTEGER_MAX_WORD_INDEX; i >= 0; --i) {
        right = result.bits[i];
        result.bits[i] = carry | (right >> rMod);
        carry = right << rModComp;
    }

    return result;
}

inline void bi_rshift_ip(BigInteger* left, BIG_INTEGER_WORD right)
{
    const int rShift64 = right >> BIG_INTEGER_WORD_POWER;
    const int rMod = right - (rShift64 << BIG_INTEGER_WORD_POWER);

    bi_rshift_word_ip(left, rShift64);

    if (!rMod) {
        return;
    }

    const int rModComp = BIG_INTEGER_WORD_BITS - rMod;
    BIG_INTEGER_WORD carry = 0U;
    for (int i = BIG_INTEGER_MAX_WORD_INDEX; i >= 0; --i) {
        right = left->bits[i];
        left->bits[i] = carry | (right >> rMod);
        carry = right << rModComp;
    }
}

inline int bi_log2(const BigInteger& n)
{
    int pw = 0;
    BigInteger p = n >> 1U;
    while (bi_compare_0(p) != 0) {
        bi_rshift_ip(&p, 1U);
        ++pw;
    }
    return pw;
}

inline int bi_and_1(const BigInteger& left) { return left.bits[0] & 1; }

inline BigInteger operator&(const BigInteger& left, const BigInteger& right)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = left.bits[i] & right.bits[i];
    }
    return result;
}

inline void bi_and_ip(BigInteger* left, const BigInteger& right)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i] &= right.bits[i];
    }
}

inline BigInteger operator|(const BigInteger& left, const BigInteger& right)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = left.bits[i] | right.bits[i];
    }
    return result;
}

inline void bi_or_ip(BigInteger* left, const BigInteger& right)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i] |= right.bits[i];
    }
}

inline BigInteger operator^(const BigInteger& left, const BigInteger& right)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = left.bits[i] ^ right.bits[i];
    }
    return result;
}

inline void bi_xor_ip(BigInteger* left, const BigInteger& right)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i] ^= right.bits[i];
    }
}

inline BigInteger operator~(const BigInteger& left)
{
    BigInteger result;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        result.bits[i] = ~(left.bits[i]);
    }
    return result;
}

inline void bi_not_ip(BigInteger* left)
{
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        left->bits[i] = ~(left->bits[i]);
    }
}

inline double bi_to_double(const BigInteger& in)
{
    double toRet = 0.0;
    for (int i = 0; i < BIG_INTEGER_WORD_SIZE; ++i) {
        if (in.bits[i]) {
            toRet += in.bits[i] * pow(2.0, BIG_INTEGER_WORD_BITS * i);
        }
    }
    return toRet;
}

inline bool operator==(const BigInteger& left, const BigInteger& right) { return bi_compare(left, right) == 0; }
inline bool operator<(const BigInteger& left, const BigInteger& right) { return bi_compare(left, right) < 0; }
inline bool operator<=(const BigInteger& left, const BigInteger& right) { return bi_compare(left, right) <= 0; }
inline bool operator>(const BigInteger& left, const BigInteger& right) { return bi_compare(left, right) > 0; }
inline bool operator>=(const BigInteger& left, const BigInteger& right) { return bi_compare(left, right) >= 0; }

inline BigInteger operator++(BigInteger& a)
{
    bi_increment(&a, 1U);
    return a;
}

inline BigInteger operator--(BigInteger& a)
{
    bi_decrement(&a, 1U);
    return a;
}

/**
 * "Schoolbook multiplication" (on half words)
 * Complexity - O(x^2)
 */
BigInteger operator*(const BigInteger& left, BIG_INTEGER_HALF_WORD right);

#if true
/**
 * Adapted from Qrack! (The fundamental algorithm was discovered before.)
 * Complexity - O(log)
 */
BigInteger operator*(const BigInteger& left, const BigInteger& right);
#else
/**
 * "Schoolbook multiplication" (on half words)
 * Complexity - O(x^2)
 */
BigInteger operator*(const BigInteger& left, const BigInteger& right);
#endif

/**
 * "Schoolbook division" (on half words)
 * Complexity - O(x^2)
 */
void bi_div_mod_small(
    const BigInteger& left, BIG_INTEGER_HALF_WORD right, BigInteger* quotient, BIG_INTEGER_HALF_WORD* rmndr);

/**
 * Adapted from Qrack! (The fundamental algorithm was discovered before.)
 * Complexity - O(log)
 */
void bi_div_mod(const BigInteger& left, const BigInteger& right, BigInteger* quotient, BigInteger* rmndr);
