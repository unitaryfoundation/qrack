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

#include "big_integer.hpp"

#include <stdexcept>

// "Schoolbook multiplication" (on half words)
// Complexity - O(x^2)
BigInteger operator*(const BigInteger& left, BIG_INTEGER_HALF_WORD right)
{
    BigInteger result = 0U;
    BIG_INTEGER_WORD carry = 0U;
    for (int i = 0; i < BIG_INTEGER_HALF_WORD_SIZE; ++i) {
        const int i2 = i >> 1;
        if (i & 1) {
            BIG_INTEGER_WORD temp = right * (left.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS) + carry;
            carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
            result.bits[i2] |= (temp & BIG_INTEGER_HALF_WORD_MASK) << BIG_INTEGER_HALF_WORD_BITS;
        } else {
            BIG_INTEGER_WORD temp = right * (left.bits[i2] & BIG_INTEGER_HALF_WORD_MASK) + carry;
            carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
            result.bits[i2] |= temp & BIG_INTEGER_HALF_WORD_MASK;
        }
    }
    return result;
}

#if true
// Adapted from Qrack! (The fundamental algorithm was discovered before.)
// Complexity - O(log of bits)
BigInteger operator*(const BigInteger& left, const BigInteger& right)
{
    int rightLog2 = bi_log2(right);

    if (rightLog2 == 0) {
        // right == 1
        return left;
    }

    int maxI = BIG_INTEGER_BITS - rightLog2;

    BigInteger result;
    bi_set_0(&result);
    for (int i = 0; i < maxI; ++i) {
        BigInteger partMul = right << i;
        if (bi_compare_0(partMul) == 0) {
            break;
        }
        const int iWord = i / BIG_INTEGER_WORD_BITS;
        if (1 & (left.bits[iWord] >> (i - (iWord * BIG_INTEGER_WORD_BITS)))) {
            for (int j = iWord; j < BIG_INTEGER_WORD_SIZE; j++) {
                BIG_INTEGER_WORD temp = result.bits[j];
                result.bits[j] += partMul.bits[j];
                int k = j;
                while ((k < BIG_INTEGER_MAX_WORD_INDEX) && (temp > result.bits[k])) {
                    temp = result.bits[++k]++;
                }
            }
        }
    }

    return result;
}
#else
// "Schoolbook multiplication" (on half words)
// Complexity - O(x^2)
BigInteger operator*(const BigInteger& left, const BigInteger& right)
{
    if (right->bits[0] <= BIG_INTEGER_HALF_WORD_MASK) {
        int wordSize;
        for (wordSize = 1; wordSize < BIG_INTEGER_WORD_SIZE; ++wordSize) {
            if (right->bits[wordSize]) {
                break;
            }
        }

        if (wordSize == BIG_INTEGER_WORD_SIZE) {
            return left * (BIG_INTEGER_HALF_WORD)(right->bits[0]);
        }
    }

    if (left.bits[0] <= BIG_INTEGER_HALF_WORD_MASK) {
        int wordSize;
        for (wordSize = 1; wordSize < BIG_INTEGER_WORD_SIZE; ++wordSize) {
            if (left.bits[wordSize]) {
                break;
            }
        }

        if (wordSize == BIG_INTEGER_WORD_SIZE) {
            return right & (BIG_INTEGER_HALF_WORD)(left.bits[0]);
        }
    }

    BigInteger result = 0U;
    for (int i = 0; i < BIG_INTEGER_HALF_WORD_SIZE; ++i) {
        BIG_INTEGER_WORD carry = 0U;
        const bool isIEven = ((i & 1) == 0);
        const int i2 = i >> 1;
        const int maxJ = BIG_INTEGER_HALF_WORD_SIZE - i;
        if (isIEven) {
            for (int j = 0; j < maxJ; ++j) {
                const bool isJEven = ((j & 1) == 0);
                const int j2 = j >> 1;
                const int i2j2 = i2 + j2;
                if (isJEven) {
                    BIG_INTEGER_WORD temp =
                        (right->bits[j2] & BIG_INTEGER_HALF_WORD_MASK) * (left.bits[i2] & BIG_INTEGER_HALF_WORD_MASK) +
                        (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK) + carry;
                    carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
                    result.bits[i2j2] =
                        (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK_NOT) | (temp & BIG_INTEGER_HALF_WORD_MASK);
                } else {
                    BIG_INTEGER_WORD temp =
                        (right->bits[j2] >> BIG_INTEGER_HALF_WORD_BITS) * (left.bits[i2] & BIG_INTEGER_HALF_WORD_MASK) +
                        (result.bits[i2j2] >> BIG_INTEGER_HALF_WORD_BITS) + carry;
                    carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
                    result.bits[i2j2] = (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK) |
                        ((temp & BIG_INTEGER_HALF_WORD_MASK) << BIG_INTEGER_HALF_WORD_BITS);
                }
            }
        } else {
            for (int j = 0; j < maxJ; ++j) {
                const bool isJEven = ((j & 1) == 0);
                const int j2 = j >> 1;
                const int i2j2 = isJEven ? (i2 + j2) : (i2 + j2 + 1);
                if (isJEven) {
                    BIG_INTEGER_WORD temp =
                        (right->bits[j2] & BIG_INTEGER_HALF_WORD_MASK) * (left.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS) +
                        (result.bits[i2j2] >> BIG_INTEGER_HALF_WORD_BITS) + carry;
                    carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
                    result.bits[i2j2] = (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK) |
                        ((temp & BIG_INTEGER_HALF_WORD_MASK) << BIG_INTEGER_HALF_WORD_BITS);
                } else {
                    BIG_INTEGER_WORD temp = (right->bits[j2] >> BIG_INTEGER_HALF_WORD_BITS) *
                            (left.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS) +
                        (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK) + carry;
                    carry = temp >> BIG_INTEGER_HALF_WORD_BITS;
                    result.bits[i2j2] =
                        (result.bits[i2j2] & BIG_INTEGER_HALF_WORD_MASK_NOT) | (temp & BIG_INTEGER_HALF_WORD_MASK);
                }
            }
        }
    }

    return result;
}
#endif

// "Schoolbook division" (on half words)
// Complexity - O(x^2)
void bi_div_mod_small(
    const BigInteger& left, BIG_INTEGER_HALF_WORD right, BigInteger* quotient, BIG_INTEGER_HALF_WORD* rmndr)
{
    if (!right) {
        throw std::invalid_argument("Division by 0 in bi_div_mod_small()!");
    }

    BIG_INTEGER_WORD carry = 0U;
    if (quotient) {
        bi_set_0(quotient);
        for (int i = BIG_INTEGER_HALF_WORD_SIZE - 1; i >= 0; --i) {
            const int i2 = i >> 1;
            carry <<= BIG_INTEGER_HALF_WORD_BITS;
            if (i & 1) {
                carry |= left.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS;
                quotient->bits[i2] |= (carry / right) << BIG_INTEGER_HALF_WORD_BITS;
            } else {
                carry |= left.bits[i2] & BIG_INTEGER_HALF_WORD_MASK;
                quotient->bits[i2] |= (carry / right);
            }
            carry %= right;
        }
    } else {
        for (int i = BIG_INTEGER_HALF_WORD_SIZE - 1; i >= 0; --i) {
            const int i2 = i >> 1;
            carry <<= BIG_INTEGER_HALF_WORD_BITS;
            if (i & 1) {
                carry |= left.bits[i2] >> BIG_INTEGER_HALF_WORD_BITS;

            } else {
                carry |= left.bits[i2] & BIG_INTEGER_HALF_WORD_MASK;
            }
            carry %= right;
        }
    }

    if (rmndr) {
        *rmndr = carry;
    }
}

// Adapted from Qrack! (The fundamental algorithm was discovered before.)
// Complexity - O(log)
void bi_div_mod(const BigInteger& left, const BigInteger& right, BigInteger* quotient, BigInteger* rmndr)
{
    if (bi_compare_0(right) == 0) {
        throw std::invalid_argument("Division by 0 in bi_div_mod()!");
    }

    const int lrCompare = bi_compare(left, right);

    if (lrCompare < 0) {
        // left < right
        if (quotient) {
            // quotient = 0
            bi_set_0(quotient);
        }
        if (rmndr) {
            // rmndr = left
            bi_copy_ip(left, rmndr);
        }

        return;
    }

    if (lrCompare == 0) {
        // left == right
        if (quotient) {
            // quotient = 1
            bi_set_0(quotient);
            quotient->bits[0] = 1;
        }
        if (rmndr) {
            // rmndr = 0
            bi_set_0(rmndr);
        }

        return;
    }

    // Otherwise, past this point, left > right.

    if (right.bits[0] <= BIG_INTEGER_HALF_WORD_MASK) {
        int wordSize;
        for (wordSize = 1; wordSize < BIG_INTEGER_WORD_SIZE; ++wordSize) {
            if (right.bits[wordSize]) {
                break;
            }
        }
        if (wordSize >= BIG_INTEGER_WORD_SIZE) {
            // We can use the small division variant.
            if (rmndr) {
                BIG_INTEGER_HALF_WORD t;
                bi_div_mod_small(left, (BIG_INTEGER_HALF_WORD)(right.bits[0]), quotient, &t);
                rmndr->bits[0] = t;
                for (int i = 1; i < BIG_INTEGER_WORD_SIZE; ++i) {
                    rmndr->bits[i] = 0U;
                }
            } else {
                bi_div_mod_small(left, (BIG_INTEGER_HALF_WORD)(right.bits[0]), quotient, 0);
            }

            return;
        }
    }

    BigInteger bi1 = 1U;
    int rightLog2 = bi_log2(right);
    BigInteger rightTest = bi1 << rightLog2;
    if (bi_compare(right, rightTest) < 0) {
        ++rightLog2;
    }
    BigInteger rem;
    bi_copy_ip(left, &rem);
    if (quotient) {
        bi_set_0(quotient);
        while (bi_compare(rem, right) >= 0) {
            int logDiff = bi_log2(rem) - rightLog2;
            if (logDiff > 0) {
                BigInteger partMul = right << logDiff;
                BigInteger partQuo = bi1 << logDiff;
                bi_sub_ip(&rem, partMul);
                bi_add_ip(quotient, partQuo);
            } else {
                bi_sub_ip(&rem, right);
                bi_increment(quotient, 1U);
            }
        }
    } else {
        while (bi_compare(rem, right) >= 0) {
            int logDiff = bi_log2(rem) - rightLog2;
            if (logDiff > 0) {
                BigInteger partMul = right << logDiff;
                bi_sub_ip(&rem, partMul);
            } else {
                bi_sub_ip(&rem, right);
            }
        }
    }
    if (rmndr) {
        *rmndr = rem;
    }
}

BigInteger operator/(const BigInteger& left, BIG_INTEGER_HALF_WORD right)
{
    BigInteger quotient;
    BIG_INTEGER_HALF_WORD rmndr;
    bi_div_mod_small(left, right, &quotient, &rmndr);

    return quotient;
}

BIG_INTEGER_HALF_WORD operator%(const BigInteger& left, BIG_INTEGER_HALF_WORD right)
{
    BigInteger quotient;
    BIG_INTEGER_HALF_WORD rmndr;
    bi_div_mod_small(left, right, &quotient, &rmndr);

    return rmndr;
}

BigInteger operator/(const BigInteger& left, const BigInteger& right)
{
    BigInteger quotient, rmndr;
    bi_div_mod(left, right, &quotient, &rmndr);

    return quotient;
}

BigInteger operator%(const BigInteger& left, const BigInteger& right)
{
    BigInteger quotient, rmndr;
    bi_div_mod(left, right, &quotient, &rmndr);

    return rmndr;
}
