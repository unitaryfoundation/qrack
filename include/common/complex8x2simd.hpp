//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a SIMD implementation of the float precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// float precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "config.h"

#if defined(_WIN32)
#include <intrin.h>
#else
#if ENABLE_SSE3
#include <pmmintrin.h>
#else
#include <xmmintrin.h>
#endif
#if ENABLE_FMA
#include <immintrin.h>
#endif
#endif

#include <complex>

namespace Qrack {

static const __m128 SIGNMASK = _mm_set_ps(0.0f, -0.0f, 0.0f, -0.0f);

/** SIMD implementation of the float precision complex vector type of 2 complex numbers, only for COMPLEX_X_2 Apply2x2.
 */

union complex2 {
    __m128 c2;
    float f[4];

    inline complex2() {}
    inline complex2(const __m128& cm2) { c2 = cm2; }
    inline complex2(const complex2& cm2) { c2 = cm2.c2; }
    inline complex2(const std::complex<float>& cm1, const std::complex<float>& cm2)
    {
        c2 = _mm_set_ps(cm2.imag(), cm2.real(), cm1.imag(), cm1.real());
    }
    inline complex2(const float& r1, const float& i1, const float& r2, const float& i2)
    {
        c2 = _mm_set_ps(i2, r2, i1, r1);
    }
    inline std::complex<float> c(const size_t& i) const { return complex(f[i << 1U], f[(i << 1U) + 1U]); }
    inline complex2 operator+(const complex2& other) const { return _mm_add_ps(c2, other.c2); }
    inline complex2 operator+=(const complex2& other)
    {
        c2 = _mm_add_ps(c2, other.c2);
        return c2;
    }
    inline complex2 operator-(const complex2& other) const { return _mm_sub_ps(c2, other.c2); }
    inline complex2 operator-=(const complex2& other)
    {
        c2 = _mm_sub_ps(c2, other.c2);
        return c2;
    }
    inline complex2 operator*(const complex2& other) const
    {
        const __m128& oVal2 = other.c2;
#if ENABLE_FMA
        // FMA proposed by Elara (OpenAI custom GPT)
        return _mm_fmadd_ps(_mm_shuffle_ps(c2, c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245)),
            _mm_mul_ps(c2, _mm_shuffle_ps(oVal2, oVal2, 160)));
#else
        return _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(c2, c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245))),
            _mm_mul_ps(c2, _mm_shuffle_ps(oVal2, oVal2, 160)));
#endif
    }
    inline complex2 operator*=(const complex2& other)
    {
        const __m128& oVal2 = other.c2;
#if ENABLE_FMA
        // FMA proposed by Elara (OpenAI custom GPT)
        c2 = _mm_fmadd_ps(_mm_shuffle_ps(c2, c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245)),
            _mm_mul_ps(c2, _mm_shuffle_ps(oVal2, oVal2, 160)));
#else
        c2 =
            _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(c2, c2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245))),
                _mm_mul_ps(c2, _mm_shuffle_ps(oVal2, oVal2, 160)));
#endif
        return c2;
    }
    inline complex2 operator*(const float& rhs) const { return _mm_mul_ps(c2, _mm_set1_ps(rhs)); }
    inline complex2 operator-() const { return _mm_mul_ps(_mm_set1_ps(-1.0f), c2); }
    inline complex2 operator*=(const float& other)
    {
        c2 = _mm_mul_ps(c2, _mm_set1_ps(other));
        return c2;
    }
};

inline complex2 mtrxColShuff(const complex2& mtrxCol) { return _mm_shuffle_ps(mtrxCol.c2, mtrxCol.c2, 177); }
inline complex2 matrixMul(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxCol1Shuff,
    const complex2& mtrxCol2Shuff, const complex2& qubit)
{
    const __m128 dupeLo = _mm_shuffle_ps(qubit.c2, qubit.c2, 68);
    const __m128 dupeHi = _mm_shuffle_ps(qubit.c2, qubit.c2, 238);
#if ENABLE_FMA
    // FMA proposed by Elara (OpenAI custom GPT)
    return _mm_add_ps(_mm_fmadd_ps(mtrxCol1Shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeLo, dupeLo, 245)),
                          _mm_mul_ps(mtrxCol1.c2, _mm_shuffle_ps(dupeLo, dupeLo, 160))),
        _mm_fmadd_ps(mtrxCol2Shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeHi, dupeHi, 245)),
            _mm_mul_ps(mtrxCol2.c2, _mm_shuffle_ps(dupeHi, dupeHi, 160))));
#else
    return _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(mtrxCol1Shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeLo, dupeLo, 245))),
            _mm_mul_ps(mtrxCol1.c2, _mm_shuffle_ps(dupeLo, dupeLo, 160))),
        _mm_add_ps(_mm_mul_ps(mtrxCol2Shuff.c2, _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeHi, dupeHi, 245))),
            _mm_mul_ps(mtrxCol2.c2, _mm_shuffle_ps(dupeHi, dupeHi, 160))));
#endif
}
inline complex2 matrixMul(const float& nrm, const complex2& mtrxCol1, const complex2& mtrxCol2,
    const complex2& mtrxCol1Shuff, const complex2& mtrxCol2Shuff, const complex2& qubit)
{
    return matrixMul(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit) * nrm;
}
inline complex2 operator*(const float& lhs, const complex2& rhs) { return _mm_mul_ps(_mm_set1_ps(lhs), rhs.c2); }

// See
// https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction#answer-35270026
inline float norm(const complex2& c)
{
    const __m128 n = _mm_mul_ps(c.c2, c.c2);
#if ENABLE_SSE3
    __m128 shuf = _mm_movehdup_ps(n);
#else
    __m128 shuf = _mm_shuffle_ps(n, n, _MM_SHUFFLE(2, 3, 0, 1));
#endif
    const __m128 sums = _mm_add_ps(n, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}
} // namespace Qrack
