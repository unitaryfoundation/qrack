//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if defined(_WIN32)
#include <intrin.h>
#else
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#endif

#include <complex>

namespace Qrack {

static const __m256d SIGNMASK = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);

/** SIMD implementation of the double precision complex vector type of 2 complex numbers, only for AVX Apply2x2. */
union complex2 {
    __m256d c2;
    double f[4];

    inline complex2() {}
    inline complex2(const __m256d& cm2) { c2 = cm2; }
    inline complex2(const complex2& cm2) { c2 = cm2.c2; }
    inline complex2(const std::complex<double>& cm1, const std::complex<double>& cm2)
    {
        c2 = _mm256_set_pd(cm2.imag(), cm2.real(), cm1.imag(), cm1.real());
    }
    inline complex2(const double& r1, const double& i1, const double& r2, const double& i2)
    {
        c2 = _mm256_set_pd(i2, r2, i1, r1);
    }
    inline std::complex<double> c(const size_t& i) const { return complex(f[i << 1U], f[(i << 1U) + 1U]); }
    inline complex2 operator+(const complex2& other) const { return _mm256_add_pd(c2, other.c2); }
    inline complex2 operator+=(const complex2& other)
    {
        c2 = _mm256_add_pd(c2, other.c2);
        return c2;
    }
    inline complex2 operator-(const complex2& other) const { return _mm256_sub_pd(c2, other.c2); }
    inline complex2 operator-=(const complex2& other)
    {
        c2 = _mm256_sub_pd(c2, other.c2);
        return c2;
    }
    inline complex2 operator*(const complex2& other) const
    {
#if ENABLE_FMA
        // FMA proposed by Elara (OpenAI custom GPT)
        return _mm256_fmadd_pd(_mm256_shuffle_pd(c2, c2, 5),
            _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
#else
        return _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(c2, c2, 5),
                                 _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15)),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
#endif
    }
    inline complex2 operator*=(const complex2& other)
    {
#if ENABLE_FMA
        // FMA proposed by Elara (OpenAI custom GPT)
        c2 = _mm256_fmadd_pd(_mm256_shuffle_pd(c2, c2, 5),
            _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
#else
        c2 = _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(c2, c2, 5),
                               _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, other.c2), other.c2, 15)),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
#endif
        return c2;
    }
    inline complex2 operator*(const double& rhs) const { return _mm256_mul_pd(c2, _mm256_set1_pd(rhs)); }
    inline complex2 operator-() const { return _mm256_mul_pd(_mm256_set1_pd(-1.0), c2); }
    inline complex2 operator*=(const double& rhs)
    {
        c2 = _mm256_mul_pd(c2, _mm256_set1_pd(rhs));
        return c2;
    }
};

inline complex2 mtrxColShuff(const complex2& mtrxCol) { return _mm256_shuffle_pd(mtrxCol.c2, mtrxCol.c2, 5); }
inline complex2 matrixMul(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxCol1Shuff,
    const complex2& mtrxCol2Shuff, const complex2& qubit)
{
    const __m256d dupeLo = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 0);
    const __m256d dupeHi = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 17);
#if ENABLE_FMA
    // FMA proposed by Elara (OpenAI custom GPT)
    return _mm256_add_pd(
        _mm256_fmadd_pd(mtrxCol1Shuff.c2, _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupeLo), dupeLo, 15),
            _mm256_mul_pd(mtrxCol1.c2, _mm256_shuffle_pd(dupeLo, dupeLo, 0))),
        _mm256_fmadd_pd(mtrxCol2Shuff.c2, _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupeHi), dupeHi, 15),
            _mm256_mul_pd(mtrxCol2.c2, _mm256_shuffle_pd(dupeHi, dupeHi, 0))));
#else
    return _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(mtrxCol1Shuff.c2, _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupeLo), dupeLo, 15)),
            _mm256_mul_pd(mtrxCol1.c2, _mm256_shuffle_pd(dupeLo, dupeLo, 0))),
        _mm256_add_pd(_mm256_mul_pd(mtrxCol2Shuff.c2, _mm256_shuffle_pd(_mm256_xor_pd(SIGNMASK, dupeHi), dupeHi, 15)),
            _mm256_mul_pd(mtrxCol2.c2, _mm256_shuffle_pd(dupeHi, dupeHi, 0))));
#endif
}
inline complex2 matrixMul(const float& nrm, const complex2& mtrxCol1, const complex2& mtrxCol2,
    const complex2& mtrxCol1Shuff, const complex2& mtrxCol2Shuff, const complex2& qubit)
{
    return matrixMul(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit) * nrm;
}
inline complex2 operator*(const double& lhs, const complex2& rhs) { return _mm256_mul_pd(_mm256_set1_pd(lhs), rhs.c2); }

inline double norm(const complex2& c)
{
    // Suggested by Elara (OpenAI custom GPT)
    const __m256d sq = _mm256_mul_pd(c.c2, c.c2);
    const __m256d sum = _mm256_hadd_pd(sq, sq);
    return _mm_cvtsd_f64(_mm_add_pd(_mm256_castpd256_pd128(sum), _mm256_extractf128_pd(sum, 1)));
}

} // namespace Qrack
