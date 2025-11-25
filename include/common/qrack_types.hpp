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

#define _USE_MATH_DEFINES

#include "config.h"

#include <cstddef>
#include <complex>
#include <functional>
#include <limits>
#include <math.h>
#include <memory>

using std::size_t;

#define IS_AMP_0(c) (norm(c) <= REAL1_EPSILON)
#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)
#define IS_SAME(c1, c2) (IS_NORM_0((c1) - (c2)))
#define IS_OPPOSITE(c1, c2) (IS_NORM_0((c1) + (c2)))

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if (FPPOW < 5) && !defined(__arm__)
#include "half.hpp"
#endif

#if QBCAPPOW < 8
#define bitLenInt uint8_t
#elif QBCAPPOW < 16
#define bitLenInt uint16_t
#elif QBCAPPOW < 32
#define bitLenInt uint32_t
#elif (QBCAPPOW < 64) || !defined(__SIZEOF_INT128__)
#define bitLenInt uint64_t
#else
#define bitLenInt unsigned __int128
#endif

#if UINTPOW < 4
#define bitCapIntOcl uint8_t
#elif UINTPOW < 5
#define bitCapIntOcl uint16_t
#elif UINTPOW < 6
#define bitCapIntOcl uint32_t
#elif (UINTPOW < 7) || !defined(__SIZEOF_INT128__)
#define bitCapIntOcl uint64_t
#else
#define bitCapIntOcl unsigned __int128
#endif

#if QBCAPPOW < 6
#define bitCapInt uint32_t
#define QRACK_MAX_QUBITS 32
#elif QBCAPPOW < 7
#define bitCapInt uint64_t
#define QRACK_MAX_QUBITS 64
#elif (QBCAPPOW < 8) && defined(__SIZEOF_INT128__)
#define bitCapInt unsigned __int128
#define QRACK_MAX_QUBITS 128
#elif BOOST_AVAILABLE
#include <boost/multiprecision/cpp_int.hpp>
typedef boost::multiprecision::cpp_int bitCapInt;
constexpr size_t QRACK_MAX_QUBITS = (1 << QBCAPPOW);
#else
#include "big_integer.hpp"
#define bitCapInt BigInteger
constexpr size_t QRACK_MAX_QUBITS = (1 << QBCAPPOW);
#endif

#if FPPOW < 5
#ifdef __arm__
namespace Qrack {
typedef __fp16 real1;
typedef float real1_f;
typedef float real1_s;
#else
#if (CPP_STD >= 23) && __has_include(<stdfloat>)
#include <stdfloat>
#endif
#if defined(__STDCPP_FLOAT16_T__)
namespace Qrack {
typedef float16_t real1;
typedef float real1_f;
typedef float real1_s;
#else
#include "half.hpp"
namespace Qrack {
typedef half_float::half real1;
typedef float real1_f;
typedef float real1_s;
#endif
#endif
#elif FPPOW < 6
namespace Qrack {
typedef float real1;
typedef float real1_f;
typedef float real1_s;
#elif FPPOW < 7
namespace Qrack {
typedef double real1;
typedef double real1_f;
typedef double real1_s;
#else
#if (CPP_STD >= 23) && __has_include(<stdfloat>)
#include <stdfloat>
#endif
#if defined(__STDCPP_FLOAT128_T__)
namespace Qrack {
typedef float128_t real1;
typedef float128_t real1_f;
typedef double real1_s;
#else
#include <boost/multiprecision/float128.hpp>
#include <quadmath.h>
namespace Qrack {
typedef boost::multiprecision::float128 real1;
typedef boost::multiprecision::float128 real1_f;
typedef double real1_s;
#endif
#endif

typedef std::complex<real1> complex;
const bitCapInt ONE_BCI = 1U;
const bitCapInt ZERO_BCI = 0U;
constexpr bitLenInt bitsInCap = ((bitLenInt)1U) << ((bitLenInt)QBCAPPOW);

typedef std::shared_ptr<complex> BitOp;

// Called once per value between begin and end.
typedef std::function<void(const bitCapIntOcl&, const unsigned& cpu)> ParallelFunc;
typedef std::function<bitCapIntOcl(const bitCapIntOcl&)> IncrementFunc;
typedef std::function<bitCapInt(const bitCapInt&)> BdtFunc;
typedef std::function<void(const bitCapInt&, const unsigned& cpu)> ParallelFuncBdt;

class StateVector;
class StateVectorArray;
class StateVectorSparse;

typedef std::shared_ptr<StateVector> StateVectorPtr;
typedef std::shared_ptr<StateVectorArray> StateVectorArrayPtr;
typedef std::shared_ptr<StateVectorSparse> StateVectorSparsePtr;

typedef std::function<void(void)> DispatchFn;

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

#define bitsInByte 8U
#define qrack_rand_gen std::mt19937_64
#define qrack_rand_gen_ptr std::shared_ptr<qrack_rand_gen>
#define QRACK_ALIGN_SIZE 64U

#if FPPOW < 6
#define ZERO_R1_F 0.0f
#define QUARTER_R1_F 0.25f
#define HALF_R1_F 0.5f
#define ONE_R1_F 1.0f
#else
#define ZERO_R1_F 0.0
#define QUARTER_R1_F 0.25
#define HALF_R1_F 0.5
#define ONE_R1_F 1.0
#endif

#if (FPPOW > 4) || defined(__arm__)
#define QRACK_CONST constexpr
#else
#define QRACK_CONST const
#endif

#define REAL1_DEFAULT_ARG -999.0f
QRACK_CONST real1 PI_R1 = (real1)M_PI;
QRACK_CONST real1 SQRT2_R1 = (real1)M_SQRT2;
QRACK_CONST real1 SQRT1_2_R1 = (real1)M_SQRT1_2;

#if (FPPOW < 5) || (FPPOW > 6)
QRACK_CONST real1 ZERO_R1 = (real1)0.0f;
QRACK_CONST real1 HALF_R1 = (real1)0.5f;
QRACK_CONST real1 ONE_R1 = (real1)1.0f;
#elif FPPOW == 5
#define ZERO_R1 0.0f
#define QUARTER_R1 0.25f
#define HALF_R1 0.5f
#define ONE_R1 1.0f
#else
#define ZERO_R1 0.0
#define QUARTER_R1 0.25
#define HALF_R1 0.5
#define ONE_R1 1.0
#endif

#if FPPOW < 5
// Half the probability in any single permutation of 20 maximally superposed qubits
QRACK_CONST real1 REAL1_EPSILON = (real1)0.000000477f;
#elif FPPOW < 6
// Half the probability in any single permutation of 48 maximally superposed qubits
#define REAL1_EPSILON 1.7763568394002505e-15f
#elif FPPOW < 7
// Half the probability in any single permutation of 96 maximally superposed qubits
#define REAL1_EPSILON 6.310887241768095e-30
#else
// Half the probability in any single permutation of 192 maximally superposed qubits
QRACK_CONST real1 REAL1_EPSILON = (real1)7.965459555662261e-59;
#endif

#if ENABLE_CUDA
#if FPPOW < 5
#include <cuda_fp16.h>
#define qCudaReal1 __half
#define qCudaReal2 __half2
#define qCudaReal4 __half2*
#define qCudaCmplx __half2
#define qCudaCmplx2 __half2*
#define qCudaReal1_f float
#define make_qCudaCmplx make_half2
#define ZERO_R1_CUDA ((qCudaReal1)0.0f)
#define REAL1_EPSILON_CUDA ((qCudaReal1)0.000000477f)
#define PI_R1_CUDA M_PI
#elif FPPOW < 6
#define qCudaReal1 float
#define qCudaReal2 float2
#define qCudaReal4 float4
#define qCudaCmplx float2
#define qCudaCmplx2 float4
#define qCudaReal1_f float
#define make_qCudaCmplx make_float2
#define make_qCudaCmplx2 make_float4
#define ZERO_R1_CUDA 0.0f
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#else
#define qCudaReal1 double
#define qCudaReal2 double2
#define qCudaReal4 double4
#define qCudaCmplx double2
#define qCudaCmplx2 double4
#define qCudaReal1_f double
#define make_qCudaCmplx make_double2
#define make_qCudaCmplx2 make_double4
#define ZERO_R1_CUDA 0.0
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#endif
#endif

constexpr size_t SPARSE_KEY_BYTES = sizeof(bitCapIntOcl) + sizeof(complex);

QRACK_CONST complex ONE_CMPLX = complex(ONE_R1, ZERO_R1);
QRACK_CONST complex ZERO_CMPLX = complex(ZERO_R1, ZERO_R1);
QRACK_CONST complex I_CMPLX = complex(ZERO_R1, ONE_R1);
QRACK_CONST complex HALF_I_HALF_CMPLX = complex(HALF_R1, HALF_R1);
QRACK_CONST complex HALF_NEG_I_HALF_CMPLX = complex(HALF_R1, -HALF_R1);
QRACK_CONST complex CMPLX_DEFAULT_ARG = complex((real1)REAL1_DEFAULT_ARG, (real1)REAL1_DEFAULT_ARG);
QRACK_CONST real1 FP_NORM_EPSILON = (real1)(std::numeric_limits<real1>::epsilon() / 4);
QRACK_CONST real1_f FP_NORM_EPSILON_F = (real1_f)FP_NORM_EPSILON;
QRACK_CONST real1_f TRYDECOMPOSE_EPSILON = 32 * FP_NORM_EPSILON_F;
const double FIDELITY_MIN = log((double)FP_NORM_EPSILON);
} // namespace Qrack
