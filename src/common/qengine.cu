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

#include "common/cuda_kernels.cuh"

namespace Qrack {

__device__ inline qCudaCmplx zmul(const qCudaCmplx lhs, const qCudaCmplx rhs)
{
    return make_qCudaCmplx((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

#if FPPOW > 4
__device__ inline qCudaCmplx2 zmatrixmul(const qCudaReal1 nrm, const qCudaReal1* lhs, const qCudaCmplx2 rhs)
{
    return (make_qCudaCmplx2(nrm * ((lhs[0] * rhs.x) - (lhs[1] * rhs.y) + (lhs[2] * rhs.z) - (lhs[3] * rhs.w)),
        nrm * ((lhs[0] * rhs.y) + (lhs[1] * rhs.x) + (lhs[2] * rhs.w) + (lhs[3] * rhs.z)),
        nrm * ((lhs[4] * rhs.x) - (lhs[5] * rhs.y) + (lhs[6] * rhs.z) - (lhs[7] * rhs.w)),
        nrm * ((lhs[4] * rhs.y) + (lhs[5] * rhs.x) + (lhs[6] * rhs.w) + (lhs[7] * rhs.z))));
}
#else
__device__ inline void zmatrixmul(const qCudaReal1 nrm, const qCudaReal1* lhs, const qCudaCmplx2 rhs, qCudaCmplx2 o)
{
    o[0] =
        (make_qCudaCmplx(nrm * ((lhs[0] * rhs[0].x) - (lhs[1] * rhs[0].y) + (lhs[2] * rhs[1].x) - (lhs[3] * rhs[1].y)),
            nrm * ((lhs[0] * rhs[0].y) + (lhs[1] * rhs[0].x) + (lhs[2] * rhs[1].y) + (lhs[3] * rhs[1].x))));
    o[1] =
        (make_qCudaCmplx(nrm * ((lhs[4] * rhs[0].x) - (lhs[5] * rhs[0].y) + (lhs[6] * rhs[1].x) - (lhs[7] * rhs[1].y)),
            nrm * ((lhs[4] * rhs[0].y) + (lhs[5] * rhs[0].x) + (lhs[6] * rhs[1].y) + (lhs[7] * rhs[1].x))));
}
#endif

__device__ inline qCudaReal1 qCudaArg(const qCudaCmplx cmp)
{
    if (cmp.x == (qCudaReal1)0.0f && cmp.y == (qCudaReal1)0.0f)
        return (qCudaReal1)0.0f;
    return (qCudaReal1)atan2((qCudaReal1_f)cmp.y, (qCudaReal1_f)cmp.x);
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal2 a, qCudaReal2 b) { return a.x * b.x + a.y * b.y; }

__device__ inline qCudaReal1 qCudaDot(qCudaReal4 a, qCudaReal4 b)
{
#if FPPOW > 4
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
#else
    return a[0].x * b[0].x + a[0].y * b[0].y + a[1].x * b[1].x + a[1].y * b[1].y;
#endif
}

#if FPPOW > 4
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta) { return make_qCudaCmplx(cos(theta), sin(theta)); }
#else
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta)
{
    return make_qCudaCmplx((qCudaReal1)cos((qCudaReal1_f)theta), (qCudaReal1)sin((qCudaReal1_f)theta));
}
#endif

__device__ inline qCudaCmplx qCudaConj(qCudaCmplx a) { return make_qCudaCmplx(a.x, -a.y); }

#define OFFSET2_ARG bitCapIntOclPtr[0]
#define OFFSET1_ARG bitCapIntOclPtr[1]
#define MAXI_ARG bitCapIntOclPtr[2]
#define BITCOUNT_ARG bitCapIntOclPtr[3]
#define ID (blockIdx.x * blockDim.x + threadIdx.x)

#define PREP_2X2()                                                                                                     \
    bitCapIntOcl Nthreads = gridDim.x * blockDim.x;                                                                    \
                                                                                                                       \
    qCudaReal1* mtrx = qCudaCmplxPtr;                                                                                  \
    qCudaReal1 nrm = qCudaCmplxPtr[8];

#define PREP_2X2_WIDE()                                                                                                \
    qCudaReal1* mtrx = qCudaCmplxPtr;                                                                                  \
    qCudaReal1 nrm = qCudaCmplxPtr[8];

#define PREP_2X2_NORM() qCudaReal1 norm_thresh = qCudaCmplxPtr[9];

#define PUSH_APART_GEN()                                                                                               \
    bitCapIntOcl i = 0U;                                                                                               \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    for (bitLenInt p = 0U; p < BITCOUNT_ARG; p++) {                                                                    \
        bitCapIntOcl iLow = iHigh & (qPowersSorted[p] - 1U);                                                           \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << 1U;                                                                                  \
    }                                                                                                                  \
    i |= iHigh;

#define PUSH_APART_1()                                                                                                 \
    bitCapIntOcl i = lcv & qMask;                                                                                      \
    i |= (lcv ^ i) << 1U;

#define PUSH_APART_2()                                                                                                 \
    bitCapIntOcl i = lcv & qMask1;                                                                                     \
    bitCapIntOcl iHigh = (lcv ^ i) << 1U;                                                                              \
    bitCapIntOcl iLow = iHigh & qMask2;                                                                                \
    i |= iLow | ((iHigh ^ iLow) << 1U);

#if FPPOW > 4
#define APPLY_AND_OUT()                                                                                                \
    qCudaCmplx2 mulRes = make_qCudaCmplx2(stateVec[i | OFFSET1_ARG].x, stateVec[i | OFFSET1_ARG].y,                    \
        stateVec[i | OFFSET2_ARG].x, stateVec[i | OFFSET2_ARG].y);                                                     \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = make_qCudaCmplx(mulRes.x, mulRes.y);                                                   \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(mulRes.z, mulRes.w);
#else
#define APPLY_AND_OUT()                                                                                                \
    __half2 mulIn[2]{ stateVec[i | OFFSET1_ARG], stateVec[i | OFFSET2_ARG] };                                          \
                                                                                                                       \
    __half2 mulRes[2];                                                                                                 \
    zmatrixmul(nrm, mtrx, mulIn, mulRes);                                                                              \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes[0];                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes[1];
#endif

#define APPLY_X()                                                                                                      \
    const qCudaCmplx Y0 = stateVec[i];                                                                                 \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                                                           \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z()                                                                                                      \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(-stateVec[i | OFFSET2_ARG].x, -stateVec[i | OFFSET2_ARG].y);

#define APPLY_PHASE()                                                                                                  \
    stateVec[i] = zmul(topLeft, stateVec[i]);                                                                          \
    stateVec[i | OFFSET2_ARG] = zmul(bottomRight, stateVec[i | OFFSET2_ARG]);

#define APPLY_INVERT()                                                                                                 \
    const qCudaCmplx Y0 = stateVec[i];                                                                                 \
    stateVec[i] = zmul(topRight, stateVec[i | OFFSET2_ARG]);                                                           \
    stateVec[i | OFFSET2_ARG] = zmul(bottomLeft, Y0);

#if FPPOW > 4
#define NORM_BODY_2X2()                                                                                                \
    qCudaCmplx mulResLo = stateVec[i | OFFSET1_ARG];                                                                   \
    qCudaCmplx mulResHi = stateVec[i | OFFSET2_ARG];                                                                   \
    qCudaCmplx2 mulRes = make_qCudaCmplx2(mulResLo.x, mulResLo.y, mulResHi.x, mulResHi.y);                             \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    qCudaCmplx mulResPart = make_qCudaCmplx(mulRes.x, mulRes.y);                                                       \
                                                                                                                       \
    qCudaReal1 dotMulRes = qCudaDot(mulResPart, mulResPart);                                                           \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.x = ZERO_R1;                                                                                            \
        mulRes.y = ZERO_R1;                                                                                            \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    mulResPart = make_qCudaCmplx(mulRes.z, mulRes.w);                                                                  \
                                                                                                                       \
    dotMulRes = qCudaDot(mulResPart, mulResPart);                                                                      \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.z = ZERO_R1;                                                                                            \
        mulRes.w = ZERO_R1;                                                                                            \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = make_qCudaCmplx(mulRes.x, mulRes.y);                                                   \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(mulRes.z, mulRes.w);
#else
#define NORM_BODY_2X2()                                                                                                \
    qCudaCmplx mulResLo = stateVec[i | OFFSET1_ARG];                                                                   \
    qCudaCmplx mulResHi = stateVec[i | OFFSET2_ARG];                                                                   \
    __half2 mulIn[2]{ mulResLo, mulResHi };                                                                            \
                                                                                                                       \
    __half2 mulRes[2];                                                                                                 \
    zmatrixmul(nrm, mtrx, mulIn, mulRes);                                                                              \
                                                                                                                       \
    qCudaCmplx mulResPart = mulRes[0];                                                                                 \
                                                                                                                       \
    qCudaReal1 dotMulRes = qCudaDot(mulResPart, mulResPart);                                                           \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes[0].x = ZERO_R1_CUDA;                                                                                    \
        mulRes[0].y = ZERO_R1_CUDA;                                                                                    \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    mulResPart = mulRes[1];                                                                                            \
                                                                                                                       \
    dotMulRes = qCudaDot(mulResPart, mulResPart);                                                                      \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes[1].x = ZERO_R1_CUDA;                                                                                    \
        mulRes[1].y = ZERO_R1_CUDA;                                                                                    \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes[0];                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes[1];
#endif

#define SUM_LOCAL(part)                                                                                                \
    extern __shared__ qCudaReal1 lBuffer[];                                                                            \
    const bitCapIntOcl locID = threadIdx.x;                                                                            \
    const bitCapIntOcl locNthreads = blockDim.x;                                                                       \
    lBuffer[locID] = part;                                                                                             \
                                                                                                                       \
    for (bitCapIntOcl lcv = (locNthreads >> 1U); lcv > 0U; lcv >>= 1U) {                                               \
        __syncthreads();                                                                                               \
        if (locID < lcv) {                                                                                             \
            lBuffer[locID] += lBuffer[locID + lcv];                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    if (locID == 0U) {                                                                                                 \
        sumBuffer[blockIdx.x] = lBuffer[0];                                                                            \
    }

__global__ void apply2x2(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted)
{
    PREP_2X2()

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_GEN()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2single(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2double(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_2()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2wide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl lcv = ID;

    PUSH_APART_GEN()
    APPLY_AND_OUT()
}

__global__ void apply2x2singlewide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_1()
    APPLY_AND_OUT()
}

__global__ void apply2x2doublewide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_2()
    APPLY_AND_OUT()
}

__global__ void apply2x2normsingle(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    PREP_2X2()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    qCudaReal1 partNrm = ZERO_R1_CUDA;
    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        NORM_BODY_2X2()
    }

    SUM_LOCAL(partNrm)
}

__global__ void apply2x2normsinglewide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    PREP_2X2_WIDE()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    qCudaReal1 partNrm = ZERO_R1_CUDA;
    PUSH_APART_1()
    NORM_BODY_2X2()

    SUM_LOCAL(partNrm)
}

__global__ void xsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_X()
    }
}

__global__ void xsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_X()
}

__global__ void xmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{

    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl setInt = lcv & mask;
        bitCapIntOcl resetInt = setInt ^ mask;

        if (setInt < resetInt) {
            continue;
        }

        setInt |= otherRes;
        resetInt |= otherRes;

        const qCudaCmplx Y0 = stateVec[resetInt];
        stateVec[resetInt] = stateVec[setInt];
        stateVec[setInt] = Y0;
    }
}

__global__ void phaseparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplxPtr)
{
    const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const qCudaCmplx phaseFac = qCudaCmplxPtr[0];
    const qCudaCmplx iPhaseFac = qCudaCmplxPtr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl setInt = lcv & mask;

        bitCapIntOcl v = setInt;
        for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        setInt |= lcv & otherMask;

        stateVec[setInt] = zmul(v ? phaseFac : iPhaseFac, stateVec[setInt]);
    }
}

__global__ void phasemask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl nPhases = bitCapIntOclPtr[2];
    const qCudaReal1 phaseAngle = (qCudaReal1)(-PI_R1_CUDA / bitCapIntOclPtr[3]);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl popCount = 0;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            popCount += v & 1;
            v >>= 1;
        }

        const bitCapIntOcl nPhaseSteps = popCount % nPhases;
        if (nPhaseSteps) {
#if FPPOW > 4
            stateVec[lcv] = zmul(polar_unit(nPhaseSteps * phaseAngle), stateVec[lcv]);
#else
            stateVec[lcv] = zmul(polar_unit(((qCudaReal1)(unsigned long long)nPhaseSteps) * phaseAngle), stateVec[lcv]);
#endif
        }
    }
}

__global__ void zsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_Z()
    }
}

__global__ void zsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_Z()
}

__global__ void phasesingle(qCudaCmplx* stateVec, qCudaCmplx* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const qCudaCmplx topLeft = qCudaCmplxPtr[0];
    const qCudaCmplx bottomRight = qCudaCmplxPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_PHASE()
    }
}

__global__ void phasesinglewide(qCudaCmplx* stateVec, qCudaCmplx* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const qCudaCmplx topLeft = qCudaCmplxPtr[0];
    const qCudaCmplx bottomRight = qCudaCmplxPtr[3];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_PHASE()
}

__global__ void invertsingle(qCudaCmplx* stateVec, qCudaCmplx* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const qCudaCmplx topRight = qCudaCmplxPtr[1];
    const qCudaCmplx bottomLeft = qCudaCmplxPtr[2];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_INVERT()
    }
}

__global__ void invertsinglewide(qCudaCmplx* stateVec, qCudaCmplx* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const qCudaCmplx topRight = qCudaCmplxPtr[1];
    const qCudaCmplx bottomLeft = qCudaCmplxPtr[2];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_INVERT()
}

__global__ void uniformlycontrolled(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowers, qCudaReal1* mtrxs, qCudaReal1* nrmIn)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl targetPower = bitCapIntOclPtr[1];
    const bitCapIntOcl targetMask = targetPower - 1U;
    const bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    const bitCapIntOcl mtrxSkipLen = bitCapIntOclPtr[3];
    const bitCapIntOcl mtrxSkipValueMask = bitCapIntOclPtr[4];
    const qCudaReal1 nrm = nrmIn[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & targetMask;
        i |= (lcv ^ i) << 1U;

        bitCapIntOcl offset = 0;
        for (bitLenInt p = 0; p < controlLen; p++) {
            if (i & qPowers[p]) {
                offset |= 1U << p;
            }
        }

        bitCapIntOcl jHigh = offset;
        bitCapIntOcl j = 0;
        for (bitCapIntOcl p = 0; p < mtrxSkipLen; p++) {
            const bitCapIntOcl jLow = jHigh & (qPowers[controlLen + p] - 1U);
            j |= jLow;
            jHigh = (jHigh ^ jLow) << 1U;
        }
        j |= jHigh;
        offset = j | mtrxSkipValueMask;

        const qCudaCmplx qubitLo = stateVec[i];
        const qCudaCmplx qubitHi = stateVec[i | targetPower];
#if FPPOW > 4
        qCudaCmplx2 qubit = make_qCudaCmplx2(qubitLo.x, qubitLo.y, qubitHi.x, qubitHi.y);
        qubit = zmatrixmul(nrm, mtrxs + (offset * 8U), qubit);
#else
        __half2 qubitIn[2]{ qubitLo, qubitHi };
        __half2 qubit[2];
        zmatrixmul(nrm, mtrxs + (offset * 8U), qubitIn, qubit);
#endif

#if FPPOW > 4
        stateVec[i] = make_qCudaCmplx(qubit.x, qubit.y);
        stateVec[i | targetPower] = make_qCudaCmplx(qubit.z, qubit.w);
#else
        stateVec[i] = qubit[0];
        stateVec[i | targetPower] = qubit[1];
#endif
    }
}

__global__ void uniformparityrz(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const qCudaCmplx phaseFac = qCudaCmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = qCudaCmplx_ptr[1];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - 1U;
        }
        stateVec[lcv] = zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

__global__ void uniformparityrznorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const qCudaCmplx phaseFac = qCudaCmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = qCudaCmplx_ptr[1];
    const qCudaCmplx nrm = qCudaCmplx_ptr[2];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - 1U;
        }
        stateVec[lcv] = zmul(nrm, zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj)));
    }
}

__global__ void cuniformparityrz(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr, bitCapIntOcl* qPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const bitCapIntOcl cMask = bitCapIntOclPtr[2];
    const bitLenInt cLen = (bitLenInt)bitCapIntOclPtr[3];
    const qCudaCmplx phaseFac = qCudaCmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = qCudaCmplx_ptr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < cLen; p++) {
            bitCapIntOcl iLow = iHigh & (qPowers[p] - 1U);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1U;
        }
        i |= iHigh | cMask;

        bitCapIntOcl perm = i & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - 1U;
        }
        stateVec[i] = zmul(stateVec[i], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

__global__ void compose(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl nMaxQPower = bitCapIntOclPtr[0];
    const bitCapIntOcl qubitCount = bitCapIntOclPtr[1];
    const bitCapIntOcl startMask = bitCapIntOclPtr[2];
    const bitCapIntOcl endMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & startMask], stateVec2[(lcv & endMask) >> qubitCount]);
    }
}

__global__ void composewide(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl lcv = ID;
    // const bitCapIntOcl nMaxQPower = bitCapIntOclPtr[0];
    const bitCapIntOcl qubitCount = bitCapIntOclPtr[1];
    const bitCapIntOcl startMask = bitCapIntOclPtr[2];
    const bitCapIntOcl endMask = bitCapIntOclPtr[3];

    nStateVec[lcv] = zmul(stateVec1[lcv & startMask], stateVec2[(lcv & endMask) >> qubitCount]);
}

__global__ void composemid(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl nMaxQPower = bitCapIntOclPtr[0];
    // const bitCapIntOcl qubitCount = bitCapIntOclPtr[1];
    const bitCapIntOcl oQubitCount = bitCapIntOclPtr[2];
    const bitCapIntOcl startMask = bitCapIntOclPtr[3];
    const bitCapIntOcl midMask = bitCapIntOclPtr[4];
    const bitCapIntOcl endMask = bitCapIntOclPtr[5];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[6];

    for (bitCapIntOcl lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] =
            zmul(stateVec1[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)], stateVec2[(lcv & midMask) >> start]);
    }
}

__global__ void decomposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* remainderStateProb,
    qCudaReal1* remainderStateAngle, qCudaReal1* partStateProb, qCudaReal1* partStateAngle)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl partPower = bitCapIntOclPtr[0];
    const bitCapIntOcl remainderPower = bitCapIntOclPtr[1];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl startMask = (1U << start) - 1U;
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        qCudaReal1 partProb = ZERO_R1_CUDA;

        for (bitCapIntOcl k = 0U; k < partPower; ++k) {
            const qCudaCmplx amp = stateVec[j | (k << start)];
            const qCudaReal1 nrm = (qCudaReal1_f)qCudaDot(amp, amp);
            partProb += nrm;
            if (nrm > REAL1_EPSILON_CUDA) {
                partStateAngle[k] += qCudaArg(amp) * (qCudaReal1)nrm;
            }
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        qCudaReal1 partProb = ZERO_R1_CUDA;

        for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
            bitCapIntOcl l = k & startMask;
            l |= j | ((k ^ l) << len);

            const qCudaCmplx amp = stateVec[l];
            const qCudaReal1 nrm = (qCudaReal1_f)qCudaDot(amp, amp);
            partProb += nrm;
            if (nrm > REAL1_EPSILON_CUDA) {
                remainderStateAngle[k] += qCudaArg(amp) * (qCudaReal1)nrm;
            }
        }

        if (partProb > REAL1_EPSILON_CUDA) {
            partStateAngle[lcv] /= partProb;
        }
        partStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const qCudaReal1 prob = remainderStateProb[lcv];
        if (prob > REAL1_EPSILON_CUDA) {
            remainderStateAngle[lcv] /= prob;
        }
    }
}

__global__ void decomposeamp(
    qCudaReal1* stateProb, qCudaReal1* stateAngle, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        const qCudaReal1 angle = stateAngle[lcv];
        const qCudaReal1 probSqrt = (qCudaReal1)sqrt((qCudaReal1_f)stateProb[lcv]);
        nStateVec[lcv] = make_qCudaCmplx(
            probSqrt * (qCudaReal1)cos((qCudaReal1_f)angle), probSqrt * (qCudaReal1)sin((qCudaReal1_f)angle));
    }
}

__global__ void disposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* remainderStateProb,
    qCudaReal1* remainderStateAngle)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl partPower = bitCapIntOclPtr[0];
    const bitCapIntOcl remainderPower = bitCapIntOclPtr[1];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl startMask = (1U << start) - 1U;
    const bitLenInt len = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        qCudaReal1 partProb = ZERO_R1_CUDA;

        for (bitCapIntOcl k = 0U; k < partPower; ++k) {
            qCudaCmplx amp = stateVec[j | (k << start)];
            qCudaReal1 nrm = qCudaDot(amp, amp);
            partProb += nrm;
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
            bitCapIntOcl l = k & startMask;
            l |= j | ((k ^ l) << len);

            const qCudaCmplx amp = stateVec[l];
            const qCudaReal1 nrm = (qCudaReal1_f)qCudaDot(amp, amp);
            if (nrm > REAL1_EPSILON_CUDA) {
                remainderStateAngle[k] += qCudaArg(amp) * (qCudaReal1)nrm;
            }
        }
    }

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const qCudaReal1 prob = remainderStateProb[lcv];
        if (prob > REAL1_EPSILON_CUDA) {
            remainderStateAngle[lcv] /= prob;
        }
    }
}

__global__ void dispose(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl remainderPower = bitCapIntOclPtr[0];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[2];
    const bitCapIntOcl disposedRes = bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & skipMask;
        bitCapIntOcl i = iLow | ((lcv ^ iLow) << (bitCapIntOcl)len) | disposedRes;
        nStateVec[lcv] = stateVec[i];
    }
}

__global__ void prob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qPower = bitCapIntOclPtr[1];
    const bitCapIntOcl qMask = qPower - 1U;

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << 1U) | qPower;
        const qCudaCmplx amp = stateVec[i];
        oneChancePart += qCudaDot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void cprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qPower = bitCapIntOclPtr[1];
    const bitCapIntOcl qControlPower = bitCapIntOclPtr[2];
    const bitCapIntOcl qControlMask = bitCapIntOclPtr[3];
    bitCapIntOcl qMask1, qMask2;
    if (qPower < qControlPower) {
        qMask1 = qPower - 1U;
        qMask2 = qControlPower - 1U;
    } else {
        qMask1 = qControlPower - 1U;
        qMask2 = qPower - 1U;
    }

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2()
        i |= qPower | qControlMask;
        const qCudaCmplx amp = stateVec[i];
        oneChancePart += qCudaDot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl perm = bitCapIntOclPtr[1];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[2];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl qMask = (1U << start) - 1U;

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << len);
        const qCudaCmplx amp = stateVec[i | perm];
        oneChancePart += qCudaDot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probregall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl maxJ = bitCapIntOclPtr[1];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[2];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl qMask = (1U << start) - 1U;

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        const bitCapIntOcl perm = lcv1 << start;
        qCudaReal1 oneChancePart = ZERO_R1_CUDA;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            bitCapIntOcl i = lcv2 & qMask;
            i |= ((lcv2 ^ i) << len);
            qCudaCmplx amp = stateVec[i | perm];
            oneChancePart += qCudaDot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

__global__ void probmask(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer, bitCapIntOcl* qPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    // const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl perm = bitCapIntOclPtr[2];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[3];

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < len; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowers[p] - 1U);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1U;
        }
        i |= iHigh;

        const qCudaCmplx amp = stateVec[i | perm];
        oneChancePart += qCudaDot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probmaskall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    bitCapIntOcl* qPowersMask, bitCapIntOcl* qPowersSkip)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl maxJ = bitCapIntOclPtr[1];
    const bitLenInt maskLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitLenInt skipLen = (bitLenInt)bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        bitCapIntOcl iHigh = lcv1;
        bitCapIntOcl perm = 0U;
        for (bitLenInt p = 0U; p < skipLen; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowersSkip[p] - 1U);
            perm |= iLow;
            iHigh = (iHigh ^ iLow) << 1U;
        }
        perm |= iHigh;

        qCudaReal1 oneChancePart = ZERO_R1_CUDA;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            iHigh = lcv2;
            bitCapIntOcl i = 0U;
            for (bitLenInt p = 0U; p < maskLen; p++) {
                bitCapIntOcl iLow = iHigh & (qPowersMask[p] - 1U);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << 1U;
            }
            i |= iHigh;

            const qCudaCmplx amp = stateVec[i | perm];
            oneChancePart += qCudaDot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

__global__ void probparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{

    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - 1U);
        }

        if (parity) {
            const qCudaCmplx amp = stateVec[lcv];
            oneChancePart += qCudaDot(amp, amp);
        }
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void forcemparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bool result = (bitCapIntOclPtr[2] == 1U);

    qCudaReal1 oneChancePart = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - 1U);
        }

        if (parity == result) {
            const qCudaCmplx amp = stateVec[lcv];
            oneChancePart += qCudaDot(amp, amp);
        } else {
            stateVec[lcv] = make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);
        }
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void expperm(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* bitPowers, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl offset = bitCapIntOclPtr[2];

    qCudaReal1 expectation = 0;
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl retIndex = 0;
        for (bitLenInt p = 0; p < len; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= (1U << p);
            }
        }
        const qCudaCmplx amp = stateVec[lcv];
        expectation += (qCudaReal1)((offset + retIndex) * (qCudaReal1_f)qCudaDot(amp, amp));
    }

    SUM_LOCAL(expectation)
}

__global__ void nrmlze(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const qCudaReal1 norm_thresh = args_ptr[0].x;
    const qCudaCmplx nrm = args_ptr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        qCudaCmplx amp = stateVec[lcv];
        if (qCudaDot(amp, amp) < norm_thresh) {
            amp = make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);
        }
        stateVec[lcv] = zmul(nrm, amp);
    }
}

__global__ void nrmlzewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr)
{
    const bitCapIntOcl lcv = ID;
    const qCudaReal1 norm_thresh = args_ptr[0].x;
    const qCudaCmplx nrm = args_ptr[1];

    qCudaCmplx amp = stateVec[lcv];
    if (qCudaDot(amp, amp) < norm_thresh) {
        amp = make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);
    }
    stateVec[lcv] = zmul(nrm, amp);
}

__global__ void updatenorm(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const qCudaReal1 norm_thresh = args_ptr[0];
    qCudaReal1 partNrm = ZERO_R1_CUDA;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const qCudaCmplx amp = stateVec[lcv];
        qCudaReal1 nrm = qCudaDot(amp, amp);
        if (nrm < norm_thresh) {
            nrm = ZERO_R1_CUDA;
        }
        partNrm += nrm;
    }

    SUM_LOCAL(partNrm)
}

__global__ void approxcompare(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* sumBuffer)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    qCudaCmplx partInner = make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const qCudaCmplx prod = zmul(qCudaConj(stateVec1[lcv]), stateVec2[lcv]);
        partInner = make_qCudaCmplx(partInner.x + prod.x, partInner.y + prod.y);
    }

    extern __shared__ qCudaCmplx lCmplxBuffer[];
    const bitCapIntOcl locID = threadIdx.x;
    const bitCapIntOcl locNthreads = blockDim.x;
    lCmplxBuffer[locID] = partInner;

    for (bitCapIntOcl lcv = (locNthreads >> 1U); lcv > 0U; lcv >>= 1U) {
        __syncthreads();
        if (locID < lcv) {
            const qCudaCmplx a = lCmplxBuffer[locID];
            const qCudaCmplx b = lCmplxBuffer[locID + lcv];
            lCmplxBuffer[locID] = make_qCudaCmplx(a.x + b.x, a.y + b.y);
        }
    }
    if (locID == 0U) {
        sumBuffer[blockIdx.x] = lCmplxBuffer[0];
    }
}

__global__ void applym(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qPower = bitCapIntOclPtr[1];
    const bitCapIntOcl qMask = qPower - 1U;
    const bitCapIntOcl savePower = bitCapIntOclPtr[2];
    const bitCapIntOcl discardPower = qPower ^ savePower;
    const qCudaCmplx nrm = qCudaCmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & qMask;
        const bitCapIntOcl i = iLow | ((lcv ^ iLow) << 1U);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);
    }
}

__global__ void applymreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl result = bitCapIntOclPtr[2];
    const qCudaCmplx nrm = qCudaCmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] =
            ((lcv & mask) == result) ? zmul(nrm, stateVec[lcv]) : make_qCudaCmplx(ZERO_R1_CUDA, ZERO_R1_CUDA);
    }
}

__global__ void shufflebuffers(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl halfMaxI = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < halfMaxI; lcv += Nthreads) {
        const qCudaCmplx amp0 = stateVec1[lcv + halfMaxI];
        stateVec1[lcv + halfMaxI] = stateVec2[lcv];
        stateVec2[lcv] = amp0;
    }
}

__global__ void rol(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - 1U;
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[4];
    const bitLenInt shift = (bitLenInt)bitCapIntOclPtr[5];
    const bitLenInt length = (bitLenInt)bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl regInt = (lcv & regMask) >> start;
        nStateVec[lcv] =
            stateVec[((((regInt >> shift) | (regInt << (length - shift))) & lengthMask) << start) | (lcv & otherMask)];
    }
}

#if ENABLE_ALU
__global__ void inc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - 1U;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[4];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    for (bitCapIntOcl i = ID; i < maxI; i += Nthreads) {
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | (i & otherMask)] =
            stateVec[i];
    }
}

__global__ void cinc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - 1U;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[4];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[7];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < controlLen; p++) {
            bitCapIntOcl iLow = iHigh & (controlPowers[p] - 1U);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1U;
        }
        i |= iHigh;

        bitCapIntOcl otherRes = i & otherMask;
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | otherRes | controlMask] =
            stateVec[i | controlMask];
    }
}

__global__ void incdecc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - 1U;
    const bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toMod = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - 1U);
        i |= (lcv ^ i) << 1U;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl outInt = (inOutRes >> inOutStart) + toMod;
        bitCapIntOcl outRes = 0U;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = carryMask;
        }
        outRes |= outInt << inOutStart;
        nStateVec[outRes | otherRes] = stateVec[i];
    }
}

__global__ void incs(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> 1U;
    const bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutRes = lcv & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? (outRes = (outInt << inOutStart) | otherRes)
                                                     : (((outInt - lengthPower) << inOutStart) | otherRes);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1U)) + 1U;
            inInt = ((~inInt) & (lengthPower - 1U)) + 1U;
            if ((inOutInt + inInt) > signMask) {
                isOverflow = true;
            }
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask) {
                isOverflow = true;
            }
        }
        qCudaCmplx amp = stateVec[lcv];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = make_qCudaCmplx(-amp.x, -amp.y);
        }
        nStateVec[outRes] = amp;
    }
}

__global__ void incdecsc1(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> 1U;
    const bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[7];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - 1U);
        i |= (lcv ^ i) << 1U;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? (outRes = (outInt << inOutStart) | otherRes)
                                                     : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1U)) + 1U;
            inInt = ((~inInt) & (lengthPower - 1U)) + 1U;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        qCudaCmplx amp = stateVec[i];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = make_qCudaCmplx(-amp.x, -amp.y);
        }
        nStateVec[outRes] = amp;
    }
}

__global__ void incdecsc2(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> 1U;
    const bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - 1U);
        i |= (lcv ^ i) << 1U;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? ((outInt << inOutStart) | otherRes)
                                                     : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1U)) + 1U;
            inInt = ((~inInt) & (lengthPower - 1U)) + 1U;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        qCudaCmplx amp = stateVec[i];
        if (isOverflow) {
            amp = make_qCudaCmplx(-amp.x, -amp.y);
        }
        nStateVec[outRes] = amp;
    }
}

__global__ void mul(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (1U << len) - 1U;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt carryStart = bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes] =
            stateVec[i];
    }
}

__global__ void div(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (1U << len) - 1U;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt carryStart = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toDiv;
        nStateVec[i] =
            stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes];
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define MODNOUT(indexIn, indexOut)                                                                                     \
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;                                                              \
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                      \
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                     \
    const bitCapIntOcl inMask = bitCapIntOclPtr[2];                                                                    \
    /* bitCapIntOcl outMask = bitCapIntOclPtr[3]; */                                                                   \
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];                                                                 \
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];                                                               \
    /* bitCapIntOcl lowMask = (1U << len) - 1U; */                                                                     \
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[6];                                                           \
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[7];                                                          \
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];                                                                  \
    const bitCapIntOcl modN = bitCapIntOclPtr[9];                                                                      \
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {                                                         \
        const bitCapIntOcl iHigh = lcv;                                                                                \
        const bitCapIntOcl iLow = iHigh & skipMask;                                                                    \
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;                                                           \
                                                                                                                       \
        const bitCapIntOcl otherRes = i & otherMask;                                                                   \
        const bitCapIntOcl inRes = i & inMask;                                                                         \
        const bitCapIntOcl outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                 \
        nStateVec[indexOut] = stateVec[indexIn];                                                                       \
    }

__global__ void mulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    MODNOUT(i, (inRes | outRes | otherRes));
}

__global__ void imulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    MODNOUT((inRes | outRes | otherRes), i);
}

__global__ void powmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl base = bitCapIntOclPtr[1];
    const bitCapIntOcl inMask = bitCapIntOclPtr[2];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    const bitCapIntOcl modN = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inRes = i & inMask;
        const bitCapIntOcl inInt = inRes >> inStart;

        bitCapIntOcl powRes = base;
        if (inInt == 0) {
            powRes = 1;
        } else {
            for (bitCapIntOcl pw = 1; pw < inInt; pw++) {
                powRes *= base;
            }
        }

        const bitCapIntOcl outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes] = stateVec[i];
    }
}

__global__ void fulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    const bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    const bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    const bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - 1U;
        qMask2 = carryOutMask - 1U;
    } else {
        qMask1 = carryOutMask - 1U;
        qMask2 = carryInSumOutMask - 1U;
    }

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit in
        const qCudaCmplx ins0c0 = stateVec[i];
        const qCudaCmplx ins0c1 = stateVec[i | carryInSumOutMask];
        const qCudaCmplx ins1c0 = stateVec[i | carryOutMask];
        const qCudaCmplx ins1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        const bool aVal = (i & input1Mask);
        const bool bVal = (i & input2Mask);

        qCudaCmplx outs0c0, outs0c1, outs1c0, outs1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                outs0c0 = ins0c0;
                outs1c0 = ins0c1;
                // Non-coding:
                outs0c1 = ins1c0;
                outs1c1 = ins1c1;
            } else {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            }
        } else {
            if (!bVal) {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            } else {
                // Coding:
                outs0c1 = ins0c0;
                outs1c1 = ins0c1;
                // Non-coding:
                outs0c0 = ins1c0;
                outs1c0 = ins1c1;
            }
        }

        stateVec[i] = outs0c0;
        stateVec[i | carryOutMask] = outs0c1;
        stateVec[i | carryInSumOutMask] = outs1c0;
        stateVec[i | carryInSumOutMask | carryOutMask] = outs1c1;
    }
}

__global__ void ifulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    const bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    const bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    const bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - 1U;
        qMask2 = carryOutMask - 1U;
    } else {
        qMask1 = carryOutMask - 1U;
        qMask2 = carryInSumOutMask - 1U;
    }

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit out
        const qCudaCmplx outs0c0 = stateVec[i];
        const qCudaCmplx outs0c1 = stateVec[i | carryOutMask];
        const qCudaCmplx outs1c0 = stateVec[i | carryInSumOutMask];
        const qCudaCmplx outs1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        const bool aVal = (i & input1Mask);
        const bool bVal = (i & input2Mask);

        qCudaCmplx ins0c0, ins0c1, ins1c0, ins1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                ins0c0 = outs0c0;
                ins0c1 = outs1c0;
                // Non-coding:
                ins1c0 = outs0c1;
                ins1c1 = outs1c1;
            } else {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            }
        } else {
            if (!bVal) {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            } else {
                // Coding:
                ins0c0 = outs0c1;
                ins0c1 = outs1c1;
                // Non-coding:
                ins1c0 = outs0c0;
                ins1c1 = outs1c0;
            }
        }

        stateVec[i] = ins0c0;
        stateVec[i | carryInSumOutMask] = ins0c1;
        stateVec[i | carryOutMask] = ins1c0;
        stateVec[i | carryInSumOutMask | carryOutMask] = ins1c1;
    }
}

#define CMOD_START()                                                                                                   \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    bitCapIntOcl i = 0U;                                                                                               \
    for (bitLenInt p = 0U; p < (controlLen + len); p++) {                                                              \
        bitCapIntOcl iLow = iHigh & (controlPowers[p] - 1U);                                                           \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << 1U;                                                                                  \
    }                                                                                                                  \
    i |= iHigh;

#define CMOD_FINISH()                                                                                                  \
    nStateVec[i] = stateVec[i];                                                                                        \
    for (bitCapIntOcl j = 1U; j < ((1U << controlLen) - 1U); j++) {                                                    \
        bitCapIntOcl partControlMask = 0U;                                                                             \
        for (bitLenInt k = 0U; k < controlLen; k++) {                                                                  \
            if (j & (1U << k)) {                                                                                       \
                partControlMask |= controlPowers[controlLen + len + k];                                                \
            }                                                                                                          \
        }                                                                                                              \
        nStateVec[i | partControlMask] = stateVec[i | partControlMask];                                                \
    }

__global__ void cmul(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl lowMask = (1U << len) - 1U;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[8];
    const bitLenInt carryStart = (bitLenInt)bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes |
            controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

__global__ void cdiv(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl lowMask = (1U << len) - 1U;
    const bitCapIntOcl highMask = lowMask << len;
    const bitCapIntOcl inOutStart = bitCapIntOclPtr[8];
    const bitCapIntOcl carryStart = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = (((i & inOutMask) >> inOutStart) * toDiv);
        nStateVec[i | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask];

        CMOD_FINISH();
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define CMODNOUT(indexIn, indexOut)                                                                                    \
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;                                                              \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                            \
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                     \
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];                                                        \
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];                                                               \
    const bitCapIntOcl inMask = bitCapIntOclPtr[4];                                                                    \
    const bitCapIntOcl outMask = bitCapIntOclPtr[5];                                                                   \
    const bitCapIntOcl modN = bitCapIntOclPtr[6];                                                                      \
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];                                                               \
    /* bitCapIntOcl lowMask = (1U << len) - 1U; */                                                                     \
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[8];                                                           \
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[9];                                                          \
                                                                                                                       \
    const bitCapIntOcl otherMask = (maxI - 1U) ^ (inMask | outMask | controlMask);                                     \
    maxI >>= (controlLen + len);                                                                                       \
                                                                                                                       \
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {                                                         \
        CMOD_START();                                                                                                  \
                                                                                                                       \
        const bitCapIntOcl otherRes = i & otherMask;                                                                   \
        const bitCapIntOcl inRes = i & inMask;                                                                         \
        const bitCapIntOcl outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                 \
        nStateVec[indexOut] = stateVec[indexIn];                                                                       \
                                                                                                                       \
        CMOD_FINISH();                                                                                                 \
    }

__global__ void cmulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    CMODNOUT((i | controlMask), (inRes | outRes | otherRes | controlMask));
}

__global__ void cimulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    CMODNOUT((inRes | outRes | otherRes | controlMask), (i | controlMask));
}

__global__ void cpowmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl base = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inMask = bitCapIntOclPtr[4];
    const bitCapIntOcl outMask = bitCapIntOclPtr[5];
    const bitCapIntOcl modN = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[8];
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[9];
    const bitCapIntOcl otherMask = (maxI - 1U) ^ (inMask | outMask | controlMask);
    maxI >>= (controlLen + len);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inRes = i & inMask;
        const bitCapIntOcl inInt = inRes >> inStart;

        bitCapIntOcl powRes = base;
        if (inInt == 0) {
            powRes = 1;
        } else {
            for (bitCapIntOcl pw = 1; pw < inInt; pw++) {
                powRes *= base;
            }
        }

        const bitCapIntOcl outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes | controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

__global__ void indexedLda(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[4];
    const bitLenInt valueLength = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (1U << outputStart) - 1U;
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & lowMask;
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << valueLength);

        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        const bitCapIntOcl outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

__global__ void indexedAdc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    const bitLenInt carryIn = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (carryMask - 1U);
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << 1U);

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputRes = i & outputMask;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt += (outputRes >> outputStart) + carryIn;

        bitCapIntOcl carryRes = 0U;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    }
}

__global__ void indexedSbc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    const bitLenInt carryIn = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (carryMask - 1U);
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << 1U);

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputRes = i & outputMask;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt = (outputRes >> outputStart) + (lengthPower - (outputInt + carryIn));

        bitCapIntOcl carryRes = 0U;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[i];
    }
}

__global__ void hash(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitCapIntOcl bytes = bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl inputRes = lcv & inputMask;
        const bitCapIntOcl inputInt = inputRes >> start;
        bitCapIntOcl outputInt = 0U;
        if (bytes == 1) {
            outputInt = values[inputInt];
        } else if (bytes == 2) {
            outputInt = ((ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < bytes; j++) {
                outputInt |= values[inputInt * bytes + j] << (8U * j);
            }
        }
        const bitCapIntOcl outputRes = outputInt << start;
        nStateVec[outputRes | (lcv & ~inputRes)] = stateVec[lcv];
    }
}

__global__ void cphaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl skipPower = bitCapIntOclPtr[2];
    const bitCapIntOcl greaterPerm = bitCapIntOclPtr[3];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[4];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (skipPower - 1U);
        const bitCapIntOcl i = (iLow | ((iHigh ^ iLow) << 1U)) | skipPower;

        if (((i & regMask) >> start) < greaterPerm) {
            const qCudaCmplx amp = stateVec[i];
            stateVec[i] = make_qCudaCmplx(-amp.x, -amp.y);
        }
    }
}

__global__ void phaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = gridDim.x * blockDim.x;
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl greaterPerm = bitCapIntOclPtr[2];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm) {
            const qCudaCmplx amp = stateVec[lcv];
            stateVec[lcv] = make_qCudaCmplx(-amp.x, -amp.y);
        }
    }
}

#if ENABLE_BCD
__global__ void incbcd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[3];
    bitCapIntOcl toAdd = bitCapIntOclPtr[4];
    int nibbleCount = bitCapIntOclPtr[5];
    bitCapIntOcl otherRes, partToAdd, inOutRes, inOutInt, outInt, outRes;
    int test1, test2;
    int j;
    // For 64 qubits, we would have 16 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    int nibbles[16];
    bool isValid;
    qCudaCmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        partToAdd = toAdd;
        inOutRes = lcv & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = partToAdd % 10;
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        amp = stateVec[lcv];
        if (isValid) {
            outInt = 0;
            outRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= ((bitCapIntOcl)nibbles[j]) << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes;
            nStateVec[outRes] = amp;
        } else {
            nStateVec[lcv] = amp;
        }
    }
}

__global__ void incdecbcdc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[4];
    bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    int nibbleCount = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, partToAdd, inOutRes, inOutInt, outInt, outRes, carryRes, i;
    int test1, test2;
    int j;
    // For 64 qubits, we would have 16 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    int nibbles[16];
    bool isValid;
    qCudaCmplx amp1, amp2;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & (carryMask - 1U);
        i |= (lcv ^ i) << 1U;

        otherRes = i & otherMask;
        partToAdd = toAdd;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        amp1 = stateVec[i];
        amp2 = stateVec[i | carryMask];
        for (j = 1; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = partToAdd % 10;
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if ((test1 > 9) || (test2 > 9)) {
                isValid = false;
            }
        }
        if (isValid) {
            outInt = 0;
            outRes = 0;
            carryRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= ((bitCapIntOcl)nibbles[j]) << (j * 4);
            }
            outRes = (outInt << inOutStart) | otherRes | carryRes;
            nStateVec[outRes] = amp1;
            outRes ^= carryMask;
            nStateVec[outRes] = amp2;
        } else {
            nStateVec[i] = amp1;
            nStateVec[i | carryMask] = amp2;
        }
    }
}
#endif
#endif
} // namespace Qrack
