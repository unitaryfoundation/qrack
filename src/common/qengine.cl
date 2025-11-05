//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline cmplx2 zmatrixmul(const real1 nrm, const cmplx4 lhs, const cmplx2 rhs)
{
    return nrm *
        ((cmplx2)((lhs.lo.x * rhs.x) - (lhs.lo.y * rhs.y) + (lhs.lo.z * rhs.z) - (lhs.lo.w * rhs.w),
            (lhs.lo.x * rhs.y) + (lhs.lo.y * rhs.x) + (lhs.lo.z * rhs.w) + (lhs.lo.w * rhs.z),
            (lhs.hi.x * rhs.x) - (lhs.hi.y * rhs.y) + (lhs.hi.z * rhs.z) - (lhs.hi.w * rhs.w),
            (lhs.hi.x * rhs.y) + (lhs.hi.y * rhs.x) + (lhs.hi.z * rhs.w) + (lhs.hi.w * rhs.z)));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return (real1)atan2((real1_f)cmp.y, (real1_f)cmp.x);
}

inline cmplx conj(const cmplx cmp)
{
    return (cmplx)(cmp.x, -cmp.y);
}

inline cmplx polar_unit(const real1 theta) {
    return sin((cmplx)(theta + (PI_R1 / 2), theta));
}

#define OFFSET2_ARG bitCapIntOclPtr[0]
#define OFFSET1_ARG bitCapIntOclPtr[1]
#define MAXI_ARG bitCapIntOclPtr[2]
#define BITCOUNT_ARG bitCapIntOclPtr[3]
#define ID get_global_id(0)

#define PREP_2X2()                                                                                                     \
    const bitCapIntOcl Nthreads = get_global_size(0);                                                                  \
                                                                                                                       \
    const cmplx4 mtrx = *((constant cmplx4*)cmplxPtr);                                                                 \
    const real1 nrm = cmplxPtr[8];

#define PREP_2X2_WIDE()                                                                                                \
    const cmplx4 mtrx = *((constant cmplx4*)cmplxPtr);                                                                 \
    const real1 nrm = cmplxPtr[8];

#define PREP_2X2_NORM()                                                                                                \
    const real1 norm_thresh = cmplxPtr[9];

#define PUSH_APART_GEN()                                                                                               \
    bitCapIntOcl i = 0U;                                                                                               \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    for (bitLenInt p = 0U; p < BITCOUNT_ARG; p++) {                                                                    \
        bitCapIntOcl iLow = iHigh & (qPowersSorted[p] - ONE_BCI);                                                      \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define PUSH_APART_1()                                                                                                 \
    bitCapIntOcl i = lcv & qMask;                                                                                      \
    i |= (lcv ^ i) << ONE_BCI;

#define PUSH_APART_2()                                                                                                 \
    bitCapIntOcl i = lcv & qMask1;                                                                                     \
    const bitCapIntOcl iHigh = (lcv ^ i) << ONE_BCI;                                                                   \
    const bitCapIntOcl iLow = iHigh & qMask2;                                                                          \
    i |= iLow | ((iHigh ^ iLow) << ONE_BCI);

#define APPLY_AND_OUT()                                                                                                \
    cmplx2 mulRes = (cmplx2)(stateVec[i | OFFSET1_ARG], stateVec[i | OFFSET2_ARG]);                                    \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

#define APPLY_X()                                                                                                      \
    const cmplx Y0 = stateVec[i];                                                                                      \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                                                           \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z() stateVec[i | OFFSET2_ARG] = -stateVec[i | OFFSET2_ARG];

#define APPLY_PHASE()                                                                                                  \
    stateVec[i] = zmul(topLeft, stateVec[i]);                                                                          \
    stateVec[i | OFFSET2_ARG] = zmul(bottomRight, stateVec[i | OFFSET2_ARG]);

#define APPLY_INVERT()                                                                                                 \
    const cmplx Y0 = stateVec[i];                                                                                      \
    stateVec[i] = zmul(topRight, stateVec[i | OFFSET2_ARG]);                                                           \
    stateVec[i | OFFSET2_ARG] = zmul(bottomLeft, Y0);

#define NORM_BODY_2X2()                                                                                                \
    cmplx2 mulRes = (cmplx2)(stateVec[i | OFFSET1_ARG], stateVec[i | OFFSET2_ARG]);                                    \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    real1 dotMulRes = dot(mulRes.lo, mulRes.lo);                                                                       \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.lo = (cmplx)(ZERO_R1, ZERO_R1);                                                                         \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    dotMulRes = dot(mulRes.hi, mulRes.hi);                                                                             \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.hi = (cmplx)(ZERO_R1, ZERO_R1);                                                                         \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

#define SUM_LOCAL(part)                                                                                                \
    const bitCapIntOcl locID = get_local_id(0);                                                                        \
    const bitCapIntOcl locNthreads = get_local_size(0);                                                                \
    lBuffer[locID] = part;                                                                                             \
                                                                                                                       \
    for (bitCapIntOcl lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \
        if (locID < lcv) {                                                                                             \
            lBuffer[locID] += lBuffer[locID + lcv];                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    if (locID == 0U) {                                                                                                 \
        sumBuffer[get_group_id(0)] = lBuffer[0];                                                                       \
    }

void kernel apply2x2(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2()

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_GEN()
        APPLY_AND_OUT()
    }
}

void kernel apply2x2single(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_AND_OUT()
    }
}

void kernel apply2x2double(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_2()
        APPLY_AND_OUT()
    }
}

void kernel apply2x2wide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl lcv = ID;

    PUSH_APART_GEN()
    APPLY_AND_OUT()
}

void kernel apply2x2singlewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_1()
    APPLY_AND_OUT()
}

void kernel apply2x2doublewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_2()
    APPLY_AND_OUT()
}

void kernel apply2x2normsingle(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    global real1* sumBuffer, local real1* lBuffer)
{
    PREP_2X2()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    real1 partNrm = ZERO_R1;
    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        NORM_BODY_2X2()
    }

    SUM_LOCAL(partNrm)
}

void kernel apply2x2normsinglewide(global cmplx* stateVec, constant real1* cmplxPtr,
    constant bitCapIntOcl* bitCapIntOclPtr, global real1* sumBuffer, local real1* lBuffer)
{
    PREP_2X2_WIDE()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    real1 partNrm = ZERO_R1;
    PUSH_APART_1()
    NORM_BODY_2X2()

    SUM_LOCAL(partNrm)
}

void kernel xsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_X()
    }
}

void kernel xsinglewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_X()
}

void kernel xmask(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{

    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;
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

        const cmplx Y0 = stateVec[resetInt];
        stateVec[resetInt] = stateVec[setInt];
        stateVec[setInt] = Y0;
    }
}

void kernel phaseparity(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx2* cmplxPtr)
{
    const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const cmplx2 phaseFac = cmplxPtr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl setInt = lcv & mask;

        bitCapIntOcl v = setInt;
        for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        setInt |= lcv & otherMask;

        stateVec[setInt] = zmul(v ? phaseFac.lo : phaseFac.hi, stateVec[setInt]);
    }
}

void kernel phasemask(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    // const bitCapIntOcl maxI = args.x;
    // const bitCapIntOcl mask = args.y;
    // const bitCapIntOcl nPhases = args.z;
    const real1 phaseAngle = -PI_R1 / args.w;

    for (bitCapIntOcl lcv = ID; lcv < args.x; lcv += Nthreads) {
        bitCapIntOcl popCount = 0;
        bitCapIntOcl v = lcv & args.y;
        while (v) {
            popCount += v & 1;
            v >>= 1;
        }

        const bitCapIntOcl nPhaseSteps = popCount % args.z;
        if (nPhaseSteps) {
            stateVec[lcv] = zmul(polar_unit(nPhaseSteps * phaseAngle), stateVec[lcv]);
        }
    }
}

void kernel zsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_Z()
    }
}

void kernel zsinglewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_Z()
}

void kernel phasesingle(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const cmplx topLeft = cmplxPtr[0];
    const cmplx bottomRight = cmplxPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_PHASE()
    }
}

void kernel phasesinglewide(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const cmplx topLeft = cmplxPtr[0];
    const cmplx bottomRight = cmplxPtr[3];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_PHASE()
}

void kernel invertsingle(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const cmplx topRight = cmplxPtr[1];
    const cmplx bottomLeft = cmplxPtr[2];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_INVERT()
    }
}

void kernel invertsinglewide(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const cmplx topRight = cmplxPtr[1];
    const cmplx bottomLeft = cmplxPtr[2];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_INVERT()
}

void kernel uniformlycontrolled(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowers, global cmplx4* mtrxs, constant real1* nrmIn)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl targetPower = args.y;
    const bitCapIntOcl targetMask = targetPower - ONE_BCI;
    const bitCapIntOcl controlLen = args.z;
    const bitCapIntOcl mtrxSkipLen = args.w;
    const bitCapIntOcl mtrxSkipValueMask = bitCapIntOclPtr[4];
    const real1 nrm = nrmIn[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & targetMask;
        i |= (lcv ^ i) << ONE_BCI;

        bitCapIntOcl offset = 0;
        for (bitLenInt p = 0; p < controlLen; p++) {
            if (i & qPowers[p]) {
                offset |= ONE_BCI << p;
            }
        }

        bitCapIntOcl jHigh = offset;
        bitCapIntOcl j = 0;
        for (bitLenInt p = 0; p < mtrxSkipLen; p++) {
            const bitCapIntOcl jLow = jHigh & (qPowers[controlLen + p] - ONE_BCI);
            j |= jLow;
            jHigh = (jHigh ^ jLow) << ONE_BCI;
        }
        j |= jHigh;
        offset = j | mtrxSkipValueMask;

        cmplx2 qubit = (cmplx2)(stateVec[i], stateVec[i | targetPower]);

        qubit = zmatrixmul(nrm, mtrxs[offset], qubit);

        stateVec[i] = qubit.lo;
        stateVec[i | targetPower] = qubit.hi;
    }
}

void kernel uniformparityrz(global cmplx* stateVec, constant bitCapIntOcl2* bitCapIntOclPtr, constant cmplx2* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qMask = args.y;
    const cmplx2 phaseFac = cmplx_ptr[0];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(stateVec[lcv], ((c & 1U) ? phaseFac.lo : phaseFac.hi));
    }
}

void kernel uniformparityrznorm(global cmplx* stateVec, constant bitCapIntOcl2* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qMask = args.y;
    const cmplx phaseFac = cmplx_ptr[0];
    const cmplx phaseFacAdj = cmplx_ptr[1];
    const cmplx nrm = cmplx_ptr[2];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(nrm, zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj)));
    }
}

void kernel cuniformparityrz(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, constant cmplx2* cmplx_ptr, constant bitCapIntOcl* qPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qMask = args.y;
    const bitCapIntOcl cMask = args.z;
    const bitLenInt cLen = (bitLenInt)args.w;
    const cmplx2 phaseFac = cmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < cLen; p++) {
            bitCapIntOcl iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh | cMask;

        bitCapIntOcl perm = i & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[i] = zmul(stateVec[i], ((c & 1U) ? phaseFac.lo : phaseFac.hi));
    }
}

void kernel compose(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl4* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    for (bitCapIntOcl lcv = ID; lcv < args.x; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
    }
}

void kernel composewide(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl4* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl lcv = ID;
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
}

void kernel composemid(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl nMaxQPower = args.x;
    const bitCapIntOcl oQubitCount = args.y;
    const bitCapIntOcl startMask = args.z;
    const bitCapIntOcl midMask = args.w;
    const bitCapIntOcl endMask = bitCapIntOclPtr[4];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[5];

    for (bitCapIntOcl lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] =
            zmul(stateVec1[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)], stateVec2[(lcv & midMask) >> start]);
    }
}

void kernel decomposeprob(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr,
    global real1* remainderStateProb, global real1* remainderStateAngle, global real1* partStateProb,
    global real1* partStateAngle)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl partPower = args.x;
    const bitCapIntOcl remainderPower = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    const bitLenInt len = (bitLenInt)args.w;

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < partPower; ++k) {
            const cmplx amp = stateVec[j | (k << start)];
            const real1 nrm = dot(amp, amp);
            partProb += nrm;
            if (nrm > REAL1_EPSILON) {
                partStateAngle[k] += arg(amp) * nrm;
            }
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
            bitCapIntOcl l = k & startMask;
            l |= j | ((k ^ l) << len);

            const cmplx amp = stateVec[l];
            const real1 nrm = dot(amp, amp);
            partProb += nrm;
            if (nrm > REAL1_EPSILON) {
                remainderStateAngle[k] += arg(amp) * nrm;
            }
        }

        if (partProb > REAL1_EPSILON) {
            partStateAngle[lcv] /= partProb;
        }
        partStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const real1 prob = remainderStateProb[lcv];
        if (prob > REAL1_EPSILON) {
            remainderStateAngle[lcv] /= prob;
        }
    }
}

void kernel decomposeamp(
    global real1* stateProb, global real1* stateAngle, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        const real1 angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((cmplx)(angle + SineShift, angle));
    }
}

void kernel disposeprob(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr,
    global real1* remainderStateProb, global real1* remainderStateAngle)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl partPower = args.x;
    const bitCapIntOcl remainderPower = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    const bitLenInt len = args.w;

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < partPower; ++k) {
            const cmplx amp = stateVec[j | (k << start)];
            partProb += dot(amp, amp);
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
            bitCapIntOcl l = k & startMask;
            l |= j | ((k ^ l) << len);

            const cmplx amp = stateVec[l];
            const real1 nrm = dot(amp, amp);
            if (nrm > REAL1_EPSILON) {
                remainderStateAngle[k] += arg(amp) * nrm;
            }
        }
    }

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const real1 prob = remainderStateProb[lcv];
        if (prob > REAL1_EPSILON) {
            remainderStateAngle[lcv] /= prob;
        }
    }
}

void kernel dispose(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl remainderPower = args.x;
    const bitLenInt len = (bitLenInt)args.y;
    const bitCapIntOcl skipMask = args.z;
    const bitCapIntOcl disposedRes = args.w;
    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & skipMask;
        bitCapIntOcl i = iLow | ((lcv ^ iLow) << (bitCapIntOcl)len) | disposedRes;
        nStateVec[lcv] = stateVec[i];
    }
}

void kernel prob(global cmplx* stateVec, constant bitCapIntOcl2* bitCapIntOclPtr, global real1* sumBuffer,
    local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qPower = args.y;
    const bitCapIntOcl qMask = qPower - ONE_BCI;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << ONE_BCI) | qPower;
        const cmplx amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

void kernel cprob(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global real1* sumBuffer,
    local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qPower = args.y;
    const bitCapIntOcl qControlPower = args.z;
    const bitCapIntOcl qControlMask = args.w;
    bitCapIntOcl qMask1, qMask2;
    if (qPower < qControlPower) {
        qMask1 = qPower - ONE_BCI;
        qMask2 = qControlPower - ONE_BCI;
    } else {
        qMask1 = qControlPower - ONE_BCI;
        qMask2 = qPower - ONE_BCI;
    }

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2()
        i |= qPower | qControlMask;
        const cmplx amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

void kernel probreg(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global real1* sumBuffer,
    local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl perm = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitLenInt len = (bitLenInt)args.w;
    const bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << len);
        const cmplx amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

void kernel probregall(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global real1* sumBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl maxJ = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitLenInt len = (bitLenInt)args.w;
    const bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        const bitCapIntOcl perm = lcv1 << start;
        real1 oneChancePart = ZERO_R1;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            bitCapIntOcl i = lcv2 & qMask;
            i |= ((lcv2 ^ i) << len);
            cmplx amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

void kernel probmask(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global real1* sumBuffer,
    constant bitCapIntOcl* qPowers, local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    //bitCapIntOcl mask = args.y;
    const bitCapIntOcl perm = args.z;
    const bitLenInt len = (bitLenInt)args.w;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < len; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        const cmplx amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

void kernel probmaskall(global cmplx* stateVec, constant bitCapIntOcl4* bitCapIntOclPtr, global real1* sumBuffer,
    constant bitCapIntOcl* qPowersMask, constant bitCapIntOcl* qPowersSkip)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl maxJ = args.y;
    const bitLenInt maskLen = (bitLenInt)args.z;
    const bitLenInt skipLen = (bitLenInt)args.w;

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        bitCapIntOcl iHigh = lcv1;
        bitCapIntOcl perm = 0U;
        for (bitLenInt p = 0U; p < skipLen; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowersSkip[p] - ONE_BCI);
            perm |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        perm |= iHigh;

        real1 oneChancePart = ZERO_R1;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            iHigh = lcv2;
            bitCapIntOcl i = 0U;
            for (bitLenInt p = 0U; p < maskLen; p++) {
                bitCapIntOcl iLow = iHigh & (qPowersMask[p] - ONE_BCI);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << ONE_BCI;
            }
            i |= iHigh;

            const cmplx amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

void kernel probparity(global cmplx* stateVec, constant bitCapIntOcl2* bitCapIntOclPtr, global real1* sumBuffer,
    local real1* lBuffer)
{

    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity) {
            const cmplx amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        }
    }

    SUM_LOCAL(oneChancePart)
}

void kernel forcemparity(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* sumBuffer,
    local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;
    const bool result = (bitCapIntOclPtr[2] == ONE_BCI);

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity == result) {
            const cmplx amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        } else {
            stateVec[lcv] = (cmplx)(ZERO_R1, ZERO_R1);
        }
    }

    SUM_LOCAL(oneChancePart)
}

void kernel expperm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant bitCapIntOcl* bitPowers, global real1* sumBuffer,
    local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitLenInt len = (bitLenInt)args.y;
    const bitCapIntOcl offset = bitCapIntOclPtr[2];

    real1 expectation = 0;
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl retIndex = 0;
        for (bitLenInt p = 0; p < len; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= (ONE_BCI << p);
            }
        }
        const cmplx amp = stateVec[lcv];
        expectation += (offset + retIndex) * dot(amp, amp);
    }

    SUM_LOCAL(expectation)
}

void kernel nrmlze(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* args_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const real1 norm_thresh = args_ptr[0].x;
    const cmplx nrm = args_ptr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        cmplx amp = stateVec[lcv];
        if (dot(amp, amp) < norm_thresh) {
            amp = (cmplx)(ZERO_R1, ZERO_R1);
        }
        stateVec[lcv] = zmul(nrm, amp);
    }
}

void kernel nrmlzewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* args_ptr)
{
    const bitCapIntOcl lcv = ID;
    const real1 norm_thresh = args_ptr[0].x;
    const cmplx nrm = args_ptr[1];

    cmplx amp = stateVec[lcv];
    if (dot(amp, amp) < norm_thresh) {
        amp = (cmplx)(ZERO_R1, ZERO_R1);
    }
    stateVec[lcv] = zmul(nrm, amp);
}

void kernel updatenorm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr,
    global real1* sumBuffer, local real1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const real1 norm_thresh = args_ptr[0];
    real1 partNrm = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const cmplx amp = stateVec[lcv];
        real1 nrm = dot(amp, amp);
        if (nrm < norm_thresh) {
            nrm = ZERO_R1;
        }
        partNrm += nrm;
    }

    SUM_LOCAL(partNrm)
}

void kernel approxcompare(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr,
    global cmplx* sumBuffer, local cmplx* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    cmplx partInner = (cmplx)(ZERO_R1, ZERO_R1);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        partInner += zmul(conj(stateVec1[lcv]), stateVec2[lcv]);
    }

    SUM_LOCAL(partInner)
}

void kernel applym(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qPower = args.y;
    const bitCapIntOcl qMask = qPower - ONE_BCI;
    const bitCapIntOcl savePower = bitCapIntOclPtr[2];
    const bitCapIntOcl discardPower = qPower ^ savePower;
    const cmplx nrm = cmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & qMask;
        const bitCapIntOcl i = iLow | ((lcv ^ iLow) << ONE_BCI);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel applymreg(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;
    const bitCapIntOcl result = bitCapIntOclPtr[2];
    const cmplx nrm = cmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = ((lcv & mask) == result) ? zmul(nrm, stateVec[lcv]) : (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel clearbuffer(global cmplx* stateVec, constant bitCapIntOcl2* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = bitCapIntOclPtr[0];
    const bitCapIntOcl maxI = args.x + args.y;
    const bitCapIntOcl offset = args.y;
    const cmplx amp0 = (cmplx)(ZERO_R1, ZERO_R1);
    for (bitCapIntOcl lcv = (ID + offset); lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = amp0;
    }
}

void kernel shufflebuffers(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl halfMaxI = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < halfMaxI; lcv += Nthreads) {
        const cmplx amp0 = stateVec1[lcv + halfMaxI];
        stateVec1[lcv + halfMaxI] = stateVec2[lcv];
        stateVec2[lcv] = amp0;
    }
}

void kernel rol(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl regMask = args.y;
    const bitCapIntOcl otherMask = args.z;
    const bitCapIntOcl lengthMask = args.w - ONE_BCI;
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[4];
    const bitLenInt shift = (bitLenInt)bitCapIntOclPtr[5];
    const bitLenInt length = (bitLenInt)bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl regInt = (lcv & regMask) >> start;
        nStateVec[lcv] = stateVec[((((regInt >> shift) | (regInt << (length - shift))) & lengthMask) << start) | (lcv & otherMask)];
    }
}
