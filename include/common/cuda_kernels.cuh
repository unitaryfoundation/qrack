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

#include "common/qrack_types.hpp"

#include "cuda_to_hip.h"

namespace Qrack {
__global__ void apply2x2(
    qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted);
__global__ void apply2x2single(qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2double(qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2wide(
    qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted);
__global__ void apply2x2singlewide(qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2doublewide(qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void apply2x2normsingle(
    qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void apply2x2normsinglewide(
    qCudaCmplx* stateVec, qCudaReal1* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void xsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void xsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void xmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void phaseparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplxPtr);
__global__ void phasemask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void zsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void zsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void phasesingle(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void phasesinglewide(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void invertsingle(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void invertsinglewide(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr);
__global__ void uniformlycontrolled(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowers, qCudaReal1* mtrxs, qCudaReal1* nrmIn);
__global__ void uniformparityrz(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr);
__global__ void uniformparityrznorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr);
__global__ void cuniformparityrz(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr, bitCapIntOcl* qPowers);
__global__ void compose(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void composewide(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void composemid(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void decomposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* remainderStateProb,
    qCudaReal1* remainderStateAngle, qCudaReal1* partStateProb, qCudaReal1* partStateAngle);
__global__ void decomposeamp(
    qCudaReal1* stateProb, qCudaReal1* stateAngle, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void disposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* remainderStateProb,
    qCudaReal1* remainderStateAngle);
__global__ void dispose(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void prob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void cprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void probreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void probregall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void probmask(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer, bitCapIntOcl* qPowers);
__global__ void probmaskall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    bitCapIntOcl* qPowersMask, bitCapIntOcl* qPowersSkip);
__global__ void probparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void forcemparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer);
__global__ void expperm(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* bitPowers, qCudaReal1* sumBuffer);
__global__ void nrmlze(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr);
__global__ void nrmlzewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr);
__global__ void updatenorm(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr, qCudaReal1* sumBuffer);
__global__ void approxcompare(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* sumBuffer);
__global__ void applym(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr);
__global__ void applymreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr);
__global__ void shufflebuffers(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr);
__global__ void rol(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
#if ENABLE_ALU
__global__ void inc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void cinc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void incdecc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incs(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecsc1(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecsc2(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void mul(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void div(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void mulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void imulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void powmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void fulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void ifulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void cmul(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cdiv(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cmulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cimulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void cpowmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers);
__global__ void indexedLda(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values);
__global__ void indexedAdc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values);
__global__ void indexedSbc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values);
__global__ void hash(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec, unsigned char* values);
__global__ void cphaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
__global__ void phaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr);
#if ENABLE_BCD
__global__ void incbcd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
__global__ void incdecbcdc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec);
#endif
#endif
} // namespace Qrack
