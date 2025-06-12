// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// (Extensively modified and adapted by Daniel Strano in unitaryfund/qrack)

#pragma once

#include "config.h"
#include "stddef.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#define MICROSOFT_QUANTUM_DECL __declspec(dllexport)
#define MICROSOFT_QUANTUM_DECL_IMPORT __declspec(dllimport)
#else
#define MICROSOFT_QUANTUM_DECL
#define MICROSOFT_QUANTUM_DECL_IMPORT
#endif

// SAL only defined in windows.
#ifndef _In_
#define _In_
#define _In_reads_(n)
#endif

typedef unsigned long long uintq;
typedef long long intq;
typedef void (*IdCallback)(uintq);
typedef bool (*ProbAmpCallback)(size_t, double, double);

#if !(FPPOW < 6 && !defined(ENABLE_COMPLEX_X2))
struct _QrackTimeEvolveOpHeader;
#endif

extern "C" {
// non-quantum
MICROSOFT_QUANTUM_DECL int get_error(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL uintq init_count_type(_In_ uintq q, _In_ bool tn, _In_ bool md, _In_ bool sd, _In_ bool sh,
    _In_ bool bdt, _In_ bool pg, _In_ bool nw, _In_ bool hy, _In_ bool oc, _In_ bool dm);
MICROSOFT_QUANTUM_DECL uintq init_count(_In_ uintq q, _In_ bool dm);
MICROSOFT_QUANTUM_DECL uintq init_count_pager(_In_ uintq q, _In_ bool dm);
MICROSOFT_QUANTUM_DECL uintq init_count_stabilizer(_In_ uintq q);
MICROSOFT_QUANTUM_DECL uintq init() { return init_count(0, false); }
MICROSOFT_QUANTUM_DECL uintq init_clone(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL void destroy(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL void seed(_In_ uintq sid, _In_ uintq s);
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ uintq sid, _In_ uintq p);
MICROSOFT_QUANTUM_DECL void set_device(_In_ uintq sid, _In_ intq did);
MICROSOFT_QUANTUM_DECL void set_device_list(_In_ uintq sid, _In_ uintq n, _In_reads_(n) intq* dids);
MICROSOFT_QUANTUM_DECL void qstabilizer_out_to_file(_In_ uintq sid, _In_ char* f);
MICROSOFT_QUANTUM_DECL void qstabilizer_in_from_file(_In_ uintq sid, _In_ char* f);

// pseudo-quantum
#if FPPOW < 6
MICROSOFT_QUANTUM_DECL void ProbAll(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, float* p);
#else
MICROSOFT_QUANTUM_DECL void ProbAll(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, double* p);
#endif
MICROSOFT_QUANTUM_DECL double Prob(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL double ProbRdm(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL double PermutationProb(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) bool* c);
MICROSOFT_QUANTUM_DECL double PermutationProbRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) bool* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double PermutationExpectation(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL double PermutationExpectationRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ bool r);
MICROSOFT_QUANTUM_DECL double FactorizedExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c);
MICROSOFT_QUANTUM_DECL double FactorizedExpectationRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double Variance(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL double VarianceRdm(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ bool r);
MICROSOFT_QUANTUM_DECL double FactorizedVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c);
MICROSOFT_QUANTUM_DECL double FactorizedVarianceRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq m, uintq* c, _In_ bool r);
#if FPPOW < 6
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, float* c);
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFpRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, float* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double UnitaryExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) float* b);
MICROSOFT_QUANTUM_DECL double MatrixExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) float* b);
MICROSOFT_QUANTUM_DECL double UnitaryExpectationEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) float* b, _In_reads_(2 * n) float* e);
MICROSOFT_QUANTUM_DECL double MatrixExpectationEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) float* b, _In_reads_(2 * n) float* e);
MICROSOFT_QUANTUM_DECL double FactorizedVarianceFp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, float* c);
MICROSOFT_QUANTUM_DECL double FactorizedVarianceFpRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, float* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double UnitaryVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) float* b);
MICROSOFT_QUANTUM_DECL double MatrixVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) float* b);
MICROSOFT_QUANTUM_DECL double UnitaryVarianceEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) float* b, _In_reads_(2 * n) float* e);
MICROSOFT_QUANTUM_DECL double MatrixVarianceEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) float* b, _In_reads_(2 * n) float* e);
#else
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, double* c);
MICROSOFT_QUANTUM_DECL double FactorizedExpectationFpRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, double* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double UnitaryExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) double* b);
MICROSOFT_QUANTUM_DECL double MatrixExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) double* b);
MICROSOFT_QUANTUM_DECL double UnitaryExpectationEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) double* b, _In_reads_(2 * n) double* e);
MICROSOFT_QUANTUM_DECL double MatrixExpectationEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) double* b, _In_reads_(2 * n) double* e);
MICROSOFT_QUANTUM_DECL double FactorizedVarianceFp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, double* c);
MICROSOFT_QUANTUM_DECL double FactorizedVarianceFpRdm(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, double* c, _In_ bool r);
MICROSOFT_QUANTUM_DECL double UnitaryVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) double* b);
MICROSOFT_QUANTUM_DECL double MatrixVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) double* b);
MICROSOFT_QUANTUM_DECL double UnitaryVarianceEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(3 * n) double* b, _In_reads_(2 * n) double* e);
MICROSOFT_QUANTUM_DECL double MatrixVarianceEigenVal(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(8 * n) double* b, _In_reads_(2 * n) double* e);
#endif
MICROSOFT_QUANTUM_DECL double PauliExpectation(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* b);
MICROSOFT_QUANTUM_DECL double PauliVariance(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* b);

MICROSOFT_QUANTUM_DECL void DumpIds(_In_ uintq sid, _In_ IdCallback callback);
MICROSOFT_QUANTUM_DECL void Dump(_In_ uintq sid, _In_ ProbAmpCallback callback);

#if FPPOW < 6
MICROSOFT_QUANTUM_DECL void InKet(_In_ uintq sid, _In_ float* ket);
MICROSOFT_QUANTUM_DECL void OutKet(_In_ uintq sid, _In_ float* ket);
MICROSOFT_QUANTUM_DECL void OutProbs(_In_ uintq sid, _In_ float* ket);
#else
MICROSOFT_QUANTUM_DECL void InKet(_In_ uintq sid, _In_ double* ket);
MICROSOFT_QUANTUM_DECL void OutKet(_In_ uintq sid, _In_ double* ket);
MICROSOFT_QUANTUM_DECL void OutProbs(_In_ uintq sid, _In_ double* ket);
#endif

MICROSOFT_QUANTUM_DECL size_t random_choice(_In_ uintq sid, _In_ size_t n, _In_reads_(n) double* p);

MICROSOFT_QUANTUM_DECL void PhaseParity(_In_ uintq sid, _In_ double lambda, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void PhaseRootN(_In_ uintq sid, _In_ uintq p, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_reads_(n) uintq* q);

MICROSOFT_QUANTUM_DECL void ResetAll(_In_ uintq sid);

// allocate and release
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ uintq sid, _In_ uintq qid);
MICROSOFT_QUANTUM_DECL bool release(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL uintq num_qubits(_In_ uintq sid);

// single-qubit gates
MICROSOFT_QUANTUM_DECL void X(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void Y(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void Z(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void H(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void S(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void SX(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void SY(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void T(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void AdjS(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void AdjSX(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void AdjSY(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void AdjT(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void U(_In_ uintq sid, _In_ uintq q, _In_ double theta, _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void Mtrx(_In_ uintq sid, _In_reads_(8) double* m, _In_ uintq q);

// multi-controlled single-qubit gates

MICROSOFT_QUANTUM_DECL void MCX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCH(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MCU(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ double theta,
    _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void MCMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q);

MICROSOFT_QUANTUM_DECL void MACX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACH(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACAdjS(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACAdjT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void MACU(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ double theta,
    _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void MACMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q);

MICROSOFT_QUANTUM_DECL void UCMtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_reads_(8) double* m, _In_ uintq q, _In_ uintq p);

MICROSOFT_QUANTUM_DECL void Multiplex1Mtrx(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, double* m);

// coalesced single qubit gates

MICROSOFT_QUANTUM_DECL void MX(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void MY(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void MZ(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);

// rotations
MICROSOFT_QUANTUM_DECL void R(_In_ uintq sid, _In_ uintq b, _In_ double phi, _In_ uintq q);

// multi-controlled rotations
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ uintq sid, _In_ uintq b, _In_ double phi, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q);

// Exponential of Pauli operators
MICROSOFT_QUANTUM_DECL void Exp(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_ double phi, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void MCExp(_In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_ double phi, _In_ uintq nc,
    _In_reads_(nc) uintq* cs, _In_reads_(n) uintq* q);

// measurements
MICROSOFT_QUANTUM_DECL uintq M(_In_ uintq sid, _In_ uintq q);
MICROSOFT_QUANTUM_DECL uintq ForceM(_In_ uintq sid, _In_ uintq q, _In_ bool r);
MICROSOFT_QUANTUM_DECL uintq MAll(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL uintq Measure(_In_ uintq sid, _In_ uintq n, _In_reads_(n) int* b, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void MeasureShots(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ uintq s, _In_reads_(s) uintq* m);

MICROSOFT_QUANTUM_DECL void SWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL void ISWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL void AdjISWAP(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL void FSim(_In_ uintq sid, _In_ double theta, _In_ double phi, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL void CSWAP(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL void ACSWAP(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq qi1, _In_ uintq qi2);

// Schmidt decomposition
MICROSOFT_QUANTUM_DECL void Compose(_In_ uintq sid1, _In_ uintq sid2, uintq* q);
MICROSOFT_QUANTUM_DECL uintq Decompose(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void Dispose(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);

MICROSOFT_QUANTUM_DECL void AND(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void OR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void XOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void NAND(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void NOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void XNOR(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLAND(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLXOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLNAND(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLNOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);
MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ uintq sid, _In_ bool ci, _In_ uintq qi, _In_ uintq qo);

MICROSOFT_QUANTUM_DECL void QFT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c);
MICROSOFT_QUANTUM_DECL void IQFT(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c);

#if ENABLE_ALU
MICROSOFT_QUANTUM_DECL void ADD(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void SUB(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void ADDS(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, uintq s, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void SUBS(
    _In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, uintq s, _In_ uintq n, _In_reads_(n) uintq* q);

MICROSOFT_QUANTUM_DECL void MCADD(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq nq, _In_reads_(nq) uintq* q);
MICROSOFT_QUANTUM_DECL void MCSUB(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq nq, _In_reads_(nq) uintq* q);

MICROSOFT_QUANTUM_DECL void MUL(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n,
    _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void DIV(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq n,
    _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void MULN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void DIVN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void POWN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_reads_(na) uintq* m,
    _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);

MICROSOFT_QUANTUM_DECL void MCMUL(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void MCDIV(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void MCMULN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void MCDIVN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);
MICROSOFT_QUANTUM_DECL void MCPOWN(_In_ uintq sid, _In_ uintq na, _In_reads_(na) uintq* a, _In_ uintq nc,
    _In_reads_(nc) uintq* c, _In_reads_(na) uintq* m, _In_ uintq n, _In_reads_(n) uintq* q, _In_reads_(n) uintq* o);

MICROSOFT_QUANTUM_DECL void LDA(
    _In_ uintq sid, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv, _In_reads_(nv) uintq* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void ADC(_In_ uintq sid, uintq s, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv,
    _In_reads_(nv) uintq* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void SBC(_In_ uintq sid, uintq s, _In_ uintq ni, _In_reads_(ni) uintq* qi, _In_ uintq nv,
    _In_reads_(nv) uintq* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void Hash(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, unsigned char* t);
#endif

MICROSOFT_QUANTUM_DECL bool TrySeparate1Qb(_In_ uintq sid, _In_ uintq qi1);
MICROSOFT_QUANTUM_DECL bool TrySeparate2Qb(_In_ uintq sid, _In_ uintq qi1, _In_ uintq qi2);
MICROSOFT_QUANTUM_DECL bool TrySeparateTol(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q, _In_ double tol);
MICROSOFT_QUANTUM_DECL void Separate(_In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL double GetUnitaryFidelity(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL void ResetUnitaryFidelity(_In_ uintq sid);
MICROSOFT_QUANTUM_DECL void SetSdrp(_In_ uintq sid, _In_ double sdrp);
MICROSOFT_QUANTUM_DECL void SetNcrp(_In_ uintq sid, _In_ double ncrp);
MICROSOFT_QUANTUM_DECL void SetReactiveSeparate(_In_ uintq sid, _In_ bool irs);
MICROSOFT_QUANTUM_DECL void SetTInjection(_In_ uintq sid, _In_ bool iti);
MICROSOFT_QUANTUM_DECL void SetNoiseParameter(_In_ uintq sid, _In_ double np);
MICROSOFT_QUANTUM_DECL void Normalize(_In_ uintq sid);

#if !(FPPOW < 6 && !defined(ENABLE_COMPLEX_X2))
MICROSOFT_QUANTUM_DECL void TimeEvolve(_In_ uintq sid, _In_ double t, _In_ uintq n,
    _In_reads_(n) _QrackTimeEvolveOpHeader* teos, uintq mn, _In_reads_(mn) double* mtrx);
#endif

MICROSOFT_QUANTUM_DECL uintq init_qneuron(
    _In_ uintq sid, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ uintq f, _In_ double a, _In_ double tol);
MICROSOFT_QUANTUM_DECL uintq clone_qneuron(_In_ uintq nid);
MICROSOFT_QUANTUM_DECL void destroy_qneuron(_In_ uintq nid);

#if FPPOW < 6
MICROSOFT_QUANTUM_DECL void set_qneuron_angles(_In_ uintq nid, _In_ float* angles);
MICROSOFT_QUANTUM_DECL void get_qneuron_angles(_In_ uintq nid, _In_ float* angles);
#else
MICROSOFT_QUANTUM_DECL void set_qneuron_angles(_In_ uintq nid, _In_ double* angles);
MICROSOFT_QUANTUM_DECL void get_qneuron_angles(_In_ uintq nid, _In_ double* angles);
#endif

MICROSOFT_QUANTUM_DECL void set_qneuron_alpha(_In_ uintq nid, _In_ double alpha);
MICROSOFT_QUANTUM_DECL double get_qneuron_alpha(_In_ uintq nid);

MICROSOFT_QUANTUM_DECL void set_qneuron_activation_fn(_In_ uintq nid, _In_ uintq f);
MICROSOFT_QUANTUM_DECL uintq get_qneuron_activation_fn(_In_ uintq nid);

MICROSOFT_QUANTUM_DECL double qneuron_predict(_In_ uintq nid, _In_ bool e, _In_ bool r);
MICROSOFT_QUANTUM_DECL double qneuron_unpredict(_In_ uintq nid, _In_ bool e);

MICROSOFT_QUANTUM_DECL double qneuron_learn_cycle(_In_ uintq nid, _In_ bool e);

MICROSOFT_QUANTUM_DECL void qneuron_learn(_In_ uintq nid, _In_ double eta, _In_ bool e, _In_ bool r);
MICROSOFT_QUANTUM_DECL void qneuron_learn_permutation(_In_ uintq nid, _In_ double eta, _In_ bool e, _In_ bool r);

MICROSOFT_QUANTUM_DECL uintq init_qcircuit(_In_ bool collapse, _In_ bool clifford);
MICROSOFT_QUANTUM_DECL uintq init_qcircuit_clone(_In_ uintq cid);
MICROSOFT_QUANTUM_DECL uintq qcircuit_inverse(_In_ uintq cid);
MICROSOFT_QUANTUM_DECL uintq qcircuit_past_light_cone(_In_ uintq cid, _In_ uintq n, _In_reads_(n) uintq* q);
MICROSOFT_QUANTUM_DECL void destroy_qcircuit(_In_ uintq cid);
MICROSOFT_QUANTUM_DECL uintq get_qcircuit_qubit_count(_In_ uintq cid);
MICROSOFT_QUANTUM_DECL void qcircuit_swap(_In_ uintq cid, _In_ uintq q1, _In_ uintq q2);
MICROSOFT_QUANTUM_DECL void qcircuit_append_1qb(_In_ uintq cid, _In_reads_(8) double* m, _In_ uintq q);
MICROSOFT_QUANTUM_DECL void qcircuit_append_mc(
    _In_ uintq cid, _In_reads_(8) double* m, _In_ uintq n, _In_reads_(n) uintq* c, _In_ uintq q, _In_ uintq p);
MICROSOFT_QUANTUM_DECL void qcircuit_run(_In_ uintq cid, _In_ uintq sid);
MICROSOFT_QUANTUM_DECL void qcircuit_out_to_file(_In_ uintq cid, _In_ char* f);
MICROSOFT_QUANTUM_DECL void qcircuit_in_from_file(_In_ uintq cid, _In_ char* f);
MICROSOFT_QUANTUM_DECL size_t qcircuit_out_to_string_length(_In_ uintq cid);
MICROSOFT_QUANTUM_DECL void qcircuit_out_to_string(_In_ uintq cid, _In_ char* f);
}
