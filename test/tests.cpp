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

#include "qfactory.hpp"

#include <atomic>
#include <iostream>
#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "qneuron.hpp"

#include "tests.hpp"

using namespace Qrack;

#define EPSILON 0.01f
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1_f __tmp_a = A;                                                                                           \
        real1_f __tmp_b = B;                                                                                           \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_a > (__tmp_b - EPSILON));                                                                        \
    } while (0);
#define REQUIRE_CMPLX(A, B)                                                                                            \
    do {                                                                                                               \
        complex __tmp_a = A;                                                                                           \
        complex __tmp_b = B;                                                                                           \
        REQUIRE(std::norm(__tmp_a - __tmp_b) < EPSILON);                                                               \
    } while (0);

#define QINTERFACE_RESTRICTED                                                                                          \
    ((testEngineType == QINTERFACE_STABILIZER_HYBRID) || (testSubEngineType == QINTERFACE_STABILIZER_HYBRID) ||        \
        (testEngineType == QINTERFACE_HYBRID) || (testSubEngineType == QINTERFACE_HYBRID) ||                           \
        (testSubSubEngineType == QINTERFACE_HYBRID) || (testEngineType == QINTERFACE_OPENCL) ||                        \
        (testSubEngineType == QINTERFACE_OPENCL) || (testSubSubEngineType == QINTERFACE_OPENCL) ||                     \
        (testEngineType == QINTERFACE_QPAGER) || (testSubEngineType == QINTERFACE_QPAGER) ||                           \
        (testEngineType == QINTERFACE_BDT) || (testSubEngineType == QINTERFACE_BDT))

#define C_SQRT1_2 complex(SQRT1_2_R1, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, SQRT1_2_R1)

#define QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

void print_bin(int bits, int d);
void log(QInterfacePtr p);

void print_bin(int bits, int d)
{
    int mask = 1 << bits;
    while (mask != 0) {
        printf("%d", !!(d & mask));
        mask >>= 1;
    }
}

void log(QInterfacePtr p) { std::cout << std::endl << std::showpoint << p << std::endl; }

QInterfacePtr MakeEngine(bitLenInt qubitCount)
{
    return CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, qubitCount, 0, rng,
        ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng, sparse, REAL1_EPSILON, devList);
}

TEST_CASE("test_complex")
{
    bool test;
    complex cmplx1(ONE_R1, -ONE_R1);
    complex cmplx2((real1)(-0.5f), (real1)0.5f);
    complex cmplx3(ZERO_R1, ZERO_R1);

    REQUIRE(cmplx1 != cmplx2);

    REQUIRE(conj(cmplx1) == complex(ONE_R1, ONE_R1));

    test = ((real1)abs(cmplx1) > (real1)(sqrt(2.0) - EPSILON)) && ((real1)abs(cmplx1) < (real1)(sqrt(2.0) + EPSILON));
    REQUIRE(test);

    cmplx3 = complex(std::polar(ONE_R1, PI_R1 / (real1)2.0f));
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 + cmplx2;
    test = (real(cmplx3) > (real1)(0.5 - EPSILON)) && (real(cmplx3) < (real1)(0.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-0.5 - EPSILON)) && (imag(cmplx3) < (real1)(-0.5 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 - cmplx2;
    test = (real(cmplx3) > (real1)(1.5 - EPSILON)) && (real(cmplx3) < (real1)(1.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-1.5 - EPSILON)) && (imag(cmplx3) < (real1)(-1.5 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 * cmplx2;
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1;
    cmplx3 *= cmplx2;
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 / cmplx2;
    test = (real(cmplx3) > (real1)(-2.0 - EPSILON)) && (real(cmplx3) < (real1)(-2.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) && (imag(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx2;
    cmplx3 /= cmplx1;
    test = (real(cmplx3) > (real1)(-0.5 - EPSILON)) && (real(cmplx3) < (real1)(-0.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) && (imag(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = ((real1)2.0) * cmplx1;
    test = (real(cmplx3) > (real1)(2.0 - EPSILON)) && (real(cmplx3) < (real1)(2.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-2.0 - EPSILON)) && (imag(cmplx3) < (real1)(-2.0 + EPSILON));
    REQUIRE(test);
}

TEST_CASE("test_push_apart_bits")
{
    bitCapInt perm = 0x13U;
    bitCapInt skipPowers[1] = { 1U << 2U };
    REQUIRE(pushApartBits(perm, skipPowers, 1U) == 0x23U);
}

#if UINTPOW > 3
TEST_CASE("test_qengine_cpu_par_for")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    const int NUM_ENTRIES = 2000;
    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for(0, NUM_ENTRIES, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[(bitCapIntOcl)lcv].exchange(old);
        REQUIRE(old == false);
        calls++;
    });

    REQUIRE(calls.load() == NUM_ENTRIES);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        REQUIRE(hit[i].load() == true);
    }
}

TEST_CASE("test_qengine_cpu_par_for_skip")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    const int NUM_ENTRIES = 2000;
    const int NUM_CALLS = 1000;

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    int skipBit = 0x4; // Skip 0b100 when counting upwards.

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 1, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[(bitCapIntOcl)lcv].exchange(old);
        REQUIRE(old == false);
        REQUIRE((lcv & skipBit) == 0);

        calls++;
    });

    REQUIRE(calls.load() == NUM_CALLS);
}

TEST_CASE("test_qengine_cpu_par_for_skip_wide")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    const size_t NUM_ENTRIES = 2000;

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    int skipBit = 0x4; // Skip 0b100 when counting upwards.

    for (size_t i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 3, [&](const bitCapInt lcv, const int cpu) {
        REQUIRE(lcv < NUM_ENTRIES);
        bool old = true;
        old = hit[(bitCapIntOcl)lcv].exchange(old);
        REQUIRE(old == false);
        REQUIRE((lcv & skipBit) == 0);

        calls++;
    });
}

TEST_CASE("test_qengine_cpu_par_for_mask")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    const int NUM_ENTRIES = 2000;

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    bitCapIntOcl skipArray[] = { 0x4, 0x100 }; // Skip bits 0b100000100
    int NUM_SKIP = sizeof(skipArray) / sizeof(skipArray[0]);

    calls.store(0);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->SetConcurrencyLevel(1);

    qengine->par_for_mask(0, NUM_ENTRIES, skipArray, 2, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[(bitCapIntOcl)lcv].exchange(old);
        REQUIRE(old == false);
        for (int i = 0; i < NUM_SKIP; i++) {
            REQUIRE((lcv & skipArray[i]) == 0);
        }
        calls++;
    });
}
#endif

TEST_CASE("test_exp2x2_log2x2")
{
    complex mtrx1[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
    complex mtrx2[4];

    exp2x2(mtrx1, mtrx2);
    REQUIRE_FLOAT(real(mtrx2[0]), M_E);
    REQUIRE_FLOAT(imag(mtrx2[0]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx2[1]), ZERO_R1);
    REQUIRE_FLOAT(imag(mtrx2[1]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx2[2]), ZERO_R1);
    REQUIRE_FLOAT(imag(mtrx2[2]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx2[3]), M_E);
    REQUIRE_FLOAT(imag(mtrx2[3]), ZERO_R1);

    log2x2(mtrx2, mtrx1);
    REQUIRE_FLOAT(real(mtrx1[0]), ONE_R1);
    REQUIRE_FLOAT(imag(mtrx1[0]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx1[1]), ZERO_R1);
    REQUIRE_FLOAT(imag(mtrx1[1]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx1[2]), ZERO_R1);
    REQUIRE_FLOAT(imag(mtrx1[2]), ZERO_R1);
    REQUIRE_FLOAT(real(mtrx1[3]), ONE_R1);
    REQUIRE_FLOAT(imag(mtrx1[3]), ZERO_R1);
}

#if ENABLE_OPENCL && !ENABLE_SNUCL
TEST_CASE_METHOD(QInterfaceTestFixture, "test_change_device")
{
    if (testEngineType == QINTERFACE_OPENCL) {
        qftReg->SetPermutation(0x55F00);
        REQUIRE_THAT(qftReg, HasProbability(0x55F00));
        std::dynamic_pointer_cast<QEngineOCL>(qftReg)->SetDevice(0);
        REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    }
}
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qengine_getmaxqpower")
{
    // Assuming default engine has 20 qubits:
    REQUIRE((qftReg->GetMaxQPower() == 1048576U));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_setconcurrency")
{
    // Make sure it doesn't throw:
    qftReg->SetConcurrency(1);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_global_phase")
{
    qftReg = CreateQuantumInterface(
        { testEngineType, testSubEngineType, testSubSubEngineType }, 1U, 0, rng, CMPLX_DEFAULT_ARG, false, false);
    qftReg->Z(0);
    qftReg->X(0);
    qftReg->Z(0);
    qftReg->X(0);
    REQUIRE_FLOAT(-ONE_R1, real(qftReg->GetAmplitude(0x00)));

    qftReg = CreateQuantumInterface(
        { testEngineType, testSubEngineType, testSubSubEngineType }, 1U, 0, rng, CMPLX_DEFAULT_ARG, false, false);
    qftReg->S(0);
    qftReg->X(0);
    qftReg->S(0);
    qftReg->X(0);
    REQUIRE_FLOAT(ONE_R1, imag(qftReg->GetAmplitude(0x00)));

    qftReg = CreateQuantumInterface(
        { testEngineType, testSubEngineType, testSubSubEngineType }, 1U, 0, rng, CMPLX_DEFAULT_ARG, false, false);
    qftReg->Y(0);
    qftReg->X(0);
    REQUIRE_FLOAT(ONE_R1, imag(qftReg->GetAmplitude(0x00)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cnot")
{
    qftReg->SetPermutation(0x01);
    qftReg->H(0, 2);
    qftReg->CNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->Z(0);
    qftReg->CNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->CNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));

    // 2022-02-16 - QBdt fails at the 11-to-12 index, 12th-to-13th bit boundary, and upwards.
    qftReg->SetPermutation(0x1000);
    qftReg->CNOT(12, 11);
    REQUIRE_THAT(qftReg, HasProbability(0x1800));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticnot")
{
    qftReg->SetPermutation(0x01);
    qftReg->H(0, 2);
    qftReg->AntiCNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->Z(0);
    qftReg->AntiCNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->AntiCNOT(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticy")
{
    bitLenInt controls[1] = { 0 };

    qftReg->SetPermutation(0x01);
    qftReg->H(0, 2);
    qftReg->AntiCY(0, 1);
    qftReg->MACInvert(controls, 1, -I_CMPLX, I_CMPLX, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->Z(0);
    qftReg->AntiCY(0, 1);
    qftReg->MACInvert(controls, 1, -I_CMPLX, I_CMPLX, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x01));

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 2);
    qftReg->AntiCY(0, 1);
    qftReg->MACInvert(controls, 1, -I_CMPLX, I_CMPLX, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccnot")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x03);
    qftReg->H(0, 3);
    qftReg->CCNOT(0, 1, 2);
    qftReg->H(2);
    qftReg->MCPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcnot")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 3);
    qftReg->AntiCCNOT(0, 1, 2);
    qftReg->H(2);
    qftReg->MACPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcy")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 3);
    qftReg->AntiCCY(0, 1, 2);
    qftReg->MACPhase(controls, 2, I_CMPLX, -I_CMPLX, 2);
    qftReg->H(2);
    qftReg->MACPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcz")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 3);
    qftReg->AntiCCZ(0, 1, 2);
    qftReg->MACPhase(controls, 2, I_CMPLX, -I_CMPLX, 2);
    qftReg->H(2);
    qftReg->MACPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap")
{
    qftReg->SetPermutation(0x80000);
    qftReg->Swap(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0x40000));

    qftReg->SetPermutation(0xc0000);
    qftReg->Swap(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0xc0000));

    qftReg->SetPermutation(0x80000);
    qftReg->Swap(17, 19);
    REQUIRE_THAT(qftReg, HasProbability(0x20000));

    qftReg->SetPermutation(0xa0000);
    qftReg->Swap(17, 19);
    REQUIRE_THAT(qftReg, HasProbability(0xa0000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_iswap")
{
    qftReg->SetPermutation(1);
    qftReg->ISwap(0, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetPermutation(0);
    qftReg->H(0, 2);
    qftReg->ISwap(0, 1);
    qftReg->ISwap(0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->CSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->SetPermutation(0x101);
    qftReg->CSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
    qftReg->H(8);
    qftReg->CSwap(control, 1, 0, 4);
    qftReg->CSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));

    QInterfacePtr qftReg2 = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 20U, 0,
        rng, ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng, sparse, REAL1_DEFAULT_ARG,
        devList, 10);

    control[0] = 9;
    qftReg2->SetPermutation((1U << 9U) | (1U << 10U));
    qftReg2->CSwap(control, 1, 10, 11);
    REQUIRE_THAT(qftReg2, HasProbability(0, 12, (1U << 9U) | (1U << 11U)));

    qftReg2->SetPermutation((1U << 9U) | (1U << 10U));
    qftReg2->CSwap(control, 1, 10, 0);
    REQUIRE_THAT(qftReg2, HasProbability(0, 12, (1U << 9U) | 1U));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->AntiCSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
    qftReg->H(8);
    qftReg->AntiCSwap(control, 1, 0, 4);
    qftReg->AntiCSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_csqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->SetPermutation(0x101);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
    qftReg->H(8);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticsqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
    qftReg->H(8);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cisqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->H(8);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticisqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->H(8);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fsim")
{
    real1_f theta = 3 * PI_R1 / 2;

    qftReg->SetPermutation(1);
    qftReg->FSim(theta, ZERO_R1, 0, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetPermutation(0);
    qftReg->H(0, 2);
    qftReg->FSim(theta, ZERO_R1, 0, 1);
    qftReg->FSim(theta, ZERO_R1, 0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    real1_f phi = PI_R1;

    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->FSim(ZERO_R1, phi, 4, 0);
    qftReg->FSim(ZERO_R1, phi, 5, 1);
    qftReg->FSim(ZERO_R1, phi, 6, 2);
    qftReg->FSim(ZERO_R1, phi, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));

    qftReg->SetPermutation(0x03);
    qftReg->H(0);
    qftReg->FSim(ZERO_R1, phi, 0, 1);
    qftReg->FSim(ZERO_R1, phi, 0, 1);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_single_bit")
{
    complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->Mtrx(pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->Mtrx(pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_controlled_single_bit")
{
    complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x8000F);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x8000F));
    qftReg->MCMtrx(controls, 3, pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x0F));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCMtrx(controls, 3, pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->MCMtrx(controls, 3, pauliX, 19);
    qftReg->MCMtrx(controls, 3, pauliX, 19);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCMtrx(NULL, 0, pauliX, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_controlled_single_invert")
{
    complex topRight = ONE_CMPLX;
    complex bottomLeft = ONE_CMPLX;
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x8000F);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x8000F));
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x0F));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCInvert(NULL, 0, topRight, bottomLeft, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_anticontrolled_single_bit")
{
    complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x80000);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
    qftReg->MACMtrx(controls, 3, pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MACMtrx(controls, 3, pauliX, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->MACMtrx(controls, 3, pauliX, 19);
    qftReg->MACMtrx(controls, 3, pauliX, 19);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MACMtrx(NULL, 0, pauliX, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->MACPhase(NULL, 0, ONE_CMPLX, -ONE_CMPLX, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->MACPhase(NULL, 0, ONE_CMPLX, -ONE_CMPLX, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_anticontrolled_single_invert")
{
    complex topRight = ONE_CMPLX;
    complex bottomLeft = ONE_CMPLX;
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x80000);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    qftReg->MCInvert(controls, 3, topRight, bottomLeft, 19);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->MCInvert(NULL, 0, topRight, bottomLeft, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->MACPhase(NULL, 0, ONE_CMPLX, -ONE_CMPLX, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->MACPhase(NULL, 0, ONE_CMPLX, -ONE_CMPLX, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_single_invert")
{
    qftReg->SetPermutation(0x01);
    qftReg->Invert(ONE_CMPLX, ONE_CMPLX, 0);
    REQUIRE_THAT(qftReg, HasProbability(0x00));

    qftReg->SetPermutation(0x00);
    qftReg->H(0);
    qftReg->Invert(ONE_CMPLX, -ONE_CMPLX, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_controlled_single_phase")
{
    bitLenInt controls[1] = { 0 };

    qftReg->SetPermutation(0x01);
    qftReg->H(1);
    qftReg->MCPhase(NULL, 0U, ONE_CMPLX, -ONE_CMPLX, 1U);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0x03));

    qftReg->SetPermutation(0x01);
    qftReg->H(1);
    qftReg->MCPhase(controls, 1U, ONE_CMPLX, -ONE_CMPLX, 1U);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0x03));

    qftReg->SetPermutation(0x01);
    qftReg->H(1);
    qftReg->MCPhase(controls, 1U, ONE_CMPLX, ONE_CMPLX, 1U);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_anti_controlled_single_phase")
{
    bitLenInt controls[1] = { 0 };

    qftReg->SetPermutation(0x00);
    qftReg->H(1);
    qftReg->MACPhase(NULL, 0U, ONE_CMPLX, -ONE_CMPLX, 1U);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0x02));

    qftReg->SetPermutation(0x00);
    qftReg->H(1);
    qftReg->MACPhase(controls, 1U, ONE_CMPLX, -ONE_CMPLX, 1U);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_u")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->U(0, M_PI / 2, 0, M_PI);
    qftReg->S(0);
    qftReg->S(0);
    qftReg->U2(0, 0, M_PI);
    qftReg->U(1, M_PI / 2, 0, M_PI);
    qftReg->S(1);
    qftReg->S(1);
    qftReg->U2(1, 0, M_PI);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_s")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->S(0);
    qftReg->S(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->S(1);
    qftReg->S(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0);
    qftReg->S(0);
    qftReg->H(0);
    qftReg->H(0);
    qftReg->S(0);
    qftReg->S(0);
    qftReg->S(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->S(0);
    qftReg->IS(0);
    qftReg->Phase(ONE_CMPLX, I_CMPLX, 0);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->Prob(0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_is")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->S(0);
    qftReg->IS(0);
    qftReg->IS(1);
    qftReg->S(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0);
    qftReg->IS(0);
    qftReg->H(0);
    qftReg->H(0);
    qftReg->IS(0);
    qftReg->IS(0);
    qftReg->IS(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_t")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sh")
{
    qftReg->SH(0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    qftReg->HIS(0);
    qftReg->HIS(1);
    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_it")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->T(0);
    qftReg->IT(0);
    qftReg->IT(1);
    qftReg->T(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cs")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0);
    qftReg->CS(4, 0);
    qftReg->CS(4, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->CS(4, 1);
    qftReg->CS(4, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0);
    qftReg->CS(4, 0);
    qftReg->H(0);
    qftReg->H(0);
    qftReg->CS(4, 0);
    qftReg->CS(4, 0);
    qftReg->CS(4, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetPermutation(2);
    qftReg->H(0);
    qftReg->CS(1, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);

    qftReg->SetPermutation(2);
    qftReg->H(0);
    qftReg->CS(1, 0);
    qftReg->CIS(1, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cis")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->CS(4, 0);
    qftReg->CIS(4, 0);
    qftReg->CIS(4, 1);
    qftReg->CS(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));

    qftReg->SetReg(0, 8, 0x11);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
    qftReg->H(0);
    qftReg->CIS(4, 0);
    qftReg->H(0);
    qftReg->H(0);
    qftReg->CIS(4, 0);
    qftReg->CIS(4, 0);
    qftReg->CIS(4, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ct")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0);
    qftReg->CT(4, 0);
    qftReg->CT(4, 0);
    qftReg->CT(4, 0);
    qftReg->CT(4, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->CT(4, 1);
    qftReg->CT(4, 1);
    qftReg->CT(4, 1);
    qftReg->CT(4, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cit")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->CT(4, 0);
    qftReg->CIT(4, 0);
    qftReg->CIT(4, 1);
    qftReg->CT(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_x")
{
    qftReg->SetPermutation(0xF0001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xF0001));
    qftReg->X(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x70001));
    qftReg->X(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xF0001));
    qftReg->H(19);
    qftReg->X(19);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xF0001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xmask")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->XMask(pow2Ocl(18) | pow2Ocl(19));
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ymask")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->YMask(pow2Ocl(18) | pow2Ocl(19));
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_zmask")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(18, 2);
    qftReg->ZMask(pow2Ocl(18) | pow2Ocl(19));
    qftReg->H(18, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40001));
    qftReg->H(18, 2);
    qftReg->ZMask(pow2Ocl(0) | pow2Ocl(18) | pow2Ocl(19));
    qftReg->H(18, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_phaseparity")
{
    qftReg->SetPermutation(0x40001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40001));
    qftReg->H(18, 2);
    qftReg->PhaseParity(PI_R1 / 2, pow2Ocl(0) | pow2Ocl(18) | pow2Ocl(19));
    qftReg->PhaseParity(PI_R1 / 2, pow2Ocl(0) | pow2Ocl(18) | pow2Ocl(19));
    qftReg->H(18, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(18, 2);
    qftReg->PhaseParity(PI_R1 / 2, pow2Ocl(0) | pow2Ocl(18) | pow2Ocl(19));
    qftReg->PhaseParity(PI_R1 / 2, pow2Ocl(18) | pow2Ocl(19));
    qftReg->H(18, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtx")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->SqrtX(19);
    qftReg->SqrtX(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->SqrtX(19);
    qftReg->SqrtX(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));

    qftReg->SqrtX(19);
    qftReg->ISqrtX(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));

    qftReg->ISqrtX(19);
    qftReg->ISqrtX(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ISqrtX(19);
    qftReg->ISqrtX(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtxconjt")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->SqrtXConjT(19);
    qftReg->SqrtXConjT(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));

    qftReg->SetPermutation(0x80001);
    qftReg->SqrtXConjT(19);
    qftReg->ISqrtXConjT(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_y")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->Y(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->Y(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrty")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->SqrtY(1);
    qftReg->SqrtY(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->SqrtH(1);
    qftReg->SqrtH(1);
    qftReg->SqrtY(1);
    qftReg->SqrtY(1);
    qftReg->SqrtH(1);
    qftReg->SqrtH(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SqrtY(1);
    qftReg->ISqrtY(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SqrtH(1);
    qftReg->SqrtH(1);
    qftReg->SqrtY(1);
    qftReg->SqrtY(1);
    qftReg->SqrtH(1);
    qftReg->SqrtH(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_z")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->Z(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->Z(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cy")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CY(4, 0);
    qftReg->CY(5, 1);
    qftReg->CY(6, 2);
    qftReg->CY(7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));

    qftReg->SetReg(0, 8, 0x10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x10));
    qftReg->H(0);
    qftReg->CY(4, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccy")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x03);
    qftReg->H(0, 3);
    qftReg->CCY(0, 1, 2);
    qftReg->MCPhase(controls, 2, I_CMPLX, -I_CMPLX, 2);
    qftReg->H(2);
    qftReg->MCPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cz")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CZ(4, 0);
    qftReg->CZ(5, 1);
    qftReg->CZ(6, 2);
    qftReg->CZ(7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));

    qftReg->SetPermutation(0x03);
    qftReg->H(0);
    qftReg->CZ(0, 1);
    qftReg->CZ(0, 1);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticz")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->X(4, 4);
    qftReg->AntiCZ(4, 0);
    qftReg->AntiCZ(5, 1);
    qftReg->AntiCZ(6, 2);
    qftReg->AntiCZ(7, 3);
    qftReg->X(4, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));

    qftReg->SetPermutation(0x03);
    qftReg->H(0);
    qftReg->X(0);
    qftReg->AntiCZ(0, 1);
    qftReg->AntiCZ(0, 1);
    qftReg->X(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ch")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CH(4, 0);
    qftReg->CH(5, 1);
    qftReg->CH(6, 2);
    qftReg->CH(7, 3);
    for (bitLenInt i = 0; i < 4; i++) {
        qftReg->Z(i);
    }
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rt")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->RT(M_PI, 0);
    qftReg->RT(M_PI, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->RT(M_PI, 1);
    qftReg->RT(M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rx")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RX(M_PI, 0);
    qftReg->RX(M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RY(M_PI, 0);
    qftReg->RY(M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry_continuous")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    for (int step = 0; step < 60; step++) {
        qftReg->RY(M_PI / 60, 0);
        qftReg->RY(M_PI / 60, 1);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rz")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->RZ(M_PI, 1);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniform_cry")
{
    bitLenInt controls[2] = { 4, 5 };
    real1 angles[4] = { PI_R1, PI_R1, ZERO_R1, ZERO_R1 };

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledRY(NULL, 0, 0, angles);
    qftReg->UniformlyControlledRY(NULL, 0, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));

    controls[0] = 5;
    controls[1] = 4;

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));

    controls[0] = 4;
    controls[1] = 5;

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(4);
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x02);
    QInterfacePtr qftReg2 = qftReg->Clone();
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    REQUIRE_THAT(qftReg2, HasProbability(0, 8, 0x02));

    if (!QINTERFACE_RESTRICTED) {
        qftReg->UniformlyControlledRY(controls, 2, 0, angles);
        qftReg2->QInterface::UniformlyControlledRY(controls, 2, 0, angles);

        REQUIRE(qftReg->ApproxCompare(qftReg2));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniform_crz")
{
    bitLenInt controls[2] = { 4, 5 };
    real1 angles[4] = { PI_R1, PI_R1, ZERO_R1, ZERO_R1 };

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(NULL, 0, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0x11);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));

    qftReg->SetReg(0, 8, 0x21);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));

    controls[0] = 5;
    controls[1] = 4;

    qftReg->SetReg(0, 8, 0x21);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x23));

    controls[0] = 4;
    controls[1] = 5;

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(4);
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniform_c_single")
{
    bitLenInt controls[2] = { 4, 5 };
    real1 angles[4] = { PI_R1, PI_R1, ZERO_R1, ZERO_R1 };
    complex pauliRYs[16];

    real1 cosine, sine;
    for (bitCapIntOcl i = 0; i < 4; i++) {
        cosine = (real1)cos(angles[i] / 2);
        sine = (real1)sin(angles[i] / 2);

        pauliRYs[0 + 4 * i] = complex(cosine, ZERO_R1);
        pauliRYs[1 + 4 * i] = complex(-sine, ZERO_R1);
        pauliRYs[2 + 4 * i] = complex(sine, ZERO_R1);
        pauliRYs[3 + 4 * i] = complex(cosine, ZERO_R1);
    }

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledSingleBit(NULL, 0, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(NULL, 0, 1, pauliRYs);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->H(4);
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));

    controls[0] = 5;
    controls[1] = 4;

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));

    controls[0] = 4;
    controls[1] = 5;

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(4);
    qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
    qftReg->UniformlyControlledSingleBit(controls, 2, 1, pauliRYs);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x02);
    QInterfacePtr qftReg2 = qftReg->Clone();
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    REQUIRE_THAT(qftReg2, HasProbability(0, 8, 0x02));

    if (!QINTERFACE_RESTRICTED) {
        qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
        qftReg2->QInterface::UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);

        REQUIRE(qftReg->ApproxCompare(qftReg2));

        qftReg->SetReg(0, 8, 0x02);
        qftReg2 = qftReg->Clone();
        REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
        REQUIRE_THAT(qftReg2, HasProbability(0, 8, 0x02));

        qftReg->UniformlyControlledSingleBit(controls, 2, 0, pauliRYs);
        qftReg2->QInterface::UniformlyControlledSingleBit(controls, 2, 0, pauliRYs, NULL, 0, 0);

        REQUIRE(qftReg->ApproxCompare(qftReg2));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ai")
{
    real1_f azimuth = 0.9f * PI_R1;
    real1_f inclination = 0.7f * PI_R1;

    real1_f probZ = (ONE_R1 / 2) - cos(inclination) / 2;
    real1_f probX = (ONE_R1 / 2) - sin(inclination) * cos(azimuth) / 2;
    real1_f probY = (ONE_R1 / 2) - sin(inclination) * sin(azimuth) / 2;

    qftReg->SetPermutation(0);
    qftReg->AI(0, azimuth, inclination);
    real1_f testZ = qftReg->Prob(0);
    qftReg->H(0);
    real1_f testX = qftReg->Prob(0);
    qftReg->S(0);
    qftReg->H(0);
    real1_f testY = qftReg->Prob(0);
    qftReg->H(0);
    qftReg->IS(0);
    qftReg->H(0);

    REQUIRE_FLOAT(probZ, testZ);
    REQUIRE_FLOAT(probX, testX);
    REQUIRE_FLOAT(probY, testY);

    qftReg->IAI(0, azimuth, inclination);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
}

#if ENABLE_ROT_API
TEST_CASE_METHOD(QInterfaceTestFixture, "test_crt")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRT(M_PI, 4, 0);
    qftReg->CRT(M_PI, 4, 0);
    qftReg->CRT(M_PI, 5, 1);
    qftReg->CRT(M_PI, 5, 1);
    qftReg->CRT(M_PI, 6, 2);
    qftReg->CRT(M_PI, 6, 2);
    qftReg->CRT(M_PI, 7, 3);
    qftReg->CRT(M_PI, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crtdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRTDyad(1, 1, 4, 0);
    qftReg->CRTDyad(1, 1, 4, 0);
    qftReg->CRTDyad(1, 1, 5, 1);
    qftReg->CRTDyad(1, 1, 5, 1);
    qftReg->CRTDyad(1, 1, 6, 2);
    qftReg->CRTDyad(1, 1, 6, 2);
    qftReg->CRTDyad(1, 1, 7, 3);
    qftReg->CRTDyad(1, 1, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rxdyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RXDyad(1, 1, 0);
    qftReg->RXDyad(1, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crx")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRX(M_PI, 4, 0);
    qftReg->CRX(M_PI, 5, 1);
    qftReg->CRX(M_PI, 6, 2);
    qftReg->CRX(M_PI, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crxdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRXDyad(1, 1, 4, 0);
    qftReg->CRXDyad(1, 1, 5, 1);
    qftReg->CRXDyad(1, 1, 6, 2);
    qftReg->CRXDyad(1, 1, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rydyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RYDyad(1, 1, 0);
    qftReg->RYDyad(1, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cry")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRY(M_PI, 4, 0);
    qftReg->CRY(M_PI, 5, 1);
    qftReg->CRY(M_PI, 6, 2);
    qftReg->CRY(M_PI, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crydyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRYDyad(1, 1, 4, 0);
    qftReg->CRYDyad(1, 1, 5, 1);
    qftReg->CRYDyad(1, 1, 6, 2);
    qftReg->CRYDyad(1, 1, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rzdyad")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZDyad(1, 1, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crz")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZ(M_PI, 4, 0);
    qftReg->CRZ(M_PI, 5, 1);
    qftReg->CRZ(M_PI, 6, 2);
    qftReg->CRZ(M_PI, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crzdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZDyad(1, 1, 4, 0);
    qftReg->CRZDyad(1, 1, 5, 1);
    qftReg->CRZDyad(1, 1, 6, 2);
    qftReg->CRZDyad(1, 1, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expx")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpX(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ExpX(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_exp")
{
    qftReg->SetPermutation(0x80001);
    qftReg->Exp(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expdyad")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpDyad(4, 1, 19);
    qftReg->SetPermutation(0x80001);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expxdyad")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpXDyad(4, 1, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ExpXDyad(4, 1, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expy")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->ExpY(2.0 * M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->ExpY(2.0 * M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expydyad")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->ExpYDyad(4, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->ExpYDyad(4, 1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expz")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->ExpZ(2.0 * M_PI, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->ExpZ(2.0 * M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expzdyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->ExpZDyad(4, 1, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->ExpZDyad(4, 1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_x_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->X(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
    qftReg->X(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x1d));
}

#if ENABLE_REG_GATES
TEST_CASE_METHOD(QInterfaceTestFixture, "test_y_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->Y(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->Y(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_z_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->Z(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_u_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->U(1, 2, M_PI / 2, 0, M_PI);
    qftReg->S(1, 2);
    qftReg->S(1, 2);
    qftReg->U2(1, 2, 0, M_PI);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_s_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->S(1, 2);
    qftReg->S(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_is_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->S(1, 2);
    qftReg->IS(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_t_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_it_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->T(1, 2);
    qftReg->IT(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cs_reg")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(1, 2);
    qftReg->CS(4, 1, 2);
    qftReg->CS(4, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cis_reg")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->CS(4, 1, 2);
    qftReg->CIS(4, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ct_reg")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(1, 2);
    qftReg->CT(4, 1, 2);
    qftReg->CT(4, 1, 2);
    qftReg->CT(4, 1, 2);
    qftReg->CT(4, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cit_reg")
{
    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->CT(4, 1, 2);
    qftReg->CIT(4, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtx_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->SqrtX(1, 4);
    qftReg->SqrtX(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
    qftReg->SqrtX(4, 1);
    qftReg->SqrtX(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x1d));

    qftReg->SqrtX(0, 4);
    qftReg->ISqrtX(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x1d));

    qftReg->ISqrtX(4, 1);
    qftReg->ISqrtX(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
    qftReg->ISqrtX(1, 4);
    qftReg->ISqrtX(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtxconjt_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->SqrtXConjT(1, 4);
    qftReg->SqrtXConjT(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetPermutation(0x1d);
    qftReg->SqrtXConjT(0, 4);
    qftReg->ISqrtXConjT(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x1d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrty_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->SqrtY(1, 4);
    qftReg->SqrtY(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->SqrtH(1, 2);
    qftReg->SqrtH(1, 2);
    qftReg->SqrtY(1, 2);
    qftReg->SqrtY(1, 2);
    qftReg->SqrtH(1, 2);
    qftReg->SqrtH(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SqrtY(0, 2);
    qftReg->ISqrtY(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SqrtH(1, 2);
    qftReg->SqrtH(1, 2);
    qftReg->SqrtY(1, 2);
    qftReg->SqrtY(1, 2);
    qftReg->SqrtH(1, 2);
    qftReg->SqrtH(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cy_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CY(4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticz_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->X(4, 4);
    qftReg->AntiCZ(4, 0, 4);
    qftReg->X(4, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ch_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CH(4, 0, 4);
    qftReg->Z(0, 4);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_phaserootn_reg")
{
    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0);
    qftReg->PhaseRootN(1, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0, 2);
    qftReg->PhaseRootN(1, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0, 2);
    qftReg->PhaseRootN(0, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0, 2);
    qftReg->PhaseRootN(2, 0, 2);
    qftReg->PhaseRootN(2, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0, 2);
    qftReg->IPhaseRootN(2, 0, 2);
    qftReg->IPhaseRootN(2, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0);
    qftReg->PhaseRootN(2, 0);
    qftReg->PhaseRootN(2, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(0, 2);
    qftReg->PhaseRootN(2, 0, 2);
    qftReg->PhaseRootN(2, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->PhaseRootN(2, 1);
    qftReg->IPhaseRootN(2, 1);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetReg(0, 16, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0, 2);
    qftReg->CPhaseRootN(1, 4, 0);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));

    qftReg->SetReg(0, 16, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0, 2);
    qftReg->CPhaseRootN(1, 4, 0);
    qftReg->CIPhaseRootN(1, 4, 0);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));

    qftReg->SetReg(0, 16, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0, 2);
    qftReg->CPhaseRootN(1, 4, 0, 1);
    qftReg->CIPhaseRootN(1, 4, 0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));

    qftReg->SetReg(0, 16, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->H(0, 2);
    qftReg->CPhaseRootN(0, 4, 0, 1);
    qftReg->CIPhaseRootN(0, 4, 0, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_reg")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->Swap(0, 1, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_iswap_reg")
{
    qftReg->SetPermutation(0);
    qftReg->H(0, 4);
    qftReg->ISwap(0, 2, 2);
    qftReg->ISwap(0, 2, 2);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0F));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtswap_reg")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->SqrtSwap(0, 1, 1);
    qftReg->SqrtSwap(0, 1, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_isqrtswap_reg")
{
    qftReg->SetPermutation(0xb2000);
    qftReg->SqrtSwap(12, 16, 4);
    qftReg->ISqrtSwap(12, 16, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xb2000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fsim_reg")
{
    real1_f theta = 3 * PI_R1 / 2;

    qftReg->SetPermutation(0);
    qftReg->H(0, 4);
    qftReg->FSim(theta, ZERO_R1, 0, 2, 2);
    qftReg->FSim(theta, ZERO_R1, 0, 2, 2);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0F));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cnot_reg")
{
    qftReg->SetPermutation(0x55F00);
    REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    qftReg->CNOT(12, 4, 8);
    REQUIRE_THAT(qftReg, HasProbability(0x55A50));
    qftReg->SetPermutation(0x40001);
    REQUIRE_THAT(qftReg, HasProbability(0x40001));
    qftReg->CNOT(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0xC0001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticnot_reg")
{
    qftReg->SetPermutation(0x55F00);
    REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    qftReg->AntiCNOT(12, 4, 8);
    REQUIRE_THAT(qftReg, HasProbability(0x555A0));
    qftReg->SetPermutation(0x00001);
    REQUIRE_THAT(qftReg, HasProbability(0x00001));
    qftReg->AntiCNOT(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticy_reg")
{
    qftReg->SetPermutation(0x55F00);
    REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    qftReg->AntiCY(12, 4, 8);
    REQUIRE_THAT(qftReg, HasProbability(0x555A0));
    qftReg->SetPermutation(0x00001);
    REQUIRE_THAT(qftReg, HasProbability(0x00001));
    qftReg->AntiCY(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0x80001));
}
TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccnot_reg")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->CCNOT(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCA400));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcnot_reg")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->AntiCCNOT(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCAD00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccy_reg")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->CCY(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCA400));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcy_reg")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->AntiCCY(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCAD00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcz_reg")
{
    bitLenInt controls[2] = { 0, 1 };

    qftReg->SetPermutation(0x00);
    qftReg->H(0, 3);
    qftReg->AntiCCZ(0, 1, 2);
    qftReg->MACPhase(controls, 2, I_CMPLX, -I_CMPLX, 2);
    qftReg->H(2);
    qftReg->MACPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_and")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLAND(0, 0x0c, 4, 4); // 0x0e & 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xce));
    qftReg->SetPermutation(0x3e);
    qftReg->AND(0, 4, 8, 4); // 0xe & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x23e));
    qftReg->SetPermutation(0x03);
    qftReg->AND(0, 0, 8, 4); // 0x3 & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x303));
    qftReg->SetPermutation(0x3e);
    qftReg->NAND(0, 4, 8, 4); // ~(0xe & 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
    qftReg->SetPermutation(0x03);
    qftReg->NAND(0, 0, 8, 4); // ~(0x3 & 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0xc03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_or")
{
    qftReg->SetPermutation(0x0c);
    REQUIRE_THAT(qftReg, HasProbability(0x0c));
    qftReg->CLOR(0, 0x0d, 4, 4); // 0x0e | 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xdc));
    qftReg->SetPermutation(0x3e);
    qftReg->OR(0, 4, 8, 4); // 0xe | 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xf3e));
    qftReg->SetPermutation(0x03);
    qftReg->OR(0, 0, 8, 4); // 0x3 | 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x303));
    qftReg->SetPermutation(0x3e);
    qftReg->NOR(0, 4, 8, 4); // ~(0xe | 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0x03e));
    qftReg->SetPermutation(0x03);
    qftReg->NOR(0, 0, 8, 4); // ~(0x3 | 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0xc03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xor")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLXOR(0, 0x0d, 4, 4); // 0x0e ^ 0x0d
    REQUIRE_THAT(qftReg, HasProbability(0x3e));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 8, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
    qftReg->SetPermutation(0xe);
    qftReg->XOR(0, 0, 0, 4); // 0xe ^ 0xe
    REQUIRE_THAT(qftReg, HasProbability(0x0));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 0, 4); // 0x3 ^ 0xe
    REQUIRE_THAT(qftReg, HasProbability(0x3d));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 4, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xde));
    qftReg->SetPermutation(0x0e);
    qftReg->CLXOR(0, 0x0d, 0, 4); // 0x0e ^ 0x0d
    REQUIRE_THAT(qftReg, HasProbability(0x03));
    qftReg->SetPermutation(0x3e);
    qftReg->XNOR(0, 4, 8, 4); // ~(0xe ^ 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0x23e));
    qftReg->SetPermutation(0xe);
    qftReg->XNOR(0, 0, 0, 4); // ~(0xe ^ 0xe)
    REQUIRE_THAT(qftReg, HasProbability(0xf));
    qftReg->SetPermutation(0x3e);
    qftReg->XNOR(0, 4, 0, 4); // ~(0xe ^ 0x3)
    REQUIRE_THAT(qftReg, HasProbability(0x32));
    qftReg->SetPermutation(0x3e);
    qftReg->XNOR(0, 4, 4, 4); // ~(0x3 ^ 0xe)
    REQUIRE_THAT(qftReg, HasProbability(0x2e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_shunts")
{
    qftReg->H(0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->Swap(0, 0, 1);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->SqrtSwap(0, 0, 1);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->ISqrtSwap(0, 0, 1);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->CSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->CSqrtSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->CISqrtSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->AntiCSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->AntiCSqrtSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->AntiCISqrtSwap(NULL, 0, 0, 0);
    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

#if ENABLE_ROT_API
TEST_CASE_METHOD(QInterfaceTestFixture, "test_crt_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRT(M_PI, 4, 0, 4);
    qftReg->CRT(M_PI, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crtdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRTDyad(1, 1, 4, 0, 4);
    qftReg->CRTDyad(1, 1, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rx_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RX(M_PI, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rxdyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RXDyad(1, 1, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crx_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRX(M_PI, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crxdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRXDyad(1, 1, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RY(M_PI, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rydyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RYDyad(1, 1, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cry_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRY(M_PI, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crydyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRYDyad(1, 1, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rz_reg")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZ(M_PI, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rzdyad_reg")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZDyad(1, 1, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crz_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZ(M_PI, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crzdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZDyad(1, 1, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_exp_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->Exp(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expdyad_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expx_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpX(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expxdyad_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpXDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expy_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpY(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpY(2.0 * M_PI, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expydyad_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpYDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpYDyad(4, 1, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expz_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpZ(2.0 * M_PI, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expzdyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpZDyad(4, 1, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}
#endif
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rol")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ROL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(3));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ror")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ROR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(192));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qft_h")
{
    bitCapInt randPerm = (bitCapInt)(qftReg->Rand() * 256U);
    qftReg->SetPermutation(randPerm);

    int i;

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    qftReg->QFT(0, 8);

    qftReg->IQFT(0, 8);

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, 8, randPerm));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_isfinished")
{
    if (QINTERFACE_RESTRICTED) {
        // Just check that this doesn't throw execption.
        // (Might be in engine initialization, still, or not.)
        qftReg->isFinished();
    } else {
        REQUIRE(qftReg->isFinished());
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_tryseparate")
{
    bitLenInt toSep[2];

    qftReg->SetPermutation(85);

    int i;

    qftReg->QFT(0, 8);

    qftReg->IQFT(0, 8);

    for (i = 0; i < 8; i++) {
        qftReg->TrySeparate(i);
        toSep[0] = i;
        qftReg->TrySeparate(toSep, 1, FP_NORM_EPSILON);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, 8, 85));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->CNOT(0, 2);
    qftReg->CNOT(0, 2);
    qftReg->TrySeparate(0, 1);
    toSep[0] = 0;
    toSep[1] = 1;
    qftReg->TrySeparate(toSep, 2, FP_NORM_EPSILON);
    qftReg->CNOT(0, 1);
    qftReg->Z(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_zero_phase_flip")
{
    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1);
    qftReg->ZeroPhaseFlip(1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    QInterfacePtr qftReg2 = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 20U, 0,
        rng, ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng, sparse, REAL1_DEFAULT_ARG,
        devList, 10);

    qftReg2->SetPermutation(3U << 9U);
    qftReg2->H(10);
    qftReg2->ZeroPhaseFlip(10, 1);
    qftReg2->H(10);
    REQUIRE_THAT(qftReg2, HasProbability(0, 12, (1U << 9U)));

    qftReg2->SetPermutation(3U << 9U);
    qftReg2->H(9, 2);
    qftReg2->ZeroPhaseFlip(9, 2);
    qftReg2->H(9, 2);
    REQUIRE_THAT(qftReg2, HasProbability(0, 12, 0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_phase_flip")
{
    qftReg->SetReg(0, 8, 0x00);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->PhaseFlip();
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m")
{
    REQUIRE(qftReg->M(0) == 0);
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE(qftReg->M(0) == true);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mreg")
{
    qftReg->SetReg(0, 8, 0);
    REQUIRE(qftReg->MReg(0, 8) == 0);
    qftReg->SetReg(0, 8, 0x2b);
    REQUIRE(qftReg->MReg(0, 8) == 0x2b);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m_array")
{
    bitLenInt bits[3] = { 0, 2, 3 };
    REQUIRE(qftReg->M(0) == 0);
    qftReg->SetReg(0, 8, 0x07);
    REQUIRE(qftReg->M(bits, 3) == 5);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x07));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_clone")
{
    qftReg->SetPermutation(0x2b);
    QInterfacePtr qftReg2 = qftReg->Clone();
    qftReg2->X(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x2b));
    REQUIRE_THAT(qftReg2, HasProbability(0, 20, 0xd4));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decompose")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x0b, rng);
    QInterfacePtr qftReg2 =
        CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x02, rng);
    qftReg->Compose(qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));

    qftReg->Decompose(0, qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));

    qftReg->Compose(qftReg2);

    // Try across device/heap allocation case:
    qftReg2 = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0, rng, ONE_CMPLX,
        enable_normalization, true, true, device_id, !disable_hardware_rng, sparse, REAL1_EPSILON, devList);

    qftReg->SetPermutation(0x2b);
    qftReg->Decompose(0, qftReg2);

    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 8, 0x33, rng);
    qftReg2 = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x02, rng);
    qftReg->H(1, 2);
    qftReg->CNOT(1, 3);
    qftReg->CNOT(2, 4);
    qftReg->CNOT(1, 6);
    qftReg->CNOT(3, 6);
    qftReg->Decompose(1, qftReg2);
    qftReg2->CNOT(0, 2);
    qftReg2->CNOT(1, 3);
    qftReg2->H(0, 2);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x3));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0x9));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dispose")
{
    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(0, 4);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));

    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(4, 4);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dispose_perm")
{
    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(0, 4, 0xb);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));

    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(4, 4, 0x2);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_compose")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x0b, rng);
    QInterfacePtr qftReg2 =
        CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x02, rng);
    qftReg->Compose(qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));

    // Try across device/heap allocation case:
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x0b, rng);
    qftReg2 = CreateQuantumInterface(
        { testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x02, rng, ONE_CMPLX, false, true, true);
    qftReg->Compose(qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_trydecompose")
{
    if (testEngineType == QINTERFACE_QUNIT_MULTI || testEngineType == QINTERFACE_QPAGER ||
        testEngineType == QINTERFACE_STABILIZER_HYBRID || testSubEngineType == QINTERFACE_STABILIZER_HYBRID ||
        testEngineType == QINTERFACE_BDT || testSubEngineType == QINTERFACE_BDT ||
        testSubSubEngineType == QINTERFACE_BDT) {
        // Not yet supported.
        return;
    }

    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 8, 0, rng, ONE_CMPLX);
    QInterfacePtr qftReg2 =
        CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0, rng, ONE_CMPLX);

    qftReg->SetPermutation(0xb);
    qftReg->H(0, 4);
    for (bitLenInt i = 0; i < 4; i++) {
        qftReg->CNOT(i, 4 + i);
    }
    REQUIRE(qftReg->TryDecompose(0, qftReg2) == false);

    qftReg->SetPermutation(0x2b);
    REQUIRE(qftReg->TryDecompose(0, qftReg2) == true);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qunit_paging")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 18, 1, rng, ONE_CMPLX);
    QInterfacePtr qftReg2 =
        CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 2, rng, ONE_CMPLX);

    qftReg->H(0, 3);
    qftReg->CCZ(0, 1, 2);

    qftReg2->H(0, 3);
    qftReg2->CCZ(0, 1, 2);

    qftReg->Compose(qftReg2);

    qftReg->CCZ(0, 1, 2);
    qftReg->H(0, 3);

    qftReg->CCZ(18, 19, 20);
    qftReg->H(18, 3);

    REQUIRE_THAT(qftReg, HasProbability(0, 22, 1 | (2 << 18)));

    qftReg->H(0, 3);
    qftReg->CCZ(0, 1, 2);

    qftReg->Decompose(18, qftReg2);

    qftReg->CCZ(0, 1, 2);
    qftReg->H(0, 3);

    REQUIRE_THAT(qftReg, HasProbability(0, 18, 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_setbit")
{
    qftReg->SetPermutation(0x02);
    qftReg->SetBit(0, true);
    qftReg->SetBit(1, false);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_proball")
{
    qftReg->SetPermutation(0x02);
    REQUIRE(qftReg->ProbAll(0x02) > 0.99);
    REQUIRE(qftReg->ProbAll(0x03) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probreg")
{
    qftReg->SetPermutation(0x20);
    REQUIRE(qftReg->ProbReg(4, 4, 0x2) > 0.99);
    REQUIRE(qftReg->ProbReg(4, 4, 0x3) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probmask")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 8, 0, rng);
    qftReg->SetPermutation(0x21);
    REQUIRE(qftReg->ProbMask(0xF0, 0x20) > 0.99);
    REQUIRE(qftReg->ProbMask(0xF0, 0x40) < 0.01);
    REQUIRE(qftReg->ProbMask(0xF3, 0x21) > 0.99);

    qftReg->SetPermutation(0);
    qftReg->X(0);
    REQUIRE(qftReg->ProbMask(0x1, 0x1) > 0.99);
    REQUIRE(qftReg->ProbMask(0x2, 0x2) < 0.01);
    REQUIRE(qftReg->ProbMask(0x3, 0x3) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probmaskall")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 1, 0, rng);
    real1 probs1[2];
    qftReg->ProbMaskAll(1U, probs1);
    REQUIRE(probs1[0] > 0.99);
    REQUIRE(probs1[1] < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probbitsall")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 3, 5, rng);
    bitLenInt bits[2] = { 2, 1 };
    real1 probs1[4];
    qftReg->ProbBitsAll(bits, 2U, probs1);
    REQUIRE(probs1[0] < 0.01);
    REQUIRE(probs1[1] > 0.99);
    REQUIRE(probs1[2] < 0.01);
    REQUIRE(probs1[3] < 0.01);

    qftReg->H(2);

    qftReg->ProbBitsAll(bits, 2U, probs1);
    REQUIRE_FLOAT(probs1[0], 0.5);
    REQUIRE_FLOAT(probs1[1], 0.5);
    REQUIRE(probs1[2] < 0.01);
    REQUIRE(probs1[3] < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expectationbitsall")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 8, 0, rng);
    bitLenInt bits[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    qftReg->H(0, 8);
    REQUIRE_FLOAT(qftReg->ExpectationBitsAll(bits, 8U), 127 + (ONE_R1 / 2))
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probparity")
{
    qftReg->SetPermutation(0x02);
    REQUIRE(QPARITY(qftReg)->ProbParity(0x7) > 0.99);
    qftReg->X(0);
    REQUIRE(QPARITY(qftReg)->ProbParity(0x7) < 0.01);

    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->X(0);
    REQUIRE(QPARITY(qftReg)->ProbParity(0x3) > 0.99);

    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    REQUIRE_FLOAT(QPARITY(qftReg)->ProbParity(0x3), ONE_R1 / 2);

    qftReg->SetPermutation(0x0);
    qftReg->H(1);
    REQUIRE_FLOAT(QPARITY(qftReg)->ProbParity(0x3), ONE_R1 / 2);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mparity")
{
    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    REQUIRE(QPARITY(qftReg)->ForceMParity(0x1, true, true));
    REQUIRE(QPARITY(qftReg)->MParity(0x1));

    qftReg->SetPermutation(0x02);
    REQUIRE(QPARITY(qftReg)->MParity(0x7));
    qftReg->X(0);
    REQUIRE(!(QPARITY(qftReg)->MParity(0x7)));

    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    REQUIRE(!(QPARITY(qftReg)->ForceMParity(0x3, false, true)));
    REQUIRE(!(QPARITY(qftReg)->MParity(0x3)));

    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->CNOT(1, 2);
    REQUIRE(!(QPARITY(qftReg)->ForceMParity(0x3, false, true)));
    REQUIRE_THAT(qftReg, HasProbability(0x0));

    qftReg->SetPermutation(0x0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->H(2);
    REQUIRE(QPARITY(qftReg)->ForceMParity(0x7, true, true));
    REQUIRE_THAT(qftReg, HasProbability(0x4));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniformparityrz")
{
    qftReg->SetPermutation(0);
    qftReg->H(0);
    QPARITY(qftReg)->UniformParityRZ(1, M_PI_2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0x1));

    qftReg->SetPermutation(0x3);
    qftReg->H(0, 3);
    QPARITY(qftReg)->UniformParityRZ(0x7, M_PI_2);
    qftReg->H(0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0x4));

    qftReg->SetPermutation(0x1);
    qftReg->H(0, 3);
    QPARITY(qftReg)->UniformParityRZ(0x7, M_PI_2);
    QPARITY(qftReg)->UniformParityRZ(0x7, M_PI_2);
    qftReg->H(0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0x1));

    qftReg->SetPermutation(0x01);
    qftReg->H(0);
    QPARITY(qftReg)->UniformParityRZ(1, M_PI_4);
    qftReg->S(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cuniformparityrz")
{
    bitLenInt controls[2] = { 3, 4 };

    qftReg->SetPermutation(0);
    qftReg->H(0);
    QPARITY(qftReg)->CUniformParityRZ(controls, 2, 1, M_PI_2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0));

    qftReg->SetPermutation(0x18);
    qftReg->H(0);
    QPARITY(qftReg)->CUniformParityRZ(controls, 2, 1, M_PI_2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0x1 | 0x18));

    qftReg->SetPermutation(0x3 | 0x18);
    qftReg->H(0, 3);
    QPARITY(qftReg)->CUniformParityRZ(controls, 1, 0x7, M_PI_2);
    qftReg->H(0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0x4 | 0x18));

    qftReg->SetPermutation(0x1 | 0x18);
    qftReg->H(0, 3);
    QPARITY(qftReg)->CUniformParityRZ(controls, 2, 0x7, M_PI_2);
    QPARITY(qftReg)->CUniformParityRZ(controls, 2, 0x7, M_PI_2);
    qftReg->H(0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0x1 | 0x18));

    qftReg->SetPermutation(0x01 | 0x18);
    qftReg->H(0);
    QPARITY(qftReg)->CUniformParityRZ(controls, 2, 1, M_PI_4);
    qftReg->S(0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0x0 | 0x18));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_multishotmeasuremask")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 8, 0, rng);

    bitCapInt qPowers[3] = { pow2(6), pow2(2), pow2(3) };

    qftReg->SetPermutation(0);
    qftReg->H(6);
    qftReg->X(2);
    qftReg->H(3);

    const std::set<bitCapInt> possibleResults = { 2, 3, 6, 7 };

    std::map<bitCapInt, int> results = qftReg->MultiShotMeasureMask(qPowers, 3U, 1000);
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end()) {
        REQUIRE(possibleResults.find(it->first) != possibleResults.end());
        it++;
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_forcem")
{
    qftReg->SetPermutation(0x0);
    qftReg->H(0, 4);

    REQUIRE_FLOAT(qftReg->ProbMask(0xF, 0), 0.0625);
    REQUIRE_FLOAT(qftReg->ProbMask(0x7, 0), 0.125);

    bitLenInt bits[3] = { 0, 1, 2 };
    bool results[3] = { 0, 1, 0 };

    qftReg->ForceM(bits, 1, results);
    qftReg->ForceM(bits, 3, results);
    qftReg->ForceM(bits, 1, NULL);
    qftReg->ForceMReg(0, 1, results[0], false);

    REQUIRE(qftReg->ProbMask(0x7, 0x2) > 0.99);
    REQUIRE_FLOAT(qftReg->ProbMask(0xF, 0x2), 0.5);

    qftReg->SetPermutation(0x0);
    qftReg->H(1);
    qftReg->CNOT(1, 2);
    qftReg->CNOT(2, 3);

    qftReg->ForceMReg(1, 2, 0x3, true);
    REQUIRE_THAT(qftReg, HasProbability(0xE));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_getamplitude")
{
    qftReg->SetPermutation(0x03);
    qftReg->H(0, 2);
    REQUIRE(norm((qftReg->GetAmplitude(0x01)) + (qftReg->GetAmplitude(0x03))) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_getquantumstate")
{
    complex state[1U << 4U];
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x0b, rng);
    qftReg->GetQuantumState(state);
    for (bitCapIntOcl i = 0; i < 16; i++) {
        if (i == 0x0b) {
            REQUIRE_FLOAT(norm(state[i]), ONE_R1);
        } else {
            REQUIRE_FLOAT(norm(state[i]), ZERO_R1);
        }
    }
    qftReg->SetQuantumState(state);

    complex state2[2] = { ZERO_CMPLX, ONE_CMPLX };
    QInterfacePtr qftReg2 =
        CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 1, 0, rng);
    qftReg2->SetQuantumState(state2);
    REQUIRE_THAT(qftReg2, HasProbability(1U));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_getprobs")
{
    real1 state[1U << 4U];
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 4, 0x0b, rng);
    qftReg->GetProbs(state);
    for (bitCapIntOcl i = 0; i < 16; i++) {
        if (i == 0x0b) {
            REQUIRE_FLOAT(state[i], ONE_R1);
        } else {
            REQUIRE_FLOAT(state[i], ZERO_R1);
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_normalize")
{
    qftReg->SetPermutation(0x03);
    qftReg->UpdateRunningNorm();
    qftReg->NormalizeState();
    REQUIRE_FLOAT(norm(qftReg->GetAmplitude(0x03)), ONE_R1);
}

#if ENABLE_ALU
TEST_CASE_METHOD(QInterfaceTestFixture, "test_asl")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ASL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(66));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_asr")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ASR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(96));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsl")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->LSL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsr")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->LSR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(64));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fulladd")
{
    qftReg->SetPermutation(0x00);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0x01);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x05));

    qftReg->SetPermutation(0x02);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x06));

    qftReg->SetPermutation(0x03);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0B));

    qftReg->SetPermutation(0x04);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SetPermutation(0x05);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x09));

    qftReg->SetPermutation(0x06);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0A));

    qftReg->SetPermutation(0x07);
    qftReg->FullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0F));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fulladd_noncoding")
{
    QInterfacePtr qftReg2 = qftReg->Clone();

    qftReg->SetPermutation(0x00 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x00 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x01 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x01 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x02 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x02 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x03 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x03 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x04 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x04 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x05 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x05 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x06 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x06 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));

    qftReg->SetPermutation(0x07 | 8);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg2->SetPermutation(0x07 | 8);
    qftReg2->QInterface::FullAdd(0, 1, 2, 3);
    REQUIRE(qftReg->MReg(0, 4) == qftReg2->MReg(0, 4));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ifulladd")
{
    // This is contingent on the previous test passing.

    qftReg->SetPermutation(0x00);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0x01);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetPermutation(0x02);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetPermutation(0x03);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetPermutation(0x04);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SetPermutation(0x05);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x05));

    qftReg->SetPermutation(0x06);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x06));

    qftReg->SetPermutation(0x07);
    qftReg->FullAdd(0, 1, 2, 3);
    qftReg->IFullAdd(0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x07));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc")
{
    qftReg->SetPermutation(0);
    qftReg->H(2, 2);
    qftReg->ADC(0, 2, 4, 2, 6);
    qftReg->IADC(0, 2, 4, 2, 6);
    qftReg->H(2, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->CNOT(0, 2);
    qftReg->ADC(0, 2, 4, 2, 6);
    qftReg->IADC(0, 2, 4, 2, 6);
    qftReg->CNOT(0, 2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(1);
    qftReg->ADC(0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 5));

    qftReg->SetPermutation(1);
    qftReg->ADC(0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_iadc")
{
    // This is contingent on the previous test passing.

    qftReg->SetPermutation(8);
    qftReg->H(2, 2);
    qftReg->ADC(0, 2, 4, 2, 6);
    qftReg->IADC(0, 2, 4, 2, 6);
    qftReg->H(2, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 8));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->CNOT(0, 2);
    qftReg->ADC(0, 2, 4, 2, 6);
    qftReg->IADC(0, 2, 4, 2, 6);
    qftReg->CNOT(0, 2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(2);
    qftReg->ADC(0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 6));
    qftReg->IADC(0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    qftReg->SetPermutation(2);
    qftReg->ADC(0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));
    qftReg->IADC(0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cfulladd")
{
    bitLenInt control[1] = { 10 };
    qftReg->SetPermutation(0x00); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0x01);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x05));

    qftReg->SetPermutation(0x02); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetPermutation(0x03);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0B));

    qftReg->SetPermutation(0x04); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SetPermutation(0x05);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x09));

    qftReg->SetPermutation(0x06); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x06));

    qftReg->SetPermutation(0x07);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0F));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cifulladd")
{
    // This is contingent on the previous test passing.

    bitLenInt control[1] = { 10 };

    qftReg->SetPermutation(0x00); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0x01);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetPermutation(0x02); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));

    qftReg->SetPermutation(0x03);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetPermutation(0x04); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));

    qftReg->SetPermutation(0x05);
    qftReg->X(control[0]); // on
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x05));

    qftReg->X(control[0]); // off

    qftReg->SetPermutation(0x06);
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x06));

    qftReg->SetPermutation(0x07); // off
    qftReg->CFullAdd(control, 1, 0, 1, 2, 3);
    qftReg->CIFullAdd(control, 1, 0, 1, 2, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x07));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cadc")
{
    bitLenInt control[1] = { 10 };

    qftReg->SetPermutation(0); // off
    qftReg->H(2, 2);
    qftReg->CADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->H(2, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(0);
    qftReg->X(control[0]); // on
    qftReg->H(0);
    qftReg->CNOT(0, 2);
    qftReg->CADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->CIADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->CNOT(0, 2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(1);
    qftReg->X(control[0]); // on
    qftReg->CADC(control, 1, 0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 5));

    qftReg->SetPermutation(1); // off
    qftReg->CADC(control, 1, 0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ciadc")
{
    // This is contingent on the previous test passing.
    bitLenInt control[1] = { 10 };

    qftReg->SetPermutation(8); // off
    qftReg->H(2, 2);
    qftReg->CADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->CIADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->H(2, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 8));

    qftReg->SetPermutation(0);
    qftReg->X(control[0]); // on
    qftReg->H(0);
    qftReg->CNOT(0, 2);
    qftReg->CADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->CIADC(control, 1, 0, 2, 4, 2, 6);
    qftReg->CNOT(0, 2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    qftReg->SetPermutation(2);
    qftReg->X(control[0]); // on
    qftReg->CADC(control, 1, 0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 6));
    qftReg->CIADC(control, 1, 0, 1, 2, 1, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    qftReg->SetPermutation(2); // off
    qftReg->CADC(control, 1, 0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));
    qftReg->CIADC(control, 1, 0, 1, 2, 0, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));
}

TEST_CASE("test_attach")
{
    if (testEngineType != QINTERFACE_BDT) {
        std::cout << ">>> 'test_attach': skipped" << std::endl;
        return;
    }
    std::cout << ">>> 'test_attach':" << std::endl;

    QInterfacePtr qftReg = CreateQuantumInterface({ QINTERFACE_BDT }, 1U, 0, rng);
    std::dynamic_pointer_cast<QBdt>(qftReg)->Attach(std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface({ QINTERFACE_STABILIZER_HYBRID }, 1U, 0, rng, CMPLX_DEFAULT_ARG, false, false)));

    qftReg->SetPermutation(0x2);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(1, 1, 0x1));
    qftReg->CZ(1, 0);
    REQUIRE_THAT(qftReg, HasProbability(1, 1, 0x1));
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0x3));

    qftReg->SetPermutation(0x2);
    qftReg->X(0);
    qftReg->H(1);
    qftReg->CNOT(1, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->CNOT(1, 0);
    qftReg->CNOT(1, 0);
    qftReg->H(1);
    qftReg->H(0);
    qftReg->CNOT(1, 0);
    qftReg->H(1);
    qftReg->X(0);
    REQUIRE(qftReg->MAll() == 0x2);

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->X(0);
    qftReg->CNOT(1, 0);
    qftReg->CNOT(1, 0);
    qftReg->X(0);
    qftReg->CNOT(0, 1);
    qftReg->H(0);
    REQUIRE(qftReg->MAll() == 0);
}

int qRand(int high, QInterfacePtr q)
{
    int rand = (int)(high * q->Rand());
    if (rand == high) {
        return high - 1;
    }
    return rand;
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_inc")
{
    int i;

    qftReg->SetPermutation(250);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->INC(1, 0, 8);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }

    for (i = 0; i < 16; i++) {
        int a = qRand(0x100, qftReg);
        int b = qRand(0x100, qftReg);
        int c = (a + b) & 0xFF;

        qftReg->SetPermutation(a);
        QALU(qftReg)->INC(b, 0, 8);
        REQUIRE_THAT(qftReg, HasProbability(0, 9, c));
    }

    qftReg->SetPermutation(255);
    qftReg->H(7);
    QALU(qftReg)->INC(1, 0, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(0));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(128));

    qftReg->SetPermutation(255);
    qftReg->H(7);
    qftReg->H(1);
    QALU(qftReg)->INC(1, 0, 8);
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(0));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(126));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(128));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(254));

    qftReg->SetPermutation(0);
    qftReg->H(7);
    qftReg->H(1);
    QALU(qftReg)->INC(1, 0, 8);
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(1));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(3));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(129));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(131));

    qftReg->SetPermutation(1);
    QALU(qftReg)->INC(8, 0, 8);
    QALU(qftReg)->DEC(8, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));

    qftReg->SetPermutation(0);
    QALU(qftReg)->INC(3, 0, 2);
    QALU(qftReg)->INC(1, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 5));

    qftReg->SetPermutation(0);
    qftReg->H(0, 8);
    QALU(qftReg)->INC(20, 0, 8);
    qftReg->H(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incs")
{
    REQUIRE(!isOverflowAdd(1, 1, 128, 256));
    REQUIRE(isOverflowAdd(127, 127, 128, 256));
    REQUIRE(isOverflowAdd(128, 128, 128, 256));

    int i;

    qftReg->SetPermutation(250);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->INCS(1, 0, 8, 9);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }

    qftReg->SetPermutation(255);
    qftReg->H(8);
    QALU(qftReg)->INCS(1, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(256));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(0));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    QALU(qftReg)->INCS(1, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(1));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(2));

    qftReg->SetPermutation(256);
    qftReg->H(7);
    QALU(qftReg)->INCS(1, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(257));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(385));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incc")
{
    int i;

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->INCC(1, 0, 8, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 2 + i - 8));
        }
    }

    qftReg->SetPermutation(255);
    qftReg->H(7);
    QALU(qftReg)->INCC(1, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(256));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(128));

    qftReg->SetPermutation(255);
    qftReg->H(7);
    QALU(qftReg)->INCC(255, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(510));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(382));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incsc")
{
    int i;

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->INCSC(1, 0, 8, 9, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 2 + i - 8));
        }
    }

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->INCSC(1, 0, 8, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 2 + i - 8));
        }
    }

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->H(1);
    QALU(qftReg)->INCSC(1, 0, 2, 3, 2);
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(1));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(2));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(3));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(4));

    qftReg->SetPermutation(0);
    qftReg->H(0);
    qftReg->H(9);
    QALU(qftReg)->INCSC(1, 0, 8, 9, 8);
    qftReg->H(9);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(1));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cinc", "[travis_xfail]")
{

    qftReg->SetPermutation(1);
    QALU(qftReg)->CINC(1, 0, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    bitLenInt controls[1] = { 8 };

    qftReg->SetPermutation(250);

    for (int i = 0; i < 8; i++) {
        // Turn control on
        qftReg->X(controls[0]);

        QALU(qftReg)->CINC(1, 0, 8, controls, 1);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }

        // Turn control off
        qftReg->X(controls[0]);

        QALU(qftReg)->CINC(1, 0, 8, controls, 1);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dec")
{
    int i;
    int start = 0x08;

    qftReg->SetPermutation(2);
    QALU(qftReg)->CDEC(1, 0, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->DEC(9, 0, 8);
        start -= 9;
        REQUIRE_THAT(qftReg, HasProbability(0, 19, 0xff - i * 9));
    }

    qftReg->SetPermutation(0);
    qftReg->H(0, 8);
    QALU(qftReg)->DEC(20, 0, 8);
    qftReg->H(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decs")
{
    REQUIRE(!isOverflowSub(1, 1, 128, 256));
    REQUIRE(isOverflowSub(1, 128, 128, 256));
    REQUIRE(isOverflowSub(128, 127, 128, 256));

    int i;
    int start = 0x08;

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->DECS(9, 0, 8, 9);
        start -= 9;
        REQUIRE_THAT(qftReg, HasProbability(0, 19, 0xff - i * 9));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decc")
{
    int i;

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->DECC(1, 0, 8, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decsc")
{
    int i;

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->DECSC(1, 0, 8, 9, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        QALU(qftReg)->DECSC(1, 0, 8, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cdec", "[travis_xfail]")
{
    int i;

    bitLenInt controls[1] = { 8 };

    qftReg->SetPermutation(0x08);
    for (i = 0; i < 8; i++) {
        // Turn control on
        qftReg->X(controls[0]);

        QALU(qftReg)->CDEC(9, 0, 8, controls, 1);
        REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xff - i * 9));

        // Turn control off
        qftReg->X(controls[0]);

        QALU(qftReg)->CDEC(9, 0, 8, controls, 1);
        REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xff - i * 9));
    }
}

#if ENABLE_BCD
TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcd")
{
    int i;

    qftReg->SetPermutation(0x95);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->INCBCD(1, 0, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x96 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 4));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcdc")
{
    int i;

    qftReg->SetPermutation(0x095);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->INCBCDC(1, 0, 8, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x096 + i));
        } else if (i == 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02 + i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcd")
{
    int i;

    qftReg->SetPermutation(0x94);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->DECBCD(1, 0, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x93 - i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x89 - i + 4));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcdc")
{
    int i;

    qftReg->SetPermutation(0x005);
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->DECBCDC(1, 0, 8, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x103 - i));
        } else if (i == 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x099));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x197 - i + 5));
        }
    }
}
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mul")
{
    int i;

    qftReg->SetPermutation(5);
    QALU(qftReg)->MUL(0, 0, 8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0));

    qftReg->SetPermutation(7);
    QALU(qftReg)->MUL(1, 0, 8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 7));

    qftReg->SetPermutation(3);
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        qftReg->SetReg(8, 8, 0x00);
        QALU(qftReg)->MUL(2, 0, 8, 8);
        res &= 0xFF;
        res *= 2;
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
    }

    qftReg->SetPermutation(0);
    qftReg->H(0);
    QALU(qftReg)->MUL(8, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(0));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(8));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_div")
{
    int i;

    qftReg->SetPermutation(256);
    bitCapInt res = 256;
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->DIV(2, 0, 8, 8);
        res /= 2;
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
    }

    qftReg->SetPermutation(0);
    qftReg->H(3);
    QALU(qftReg)->DIV(8, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(0));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mulmodnout")
{
    qftReg->SetPermutation(65);
    QALU(qftReg)->MULModNOut(5, 256U, 0, 8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 65 | (69 << 8)));

    qftReg->SetPermutation(0);
    qftReg->H(3);
    QALU(qftReg)->MULModNOut(2, 256U, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(0));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(8 | (16 << 8)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_imulmodnout")
{
    qftReg->SetPermutation(65 | (69 << 8));
    QALU(qftReg)->IMULModNOut(5, 256U, 0, 8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 65));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_powmodnout")
{
    qftReg->SetPermutation(6);
    QALU(qftReg)->POWModNOut(3, 256U, 0, 8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 6 | (217 << 8)));

    qftReg->SetPermutation(0);
    qftReg->H(1);
    QALU(qftReg)->POWModNOut(2, 256U, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(1 << 8));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(2 | (4 << 8)));

    qftReg->SetPermutation(0);
    qftReg->H(0, 2);
    QALU(qftReg)->POWModNOut(2, 256U, 0, 8, 8);
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(0 | (1 << 8)));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(1 | (2 << 8)));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(2 | (4 << 8)));
    REQUIRE_FLOAT(ONE_R1 / 4, qftReg->ProbAll(3 | (8 << 8)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cmul")
{
    int i;

    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(1);
    QALU(qftReg)->CMUL(2, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    qftReg->SetPermutation(3 | (1 << 16));
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->CMUL(2, 0, 8, 8, controls, 1);
        if ((i % 2) == 0) {
            res *= 2;
        }
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        res &= 255;
        qftReg->X(16);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cdiv")
{
    int i;

    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(2);
    QALU(qftReg)->CDIV(2, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));

    qftReg->SetPermutation(256 | (1 << 16));
    bitCapInt res = 256;
    for (i = 0; i < 8; i++) {
        QALU(qftReg)->CDIV(2, 0, 8, 8, controls, 1);
        if ((i % 2) == 0) {
            res /= 2;
        }
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        qftReg->X(16);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cmulmodnout", "[travis_xfail]")
{
    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(1);
    QALU(qftReg)->CMULModNOut(2, 256U, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1 | (2 << 8)));

    qftReg->SetPermutation(3 | (1 << 16));
    QALU(qftReg)->CMULModNOut(3, 256U, 0, 8, 8, controls, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 3 | (9 << 8)));

    qftReg->SetPermutation(3);
    qftReg->H(16);
    QALU(qftReg)->CMULModNOut(3, 256U, 0, 8, 8, controls, 1);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(3));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(3 | (9 << 8) | (1 << 16)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cimulmodnout")
{
    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(1);
    QALU(qftReg)->CMULModNOut(2, 256U, 0, 8, 8, NULL, 0);
    QALU(qftReg)->CIMULModNOut(2, 256U, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1));

    qftReg->SetPermutation(3 | (1 << 16));
    QALU(qftReg)->CMULModNOut(3, 256U, 0, 8, 8, controls, 1);
    QALU(qftReg)->CIMULModNOut(3, 256U, 0, 8, 8, controls, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 3 | (1 << 16)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cpowmodnout", "[travis_xfail]")
{
    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(1);
    QALU(qftReg)->CPOWModNOut(2, 256U, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1 | (2 << 8)));

    qftReg->SetPermutation(3 | (1 << 16));
    QALU(qftReg)->CPOWModNOut(3, 256U, 0, 8, 8, controls, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 3 | (27 << 8)));

    qftReg->SetPermutation(3);
    qftReg->H(16);
    QALU(qftReg)->CPOWModNOut(3, 256U, 0, 8, 8, controls, 1);
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(3));
    REQUIRE_FLOAT(ONE_R1 / 2, qftReg->ProbAll(3 | (27 << 8) | (1 << 16)));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_c_phase_flip_if_less", "[travis_xfail]")
{
    qftReg->SetReg(0, 20, 0x40000);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40000));
    qftReg->H(19);
    QALU(qftReg)->CPhaseFlipIfLess(1, 19, 1, 18);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xC0000));

    qftReg->SetReg(0, 20, 0x00);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00000));
    qftReg->H(19);
    QALU(qftReg)->CPhaseFlipIfLess(1, 19, 1, 18);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00000));

    qftReg->SetPermutation(0);
    qftReg->H(19);
    qftReg->H(18);
    QALU(qftReg)->CPhaseFlipIfLess(1, 19, 1, 18);
    qftReg->CZ(19, 18);
    qftReg->H(18);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1U << 18));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg")
{
    int j;

    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    QALU(qftReg)->IndexedLDA(0, 8, 8, 8, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x303));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0));

    qftReg->H(8, 8);
    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }

    QALU(qftReg)->IndexedLDA(8, 8, 0, 8, testPage);

    for (j = 0; j < 256; j++) {
        testPage[j] = 255 - j;
    }
    QALU(qftReg)->IndexedADC(8, 8, 0, 8, 16, testPage);
    qftReg->H(8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 0xff));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(1 << 16);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1 << 16));

    qftReg->H(8, 8);
    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    QALU(qftReg)->IndexedLDA(8, 8, 0, 8, testPage);

    QALU(qftReg)->IndexedSBC(8, 8, 0, 8, 16, testPage);
    qftReg->H(8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 1 << 16));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg_long")
{
    int j;

    qftReg->SetReg(0, 9, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    QALU(qftReg)->IndexedLDA(0, 9, 9, 9, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 0x603));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg_long_index")
{
    int j;

    qftReg->SetPermutation(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 18, 0));

    qftReg->H(9, 9);
    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }

    QALU(qftReg)->IndexedLDA(9, 9, 0, 9, testPage);

    for (j = 0; j < 512; j++) {
        testPage[j * 2] = (511 - j) & 0xff;
        testPage[j * 2 + 1] = ((511 - j) & 0x100) >> 8;
    }
    QALU(qftReg)->IndexedADC(9, 9, 0, 9, 18, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x1ff));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg_long_index")
{
    int j;

    qftReg->SetPermutation(1 << 18);
    REQUIRE_THAT(qftReg, HasProbability(0, 18, 1 << 18));

    qftReg->H(9, 9);
    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    QALU(qftReg)->IndexedLDA(9, 9, 0, 9, testPage);

    QALU(qftReg)->IndexedSBC(9, 9, 0, 9, 18, testPage);
    qftReg->H(9, 9);
    REQUIRE_THAT(qftReg, HasProbability(0, 19, 1 << 18));
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_hash")
{
    const bitCapIntOcl INPUT_KEY = 126;

    int j;

    qftReg->SetPermutation(INPUT_KEY);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, INPUT_KEY));

    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    std::random_shuffle(testPage, testPage + 256);

    QALU(qftReg)->Hash(0, 8, testPage);

    REQUIRE_THAT(qftReg, HasProbability(0, 8, testPage[INPUT_KEY]));

    qftReg->SetPermutation(0);
    qftReg->H(0, 8);
    QALU(qftReg)->Hash(0, 8, testPage);
    qftReg->H(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));

    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_grover")
{
    int i;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.

    const bitCapInt TARGET_PROB = 100;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(0, 8);

    // std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        QALU(qftReg)->DEC(100, 0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        QALU(qftReg)->INC(100, 0, 8);
        // This ends the "oracle."
        qftReg->H(0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->H(0, 8);
        qftReg->PhaseFlip();
        // std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qftReg->ProbAll(TARGET_PROB) << std::endl;
    }

    // std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    // std::cout << "Full Result:    " << qftReg << std::endl;
    // std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_grover_lookup")
{
    int i;

    // Grover's search to find a value in a lookup table.
    // We search for 100. All values in lookup table are 1 except a single match.

    const bitLenInt indexLength = 8;
    const bitLenInt valueLength = 9;
    const bitLenInt carryIndex = indexLength + valueLength;
    const bitCapIntOcl TARGET_VALUE = 100;
    const bitCapIntOcl TARGET_KEY = 230;

    unsigned char* toLoad = cl_alloc(2 * (1 << indexLength));
    for (i = 0; i < (2 * (1 << indexLength)); i += 2) {
        toLoad[i] = 1;
        toLoad[i + 1] = 0;
    }
    toLoad[2 * TARGET_KEY] = TARGET_VALUE;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(valueLength, indexLength);
    QALU(qftReg)->IndexedLDA(valueLength, indexLength, 0, valueLength, toLoad);

    // Twelve iterations maximizes the probablity for 256 searched elements, for example.
    // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
    int optIter = M_PI / (4.0 * asin(1.0 / sqrt(1 << indexLength)));

    for (i = 0; i < optIter; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        QALU(qftReg)->DEC(TARGET_VALUE, 0, valueLength);
        qftReg->ZeroPhaseFlip(0, valueLength);
        QALU(qftReg)->INC(TARGET_VALUE, 0, valueLength);
        // This ends the "oracle."
        qftReg->X(carryIndex);
        QALU(qftReg)->IndexedSBC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        qftReg->X(carryIndex);
        qftReg->H(valueLength, indexLength);
        qftReg->ZeroPhaseFlip(valueLength, indexLength);
        qftReg->H(valueLength, indexLength);
        // qftReg->PhaseFlip();
        QALU(qftReg)->IndexedADC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, indexLength + valueLength, TARGET_VALUE | (TARGET_KEY << valueLength)));
    cl_free(toLoad);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fast_grover")
{
    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    const bitLenInt length = 10;
    const bitCapInt TARGET_PROB = 100;
    bitLenInt i;
    bitLenInt partStart;
    // Start in a superposition of all inputs.
    qftReg->SetPermutation(0);
    // For Grover's search, our black box "oracle" would secretly return true for TARGET_PROB and false for all other
    // inputs. This is the function we are trying to invert. For an improvement in search speed, we require n/2 oracles
    // for an n bit search target. Each oracle marks 2 bits of the n total. This method might be applied to an ORDERED
    // lookup table search, in which a series of quaternary decisions can ultimately select any result in the list.
    for (i = 0; i < (length / 2); i++) {
        // This is the number of bits not yet fixed.
        partStart = length - ((i + 1) * 2);
        qftReg->H(partStart, 2);
        // We map from input to output.
        QALU(qftReg)->DEC(TARGET_PROB & (3 << partStart), 0, length);
        // Phase flip the target state.
        qftReg->ZeroPhaseFlip(partStart, 2);
        // We map back from outputs to inputs.
        QALU(qftReg)->INC(TARGET_PROB & (3 << partStart), 0, length);
        // Phase flip the input state from the previous iteration.
        qftReg->H(partStart, 2);
        qftReg->ZeroPhaseFlip(partStart, 2);
        qftReg->H(partStart, 2);
        // Now, we have one quarter as many states to look for.
    }

    REQUIRE_THAT(qftReg, HasProbability(0, length, TARGET_PROB));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_basis_change")
{
    int i;
    unsigned char* toSearch = cl_alloc(256);

    // Create the lookup table
    for (i = 0; i < 256; i++) {
        toSearch[i] = 100;
    }

    // Divide qftReg into two registers of 8 bits each
    qftReg->SetPermutation(0);
    qftReg->H(8, 8);
    QALU(qftReg)->IndexedLDA(8, 8, 0, 8, toSearch);
    qftReg->H(8, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, 100));
    cl_free(toSearch);
}
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_amplitude_amplification")
{
    int i;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true for an input of 000... or 110...

    int optIter = M_PI / (4.0 * asin(2.0 / sqrt(1U << 8U)));

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(0, 8);

    for (i = 0; i < optIter; i++) {
        // Our "oracle" is true for an input of "000..." or "110..." and false for all other inputs.
        qftReg->CNOT(0, 1);
        qftReg->H(0);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->H(0);
        qftReg->CNOT(0, 1);
        // This ends the "oracle."
        qftReg->H(0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->H(0, 8);
        qftReg->PhaseFlip();
    }

    qftReg->CNOT(0, 1);
    qftReg->H(0);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_set_reg")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
    qftReg->SetReg(0, 8, 10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_entanglement")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x0));
    for (bitLenInt i = 0; i < qftReg->GetQubitCount(); i += 2) {
        qftReg->X(i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x55555));
    for (bitLenInt i = 0; i < (qftReg->GetQubitCount() - 1); i += 2) {
        qftReg->CNOT(i, i + 1);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xfffff));
    for (bitLenInt i = qftReg->GetQubitCount() - 2; i > 0; i -= 2) {
        qftReg->CNOT(i - 1, i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xAAAAB));
    for (bitLenInt i = 1; i < qftReg->GetQubitCount(); i += 2) {
        qftReg->X(i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_bit")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->Swap(0, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtswap_bit")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->SqrtSwap(0, 1);
    qftReg->SqrtSwap(0, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_timeevolve")
{
    real1 aParam = (real1)1e-4f;
    real1 tDiff = (real1)2.1f;
    real1 e0 = (real1)sqrt(ONE_R1 - aParam * aParam);

    BitOp o2neg1(new complex[4], std::default_delete<complex[]>());
    o2neg1.get()[0] = complex(e0, ZERO_R1);
    o2neg1.get()[1] = complex(-aParam, ZERO_R1);
    o2neg1.get()[2] = complex(-aParam, ZERO_R1);
    o2neg1.get()[3] = complex(e0, ZERO_R1);

    HamiltonianOpPtr h0 = std::make_shared<HamiltonianOp>(0, o2neg1);
    Hamiltonian h(1);
    h[0] = h0;

    qftReg->SetPermutation(0);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);

    bitLenInt controls[1] = { 1 };
    bool controlToggles[1] = { false };

    HamiltonianOpPtr h1 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, false, controlToggles);
    h[0] = h1;

    // The point of this "toggle" behavior is to allow enumeration of arbitrary local Hamiltonian terms with
    // permutations of a set of control bits. For example, a Hamiltonian might represent an array of local
    // electromagnetic potential wells. If there are 4 wells, each with independent potentials, control "toggles" could
    // be used on two control bits, to enumerate all four permutations of two control bits with four different local
    // Hamiltonian terms.

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);

    controlToggles[0] = true;
    HamiltonianOpPtr h2 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, false, controlToggles);
    h[0] = h2;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(qftReg->Prob(0), ZERO_R1);

    controlToggles[0] = false;
    HamiltonianOpPtr h3 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, true, controlToggles);
    h[0] = h3;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(qftReg->Prob(0), ZERO_R1);

    controlToggles[0] = true;
    HamiltonianOpPtr h4 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, true, controlToggles);
    h[0] = h4;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_timeevolve_uniform")
{
    real1 aParam = (real1)1e-4f;
    real1 tDiff = (real1)2.1f;
    real1 e0 = (real1)sqrt(ONE_R1 - aParam * aParam);

    BitOp o2neg1(new complex[8], std::default_delete<complex[]>());
    o2neg1.get()[0] = ONE_CMPLX;
    o2neg1.get()[1] = ZERO_CMPLX;
    o2neg1.get()[2] = ZERO_CMPLX;
    o2neg1.get()[3] = ONE_CMPLX;
    o2neg1.get()[4] = complex(e0, ZERO_R1);
    o2neg1.get()[5] = complex(-aParam, ZERO_R1);
    o2neg1.get()[6] = complex(-aParam, ZERO_R1);
    o2neg1.get()[7] = complex(e0, ZERO_R1);

    bitLenInt controls[1] = { 1 };

    HamiltonianOpPtr h0 = std::make_shared<UniformHamiltonianOp>(controls, 1, 0, o2neg1);
    Hamiltonian h(1);
    h[0] = h0;

    REQUIRE(h0->uniform);

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qfusion_controlled")
{
    if (QINTERFACE_RESTRICTED) {
        return;
    }

    bitLenInt controls[2] = { 1, 2 };
    real1 angles[4] = { (real1)3.0f, (real1)0.8f, (real1)1.2f, (real1)0.7f };

    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 3, 0, rng);
    qftReg->SetPermutation(2);
    QInterfacePtr qftReg2 = qftReg->Clone();

    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg2->QInterface::UniformlyControlledRY(controls, 2, 0, angles);

    complex a, b;
    for (bitCapInt i = 0; i < 8; i++) {
        a = qftReg->GetAmplitude(i);
        b = qftReg2->GetAmplitude(i);
        REQUIRE_FLOAT(real(a), real(b));
        REQUIRE_FLOAT(imag(a), imag(b));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qneuron")
{
    const bitLenInt InputCount = 4;
    const bitLenInt OutputCount = 4;
    const bitCapInt InputPower = 1U << InputCount;
    const bitCapInt OutputPower = 1U << OutputCount;
    const real1_f eta = 0.5f;

    qftReg->Dispose(0, qftReg->GetQubitCount() - (InputCount + OutputCount));

    bitLenInt inputIndices[InputCount];
    for (bitLenInt i = 0; i < InputCount; i++) {
        inputIndices[i] = i;
    }

    std::vector<QNeuronPtr> outputLayer;
    for (bitLenInt i = 0; i < OutputCount; i++) {
        outputLayer.push_back(std::make_shared<QNeuron>(qftReg, inputIndices, InputCount, InputCount + i));
    }

    // Train the network to associate powers of 2 with their log2()
    bitCapInt perm, comp, test;
    bool bit;
    for (perm = 0; perm < InputPower; perm++) {
        comp = (~perm) + 1U;
        for (bitLenInt i = 0; i < OutputCount; i++) {
            qftReg->SetPermutation(perm);
            bit = (comp & pow2(i)) != 0;
            outputLayer[i]->LearnPermutation(bit, eta);
        }
    }

    for (perm = 0; perm < InputPower; perm++) {
        qftReg->SetPermutation(perm);
        for (bitLenInt i = 0; i < OutputCount; i++) {
            outputLayer[i]->Predict();
        }
        comp = qftReg->MReg(InputCount, OutputCount);
        test = ((~perm) + 1U) & (OutputPower - 1);
        REQUIRE(comp == test);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_bell_m")
{
    bitCapInt qPowers[2] = { 1, 2 };
    const std::set<bitCapInt> possibleResults = { 0, 3 };

    qftReg->SetPermutation(0);
    qftReg->H(0, 2);
    qftReg->CZ(0, 1);
    qftReg->H(0);
    std::map<bitCapInt, int> results = qftReg->MultiShotMeasureMask(qPowers, 2U, 1000);
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end()) {
        REQUIRE(possibleResults.find(it->first) != possibleResults.end());
        it++;
    }

    qftReg->SetPermutation(0);
    qftReg->H(0, 2);
    qftReg->CZ(0, 1);
    qftReg->H(1);
    results = qftReg->MultiShotMeasureMask(qPowers, 2U, 1000);
    it = results.begin();
    while (it != results.end()) {
        REQUIRE(possibleResults.find(it->first) != possibleResults.end());
        it++;
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_n_bell")
{
    int i;

    qftReg->SetPermutation(10);

    qftReg->H(0);
    for (i = 0; i <= 18; i++) {
        qftReg->CNOT(i, i + 1U);
    }
    for (i = 18; i >= 0; i--) {
        qftReg->CNOT(i, i + 1U);
    }
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_repeat_h_cnot")
{
    int i;

    qftReg->SetPermutation(10);

    for (i = 0; i <= 3; i++) {
        qftReg->H(i);
        qftReg->CNOT(i, i + 1U);
    }
    for (i = 3; i >= 0; i--) {
        qftReg->CNOT(i, i + 1U);
        qftReg->H(i);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, 20, 10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_invert_anti_pair")
{
    qftReg->SetPermutation(3);

    qftReg->H(0);
    qftReg->H(1);
    qftReg->CNOT(1, 0);
    qftReg->AntiCNOT(1, 0);
    qftReg->H(1);
    qftReg->H(0);

    REQUIRE_THAT(qftReg, HasProbability(0, 20, 3));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_universal_set")
{
    // Using any gate in this test, with any phase arguments, should form a universal algebra.

    qftReg->SetPermutation(0);

    qftReg->H(0);
    qftReg->Phase(ONE_CMPLX, -ONE_CMPLX, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));

    qftReg->Invert(ONE_CMPLX, ONE_CMPLX, 1);
    qftReg->H(0);
    qftReg->CZ(1, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 2));

    qftReg->CNOT(1, 0);
    qftReg->MAll();
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 3));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_teleport")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 3, 0);

    qftReg->SetPermutation(0);

    qftReg->H(1);
    qftReg->CNOT(1, 2);
    qftReg->CNOT(0, 1);
    qftReg->H(0);

    for (int i = 0; i < 10; i++) {
        QInterfacePtr suffix = qftReg->Clone();
        bool c0 = suffix->M(0);
        bool c1 = suffix->M(1);

        if (c0) {
            suffix->Z(2);
        }
        if (c1) {
            suffix->X(2);
        }

        REQUIRE(!suffix->M(2));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_h_cnot_rand")
{
    qftReg = CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, 2, 0);
    qftReg->H(0);
    qftReg->CNOT(0, 1);

    complex state[4];
    qftReg->GetQuantumState(state);
    REQUIRE_FLOAT(norm(state[0]), ONE_R1 / 2);
    REQUIRE_FLOAT(norm(state[1]), ZERO_R1);
    REQUIRE_FLOAT(norm(state[2]), ZERO_R1);
    REQUIRE_FLOAT(norm(state[3]), ONE_R1 / 2);

    bitCapInt qPowers[2] = { 1, 2 };
    std::map<bitCapInt, int> results = qftReg->MultiShotMeasureMask(qPowers, 2U, 1000);

    REQUIRE(results.find(1) == results.end());
    REQUIRE(results.find(2) == results.end());
    REQUIRE(results[0] > 450);
    REQUIRE(results[0] < 550);
    REQUIRE(results[3] > 450);
    REQUIRE(results[3] < 550);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_1", "[mirror]")
{
    qftReg->SetPermutation(7);

    qftReg->H(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->CNOT(1, 0);
    qftReg->CNOT(1, 0);
    qftReg->IT(0);
    qftReg->IT(0);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 7);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_2", "[mirror]")
{
    qftReg->SetPermutation(3);

    qftReg->H(0);
    qftReg->H(1);
    qftReg->CZ(1, 0);
    qftReg->CZ(1, 0);
    qftReg->H(1);
    qftReg->CNOT(1, 0);
    qftReg->CNOT(1, 0);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 3);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_3", "[mirror]")
{
    qftReg->SetReactiveSeparate(true);
    qftReg->SetPermutation(15);

    qftReg->H(1);
    qftReg->CNOT(1, 2);
    qftReg->X(1);
    qftReg->CCNOT(1, 2, 0);
    qftReg->CCNOT(1, 2, 0);
    qftReg->X(1);
    qftReg->CNOT(1, 2);
    qftReg->H(1);

    REQUIRE(qftReg->MAll() == 15);
}

// Broken with QUnit over QStabilizerHybrid
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_4", "[mirror]")
{
    qftReg->SetPermutation(1);

    qftReg->H(0);
    qftReg->T(0);
    qftReg->CNOT(0, 1);
    qftReg->Z(1);
    qftReg->T(0);
    qftReg->CNOT(1, 0);
    qftReg->CNOT(1, 0);
    qftReg->IT(0);
    qftReg->Z(1);
    qftReg->CNOT(0, 1);
    qftReg->IT(0);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 1);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_5", "[mirror]")
{
    qftReg->SetPermutation(4);

    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->H(1);
    qftReg->AntiCNOT(0, 1);
    qftReg->Z(1);
    qftReg->CNOT(1, 2);
    qftReg->CNOT(1, 2);
    qftReg->Z(1);
    qftReg->AntiCNOT(0, 1);
    qftReg->H(1);
    qftReg->CNOT(0, 1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 4);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_6", "[mirror]")
{
    qftReg->SetPermutation(0);

    qftReg->H(0);
    qftReg->T(0);
    qftReg->CNOT(0, 1);
    qftReg->T(0);
    qftReg->Z(1);
    qftReg->CZ(0, 1);
    qftReg->CZ(0, 1);
    qftReg->Z(1);
    qftReg->IT(0);
    qftReg->CNOT(0, 1);
    qftReg->IT(0);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 0);
}

// QUnit -> QStabilizerHybrid bug with QUnit::CNOT "pmBasis," ApplyEitherControlled "inCurrentBasis"
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_7", "[mirror]")
{
    qftReg->SetPermutation(10);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(0);
    qftReg->H(2);
    qftReg->CNOT(2, 3);
    qftReg->CNOT(0, 3);
    qftReg->CNOT(0, 3);
    qftReg->CNOT(2, 3);
    qftReg->H(2);
    qftReg->H(3);
    qftReg->CNOT(2, 3);
    qftReg->H(3);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 10);
}

// QUnit -> QStabilizerHybrid CZ/CY decomposition bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_8", "[mirror]")
{
    qftReg->SetPermutation(11);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(3);
    qftReg->CCNOT(0, 3, 2);
    qftReg->T(2);
    qftReg->CZ(1, 3);
    qftReg->CZ(2, 0);
    qftReg->T(2);
    qftReg->H(3);
    qftReg->H(3);
    qftReg->IT(2);
    qftReg->CZ(2, 0);
    qftReg->CZ(1, 3);
    qftReg->IT(2);
    qftReg->CCNOT(0, 3, 2);
    qftReg->H(3);

    REQUIRE(qftReg->MAll() == 11);
}

// QUnit -> QStabilizerHybrid CZ/CY decomposition bug (another)
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_9", "[mirror]")
{
    qftReg->SetPermutation(0);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->S(1);
    qftReg->T(0);
    qftReg->X(0);
    qftReg->T(0);
    qftReg->X(1);
    qftReg->CZ(0, 1);
    qftReg->CZ(0, 1);
    qftReg->X(1);
    qftReg->IT(0);
    qftReg->X(0);
    qftReg->IT(0);
    qftReg->IS(1);
    qftReg->CNOT(0, 1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 0);
}

// QUnit -> QStabilizerHybrid separability bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_10", "[mirror]")
{
    qftReg->SetPermutation(9);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(0);
    qftReg->H(1);
    qftReg->CCNOT(1, 3, 0);
    qftReg->Y(0);
    qftReg->Y(1);
    qftReg->H(3);
    qftReg->CNOT(0, 3);
    qftReg->X(3);
    qftReg->CZ(1, 3);
    qftReg->Y(0);
    qftReg->H(1);
    qftReg->H(1);
    qftReg->Y(0);
    qftReg->CZ(1, 3);
    qftReg->X(3);
    qftReg->CNOT(0, 3);
    qftReg->H(3);
    qftReg->Y(1);
    qftReg->Y(0);
    qftReg->CCNOT(1, 3, 0);
    qftReg->H(1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 9);
}

// QUnit -> QStabilizerHybrid TrimControls() bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_11", "[mirror]")
{
    qftReg->SetPermutation(1);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->X(0);
    qftReg->CNOT(0, 1);
    qftReg->H(1);
    qftReg->T(1);
    qftReg->IT(1);
    qftReg->H(1);
    qftReg->CNOT(0, 1);
    qftReg->X(0);
    qftReg->CNOT(0, 1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 1);
}

// QUnit -> QStabilizerHybrid bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_12", "[mirror]")
{
    qftReg->SetPermutation(0);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->H(0);
    qftReg->CNOT(1, 2);
    qftReg->H(1);
    qftReg->CZ(2, 1);
    qftReg->CNOT(0, 1);
    qftReg->T(1);
    qftReg->IT(1);
    qftReg->CNOT(0, 1);
    qftReg->CZ(2, 1);
    qftReg->H(1);
    qftReg->CNOT(1, 2);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 0);
}

// QUnit -> QStabilizerHybrid bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_13", "[mirror]")
{
    qftReg->SetPermutation(12);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(2);
    qftReg->H(1);
    qftReg->CCNOT(2, 1, 0);
    qftReg->T(2);
    qftReg->IT(2);
    qftReg->CCNOT(2, 1, 0);
    qftReg->H(1);
    qftReg->Y(0);
    qftReg->CNOT(2, 0);
    qftReg->CNOT(0, 2);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 13);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_14", "[mirror]")
{
    qftReg->SetPermutation(2);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(1);
    qftReg->H(3);
    qftReg->CNOT(3, 2);
    qftReg->Y(2);
    qftReg->CNOT(2, 3);
    qftReg->H(0);
    qftReg->CCNOT(0, 2, 3);
    qftReg->T(0);
    qftReg->IT(0);
    qftReg->CCNOT(0, 2, 3);
    qftReg->H(0);
    qftReg->CNOT(2, 3);
    qftReg->Y(2);
    qftReg->CNOT(3, 2);
    qftReg->H(3);
    qftReg->CCNOT(3, 1, 2);
    qftReg->H(1);

    REQUIRE(qftReg->MAll() == 2);
}

// If QUnit->QPager minPageQubits paging threshold is 1, this used to fail.
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_15", "[mirror]")
{
    qftReg->SetPermutation(5);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(3);
    qftReg->CNOT(3, 2);
    qftReg->CNOT(2, 3);
    qftReg->H(3);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->H(1);
    qftReg->CCNOT(3, 1, 0);
    qftReg->CCNOT(3, 1, 0);
    qftReg->H(1);
    qftReg->CNOT(0, 1);
    qftReg->H(0);
    qftReg->H(3);
    qftReg->CNOT(2, 3);
    qftReg->CNOT(3, 2);
    qftReg->H(3);

    REQUIRE(qftReg->MAll() == 5);
}

// If QUnit->QPager minPageQubits paging threshold is 1, this used to fail.
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_16", "[mirror]")
{
    qftReg->SetPermutation(2);

    qftReg->H(3);
    qftReg->CNOT(3, 0);
    qftReg->CCNOT(1, 0, 2);
    qftReg->H(0);
    qftReg->X(2);
    qftReg->T(3);
    qftReg->CNOT(0, 1);
    qftReg->H(0);
    qftReg->H(3);
    qftReg->H(2);
    qftReg->CZ(1, 3);
    qftReg->CNOT(0, 2);
    qftReg->CNOT(0, 2);
    qftReg->CZ(1, 3);
    qftReg->H(2);
    qftReg->H(3);
    qftReg->H(0);
    qftReg->CNOT(0, 1);
    qftReg->IT(3);
    qftReg->X(2);
    qftReg->H(0);
    qftReg->CCNOT(1, 0, 2);
    qftReg->CNOT(3, 0);
    qftReg->H(3);

    REQUIRE(qftReg->MAll() == 2);
}

// Deterministic QUnit bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_17", "[mirror]")
{
    qftReg->SetPermutation(7);
    qftReg->SetReactiveSeparate(true);

    qftReg->T(0);
    qftReg->Y(1);
    qftReg->H(2);
    qftReg->H(3);
    qftReg->Y(4);
    qftReg->H(5);
    qftReg->T(6);
    qftReg->H(7);
    qftReg->Swap(6, 0);
    qftReg->Swap(5, 4);
    qftReg->CZ(7, 1);
    qftReg->CZ(2, 3);
    qftReg->Y(0);
    qftReg->T(1);
    qftReg->Y(2);
    qftReg->X(3);
    qftReg->X(4);
    qftReg->T(5);
    qftReg->T(6);
    qftReg->X(7);
    qftReg->CNOT(7, 6);
    qftReg->CZ(1, 2);
    qftReg->Swap(5, 3);
    qftReg->CZ(4, 0);
    qftReg->Y(0);
    qftReg->Y(1);
    qftReg->Y(4);
    qftReg->H(5);
    qftReg->Y(7);
    qftReg->CZ(6, 0);
    qftReg->CCNOT(4, 3, 7);
    qftReg->CNOT(2, 5);
    qftReg->H(0);
    qftReg->T(1);
    qftReg->T(2);
    qftReg->Y(3);
    qftReg->H(5);
    qftReg->Y(6);
    qftReg->X(7);
    qftReg->CCNOT(2, 0, 3);
    qftReg->Swap(4, 7);
    qftReg->CZ(5, 6);
    qftReg->T(0);
    qftReg->X(1);
    qftReg->X(2);
    qftReg->T(3);
    qftReg->H(4);
    qftReg->H(5);
    qftReg->X(6);
    qftReg->Swap(6, 0);
    qftReg->Swap(5, 1);
    qftReg->CNOT(4, 7);
    qftReg->CNOT(2, 3);
    qftReg->T(0);
    qftReg->Y(1);
    qftReg->X(5);
    qftReg->H(6);
    qftReg->CNOT(5, 6);
    qftReg->CNOT(1, 2);
    qftReg->CZ(4, 3);
    qftReg->Swap(0, 7);
    qftReg->Swap(0, 7);
    qftReg->CZ(4, 3);
    qftReg->CNOT(1, 2);
    qftReg->CNOT(5, 6);
    qftReg->H(6);
    qftReg->X(5);
    qftReg->Y(1);
    qftReg->IT(0);
    qftReg->CNOT(2, 3);
    qftReg->CNOT(4, 7);
    qftReg->Swap(5, 1);
    qftReg->Swap(6, 0);
    qftReg->X(6);
    qftReg->H(5);
    qftReg->H(4);
    qftReg->IT(3);
    qftReg->X(2);
    qftReg->X(1);
    qftReg->IT(0);
    qftReg->CZ(5, 6);
    qftReg->Swap(4, 7);
    qftReg->CCNOT(2, 0, 3);
    qftReg->X(7);
    qftReg->Y(6);
    qftReg->H(5);
    qftReg->Y(3);
    qftReg->IT(2);
    qftReg->IT(1);
    qftReg->H(0);
    qftReg->CNOT(2, 5);
    qftReg->CCNOT(4, 3, 7);
    qftReg->CZ(6, 0);
    qftReg->Y(7);
    qftReg->H(5);
    qftReg->Y(4);
    qftReg->Y(1);
    qftReg->Y(0);
    qftReg->CZ(4, 0);
    qftReg->Swap(5, 3);
    qftReg->CZ(1, 2);
    qftReg->CNOT(7, 6);
    qftReg->X(7);
    qftReg->IT(6);
    qftReg->IT(5);
    qftReg->X(4);
    qftReg->X(3);
    qftReg->Y(2);
    qftReg->IT(1);
    qftReg->Y(0);
    qftReg->CZ(2, 3);
    qftReg->CZ(7, 1);
    qftReg->Swap(5, 4);
    qftReg->Swap(6, 0);
    qftReg->H(7);
    qftReg->IT(6);
    qftReg->H(5);
    qftReg->Y(4);
    qftReg->H(3);
    qftReg->H(2);
    qftReg->Y(1);
    qftReg->IT(0);

    REQUIRE(qftReg->MAll() == 7);
}

// Deterministic QUnit->QPager bug, when thresholds are low enough
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_18", "[mirror]")
{
    qftReg->SetPermutation(18);
    qftReg->SetReactiveSeparate(true);

    qftReg->T(1);
    qftReg->X(2);
    qftReg->T(3);
    qftReg->H(5);
    qftReg->CCNOT(2, 0, 1);
    qftReg->CZ(5, 4);
    qftReg->Y(1);
    qftReg->H(2);
    qftReg->X(3);
    qftReg->T(4);
    qftReg->X(5);
    qftReg->Swap(5, 4);
    qftReg->CNOT(1, 0);
    qftReg->Swap(2, 3);
    qftReg->X(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->X(4);
    qftReg->H(5);
    qftReg->CNOT(0, 5);
    qftReg->CNOT(4, 3);
    qftReg->CNOT(1, 2);
    qftReg->H(1);
    qftReg->X(4);
    qftReg->X(5);
    qftReg->Swap(1, 3);
    qftReg->CNOT(2, 0);
    qftReg->CNOT(5, 4);
    qftReg->X(1);
    qftReg->H(2);
    qftReg->X(3);
    qftReg->H(4);
    qftReg->H(5);
    qftReg->CCNOT(3, 4, 5);
    qftReg->CCNOT(0, 1, 2);
    qftReg->Y(1);
    qftReg->X(2);
    qftReg->T(3);
    qftReg->Y(4);
    qftReg->Swap(3, 5);
    qftReg->CNOT(2, 4);
    qftReg->Swap(0, 1);
    qftReg->Swap(0, 1);
    qftReg->CNOT(2, 4);
    qftReg->Swap(3, 5);
    qftReg->Y(4);
    qftReg->IT(3);
    qftReg->X(2);
    qftReg->Y(1);
    qftReg->CCNOT(0, 1, 2);
    qftReg->CCNOT(3, 4, 5);
    qftReg->H(5);
    qftReg->H(4);
    qftReg->X(3);
    qftReg->H(2);
    qftReg->X(1);
    qftReg->CNOT(5, 4);
    qftReg->CNOT(2, 0);
    qftReg->Swap(1, 3);
    qftReg->X(5);
    qftReg->X(4);
    qftReg->H(1);
    qftReg->CNOT(1, 2);
    qftReg->CNOT(4, 3);
    qftReg->CNOT(0, 5);
    qftReg->H(5);
    qftReg->X(4);
    qftReg->H(3);
    qftReg->H(1);
    qftReg->X(0);
    qftReg->Swap(2, 3);
    qftReg->CNOT(1, 0);
    qftReg->Swap(5, 4);
    qftReg->X(5);
    qftReg->IT(4);
    qftReg->X(3);
    qftReg->H(2);
    qftReg->Y(1);
    qftReg->CZ(5, 4);
    qftReg->CCNOT(2, 0, 1);
    qftReg->H(5);
    qftReg->IT(3);
    qftReg->X(2);
    qftReg->IT(1);

    REQUIRE(qftReg->MAll() == 18);
}

// Probabilistic QUnit bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_19", "[mirror]")
{
    qftReg->SetPermutation(11);
    qftReg->SetReactiveSeparate(true);

    qftReg->H(2);
    qftReg->T(2);
    qftReg->H(2);
    qftReg->H(0);
    qftReg->CNOT(2, 0);
    qftReg->H(3);
    qftReg->T(0);
    qftReg->CNOT(0, 1);
    qftReg->T(3);
    qftReg->T(1);
    qftReg->T(0);
    qftReg->CNOT(0, 3);
    qftReg->H(3);
    qftReg->CNOT(3, 1);
    qftReg->CNOT(3, 1);
    qftReg->H(3);
    qftReg->CNOT(0, 3);
    qftReg->IT(0);
    qftReg->IT(1);
    qftReg->IT(3);
    qftReg->CNOT(0, 1);
    qftReg->IT(0);
    qftReg->H(3);
    qftReg->CNOT(2, 0);
    qftReg->H(0);
    qftReg->H(2);
    qftReg->IT(2);
    qftReg->CNOT(0, 2);
    qftReg->H(2);

    REQUIRE(qftReg->MAll() == 11);
}

// QBinaryDecisionTree bug
TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_20", "[mirror]")
{
    qftReg = MakeEngine(2);
    qftReg->SetPermutation(2);

    qftReg->H(0);
    qftReg->H(1);
    qftReg->X(1);
    qftReg->CZ(0, 1);
    qftReg->H(0);
    qftReg->H(0);
    qftReg->CZ(0, 1);
    qftReg->X(1);
    qftReg->H(1);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 2);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_21", "[mirror]")
{
    qftReg = MakeEngine(6);
    qftReg->SetPermutation(34);

    qftReg->Z(0);
    qftReg->H(1);
    qftReg->T(2);
    qftReg->IT(3);
    qftReg->S(4);
    qftReg->X(5);
    qftReg->CNOT(4, 5);
    qftReg->AntiCNOT(2, 0);
    qftReg->CNOT(1, 3);
    qftReg->T(0);
    qftReg->T(1);
    qftReg->IS(2);
    qftReg->IT(3);
    qftReg->Y(4);
    qftReg->H(5);
    qftReg->AntiCY(0, 4);
    qftReg->AntiCZ(1, 3);
    qftReg->AntiCNOT(5, 2);
    qftReg->Y(0);
    qftReg->Y(1);
    qftReg->Y(2);
    qftReg->Y(3);
    qftReg->Z(4);
    qftReg->T(5);
    qftReg->CCNOT(4, 2, 5);
    qftReg->CY(1, 3);
    qftReg->T(0);
    qftReg->X(1);
    qftReg->Y(2);
    qftReg->T(3);
    qftReg->X(4);
    qftReg->IS(5);
    qftReg->AntiCCY(0, 2, 4);
    qftReg->AntiCCY(1, 3, 5);
    qftReg->IT(0);
    qftReg->T(1);
    qftReg->IS(2);
    qftReg->IT(3);
    qftReg->H(4);
    qftReg->X(5);
    qftReg->CCZ(3, 1, 4);
    qftReg->AntiCCNOT(5, 0, 2);
    qftReg->IT(0);
    qftReg->H(1);
    qftReg->X(2);
    qftReg->S(3);
    qftReg->Z(4);
    qftReg->Z(5);
    qftReg->AntiCNOT(5, 1);
    qftReg->AntiCCNOT(2, 3, 0);
    qftReg->AntiCCNOT(2, 3, 0);
    qftReg->AntiCNOT(5, 1);
    qftReg->Z(5);
    qftReg->Z(4);
    qftReg->IS(3);
    qftReg->X(2);
    qftReg->H(1);
    qftReg->T(0);
    qftReg->AntiCCNOT(5, 0, 2);
    qftReg->CCZ(3, 1, 4);
    qftReg->X(5);
    qftReg->H(4);
    qftReg->T(3);
    qftReg->S(2);
    qftReg->IT(1);
    qftReg->T(0);
    qftReg->AntiCCY(1, 3, 5);
    qftReg->AntiCCY(0, 2, 4);
    qftReg->S(5);
    qftReg->X(4);
    qftReg->IT(3);
    qftReg->Y(2);
    qftReg->X(1);
    qftReg->IT(0);
    qftReg->CY(1, 3);
    qftReg->CCNOT(4, 2, 5);
    qftReg->IT(5);
    qftReg->Z(4);
    qftReg->Y(3);
    qftReg->Y(2);
    qftReg->Y(1);
    qftReg->Y(0);
    qftReg->AntiCNOT(5, 2);
    qftReg->AntiCZ(1, 3);
    qftReg->AntiCY(0, 4);
    qftReg->H(5);
    qftReg->Y(4);
    qftReg->T(3);
    qftReg->S(2);
    qftReg->IT(1);
    qftReg->IT(0);
    qftReg->CNOT(1, 3);
    qftReg->AntiCNOT(2, 0);
    qftReg->CNOT(4, 5);
    qftReg->X(5);
    qftReg->IS(4);
    qftReg->T(3);
    qftReg->IT(2);
    qftReg->H(1);
    qftReg->Z(0);

    REQUIRE(qftReg->MAll() == 34);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_22", "[mirror]")
{
    qftReg = MakeEngine(6);
    qftReg->SetPermutation(37);

    qftReg->H(1);
    qftReg->H(2);
    qftReg->Y(3);
    qftReg->H(4);
    qftReg->CZ(3, 2);
    qftReg->CNOT(0, 4);
    qftReg->Swap(1, 5);
    qftReg->X(0);
    qftReg->T(1);
    qftReg->Y(2);
    qftReg->T(3);
    qftReg->H(4);
    qftReg->CCNOT(3, 2, 5);
    qftReg->CCNOT(4, 0, 1);
    qftReg->X(0);
    qftReg->X(1);
    qftReg->T(2);
    qftReg->H(4);
    qftReg->T(5);
    qftReg->CZ(0, 5);
    qftReg->CZ(2, 4);
    qftReg->Swap(1, 3);
    qftReg->T(0);
    qftReg->Y(1);
    qftReg->Y(3);
    qftReg->X(4);
    qftReg->T(5);
    qftReg->CNOT(3, 0);
    qftReg->CZ(5, 2);
    qftReg->CNOT(1, 4);
    qftReg->X(0);
    qftReg->T(2);
    qftReg->X(3);
    qftReg->X(4);
    qftReg->X(5);
    qftReg->CCNOT(2, 5, 1);
    qftReg->CZ(0, 3);
    qftReg->H(0);
    qftReg->X(1);
    qftReg->H(2);
    qftReg->X(3);
    qftReg->H(4);
    qftReg->H(5);
    qftReg->Swap(1, 2);
    qftReg->CNOT(5, 4);
    qftReg->CNOT(0, 3);
    qftReg->CNOT(0, 3);
    qftReg->CNOT(5, 4);
    qftReg->Swap(1, 2);
    qftReg->H(5);
    qftReg->H(4);
    qftReg->X(3);
    qftReg->H(2);
    qftReg->X(1);
    qftReg->H(0);
    qftReg->CZ(0, 3);
    qftReg->CCNOT(2, 5, 1);
    qftReg->X(5);
    qftReg->X(4);
    qftReg->X(3);
    qftReg->IT(2);
    qftReg->X(0);
    qftReg->CNOT(1, 4);
    qftReg->CZ(5, 2);
    qftReg->CNOT(3, 0);
    qftReg->IT(5);
    qftReg->X(4);
    qftReg->Y(3);
    qftReg->Y(1);
    qftReg->IT(0);
    qftReg->Swap(1, 3);
    qftReg->CZ(2, 4);
    qftReg->CZ(0, 5);
    qftReg->IT(5);
    qftReg->H(4);
    qftReg->IT(2);
    qftReg->X(1);
    qftReg->X(0);
    qftReg->CCNOT(4, 0, 1);
    qftReg->CCNOT(3, 2, 5);
    qftReg->H(4);
    qftReg->IT(3);
    qftReg->Y(2);
    qftReg->IT(1);
    qftReg->X(0);
    qftReg->Swap(1, 5);
    qftReg->CNOT(0, 4);
    qftReg->CZ(3, 2);
    qftReg->H(4);
    qftReg->Y(3);
    qftReg->H(2);
    qftReg->H(1);

    REQUIRE(qftReg->MAll() == 37);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_23", "[mirror]")
{
    qftReg = MakeEngine(6);
    qftReg->SetPermutation(0);

    qftReg->Y(0);
    qftReg->X(1);
    qftReg->Y(2);
    qftReg->X(3);
    qftReg->IT(4);
    qftReg->S(5);
    qftReg->CY(0, 2);
    qftReg->AntiCCZ(3, 5, 4);
    qftReg->T(0);
    qftReg->H(1);
    qftReg->IT(2);
    qftReg->X(3);
    qftReg->S(4);
    qftReg->X(5);
    qftReg->AntiCZ(0, 1);
    qftReg->AntiCCNOT(4, 2, 3);
    qftReg->IT(0);
    qftReg->Z(1);
    qftReg->Z(2);
    qftReg->T(3);
    qftReg->Y(4);
    qftReg->Z(5);
    qftReg->CCZ(3, 2, 4);
    qftReg->CCNOT(5, 1, 0);
    qftReg->IS(0);
    qftReg->Y(1);
    qftReg->H(2);
    qftReg->Z(3);
    qftReg->Z(4);
    qftReg->T(5);
    qftReg->CCNOT(4, 5, 2);
    qftReg->CCNOT(3, 1, 0);
    qftReg->H(0);
    qftReg->IT(1);
    qftReg->X(2);
    qftReg->IS(3);
    qftReg->Y(4);
    qftReg->IS(5);
    qftReg->AntiCZ(5, 4);
    qftReg->CNOT(1, 0);
    qftReg->AntiCNOT(2, 3);
    qftReg->IT(0);
    qftReg->H(1);
    qftReg->S(2);
    qftReg->H(3);
    qftReg->IS(4);
    qftReg->T(5);
    qftReg->AntiCCNOT(1, 5, 4);
    qftReg->CZ(2, 0);
    qftReg->CZ(2, 0);
    qftReg->AntiCCNOT(1, 5, 4);
    qftReg->IT(5);
    qftReg->S(4);
    qftReg->H(3);
    qftReg->IS(2);
    qftReg->H(1);
    qftReg->T(0);
    qftReg->AntiCNOT(2, 3);
    qftReg->CNOT(1, 0);
    qftReg->AntiCZ(5, 4);
    qftReg->S(5);
    qftReg->Y(4);
    qftReg->S(3);
    qftReg->X(2);
    qftReg->T(1);
    qftReg->H(0);
    qftReg->CCNOT(3, 1, 0);
    qftReg->CCNOT(4, 5, 2);
    qftReg->IT(5);
    qftReg->Z(4);
    qftReg->Z(3);
    qftReg->H(2);
    qftReg->Y(1);
    qftReg->S(0);
    qftReg->CCNOT(5, 1, 0);
    qftReg->CCZ(3, 2, 4);
    qftReg->Z(5);
    qftReg->Y(4);
    qftReg->IT(3);
    qftReg->Z(2);
    qftReg->Z(1);
    qftReg->T(0);
    qftReg->AntiCCNOT(4, 2, 3);
    qftReg->AntiCZ(0, 1);
    qftReg->X(5);
    qftReg->IS(4);
    qftReg->X(3);
    qftReg->T(2);
    qftReg->H(1);
    qftReg->IT(0);
    qftReg->AntiCCZ(3, 5, 4);
    qftReg->CY(0, 2);
    qftReg->IS(5);
    qftReg->T(4);
    qftReg->X(3);
    qftReg->Y(2);
    qftReg->X(1);
    qftReg->Y(0);

    REQUIRE(qftReg->MAll() == 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_24", "[mirror]")
{
    qftReg = MakeEngine(6);
    qftReg->SetPermutation(48);

    qftReg->H(1);
    qftReg->H(2);
    qftReg->H(4);
    qftReg->CNOT(2, 4);
    qftReg->CZ(1, 5);
    qftReg->H(2);
    qftReg->H(4);
    qftReg->T(5);
    qftReg->CNOT(4, 5);
    qftReg->CZ(1, 2);
    qftReg->H(3);
    qftReg->H(1);
    qftReg->T(2);
    qftReg->T(5);
    qftReg->CCNOT(1, 0, 2);
    qftReg->X(4);
    qftReg->X(1);
    qftReg->X(2);
    qftReg->H(0);
    qftReg->T(3);
    qftReg->CCNOT(0, 3, 4);
    qftReg->CNOT(1, 2);
    qftReg->X(1);
    qftReg->H(5);
    qftReg->CZ(5, 1);
    qftReg->CZ(0, 2);
    qftReg->CZ(0, 2);
    qftReg->CZ(5, 1);
    qftReg->H(5);
    qftReg->X(1);
    qftReg->CNOT(1, 2);
    qftReg->CCNOT(0, 3, 4);
    qftReg->IT(3);
    qftReg->H(0);
    qftReg->X(2);
    qftReg->X(1);
    qftReg->X(4);
    qftReg->CCNOT(1, 0, 2);
    qftReg->IT(5);
    qftReg->IT(2);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->CZ(1, 2);
    qftReg->CNOT(4, 5);
    qftReg->IT(5);
    qftReg->H(4);
    qftReg->H(2);
    qftReg->CZ(1, 5);
    qftReg->CNOT(2, 4);
    qftReg->H(4);
    qftReg->H(2);
    qftReg->H(1);

    REQUIRE(qftReg->MAll() == 48);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mirror_circuit_25", "[mirror]")
{
    qftReg = MakeEngine(2);
    qftReg->SetPermutation(2);

    qftReg->H(0);
    qftReg->IT(0);
    qftReg->IS(0);
    qftReg->AntiCNOT(0, 1);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->X(1);
    qftReg->X(1);
    qftReg->H(1);
    qftReg->H(0);
    qftReg->AntiCNOT(0, 1);
    qftReg->S(0);
    qftReg->T(0);
    qftReg->H(0);

    REQUIRE(qftReg->MAll() == 2);
}

TEST_CASE("test_mirror_circuit_26", "[mirror]")
{
    if (testEngineType != QINTERFACE_BDT) {
        std::cout << ">>> 'test_mirror_circuit_26': skipped" << std::endl;
        return;
    }
    std::cout << ">>> 'test_mirror_circuit_26':" << std::endl;

    QInterfacePtr qftReg = CreateQuantumInterface({ QINTERFACE_BDT }, 1U, 0, rng);
    std::dynamic_pointer_cast<QBdt>(qftReg)->Attach(std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface({ QINTERFACE_STABILIZER_HYBRID }, 2U, 0, rng, CMPLX_DEFAULT_ARG, false, false)));

    qftReg->SetPermutation(7);

    qftReg->H(1);
    qftReg->Z(0);
    qftReg->S(1);
    qftReg->Swap(2, 1);
    qftReg->Swap(2, 1);
    qftReg->IS(1);
    qftReg->Z(0);
    qftReg->H(1);

    REQUIRE(qftReg->MAll() == 7);
}

bitLenInt pickRandomBit(QInterfacePtr qReg, std::set<bitLenInt>* unusedBitsPtr)
{
    std::set<bitLenInt>::iterator bitIterator = unusedBitsPtr->begin();
    bitLenInt bitRand = qReg->Rand() * unusedBitsPtr->size();
    if (bitRand >= unusedBitsPtr->size()) {
        bitRand = unusedBitsPtr->size() - 1U;
    }
    std::advance(bitIterator, bitRand);
    bitLenInt bit = *bitIterator;
    unusedBitsPtr->erase(bitIterator);
    return bit;
}

struct MultiQubitGate {
    int gate;
    bitLenInt b1;
    bitLenInt b2;
    bitLenInt b3;
};

TEST_CASE("test_mirror_circuit", "[mirror]")
{
    std::cout << ">>> 'test_mirror_circuit':" << std::endl;

    const int GateCount1Qb = 8;
    const int GateCountMultiQb = 13;
    const int GateCount2Qb = 7;
    const int Depth = 6;

    const int TRIALS = 100;
    const int n = 6;

    int d;
    int i;
    int maxGates;

    int gate;

    for (int trial = 0; trial < TRIALS; trial++) {
        QInterfacePtr testCase =
            CreateQuantumInterface({ testEngineType, testSubEngineType, testSubSubEngineType }, n, 0);

        std::vector<std::vector<int>> gate1QbRands(Depth);
        std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(Depth);

        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < n; i++) {
                gate = (int)(testCase->Rand() * GateCount1Qb);
                if (gate >= GateCount1Qb) {
                    gate = (GateCount1Qb - 1U);
                }
                layer1QbRands.push_back(gate);
            }

            std::set<bitLenInt> unusedBits;
            for (i = 0; i < n; i++) {
                unusedBits.insert(i);
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            while (unusedBits.size() > 1) {
                MultiQubitGate multiGate;
                multiGate.b1 = pickRandomBit(testCase, &unusedBits);
                multiGate.b2 = pickRandomBit(testCase, &unusedBits);
                multiGate.b3 = 0;

                if (unusedBits.size() > 0) {
                    maxGates = GateCountMultiQb;
                } else {
                    maxGates = GateCount2Qb;
                }

                gate = (int)(testCase->Rand() * maxGates);
                if (gate >= maxGates) {
                    gate = (maxGates - 1U);
                }

                multiGate.gate = gate;

                if (multiGate.gate >= GateCount2Qb) {
                    multiGate.b3 = pickRandomBit(testCase, &unusedBits);
                }

                layerMultiQbRands.push_back(multiGate);
            }
        }

        bitCapIntOcl randPerm = testCase->Rand() * (bitCapIntOcl)testCase->GetMaxQPower();
        if (randPerm >= testCase->GetMaxQPower()) {
            randPerm = (bitCapIntOcl)testCase->GetMaxQPower() - 1U;
        }
        testCase->SetPermutation(randPerm);

        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->S(i);
                } else if (gate1Qb == 5) {
                    testCase->IS(i);
                } else if (gate1Qb == 6) {
                    testCase->T(i);
                } else {
                    testCase->IT(i);
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 6) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 7) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 8) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 9) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 10) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }
        }

        // Mirror the circuit
        for (d = Depth - 1U; d >= 0; d--) {
            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = (layerMultiQbRands.size() - 1U); i >= 0; i--) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 6) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 7) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 8) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 9) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 10) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }

            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = (layer1QbRands.size() - 1U); i >= 0; i--) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->IS(i);
                } else if (gate1Qb == 5) {
                    testCase->S(i);
                } else if (gate1Qb == 6) {
                    testCase->IT(i);
                } else {
                    testCase->T(i);
                }
            }
        }

        bitCapInt result = testCase->MAll();

        if (result != randPerm) {
            for (d = 0; d < Depth; d++) {
                std::vector<int>& layer1QbRands = gate1QbRands[d];
                for (i = 0; i < (int)layer1QbRands.size(); i++) {
                    int gate1Qb = layer1QbRands[i];
                    if (gate1Qb == 0) {
                        std::cout << "qftReg->H(" << (int)i << ");" << std::endl;
                        // testCase->H(i);
                    } else if (gate1Qb == 1) {
                        std::cout << "qftReg->X(" << (int)i << ");" << std::endl;
                        // testCase->X(i);
                    } else if (gate1Qb == 2) {
                        std::cout << "qftReg->Y(" << (int)i << ");" << std::endl;
                        // testCase->Y(i);
                    } else if (gate1Qb == 3) {
                        std::cout << "qftReg->Z(" << (int)i << ");" << std::endl;
                        // testCase->Z(i);
                    } else if (gate1Qb == 4) {
                        std::cout << "qftReg->S(" << (int)i << ");" << std::endl;
                        // testCase->S(i);
                    } else if (gate1Qb == 5) {
                        std::cout << "qftReg->IS(" << (int)i << ");" << std::endl;
                        // testCase->T(i);
                    } else if (gate1Qb == 6) {
                        std::cout << "qftReg->T(" << (int)i << ");" << std::endl;
                        // testCase->IS(i);
                    } else {
                        std::cout << "qftReg->IT(" << (int)i << ");" << std::endl;
                        // testCase->IT(i);
                    }
                }

                std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
                for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                    MultiQubitGate multiGate = layerMultiQbRands[i];
                    if (multiGate.gate == 0) {
                        std::cout << "qftReg->Swap(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->Swap(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 1) {
                        std::cout << "qftReg->CNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CNOT(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 2) {
                        std::cout << "qftReg->CY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CY(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 3) {
                        std::cout << "qftReg->CZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CZ(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 4) {
                        std::cout << "qftReg->AntiCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 5) {
                        std::cout << "qftReg->AntiCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCY(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 6) {
                        std::cout << "qftReg->AntiCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 7) {
                        std::cout << "qftReg->CCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 8) {
                        std::cout << "qftReg->CCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 9) {
                        std::cout << "qftReg->CCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 10) {
                        std::cout << "qftReg->AntiCCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 11) {
                        std::cout << "qftReg->AntiCCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else {
                        std::cout << "qftReg->AntiCCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    }
                }
            }

            for (d = (Depth - 1U); d >= 0; d--) {
                std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
                for (i = (layerMultiQbRands.size() - 1U); i >= 0; i--) {
                    MultiQubitGate multiGate = layerMultiQbRands[i];
                    if (multiGate.gate == 0) {
                        std::cout << "qftReg->Swap(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->Swap(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 1) {
                        std::cout << "qftReg->CNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CNOT(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 2) {
                        std::cout << "qftReg->CY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CY(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 3) {
                        std::cout << "qftReg->CZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->CZ(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 4) {
                        std::cout << "qftReg->AntiCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 5) {
                        std::cout << "qftReg->AntiCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCY(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 6) {
                        std::cout << "qftReg->AntiCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ");"
                                  << std::endl;
                        // testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    } else if (multiGate.gate == 7) {
                        std::cout << "qftReg->CCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 8) {
                        std::cout << "qftReg->CCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 9) {
                        std::cout << "qftReg->CCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 10) {
                        std::cout << "qftReg->AntiCCNOT(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else if (multiGate.gate == 11) {
                        std::cout << "qftReg->AntiCCY(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    } else {
                        std::cout << "qftReg->AntiCCZ(" << (int)multiGate.b1 << "," << (int)multiGate.b2 << ","
                                  << (int)multiGate.b3 << ");" << std::endl;
                        // testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    }
                }

                std::vector<int>& layer1QbRands = gate1QbRands[d];
                for (i = (layer1QbRands.size() - 1U); i >= 0; i--) {
                    int gate1Qb = layer1QbRands[i];
                    if (gate1Qb == 0) {
                        std::cout << "qftReg->H(" << (int)i << ");" << std::endl;
                        // testCase->H(i);
                    } else if (gate1Qb == 1) {
                        std::cout << "qftReg->X(" << (int)i << ");" << std::endl;
                        // testCase->X(i);
                    } else if (gate1Qb == 2) {
                        std::cout << "qftReg->Y(" << (int)i << ");" << std::endl;
                        // testCase->Y(i);
                    } else if (gate1Qb == 3) {
                        std::cout << "qftReg->Z(" << (int)i << ");" << std::endl;
                        // testCase->Z(i);
                    } else if (gate1Qb == 4) {
                        std::cout << "qftReg->IS(" << (int)i << ");" << std::endl;
                        // testCase->IS(i);
                    } else if (gate1Qb == 5) {
                        std::cout << "qftReg->S(" << (int)i << ");" << std::endl;
                        // testCase->IT(i);
                    } else if (gate1Qb == 6) {
                        std::cout << "qftReg->IT(" << (int)i << ");" << std::endl;
                        // testCase->S(i);
                    } else {
                        std::cout << "qftReg->T(" << (int)i << ");" << std::endl;
                        // testCase->T(i);
                    }
                }
            }
        }

        REQUIRE(result == randPerm);
    }
}

TEST_CASE("test_mirror_near_clifford", "[mirror]")
{
    if (testEngineType != QINTERFACE_BDT) {
        std::cout << ">>> 'test_mirror_near_clifford': skipped" << std::endl;
        return;
    }
    std::cout << ">>> 'test_mirror_near_clifford':" << std::endl;

    const int GateCount1Qb = 8;
    const int GateCount2Qb = 7;

    const int TRIALS = benchmarkSamples;
    const int Depth = benchmarkDepth;
    const int n = benchmarkDepth;
    const int magic = (benchmarkMaxMagic < 0) ? 3U : benchmarkMaxMagic;

    std::cout << "Width/Depth (with x2 depth mirror): " << n << std::endl;
    std::cout << "\"Magic\": " << magic << std::endl;
    std::cout << "Trials: " << TRIALS << std::endl;

    int failureCount = 0;

    int d;
    int i;
    int maxGates;

    int gate;

    real1_f tRate = ZERO_R1;
    for (int trial = 0; trial < TRIALS; trial++) {
        QInterfacePtr testCase = CreateQuantumInterface({ QINTERFACE_BDT }, magic, 0, rng);
        std::dynamic_pointer_cast<QBdt>(testCase)->Attach(std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(
            { QINTERFACE_STABILIZER_HYBRID }, n - magic, 0, rng, CMPLX_DEFAULT_ARG, false, false)));

        std::vector<std::vector<int>> gate1QbRands(Depth);
        std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(Depth);

        int tGateCount = 0;
        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < n; i++) {
                if ((n * Depth * testCase->Rand() / (n + 2)) < ONE_R1) {
                    if ((2 * testCase->Rand()) < ONE_R1) {
                        gate = (GateCount1Qb - 2);
                    } else {
                        gate = (GateCount1Qb - 1);
                    }
                    tGateCount++;
                } else {
                    gate = (int)(testCase->Rand() * (GateCount1Qb - 2U));
                    if (gate >= (GateCount1Qb - 2)) {
                        gate = (GateCount1Qb - 3);
                    }
                    layer1QbRands.push_back(gate);
                }
            }

            std::set<bitLenInt> unusedBits;
            for (i = 0; i < n; i++) {
                unusedBits.insert(i);
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            while (unusedBits.size() > 1) {
                MultiQubitGate multiGate;
                multiGate.b1 = pickRandomBit(testCase, &unusedBits);
                multiGate.b2 = pickRandomBit(testCase, &unusedBits);
                multiGate.b3 = 0;

                maxGates = GateCount2Qb;

                gate = (int)(testCase->Rand() * maxGates);
                if (gate >= maxGates) {
                    gate = (maxGates - 1U);
                }

                multiGate.gate = gate;

                if (multiGate.gate >= GateCount2Qb) {
                    multiGate.b3 = pickRandomBit(testCase, &unusedBits);
                }

                layerMultiQbRands.push_back(multiGate);
            }
        }

        bitCapIntOcl randPerm = testCase->Rand() * (bitCapIntOcl)testCase->GetMaxQPower();
        if (randPerm >= testCase->GetMaxQPower()) {
            randPerm = (bitCapIntOcl)testCase->GetMaxQPower() - 1U;
        }
        testCase->SetPermutation(randPerm);

        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->S(i);
                } else if (gate1Qb == 5) {
                    testCase->IS(i);
                } else if (gate1Qb == 6) {
                    testCase->T(i);
                } else {
                    testCase->IT(i);
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                }
            }
        }

        // Mirror the circuit
        for (d = Depth - 1U; d >= 0; d--) {
            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = (layerMultiQbRands.size() - 1U); i >= 0; i--) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                }
            }

            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = (layer1QbRands.size() - 1U); i >= 0; i--) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->IS(i);
                } else if (gate1Qb == 5) {
                    testCase->S(i);
                } else if (gate1Qb == 6) {
                    testCase->IT(i);
                } else {
                    testCase->T(i);
                }
            }
        }

        tRate += ((real1_f)tGateCount) / TRIALS;

        bitCapInt result = testCase->MAll();

        if (result != randPerm) {
            failureCount++;
        }
    }

    const real1_f succesRate = ((real1_f)(TRIALS - failureCount)) / TRIALS;
    std::cout << "Success rate: " << (TRIALS - failureCount) << " / " << TRIALS << std::endl;
    if (succesRate >= (2.0f / 3.0f)) {
        std::cout << "Success! Mirrored correctly. (This does not check heavy outputs, though.)" << std::endl;
    } else {
        std::cout << "Failure. Mirrored incorrectly." << std::endl;
    }
    std::cout << "Average T gates per trial: " << tRate << std::endl;
    REQUIRE(succesRate >= (2.0f / 3.0f));
}

TEST_CASE("test_mirror_quantum_volume", "[mirror]")
{
    if (testEngineType != QINTERFACE_BDT) {
        std::cout << ">>> 'test_mirror_quantum_volume': skipped" << std::endl;
        return;
    }
    std::cout << ">>> 'test_mirror_quantum_volume':" << std::endl;

    const int GateCount1Qb = 8;
    const int GateCountMultiQb = 13;
    const int GateCount2Qb = 7;

    const int TRIALS = benchmarkSamples;
    const int Depth = benchmarkDepth;
    const int n = benchmarkDepth;
    const int magic = (benchmarkMaxMagic < 0) ? 3U : benchmarkMaxMagic;

    std::cout << "Width/Depth (with x2 depth mirror): " << n << std::endl;
    std::cout << "\"Magic\": " << magic << std::endl;
    std::cout << "Trials: " << TRIALS << std::endl;

    int failureCount = 0;

    int d;
    int i;
    int maxGates;

    int gate;

    for (int trial = 0; trial < TRIALS; trial++) {
        QInterfacePtr testCase = CreateQuantumInterface({ QINTERFACE_BDT }, magic, 0, rng);
        std::dynamic_pointer_cast<QBdt>(testCase)->Attach(std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(
            { QINTERFACE_STABILIZER_HYBRID }, n - magic, 0, rng, CMPLX_DEFAULT_ARG, false, false)));

        std::vector<std::vector<int>> gate1QbRands(Depth);
        std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(Depth);

        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < n; i++) {
                gate = (int)(testCase->Rand() * GateCount1Qb);
                if (gate >= GateCount1Qb) {
                    gate = (GateCount1Qb - 1U);
                }
                layer1QbRands.push_back(gate);
            }

            std::set<bitLenInt> unusedBits;
            for (i = 0; i < n; i++) {
                unusedBits.insert(i);
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            while (unusedBits.size() > 1) {
                MultiQubitGate multiGate;
                multiGate.b1 = pickRandomBit(testCase, &unusedBits);
                multiGate.b2 = pickRandomBit(testCase, &unusedBits);
                multiGate.b3 = 0;

                if (unusedBits.size() > 0) {
                    maxGates = GateCountMultiQb;
                } else {
                    maxGates = GateCount2Qb;
                }

                gate = (int)(testCase->Rand() * maxGates);
                if (gate >= maxGates) {
                    gate = (maxGates - 1U);
                }

                multiGate.gate = gate;

                if (multiGate.gate >= GateCount2Qb) {
                    multiGate.b3 = pickRandomBit(testCase, &unusedBits);
                }

                layerMultiQbRands.push_back(multiGate);
            }
        }

        bitCapIntOcl randPerm = testCase->Rand() * (bitCapIntOcl)testCase->GetMaxQPower();
        if (randPerm >= testCase->GetMaxQPower()) {
            randPerm = (bitCapIntOcl)testCase->GetMaxQPower() - 1U;
        }
        testCase->SetPermutation(randPerm);

        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->S(i);
                } else if (gate1Qb == 5) {
                    testCase->IS(i);
                } else if (gate1Qb == 6) {
                    testCase->T(i);
                } else {
                    testCase->IT(i);
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 6) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 7) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 8) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 9) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 10) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }
        }

        // Mirror the circuit
        for (d = Depth - 1U; d >= 0; d--) {
            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = (layerMultiQbRands.size() - 1U); i >= 0; i--) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 3) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 4) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 5) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 6) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 7) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 8) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 9) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 10) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }

            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = (layer1QbRands.size() - 1U); i >= 0; i--) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else if (gate1Qb == 3) {
                    testCase->Z(i);
                } else if (gate1Qb == 4) {
                    testCase->IS(i);
                } else if (gate1Qb == 5) {
                    testCase->S(i);
                } else if (gate1Qb == 6) {
                    testCase->IT(i);
                } else {
                    testCase->T(i);
                }
            }
        }

        bitCapInt result = testCase->MAll();

        if (result != randPerm) {
            failureCount++;
        }
    }

    const real1_f succesRate = ((real1_f)(TRIALS - failureCount)) / TRIALS;
    std::cout << "Success rate: " << (TRIALS - failureCount) << " / " << TRIALS << std::endl;
    if (succesRate >= (2.0f / 3.0f)) {
        std::cout << "Success! Mirrored correctly. (This does not check heavy outputs, though.)" << std::endl;
    } else {
        std::cout << "Failure. Mirrored incorrectly." << std::endl;
    }
    REQUIRE(succesRate >= (2.0f / 3.0f));
}
