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

#include "qcircuit.hpp"
#include "qfactory.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <list>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "catch.hpp"

#include "tests.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#if ENABLE_OPENCL
#define QRACK_GPU_SINGLETON (OCLEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define QRACK_GPU_SINGLETON (CUDAEngine::Instance())
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1_f __tmp_a = A;                                                                                           \
        real1_f __tmp_b = B;                                                                                           \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                                                                        \
    } while (0);

#define QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)

using namespace Qrack;

const double clockFactor = 1.0 / 1000.0; // Report in ms

double formatTime(double t, bool logNormal)
{
    if (logNormal) {
        return pow(2.0, t);
    } else {
        return t;
    }
}

void RandomInitQubit(QInterfacePtr sim, bitLenInt i)
{
    const real1_f theta = 4 * M_PI * sim->Rand();
    const real1_f phi = 2 * M_PI * sim->Rand();
    const real1_f lambda = 2 * M_PI * sim->Rand();

    sim->U(i, theta, phi, lambda);
}

std::vector<QInterfaceEngine> BuildEngineStack()
{
    std::vector<QInterfaceEngine> engineStack;
    if (optimal) {
        engineStack.push_back(QINTERFACE_TENSOR_NETWORK);
#if ENABLE_OPENCL || ENABLE_CUDA
        engineStack.push_back(
            (QRACK_GPU_SINGLETON.GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#else
        engineStack.push_back(QINTERFACE_OPTIMAL);
#endif
    } else if (optimal_single) {
        engineStack.push_back(QINTERFACE_TENSOR_NETWORK);
        engineStack.push_back(QINTERFACE_OPTIMAL);
    } else {
        engineStack.push_back(testEngineType);
        engineStack.push_back(testSubEngineType);
        engineStack.push_back(testSubSubEngineType);
        engineStack.push_back(testSubSubSubEngineType);
    }

    return engineStack;
}

void benchmarkLoopVariable(std::function<void(QInterfacePtr, bitLenInt)> fn, bitLenInt mxQbts,
    bool resetRandomPerm = true, bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
{
    std::cout << std::endl;
    std::cout << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
    std::cout << benchmarkSamples << " iterations" << std::endl;
    std::cout << "# of Qubits, ";
    std::cout << "Average Time (ms), ";
    std::cout << "Sample Std. Deviation (ms), ";
    std::cout << "Fastest (ms), ";
    std::cout << "1st Quartile (ms), ";
    std::cout << "Median (ms), ";
    std::cout << "3rd Quartile (ms), ";
    std::cout << "Slowest (ms), ";
    std::cout << "Failure count, ";
    std::cout << "Average SDRP Fidelity" << std::endl;

    std::vector<double> trialClocks;

    bitLenInt mnQbts;
    if (single_qubit_run) {
        mnQbts = mxQbts;
    } else {
        mnQbts = min_qubits;
    }

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    for (bitLenInt numBits = mnQbts; numBits <= mxQbts; numBits++) {
        QInterfacePtr qftReg = CreateQuantumInterface(engineStack, numBits, ZERO_BCI, rng, CMPLX_DEFAULT_ARG,
            enable_normalization, true, use_host_dma, device_id, !disable_hardware_rng, sparse, REAL1_EPSILON, devList);
        if (disable_t_injection) {
            qftReg->SetTInjection(false);
        }
        if (disable_reactive_separation) {
            qftReg->SetReactiveSeparate(false);
        }
        double avgt = 0.0;
        double avgf = 0.0;
        int sampleFailureCount = 0;
        trialClocks.clear();

        std::vector<bitCapInt> qPowers;
        for (bitLenInt i = 0U; i < numBits; ++i) {
            qPowers.push_back(pow2(i));
        }

        for (int sample = 0; sample < benchmarkSamples; sample++) {
            try {
                qftReg->ResetUnitaryFidelity();
                if (!qUniverse) {
                    if (resetRandomPerm) {
                        bitCapInt perm = (bitCapIntOcl)(qftReg->Rand() * bi_to_double(qftReg->GetMaxQPower()));
                        if (bi_compare(perm, qftReg->GetMaxQPower()) >= 0) {
                            perm = qftReg->GetMaxQPower() - ONE_BCI;
                        }
                        qftReg->SetPermutation(perm);
                    } else {
                        qftReg->SetPermutation(ZERO_BCI);
                    }
                    if (hadamardRandomBits) {
                        for (bitLenInt j = 0; j < numBits; j++) {
                            if (qftReg->Rand() >= ONE_R1 / 2) {
                                qftReg->H(j);
                            }
                        }
                    }
                } else {
                    qftReg->SetPermutation(ZERO_BCI);
                    for (bitLenInt i = 0; i < numBits; i++) {
                        RandomInitQubit(qftReg, i);
                    }
                }
                qftReg->Finish();

                const auto iterClock = std::chrono::high_resolution_clock::now();

                // Run loop body
                fn(qftReg, numBits);
                if (!async_time && qftReg) {
                    qftReg->Finish();
                }

                // Collect interval data
                double fidelity = qftReg->GetUnitaryFidelity();
                if (!disable_terminal_measurement) {
                    if (benchmarkShots == 1) {
                        bitCapInt result;
                        result = qftReg->MAll();
                        if (mOutputFileName.compare("")) {
                            mOutputFile << result << std::endl;
                        }
                    } else if (benchmarkShots) {
                        std::unique_ptr<unsigned long long[]> results(new unsigned long long[benchmarkShots]);
                        qftReg->MultiShotMeasureMask(qPowers, benchmarkShots, results.get());
                        for (int i = 0U; i < benchmarkShots; ++i) {
                            mOutputFile << results.get()[i] << std::endl;
                        }
                    }
                }

                if (fidelity > qftReg->GetUnitaryFidelity()) {
                    fidelity = qftReg->GetUnitaryFidelity();
                }
                avgf += fidelity;

                // Clock stops after benchmark definition plus measurement sampling.
                auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - iterClock);

                if (tClock.count() < 0) {
                    trialClocks.push_back(0);
                } else if (logNormal) {
                    trialClocks.push_back(std::log2(tClock.count() * clockFactor));
                } else {
                    trialClocks.push_back(tClock.count() * clockFactor);
                }
                avgt += trialClocks.back();

                if (async_time && qftReg) {
                    qftReg->Finish();
                }

            } catch (const std::exception& e) {
                // Release before re-alloc:
                qftReg = nullptr;

                // Re-alloc:
                qftReg =
                    CreateQuantumInterface(engineStack, numBits, ZERO_BCI, rng, CMPLX_DEFAULT_ARG, enable_normalization,
                        true, use_host_dma, device_id, !disable_hardware_rng, sparse, REAL1_EPSILON, devList);
                if (disable_t_injection) {
                    qftReg->SetTInjection(false);
                }
                if (disable_reactive_separation) {
                    qftReg->SetReactiveSeparate(false);
                }

                ++sampleFailureCount;
            }
        }

        if (sampleFailureCount >= benchmarkSamples) {
            std::cout << "All samples at width failed. Terminating..." << std::endl;
            return;
        }

        avgt /= trialClocks.size();
        avgf /= trialClocks.size();

        double stdet = 0.0;
        for (int sample = 0; sample < (int)trialClocks.size(); sample++) {
            stdet += (trialClocks[sample] - avgt) * (trialClocks[sample] - avgt);
        }
        stdet = sqrt(stdet / trialClocks.size());

        std::sort(trialClocks.begin(), trialClocks.end());

        std::cout << (int)numBits << ", "; /* # of Qubits */
        std::cout << formatTime(avgt, logNormal) << ","; /* Average Time (ms) */
        std::cout << formatTime(stdet, logNormal) << ","; /* Sample Std. Deviation (ms) */

        // Fastest (ms)
        std::cout << formatTime(trialClocks[0], logNormal) << ",";

        if (trialClocks.size() == 1) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << sampleFailureCount << ",";
            std::cout << avgf << std::endl;
            continue;
        }

        // 1st Quartile (ms)
        if (trialClocks.size() < 8) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
        } else if (trialClocks.size() % 4 == 0) {
            std::cout << formatTime((trialClocks[trialClocks.size() / 4 - 1] + trialClocks[trialClocks.size() / 4]) / 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() / 4 - 1] / 2, logNormal) << ",";
        }

        // Median (ms) (2nd quartile)
        if (trialClocks.size() < 4) {
            std::cout << formatTime(trialClocks[trialClocks.size() / 2], logNormal) << ",";
        } else if (trialClocks.size() % 2 == 0) {
            std::cout << formatTime((trialClocks[trialClocks.size() / 2 - 1] + trialClocks[trialClocks.size() / 2]) / 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() / 2 - 1] / 2, logNormal) << ","; /* Median (ms) */
        }

        // 3rd Quartile (ms)
        if (trialClocks.size() < 8) {
            std::cout << formatTime(trialClocks[(3 * trialClocks.size()) / 4], logNormal) << ",";
        } else if (trialClocks.size() % 4 == 0) {
            std::cout << formatTime((trialClocks[(3 * trialClocks.size()) / 4 - 1] +
                                        trialClocks[(3 * trialClocks.size()) / 4]) /
                                 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[(3 * trialClocks.size()) / 4 - 1] / 2, logNormal) << ",";
        }

        // Slowest (ms)
        if (trialClocks.size() <= 1) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() - 1], logNormal) << ",";
        }

        // Failure count
        std::cout << sampleFailureCount << ",";
        // Average SDRP fidelity
        std::cout << avgf << std::endl;
    }
}

void benchmarkLoop(std::function<void(QInterfacePtr, bitLenInt)> fn, bool resetRandomPerm = true,
    bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
{
    benchmarkLoopVariable(fn, max_qubits, resetRandomPerm, hadamardRandomBits, logNormal, qUniverse);
}

TEST_CASE("test_cnot_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CNOT(0, 1); });
}

TEST_CASE("test_x_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->X(0); });
}

TEST_CASE("test_y_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Y(0); });
}

TEST_CASE("test_z_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Z(0); });
}

TEST_CASE("test_swap_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Swap(0, 1); });
}

#if ENABLE_REG_GATES
TEST_CASE("test_cnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CNOT(0, n / 2, n / 2); });
}

TEST_CASE("test_x_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->XMask(pow2(n) - 1U); });
}

TEST_CASE("test_y_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->YMask(pow2(n) - 1U); });
}

TEST_CASE("test_z_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ZMask(pow2(n) - 1U); });
}

TEST_CASE("test_swap_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Swap(0, n / 2, n / 2); });
}

TEST_CASE("test_ccnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CCNOT(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_and_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->AND(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_or_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->OR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_xor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->XOR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_cland_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLAND(0, 0x0c, 0, n); });
}

TEST_CASE("test_clor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_clxor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLXOR(0, 0x0d, 0, n); });
}

#if ENABLE_ROT_API
TEST_CASE("test_rt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->RT(M_PI, 0, n); });
}

TEST_CASE("test_crt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CRT(M_PI, 0, n / 2, n / 2); });
}
#endif
#endif

TEST_CASE("test_m", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->M(n - 1); });
}

TEST_CASE("test_mreg", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->MReg(0, n); });
}

TEST_CASE("test_rol", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ROL(1, 0, n); });
}

TEST_CASE("test_inc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->INC(ONE_BCI, 0, n); });
}

#if ENABLE_ALU
TEST_CASE("test_incs", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { QALU(qftReg)->INCS(ONE_BCI, 0, n - 1, n - 1); });
}

TEST_CASE("test_incc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { QALU(qftReg)->INCC(ONE_BCI, 0, n - 1, n - 1); });
}

TEST_CASE("test_incsc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { QALU(qftReg)->INCSC(ONE_BCI, 0, n - 2, n - 2, n - 1); });
}

TEST_CASE("test_c_phase_flip_if_less", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { QALU(qftReg)->CPhaseFlipIfLess(ONE_BCI, 0, n - 1, n - 1); });
}
#endif

TEST_CASE("test_zero_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ZeroPhaseFlip(0, n); });
}

TEST_CASE("test_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->PhaseFlip(); });
}

#if ENABLE_ALU
void benchmarkSuperpose(std::function<void(QInterfacePtr, int, unsigned char*)> fn)
{
    const bitCapIntOcl wordLength = (max_qubits / 16U + 1U);
    const bitCapIntOcl indexLength = pow2Ocl(max_qubits / 2U);
    std::unique_ptr<unsigned char[]> testPage(new unsigned char[wordLength * indexLength]);
    for (bitCapIntOcl j = 0; j < indexLength; j++) {
        for (bitCapIntOcl i = 0; i < wordLength; i++) {
            testPage.get()[j * wordLength + i] = (j & (0xff << (8U * i))) >> (8U * i);
        }
    }
    unsigned char* tp = testPage.get();
    benchmarkLoop([fn, tp](QInterfacePtr qftReg, bitLenInt n) { fn(qftReg, n, tp); });
}

TEST_CASE("test_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        QALU(qftReg)->IndexedLDA(0, n / 2, n / 2, n / 2, testPage);
    });
}

TEST_CASE("test_adc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        QALU(qftReg)->IndexedADC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_sbc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        QALU(qftReg)->IndexedSBC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}
#endif

TEST_CASE("test_setbit", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->SetBit(0, true); });
}

TEST_CASE("test_proball", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ProbAll(0x02); });
}

TEST_CASE("test_set_reg", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->SetReg(0, n, ONE_BCI); });
}

TEST_CASE("test_ghz", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        qftReg->H(0U);
        for (bitLenInt i = 1U; i < n; ++i) {
            qftReg->CNOT(i - 1U, i);
        }
    });
}

#if ENABLE_ALU
TEST_CASE("test_grover", "[grover]")
{

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 3.

    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        // Twelve iterations maximizes the probablity for 256 searched elements, for example.
        // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
        const int optIter = M_PI / (4.0 * asin(1.0 / sqrt((real1_s)pow2Ocl(n))));

        // Our input to the subroutine "oracle" is 8 bits.
        qftReg->SetPermutation(ZERO_BCI);
        qftReg->H(0, n);

        for (int i = 0; i < optIter; i++) {
            // Our "oracle" is true for an input of "3" and false for all other inputs.
            QALU(qftReg)->DEC(3, 0, n);
            qftReg->ZeroPhaseFlip(0, n);
            QALU(qftReg)->INC(3, 0, n);
            // This ends the "oracle."
            qftReg->H(0, n);
            qftReg->ZeroPhaseFlip(0, n);
            qftReg->H(0, n);
            // Global phase flip has no physically measurable effect:
            // qftReg->PhaseFlip();
        }

        REQUIRE_THAT(qftReg, HasProbability(0x3));

        qftReg->MReg(0, n);
    });
}
#endif

TEST_CASE("test_qft_ideal_init", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, false, false);
}

TEST_CASE("test_qft_permutation_init", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, true, false,
        testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_permutation_round_trip_entangled", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, bitLenInt n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_one_way", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, true, true,
        testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_round_trip", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, bitLenInt n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, true, testEngineType == QINTERFACE_QUNIT);
}

bitLenInt pickRandomBit(real1_f rand, std::set<bitLenInt>* unusedBitsPtr)
{
    std::set<bitLenInt>::iterator bitIterator = unusedBitsPtr->begin();
    bitLenInt bitRand = (bitLenInt)(unusedBitsPtr->size() * rand);
    if (bitRand >= unusedBitsPtr->size()) {
        bitRand = unusedBitsPtr->size() - 1U;
    }
    std::advance(bitIterator, bitRand);
    bitRand = *bitIterator;
    unusedBitsPtr->erase(bitIterator);
    return bitRand;
}

TEST_CASE("test_quantum_triviality", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 5;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
            for (int d = 0; d < depth; d++) {

                bitCapInt xMask = ZERO_BCI;
                bitCapInt yMask = ZERO_BCI;
                for (bitLenInt i = 0; i < n; i++) {
                    const real1_f gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        // qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&xMask, pow2(i));
                    } else if (gateRand < (3 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&yMask, pow2(i));
                    } else {
                        qReg->T(i);
                    }
                }
                qReg->XMask(xMask);
                qReg->YMask(yMask);

                std::set<bitLenInt> unusedBits;
                for (bitLenInt i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                    const int maxGates = (unusedBits.size() > 0) ? GateCountMultiQb : GateCountMultiQb - 2U;
                    const real1_f gateRand = maxGates * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->CZ(b1, b2);
                    } else if ((unusedBits.size() == 0) || (gateRand < (3 * ONE_R1))) {
                        qReg->CNOT(b1, b2);
                    } else if (gateRand < (4 * ONE_R1)) {
                        const bitLenInt b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCZ(b1, b2, b3);
                    } else {
                        const bitLenInt b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_stabilizer", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 2;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
            for (int d = 0; d < depth; d++) {
                bitCapInt xMask = ZERO_BCI;
                bitCapInt yMask = ZERO_BCI;
                for (bitLenInt i = 0; i < n; i++) {
                    const real1_f gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&xMask, pow2(i));
                    } else if (gateRand < (3 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&yMask, pow2(i));
                    } else {
                        qReg->S(i);
                    }
                }
                qReg->XMask(xMask);
                qReg->YMask(yMask);

                std::set<bitLenInt> unusedBits;
                for (bitLenInt i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->CNOT(b1, b2);
                    } else {
                        qReg->CZ(b1, b2);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_stabilizer_t", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        for (int d = 0; d < depth; d++) {
            bitCapInt zMask = ZERO_BCI;
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->H(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                }
                // else - identity

                // "Position transforms:

                // Continuous Z root gates option:
                // gateRand = 4 * PI_R1 * qReg->Rand();
                // qReg->Phase(ONE_R1, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    bi_or_ip(&zMask, pow2(i));
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    bi_or_ip(&zMask, pow2(i));
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    bi_or_ip(&zMask, pow2(i));
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
            }
            qReg->ZMask(zMask);

            std::set<bitLenInt> unusedBits;
            for (bitLenInt i = 0; i < n; i++) {
                // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                // the method can give performance advantages, under opportune conditions, but it does not, here.
                unusedBits.insert(unusedBits.end(), i);
            }

            while (unusedBits.size() > 1) {
                const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                real1_f gateRand = GateCountMultiQb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    gateRand = 4 * qReg->Rand();
                    if (gateRand < (3 * ONE_R1)) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CNOT(b1, b2);
                        } else {
                            qReg->AntiCNOT(b1, b2);
                        }
                    } else {
                        qReg->Swap(b1, b2);
                    }
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 4 * qReg->Rand();
                    if (gateRand < (3 * ONE_R1)) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CY(b1, b2);
                        } else {
                            qReg->AntiCY(b1, b2);
                        }
                    } else {
                        qReg->Swap(b1, b2);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->CZ(b1, b2);
                    } else {
                        qReg->AntiCZ(b1, b2);
                    }
                }
                // else - identity
            }
        }
    });
}

TEST_CASE("test_stabilizer_t_cc", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int DimCount1Qb = 4;
    const int DimCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        for (int d = 0; d < depth; d++) {
            bitCapInt zMask = ZERO_BCI;
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->H(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                }
                // else - identity

                // "Position transforms:

                // Continuous Z root gates option:
                // gateRand = 4 * PI_R1 * qReg->Rand();
                // qReg->Phase(ONE_R1, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    bi_or_ip(&zMask, pow2(i));
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    bi_or_ip(&zMask, pow2(i));
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    bi_or_ip(&zMask, pow2(i));
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
            }
            qReg->ZMask(zMask);

            std::set<bitLenInt> unusedBits;
            for (bitLenInt i = 0; i < n; i++) {
                unusedBits.insert(unusedBits.end(), i);
            }

            while (unusedBits.size() > 1) {
                const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                const real1_f gateRand = 2 * qReg->Rand();
                if ((gateRand < ONE_R1) || !unusedBits.size()) {
                    real1_f gateRand = DimCountMultiQb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        gateRand = 4 * qReg->Rand();
                        if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CNOT(b1, b2);
                            } else {
                                qReg->AntiCNOT(b1, b2);
                            }
                        } else {
                            qReg->Swap(b1, b2);
                        }
                    } else if (gateRand < (2 * ONE_R1)) {
                        gateRand = 4 * qReg->Rand();
                        if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CY(b1, b2);
                            } else {
                                qReg->AntiCY(b1, b2);
                            }
                        } else {
                            qReg->Swap(b1, b2);
                        }
                    } else if (gateRand < (3 * ONE_R1)) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CZ(b1, b2);
                        } else {
                            qReg->AntiCZ(b1, b2);
                        }
                    }
                    // else - identity
                } else {
                    const bitLenInt b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                    real1_f gateRand = DimCountMultiQb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CCNOT(b1, b2, b3);
                        } else {
                            qReg->AntiCCNOT(b1, b2, b3);
                        }
                    } else if (gateRand < (2 * ONE_R1)) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CCY(b1, b2, b3);
                        } else {
                            qReg->AntiCCY(b1, b2, b3);
                        }
                    } else if (gateRand < (3 * ONE_R1)) {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            qReg->CCZ(b1, b2, b3);
                        } else {
                            qReg->AntiCCZ(b1, b2, b3);
                        }
                    }
                    // else - identity
                }
            }
        }
    });
}

TEST_CASE("test_stabilizer_t_nn", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        const int rowLen = n / colLen;

        const auto iterClock = std::chrono::high_resolution_clock::now();

        for (int d = 0; d < depth; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if ((2 * qReg->Rand()) < ONE_R1) {
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->S(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->H(i);
                        qReg->S(i);
                    }
                    // else - identity
                } else {
                    gateRand = DimCount1Qb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->IS(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                    // else - identity
                }

                //"Position" transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                /*
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    qReg->Z(i);
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    qReg->Z(i);
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    qReg->Z(i);
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
                */

                if (timeout >= 0) {
                    auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - iterClock);
                    if ((tClock.count() * clockFactor) > timeout) {
                        throw std::runtime_error("Timeout");
                    }
                }
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        qReg->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            qReg->AntiCNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->AntiCY(b1, b2);
                        } else {
                            qReg->AntiCZ(b1, b2);
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            qReg->CNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->CY(b1, b2);
                        } else {
                            qReg->CZ(b1, b2);
                        }
                    }

                    if (timeout >= 0) {
                        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - iterClock);
                        if ((tClock.count() * clockFactor) > timeout) {
                            throw std::runtime_error("Timeout");
                        }
                    }
                }
            }
        }
    });
}

TEST_CASE("test_stabilizer_t_nn_d", "[supreme]")
{
    // Try with environment variable
    // QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466
    // for clamping of single bit states to Pauli basis axes.

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    if (benchmarkMaxMagic >= 0) {
        std::cout << "(max quantum \"magic\": " << benchmarkMaxMagic << ")";
    } else {
        std::cout << "(max quantum \"magic\": default, ceiling equal to qubit count +2)";
    }

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        const int tMax = (benchmarkMaxMagic >= 0) ? benchmarkMaxMagic : (n + 2);
        int tCount = 0;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        const int rowLen = n / colLen;

        const auto iterClock = std::chrono::high_resolution_clock::now();

        for (int d = 0; d < depth; d++) {
            bitCapInt zMask = ZERO_BCI;
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if ((2 * qReg->Rand()) < ONE_R1) {
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->S(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->H(i);
                        qReg->S(i);
                    }
                    // else - identity
                } else {
                    gateRand = DimCount1Qb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->IS(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                    // else - identity
                }

                //"Position" transforms:

                // Discrete Z root gates option:
                gateRand = 2 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    bi_or_ip(&zMask, pow2(i));
                }

                gateRand = 2 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    if ((2 * qReg->Rand()) < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                }

                if (tCount < tMax) {
                    gateRand = n * depth * qReg->Rand() / (n + 2);
                    if (gateRand < ONE_R1) {
                        if ((2 * qReg->Rand()) < ONE_R1) {
                            qReg->T(i);
                        } else {
                            qReg->IT(i);
                        }
                        tCount++;
                    }
                }

                if (timeout >= 0) {
                    auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - iterClock);
                    if ((tClock.count() * clockFactor) > timeout) {
                        throw std::runtime_error("Timeout");
                    }
                }
            }
            qReg->ZMask(zMask);

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        qReg->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            qReg->AntiCNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->AntiCY(b1, b2);
                        } else {
                            qReg->AntiCZ(b1, b2);
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            qReg->CNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->CY(b1, b2);
                        } else {
                            qReg->CZ(b1, b2);
                        }
                    }

                    if (timeout >= 0) {
                        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - iterClock);
                        if ((tClock.count() * clockFactor) > timeout) {
                            throw std::runtime_error("Timeout");
                        }
                    }
                }
            }
        }
    });
}

TEST_CASE("test_stabilizer_rz", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    if (benchmarkMaxMagic >= 0) {
        std::cout << "(max quantum \"magic\": " << benchmarkMaxMagic << ")";
    } else {
        std::cout << "(max quantum \"magic\": default, ceiling equal to qubit count +2)";
    }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        const int tMax = (benchmarkMaxMagic >= 0) ? benchmarkMaxMagic : (n + 2);
        int tCount = 0U;

        std::set<bitLenInt> qubitSet;
        for (bitLenInt i = 0; i < n; ++i) {
            qubitSet.insert(i);
        }

        for (int d = 0; d < depth; ++d) {
            for (bitLenInt i = 0; i < n; ++i) {
                for (int j = 0; j < 3; ++j) {
                    // We're trying to cover 3 Pauli axes
                    // with Euler angle axes x-z-x.
                    qReg->H(i);

                    // We can trace out a quarter rotations around the Bloch sphere with stabilizer.
                    const int gate1Qb = (int)(4 * qReg->Rand());
                    if (gate1Qb & 1) {
                        qReg->S(i);
                    }
                    if (gate1Qb & 2) {
                        qReg->Z(i);
                    }

                    if ((tCount < tMax) && (((3 * n * depth * qReg->Rand()) / benchmarkMaxMagic) < ONE_R1)) {
                        real1_f angle = ZERO_R1_F;
                        do {
                            angle = qReg->Rand();
                        } while (angle <= FP_NORM_EPSILON);
                        angle *= PI_R1 / 2;

                        qReg->RZ(angle, i);

                        tCount++;
                    }
                }
            }

            std::set<bitLenInt> unusedBits = qubitSet;
            while (unusedBits.size() > 1) {
                const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                qReg->CNOT(b1, b2);
            }
        }
    });
}

TEST_CASE("test_stabilizer_rz_nn", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    if (benchmarkMaxMagic >= 0) {
        std::cout << "(max quantum \"magic\": " << benchmarkMaxMagic << ")";
    } else {
        std::cout << "(max quantum \"magic\": default, ceiling equal to qubit count +2)";
    }

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        const int tMax = (benchmarkMaxMagic >= 0) ? benchmarkMaxMagic : (n + 2);
        int tCount = 0;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        const auto iterClock = std::chrono::high_resolution_clock::now();

        for (int d = 0; d < depth; d++) {
            bitCapInt zMask = ZERO_BCI;
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if ((2 * qReg->Rand()) < ONE_R1) {
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->S(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->H(i);
                        qReg->S(i);
                    }
                    // else - identity
                } else {
                    gateRand = DimCount1Qb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->IS(i);
                    } else if (gateRand < (3 * ONE_R1)) {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                    // else - identity
                }

                //"Position" transforms:

                // Discrete Z root gates option:
                gateRand = 2 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    bi_or_ip(&zMask, pow2(i));
                }

                gateRand = 2 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    if ((2 * qReg->Rand()) < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                }

                if (tCount < tMax) {
                    gateRand = n * depth * qReg->Rand() / (n + 2);
                    if (gateRand < ONE_R1) {
                        qReg->RZ(4 * PI_R1 * qReg->Rand(), i);
                        tCount++;
                    }
                }

                if (timeout >= 0) {
                    auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - iterClock);
                    if ((tClock.count() * clockFactor) > timeout) {
                        throw std::runtime_error("Timeout");
                    }
                }
            }
            qReg->ZMask(zMask);

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        qReg->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            qReg->AntiCNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->AntiCY(b1, b2);
                        } else {
                            qReg->AntiCZ(b1, b2);
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            qReg->CNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->CY(b1, b2);
                        } else {
                            qReg->CZ(b1, b2);
                        }
                    }

                    if (timeout >= 0) {
                        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - iterClock);
                        if ((tClock.count() * clockFactor) > timeout) {
                            throw std::runtime_error("Timeout");
                        }
                    }
                }
            }
        }
    });
}

TEST_CASE("test_dense", "[supreme]")
{
    // Try with environment variable
    // QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466
    // for clamping of single bit states to Pauli basis axes.

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        const int rowLen = n / colLen;

        for (int d = 0; d < depth; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                const real1_f theta = 4 * (real1_f)PI_R1 * qReg->Rand();
                const real1_f phi = 4 * (real1_f)PI_R1 * qReg->Rand();
                const real1_f lambda = 4 * (real1_f)PI_R1 * qReg->Rand();
                qReg->U(i, (real1_f)theta, (real1_f)phi, (real1_f)lambda);
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        qReg->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            qReg->AntiCNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->AntiCY(b1, b2);
                        } else {
                            qReg->AntiCZ(b1, b2);
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            qReg->CNOT(b1, b2);
                        } else if (gateRand < (2 * ONE_R1)) {
                            qReg->CY(b1, b2);
                        } else {
                            qReg->CZ(b1, b2);
                        }
                    }
                }
            }
        }
    });
}

TEST_CASE("test_stabilizer_t_cc_nn", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to
        // the paper.
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        const int rowLen = n / colLen;

        // qReg->SetReactiveSeparate(n > maxShardQubits);
        qReg->SetReactiveSeparate(true);

        for (int d = 0; d < depth; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->H(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                }
                // else - identity

                // "Position transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                /*
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    qReg->Z(i);
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    qReg->Z(i);
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    qReg->Z(i);
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
                */
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;

                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    int tempRow = row + ((gate & 2U) ? 1 : -1);
                    int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    // Try to pack 3-qubit gates as "greedily" as we can:
                    bitLenInt tempGate = 0U;
                    bitLenInt b3 = 0U;
                    do {
                        tempRow = row + ((tempGate & 2U) ? 1 : -1);
                        tempCol = col + ((colLen == 1) ? 0 : ((tempGate & 1U) ? 1 : 0));

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));

                    const bool is3Qubit = (tempGate < 4) && ((qReg->Rand() * 2) >= ONE_R1);
                    if (is3Qubit) {
                        usedBits.push_back(b3);
                    }

                    if ((qReg->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if (is3Qubit) {
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b1, b3);
                        }
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b2, b3);
                        }
                    }

                    real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (is3Qubit) {
                        if ((gateRand < (3 * ONE_R1)) && ((8 * qReg->Rand()) < ONE_R1)) {
                            const std::vector<bitLenInt> controls{ b1 };
                            qReg->CSwap(controls, b2, b3);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCNOT(b1, b2, b3);
                            } else {
                                qReg->AntiCCNOT(b1, b2, b3);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCY(b1, b2, b3);
                            } else {
                                qReg->AntiCCY(b1, b2, b3);
                            }
                        } else if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCZ(b1, b2, b3);
                            } else {
                                qReg->AntiCCZ(b1, b2, b3);
                            }
                        }
                        // else - identity

                        // std::cout << "(b1, b2, b3) = (" << (int)b1 << ", " << (int)b2 << ", " << (int)b3 << ")"
                        //           << std::endl;
                    } else {
                        if ((gateRand < (3 * ONE_R1)) && ((4 * qReg->Rand()) < ONE_R1)) {
                            // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                            qReg->Swap(b1, b2);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CNOT(b1, b2);
                                } else {
                                    qReg->AntiCNOT(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CY(b1, b2);
                                } else {
                                    qReg->AntiCY(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CZ(b1, b2);
                            } else {
                                qReg->AntiCZ(b1, b2);
                            }
                        }
                        // else - identity

                        // std::cout << "(b1, b2) = (" << (int)b1 << ", " << (int)b2 << ")" << std::endl;
                    }
                }
            }
        }
    });
}

TEST_CASE("test_circuit_t_nn", "[supreme]")
{
    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    const complex h[4]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        QCircuitPtr circuit = std::make_shared<QCircuit>();

        for (int d = 0; d < n; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if ((2 * qReg->Rand()) < ONE_R1) {
                    if (gateRand < ONE_R1) {
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    } else if (gateRand < (2 * ONE_R1)) {
                        // qReg->S(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                    } else if (gateRand < (3 * ONE_R1)) {
                        // qReg->H(i);
                        // qReg->S(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                    }
                    // else - identity
                } else {
                    gateRand = DimCount1Qb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    } else if (gateRand < (2 * ONE_R1)) {
                        // qReg->IS(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, is));
                    } else if (gateRand < (3 * ONE_R1)) {
                        // qReg->IS(i);
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, is));
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    }
                    // else - identity
                }

                //"Position" transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                const complex p[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, std::polar(ONE_R1, (real1)gateRand) };
                circuit->AppendGate(std::make_shared<QCircuitGate>(i, p));
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        circuit->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    const std::set<bitLenInt> controls{ b1 };
                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            // qReg->AntiCNOT(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, controls, ZERO_BCI));
                        } else if (gateRand < (2 * ONE_R1)) {
                            // qReg->AntiCY(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, controls, ZERO_BCI));
                        } else {
                            // qReg->AntiCZ(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, controls, ZERO_BCI));
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            // qReg->CNOT(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, controls, ONE_BCI));
                        } else if (gateRand < (2 * ONE_R1)) {
                            // qReg->CY(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, controls, ONE_BCI));
                        } else {
                            // qReg->CZ(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, controls, ONE_BCI));
                        }
                    }
                }
            }
        }

        circuit->Run(qReg);
    });
}

TEST_CASE("test_circuit_t_nn_generate_and_load", "[supreme]")
{
    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    const complex h[4]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    int iter = 0;

    const std::string path = "circuit_t_nn";
#if defined(_WIN32) && !defined(__CYGWIN__)
    int err = _mkdir(path.c_str());
#else
    int err = mkdir(path.c_str(), 0700);
#endif
    if (err != -1) {
        std::cout << "Making directory: " << path << std::endl;
    }

    std::cout << std::endl << "Generating optimized circuits..." << std::endl;
    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of "Sycamore circuits."
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        const int rowLen = n / colLen;

        QCircuitPtr circuit = std::make_shared<QCircuit>();

        for (int d = 0; d < n; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if ((2 * qReg->Rand()) < ONE_R1) {
                    if (gateRand < ONE_R1) {
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    } else if (gateRand < (2 * ONE_R1)) {
                        // qReg->S(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                    } else if (gateRand < (3 * ONE_R1)) {
                        // qReg->H(i);
                        // qReg->S(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                    }
                    // else - identity
                } else {
                    gateRand = DimCount1Qb * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    } else if (gateRand < (2 * ONE_R1)) {
                        // qReg->IS(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, is));
                    } else if (gateRand < (3 * ONE_R1)) {
                        // qReg->IS(i);
                        // qReg->H(i);
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, is));
                        circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    }
                    // else - identity
                }

                //"Position" transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                const complex p[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, std::polar(ONE_R1, (real1)gateRand) };
                circuit->AppendGate(std::make_shared<QCircuitGate>(i, p));
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    const int tempRow = row + ((gate & 2U) ? 1 : -1);
                    const int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    const real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (gateRand >= (3 * ONE_R1)) {
                        // 1/4 chance of identity
                        continue;
                    }

                    if ((4 * qReg->Rand()) < ONE_R1) {
                        // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                        circuit->Swap(b1, b2);
                        continue;
                    }

                    if ((qReg->Rand() * 2) < ONE_R1) {
                        std::swap(b1, b2);
                    }

                    const std::set<bitLenInt> controls{ b1 };
                    if ((2 * qReg->Rand()) < ONE_R1) {
                        if (gateRand < ONE_R1) {
                            // qReg->AntiCNOT(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, controls, ZERO_BCI));
                        } else if (gateRand < (2 * ONE_R1)) {
                            // qReg->AntiCY(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, controls, ZERO_BCI));
                        } else {
                            // qReg->AntiCZ(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, controls, ZERO_BCI));
                        }
                    } else {
                        if (gateRand < ONE_R1) {
                            // qReg->CNOT(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, controls, ONE_BCI));
                        } else if (gateRand < (2 * ONE_R1)) {
                            // qReg->CY(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, controls, ONE_BCI));
                        } else {
                            // qReg->CZ(b1, b2);
                            circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, controls, ONE_BCI));
                        }
                    }
                }
            }
        }

        // circuit->Run(qReg);

        std::ofstream ofile;
        std::string nstr = std::to_string(n);
        std::string istr = std::to_string(iter);
        ofile.open(path + "/qcircuit_test_" + nstr + "_" + istr + ".qgc");
        ofile << circuit;
        ofile.close();

        ++iter;
        if (iter == benchmarkSamples) {
            iter = 0;
        }
    });

    std::cout << std::endl << "Loading optimized circuits from disk and running..." << std::endl;
    iter = 0;
    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        std::ifstream ifile;
        std::string nstr = std::to_string(n);
        std::string istr = std::to_string(iter);
        ifile.open(path + "/qcircuit_test_" + nstr + "_" + istr + ".qgc");
        QCircuitPtr circuit = std::make_shared<QCircuit>();
        ifile >> circuit;
        ifile.close();

        circuit->Run(qReg);
        ++iter;
        if (iter == benchmarkSamples) {
            iter = 0;
        }
    });
}

void inject_1qb_u3_noise(QInterfacePtr qReg, bitLenInt qubit, real1_f distance)
{
    distance = 2 * PI_R1 * distance * qReg->Rand();
    distance *= distance;
    real1_f th = qReg->Rand();
    real1_f ph = qReg->Rand();
    real1_f lm = qReg->Rand();
    real1_f nrm = th * th + ph * ph + lm * lm;
    th = distance * th / nrm;
    ph = distance * ph / nrm;
    lm = distance * lm / nrm;
    // Displace the qubit by the distance in a unitary manner:
    qReg->U(qubit, th, ph, lm);
}

TEST_CASE("test_noisy_stabilizer_t_cc_nn", "[supreme]")
{
    real1_f noiseParam = ONE_R1 / 5;
#if ENABLE_ENV_VARS
    if (getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")) {
        noiseParam = (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")));
    }
#endif

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    std::cout << "(noise parameter: " << noiseParam << ")";

    const int DimCount1Qb = 4;
    const int GateCountMultiQb = 4;

    // bitLenInt maxShardQubits = -1;
    // if (getenv("QRACK_MAX_PAGING_QB")) {
    //     maxShardQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    // }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to
        // the paper.
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        // qReg->SetReactiveSeparate(n > maxShardQubits);
        qReg->SetReactiveSeparate(true);

        for (int d = 0; d < depth; d++) {
            for (bitLenInt i = 0; i < n; i++) {
                // "Phase" transforms:
                real1_f gateRand = DimCount1Qb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->H(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                }
                // else - identity

                // "Position transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                /*
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    qReg->Z(i);
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    qReg->Z(i);
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    qReg->Z(i);
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
                */
            }

            const bitLenInt gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    bitLenInt b1 = row * colLen + col;

                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    int tempRow = row + ((gate & 2U) ? 1 : -1);
                    int tempCol = col + ((colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0));

                    bitLenInt b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    // Try to pack 3-qubit gates as "greedily" as we can:
                    bitLenInt tempGate = 0U;
                    bitLenInt b3 = 0U;
                    do {
                        tempRow += row + ((tempGate & 2U) ? 1 : -1);
                        tempCol += col + ((colLen == 1) ? 0 : ((tempGate & 1U) ? 1 : 0));

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));

                    const bool is3Qubit = (tempGate < 4) && ((qReg->Rand() * 2) >= ONE_R1);
                    if (is3Qubit) {
                        usedBits.push_back(b3);
                    }

                    if ((qReg->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if (is3Qubit) {
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b1, b3);
                        }
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b2, b3);
                        }
                    }

                    real1_f gateRand = GateCountMultiQb * qReg->Rand();

                    if (is3Qubit) {
                        if (gateRand < (3 * ONE_R1)) {
                            inject_1qb_u3_noise(qReg, b1, noiseParam);
                            inject_1qb_u3_noise(qReg, b2, noiseParam);
                            inject_1qb_u3_noise(qReg, b3, noiseParam);
                        }

                        if ((gateRand < (3 * ONE_R1)) && ((8 * qReg->Rand()) < ONE_R1)) {
                            const std::vector<bitLenInt> controls{ b1 };
                            qReg->CSwap(controls, b2, b3);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCNOT(b1, b2, b3);
                            } else {
                                qReg->AntiCCNOT(b1, b2, b3);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCY(b1, b2, b3);
                            } else {
                                qReg->AntiCCY(b1, b2, b3);
                            }
                        } else if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCZ(b1, b2, b3);
                            } else {
                                qReg->AntiCCZ(b1, b2, b3);
                            }
                        }
                        // else - identity

                        // std::cout << "(b1, b2, b3) = (" << (int)b1 << ", " << (int)b2 << ", " << (int)b3 << ")"
                        //           << std::endl;
                    } else {
                        if ((gateRand < (3 * ONE_R1)) && ((4 * qReg->Rand()) < ONE_R1)) {
                            // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                            qReg->Swap(b1, b2);
                            continue;
                        }

                        if (gateRand < (3 * ONE_R1)) {
                            inject_1qb_u3_noise(qReg, b1, noiseParam);
                            inject_1qb_u3_noise(qReg, b2, noiseParam);
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CNOT(b1, b2);
                                } else {
                                    qReg->AntiCNOT(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CY(b1, b2);
                                } else {
                                    qReg->AntiCY(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (3 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CZ(b1, b2);
                            } else {
                                qReg->AntiCZ(b1, b2);
                            }
                        }
                        // else - identity

                        // std::cout << "(b1, b2) = (" << (int)b1 << ", " << (int)b2 << ")" << std::endl;
                    }
                }
            }
        }
    });
}

TEST_CASE("test_dense_cc_nn", "[supreme]")
{
    // Try with environment variable
    // QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466
    // for clamping of single bit states to Pauli basis axes.

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCountMultiQb = 3;

    // bitLenInt maxShardQubits = -1;
    // if (getenv("QRACK_MAX_PAGING_QB")) {
    //     maxShardQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    // }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        int d;
        bitLenInt i;
        real1_f gateRand;
        bitLenInt b1, b2, b3 = 0;
        bool is3Qubit;
        int row, col;
        int tempRow, tempCol;
        bitLenInt gate, tempGate;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to
        // the paper.
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        // qReg->SetReactiveSeparate(n > maxShardQubits);
        qReg->SetReactiveSeparate(true);

        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                // Random general 3-parameter unitary gate via Euler angles
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->H(i);
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->S(i);
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->IS(i);
                qReg->H(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    b1 = row * colLen + col;

                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    // Try to pack 3-qubit gates as "greedily" as we can:
                    tempGate = 0U;
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2U) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1U) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));

                    is3Qubit = (tempGate < 4) && ((qReg->Rand() * 2) >= ONE_R1);
                    if (is3Qubit) {
                        usedBits.push_back(b3);
                    }

                    if ((qReg->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if (is3Qubit) {
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b1, b3);
                        }
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b2, b3);
                        }
                    }

                    gateRand = GateCountMultiQb * qReg->Rand();

                    if (is3Qubit) {
                        if ((8 * qReg->Rand()) < ONE_R1) {
                            const std::vector<bitLenInt> controls{ b1 };
                            qReg->CSwap(controls, b2, b3);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCNOT(b1, b2, b3);
                            } else {
                                qReg->AntiCCNOT(b1, b2, b3);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCY(b1, b2, b3);
                            } else {
                                qReg->AntiCCY(b1, b2, b3);
                            }
                        } else {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCZ(b1, b2, b3);
                            } else {
                                qReg->AntiCCZ(b1, b2, b3);
                            }
                        }

                        // std::cout << "(b1, b2, b3) = (" << (int)b1 << ", " << (int)b2 << ", " << (int)b3 << ")"
                        //           << std::endl;
                    } else {
                        if ((4 * qReg->Rand()) < ONE_R1) {
                            // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                            qReg->Swap(b1, b2);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CNOT(b1, b2);
                                } else {
                                    qReg->AntiCNOT(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CY(b1, b2);
                                } else {
                                    qReg->AntiCY(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CZ(b1, b2);
                            } else {
                                qReg->AntiCZ(b1, b2);
                            }
                        }

                        // std::cout << "(b1, b2) = (" << (int)b1 << ", " << (int)b2 << ")" << std::endl;
                    }
                }
            }
        }
    });
}

TEST_CASE("test_noisy_dense_cc_nn", "[supreme]")
{
    real1_f noiseParam = ONE_R1 / 5;
#if ENABLE_ENV_VARS
    if (getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")) {
        noiseParam = (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")));
    }
#endif

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    std::cout << "(noise parameter: " << noiseParam << ")";

    const int GateCountMultiQb = 3;

    // bitLenInt maxShardQubits = -1;
    // if (getenv("QRACK_MAX_PAGING_QB")) {
    //     maxShardQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    // }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        int d;
        bitLenInt i;
        real1_f gateRand;
        bitLenInt b1, b2, b3 = 0;
        bool is3Qubit;
        int row, col;
        int tempRow, tempCol;
        bitLenInt gate, tempGate;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to
        // the paper.
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        // qReg->SetReactiveSeparate(n > maxShardQubits);
        qReg->SetReactiveSeparate(true);

        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                // Random general 3-parameter unitary gate via Euler angles
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->H(i);
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->S(i);
                gateRand = (real1_f)(4 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);
                qReg->IS(i);
                qReg->H(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    b1 = row * colLen + col;

                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    // Try to pack 3-qubit gates as "greedily" as we can:
                    tempGate = 0U;
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2U) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1U) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));

                    is3Qubit = (tempGate < 4) && ((qReg->Rand() * 2) >= ONE_R1);
                    if (is3Qubit) {
                        usedBits.push_back(b3);
                    }

                    if ((qReg->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if (is3Qubit) {
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b1, b3);
                        }
                        if ((qReg->Rand() * 2) >= ONE_R1) {
                            std::swap(b2, b3);
                        }
                    }

                    gateRand = GateCountMultiQb * qReg->Rand();

                    if (is3Qubit) {
                        inject_1qb_u3_noise(qReg, b1, noiseParam);
                        inject_1qb_u3_noise(qReg, b2, noiseParam);
                        inject_1qb_u3_noise(qReg, b3, noiseParam);

                        if ((8 * qReg->Rand()) < ONE_R1) {
                            const std::vector<bitLenInt> controls{ b1 };
                            qReg->CSwap(controls, b2, b3);
                            continue;
                        }

                        if (gateRand < ONE_R1) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCNOT(b1, b2, b3);
                            } else {
                                qReg->AntiCCNOT(b1, b2, b3);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCY(b1, b2, b3);
                            } else {
                                qReg->AntiCCY(b1, b2, b3);
                            }
                        } else {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CCZ(b1, b2, b3);
                            } else {
                                qReg->AntiCCZ(b1, b2, b3);
                            }
                        }

                        // std::cout << "(b1, b2, b3) = (" << (int)b1 << ", " << (int)b2 << ", " << (int)b3 << ")"
                        //           << std::endl;
                    } else {
                        if ((4 * qReg->Rand()) < ONE_R1) {
                            // In 3 CNOT(a,b) sequence, for example, 1/4 of sequences on average are equivalent to SWAP.
                            qReg->Swap(b1, b2);
                            continue;
                        }

                        inject_1qb_u3_noise(qReg, b1, noiseParam);
                        inject_1qb_u3_noise(qReg, b2, noiseParam);

                        if (gateRand < ONE_R1) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CNOT(b1, b2);
                                } else {
                                    qReg->AntiCNOT(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else if (gateRand < (2 * ONE_R1)) {
                            gateRand = 4 * qReg->Rand();
                            if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CY(b1, b2);
                                } else {
                                    qReg->AntiCY(b1, b2);
                                }
                            } else {
                                qReg->Swap(b1, b2);
                            }
                        } else {
                            gateRand = 2 * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CZ(b1, b2);
                            } else {
                                qReg->AntiCZ(b1, b2);
                            }
                        }

                        // std::cout << "(b1, b2) = (" << (int)b1 << ", " << (int)b2 << ")" << std::endl;
                    }
                }
            }
        }
    });
}

TEST_CASE("test_stabilizer_ct_nn", "[supreme]")
{
    // Try with environment variable
    // QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466
    // for clamping of single bit states to Pauli basis axes.

    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int DimCount1Qb = 4;
    const int DimCountMultiQb = 4;

    // bitLenInt maxShardQubits = -1;
    // if (getenv("QRACK_MAX_PAGING_QB")) {
    //     maxShardQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    // }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        int d;
        bitLenInt i, gate;
        real1_f gateRand;
        complex top, bottom;
        bitLenInt b1, b2;
        int row, col;
        int tempRow, tempCol;
        std::vector<bitLenInt> controls(1);

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to
        // the paper.
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        // qReg->SetReactiveSeparate(n > maxShardQubits);
        qReg->SetReactiveSeparate(true);

        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                // "Phase" transforms:
                gateRand = DimCount1Qb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->H(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                    }
                } else if (gateRand < (3 * ONE_R1)) {
                    gateRand = 2 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->H(i);
                        qReg->S(i);
                    } else {
                        qReg->IS(i);
                        qReg->H(i);
                    }
                }
                // else - identity

                // "Position transforms:

                // Continuous Z root gates option:
                gateRand = (real1_f)(2 * PI_R1 * qReg->Rand());
                qReg->Phase(ONE_CMPLX, std::polar(ONE_R1, (real1)gateRand), i);

                // Discrete Z root gates option:
                /*
                gateRand = 8 * qReg->Rand();
                if (gateRand < ONE_R1) {
                    // Z^(1/4)
                    qReg->T(i);
                } else if (gateRand < (2 * ONE_R1)) {
                    // Z^(1/2)
                    qReg->S(i);
                } else if (gateRand < (3 * ONE_R1)) {
                    // Z^(3/4)
                    qReg->Z(i);
                    qReg->IT(i);
                } else if (gateRand < (4 * ONE_R1)) {
                    // Z
                    qReg->Z(i);
                } else if (gateRand < (5 * ONE_R1)) {
                    // Z^(-3/4)
                    qReg->Z(i);
                    qReg->T(i);
                } else if (gateRand < (6 * ONE_R1)) {
                    // Z^(-1/2)
                    qReg->IS(i);
                } else if (gateRand < (7 * ONE_R1)) {
                    // Z^(-1/4)
                    qReg->IT(i);
                }
                // else - identity
                */
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            std::vector<bitLenInt> usedBits;

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    b1 = row * colLen + col;
                    if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                        continue;
                    }

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    b2 = tempRow * colLen + tempCol;

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                        (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                        continue;
                    }

                    usedBits.push_back(b1);
                    usedBits.push_back(b2);

                    if ((qReg->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }

                    gateRand = 4 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        // 1 out of 4 chance of producing swap from 3 CNOTs, for example.
                        gateRand = DimCount1Qb * qReg->Rand();
                        if (gateRand < (3 * ONE_R1)) {
                            if (gateRand < ONE_R1) {
                                qReg->Swap(b1, b2);
                            } else {
                                qReg->ISwap(b1, b2);
                            }
                        }
                        // else - identity
                    } else {
                        gateRand = 2 * qReg->Rand();
                        if (gateRand < ONE_R1) {
                            // "Phase" transforms:
                            gateRand = DimCountMultiQb * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->CH(b1, b2);
                            } else if (gateRand < (2 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CS(b1, b2);
                                } else {
                                    qReg->CIS(b1, b2);
                                }
                            } else if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->CH(b1, b2);
                                    qReg->CS(b1, b2);
                                } else {
                                    qReg->CIS(b1, b2);
                                    qReg->CH(b1, b2);
                                }
                            }
                            // else - identity

                            // "Position transforms:

                            // Continuous Z root gates option:
                            controls[0] = b1;
                            top = std::polar(ONE_R1, (real1)(2 * PI_R1 * qReg->Rand()));
                            bottom = std::conj(top);
                            qReg->MCPhase(controls, top, bottom, b2);
                        } else {
                            // "Phase" transforms:
                            gateRand = DimCountMultiQb * qReg->Rand();
                            if (gateRand < ONE_R1) {
                                qReg->AntiCH(b1, b2);
                            } else if (gateRand < (2 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->AntiCS(b1, b2);
                                } else {
                                    qReg->AntiCIS(b1, b2);
                                }
                            } else if (gateRand < (3 * ONE_R1)) {
                                gateRand = 2 * qReg->Rand();
                                if (gateRand < ONE_R1) {
                                    qReg->AntiCH(b1, b2);
                                    qReg->AntiCS(b1, b2);
                                } else {
                                    qReg->AntiCIS(b1, b2);
                                    qReg->AntiCH(b1, b2);
                                }
                            }
                            // else - identity

                            // "Position transforms:

                            // Continuous Z root gates option:
                            controls[0] = b1;
                            top = std::polar(ONE_R1, (real1)(2 * PI_R1 * qReg->Rand()));
                            bottom = std::conj(top);
                            qReg->MACPhase(controls, top, bottom, b2);
                        }
                    }
                }
            }
        }
    });
}

TEST_CASE("test_universal_circuit_continuous", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCountMultiQb = 2;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

            int d;
            bitLenInt i;
            real1_f theta, phi, lambda;
            bitLenInt b1, b2;

            for (d = 0; d < depth; d++) {

                for (i = 0; i < n; i++) {
                    theta = (real1_f)(2 * PI_R1 * qReg->Rand());
                    phi = (real1_f)(2 * PI_R1 * qReg->Rand());
                    lambda = (real1_f)(2 * PI_R1 * qReg->Rand());

                    qReg->U(i, theta, phi, lambda);
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    b2 = pickRandomBit(qReg->Rand(), &unusedBits);

                    if ((GateCountMultiQb * qReg->Rand()) < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else {
                        qReg->CNOT(b1, b2);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_discrete", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCount1Qb = 2;
    const int GateCountMultiQb = 2;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

            int d;
            bitLenInt i;
            real1_f gateRand;
            bitLenInt b1, b2, b3;
            int maxGates;

            for (d = 0; d < depth; d++) {

                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    }
                    // Otherwise, no H gate
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    b2 = pickRandomBit(qReg->Rand(), &unusedBits);

                    if (unusedBits.size() > 0) {
                        maxGates = GateCountMultiQb;
                    } else {
                        maxGates = GateCountMultiQb - 1U;
                    }

                    gateRand = maxGates * qReg->Rand();

                    if ((unusedBits.size() == 0) || (gateRand < ONE_R1)) {
                        qReg->Swap(b1, b2);
                    } else {
                        b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_digital", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

            int d;
            bitLenInt i;
            real1_f gateRand;
            bitLenInt b1, b2, b3;
            int maxGates;

            for (d = 0; d < depth; d++) {

                bitCapInt xMask = ZERO_BCI;
                bitCapInt yMask = ZERO_BCI;
                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&xMask, pow2(i));
                    } else if (gateRand < (3 * ONE_R1 / GateCount1Qb)) {
                        bi_or_ip(&yMask, pow2(i));
                    } else {
                        qReg->T(i);
                    }
                }
                qReg->XMask(xMask);
                qReg->YMask(yMask);

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    b2 = pickRandomBit(qReg->Rand(), &unusedBits);

                    if (unusedBits.size() > 0) {
                        maxGates = GateCountMultiQb;
                    } else {
                        maxGates = GateCountMultiQb - 1U;
                    }

                    gateRand = maxGates * qReg->Rand();

                    if (gateRand < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->CZ(b1, b2);
                    } else if ((unusedBits.size() == 0) || (gateRand < (3 * ONE_R1))) {
                        qReg->CNOT(b1, b2);
                    } else {
                        b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_analog", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCount1Qb = 3;
    const int GateCountMultiQb = 4;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

            int d;
            bitLenInt i;
            real1_f gateRand;
            bitLenInt b1, b2, b3;
            std::vector<bitLenInt> control;
            complex polar0;
            bool canDo3;
            int gateThreshold, gateMax;

            for (d = 0; d < depth; d++) {

                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    polar0 = complex(std::polar(ONE_R1, (real1)(2 * M_PI * qReg->Rand())));
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        qReg->Phase(ONE_CMPLX, polar0, i);
                    } else {
                        qReg->Invert(ONE_CMPLX, polar0, i);
                    }
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // TrySeparate hurts average time, in this case, but it majorly benefits statistically common
                    // worse cases, on these random circuits.
                    qReg->TrySeparate(i);
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    b2 = pickRandomBit(qReg->Rand(), &unusedBits);

                    canDo3 = (unusedBits.size() > 0);
                    if (canDo3) {
                        gateThreshold = 3;
                        gateMax = GateCountMultiQb;
                    } else {
                        gateThreshold = 2;
                        gateMax = GateCountMultiQb - 1;
                    }

                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / gateMax)) {
                        qReg->Swap(b1, b2);
                    } else if (canDo3 && (gateRand < (2 * ONE_R1 / GateCountMultiQb))) {
                        b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    } else {
                        control[0] = b1;
                        polar0 = complex(std::polar(ONE_R1, (real1)(2 * M_PI * qReg->Rand())));
                        if (gateRand < (gateThreshold * ONE_R1 / gateMax)) {
                            qReg->MCPhase(control, polar0, -polar0, b2);
                        } else {
                            qReg->MCInvert(control, polar0, polar0, b2);
                        }
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_ccz_ccx_h", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 4;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {
            const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

            int d;
            bitLenInt i;
            real1_f gateRand;
            bitLenInt b1, b2, b3;
            int maxGates;

            for (d = 0; d < depth; d++) {

                bitCapInt zMask = ZERO_BCI;
                bitCapInt xMask = ZERO_BCI;
                for (i = 0; i < n; i++) {
                    gateRand = GateCount1Qb * qReg->Rand();
                    if (gateRand < 1) {
                        qReg->H(i);
                    } else if (gateRand < 2) {
                        bi_or_ip(&zMask, pow2(i));
                    } else if (gateRand < 3) {
                        bi_or_ip(&xMask, pow2(i));
                    } else {
                        // Identity;
                    }
                }
                qReg->ZMask(zMask);
                qReg->XMask(xMask);

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                    b2 = pickRandomBit(qReg->Rand(), &unusedBits);

                    if (unusedBits.size() > 0) {
                        maxGates = GateCountMultiQb;
                    } else {
                        maxGates = GateCountMultiQb - 2U;
                    }

                    gateRand = maxGates * qReg->Rand();

                    if (gateRand < ONE_R1) {
                        qReg->CZ(b1, b2);
                    } else if ((unusedBits.size() == 0) || (gateRand < 2)) {
                        qReg->CNOT(b1, b2);
                    } else if (gateRand < 3) {
                        b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCZ(b1, b2, b3);
                    } else {
                        b3 = pickRandomBit(qReg->Rand(), &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_quantum_supremacy", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit.";

    // This is an attempt to simulate the circuit argued to establish quantum supremacy.
    // See https://doi.org/10.1038/s41586-019-1666-5

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
        // paper.
        const bitLenInt deadQubit = 3U;
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;

        // std::cout<<"n="<<(int)n<<std::endl;
        // std::cout<<"rowLen="<<(int)rowLen<<std::endl;
        // std::cout<<"colLen="<<(int)colLen<<std::endl;

        real1_f gateRand;
        bitLenInt gate;
        int b1, b2;
        bitLenInt i;
        int d;
        int row, col;
        int tempRow, tempCol;

        std::vector<bitLenInt> controls(1);

        std::vector<int> lastSingleBitGates;
        std::set<int>::iterator gateChoiceIterator;
        int gateChoice;

        // We repeat the entire prepartion for "depth" iterations.
        // We can avoid maximal representational entanglement of the state as a single Schr{\"o}dinger method unit.
        // See https://arxiv.org/abs/1710.05867
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                if ((n == 54U) && (i == deadQubit)) {
                    if (d == 0) {
                        lastSingleBitGates.push_back(0);
                    }
                    continue;
                }

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
                // are handled as general single bit gates.

                // The same gate is not applied twice consecutively in sequence.

                if (d == 0) {
                    // For the first iteration, we can pick any gate.

                    gateRand = 3 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(0);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(1);
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(2);
                    }
                } else {
                    // For all subsequent iterations after the first, we eliminate the choice of the same gate applied
                    // on the immediately previous iteration.

                    gateChoice = (int)(2 * qReg->Rand());
                    if (gateChoice >= 2) {
                        gateChoice = 1;
                    }
                    if (gateChoice >= lastSingleBitGates[i]) {
                        ++gateChoice;
                    }

                    if (gateChoice == 0) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 0;
                    } else if (gateChoice == 1) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 1;
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 2;
                    }
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose")
                // the representation without changing the logical state of the QUnit, up to float error:
                // qReg->TrySeparate(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                        continue;
                    }

                    b1 = row * colLen + col;
                    b2 = tempRow * colLen + tempCol;

                    if ((n == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                        continue;
                    }

                    // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
                    // std::endl;

                    qReg->TrySeparate(b1, b2);
                    qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                    qReg->TrySeparate(b1, b2);
                }
            }
            // std::cout<<"Depth++"<<std::endl;
        }
    });
}

TEST_CASE("test_quantum_supremacy_patch", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit.";

    // This is an attempt to simulate the circuit argued to establish quantum supremacy.
    // See https://doi.org/10.1038/s41586-019-1666-5

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
        // paper.
        const bitLenInt deadQubit = 3U;
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;
        const bitLenInt patchBound = rowLen >> 1U;

        // std::cout<<"n="<<(int)n<<std::endl;
        // std::cout<<"rowLen="<<(int)rowLen<<std::endl;
        // std::cout<<"colLen="<<(int)colLen<<std::endl;

        real1_f gateRand;
        bitLenInt gate;
        int b1, b2;
        bitLenInt i;
        int d;
        int row, col;
        int tempRow, tempCol;

        std::vector<bitLenInt> controls(1);

        std::vector<int> lastSingleBitGates;
        std::set<int>::iterator gateChoiceIterator;
        int gateChoice;

        // We repeat the entire prepartion for "depth" iterations.
        // We can avoid maximal representational entanglement of the state as a single Schr{\"o}dinger method unit.
        // See https://arxiv.org/abs/1710.05867
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                if ((n == 54U) && (i == deadQubit)) {
                    if (d == 0) {
                        lastSingleBitGates.push_back(0);
                    }
                    continue;
                }

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
                // are handled as general single bit gates.

                // The same gate is not applied twice consecutively in sequence.

                if (d == 0) {
                    // For the first iteration, we can pick any gate.

                    gateRand = 3 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(0);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(1);
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(2);
                    }
                } else {
                    // For all subsequent iterations after the first, we eliminate the choice of the same gate applied
                    // on the immediately previous iteration.

                    gateChoice = (int)(2 * qReg->Rand());
                    if (gateChoice >= 2) {
                        gateChoice = 1;
                    }
                    if (gateChoice >= lastSingleBitGates[i]) {
                        ++gateChoice;
                    }

                    if (gateChoice == 0) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 0;
                    } else if (gateChoice == 1) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 1;
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 2;
                    }
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose")
                // the representation without changing the logical state of the QUnit, up to float error:
                // qReg->TrySeparate(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                        continue;
                    }

                    b1 = row * colLen + col;
                    b2 = tempRow * colLen + tempCol;

                    if ((n == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                        continue;
                    }

                    // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
                    // std::endl;

                    if (((row < patchBound) && (tempRow >= patchBound)) ||
                        ((tempRow < patchBound) && (row >= patchBound))) {
                        continue;
                    }

                    qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                }
            }
            // std::cout<<"Depth++"<<std::endl;
        }
    });
}

TEST_CASE("test_quantum_supremacy_elided", "[supreme]")
{
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit.";

    // This is an attempt to simulate the circuit argued to establish quantum supremacy.
    // See https://doi.org/10.1038/s41586-019-1666-5

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
        // paper.
        const bitLenInt deadQubit = 3U;
        std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int colLen = std::sqrt(n);
        while (((n / colLen) * colLen) != n) {
            colLen--;
        }
        int rowLen = n / colLen;
        const bitLenInt patchBound = rowLen >> 1U;

        // std::cout<<"n="<<(int)n<<std::endl;
        // std::cout<<"rowLen="<<(int)rowLen<<std::endl;
        // std::cout<<"colLen="<<(int)colLen<<std::endl;

        real1_f gateRand;
        bitLenInt gate;
        int b1, b2;
        bitLenInt i;
        int d;
        int row, col;
        int tempRow, tempCol;

        std::vector<bitLenInt> controls(1);

        std::vector<int> lastSingleBitGates;
        std::set<int>::iterator gateChoiceIterator;
        int gateChoice;

        // We repeat the entire prepartion for "depth" iterations.
        // We can avoid maximal representational entanglement of the state as a single Schr{\"o}dinger method unit.
        // See https://arxiv.org/abs/1710.05867
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                if ((n == 54U) && (i == deadQubit)) {
                    if (d == 0) {
                        lastSingleBitGates.push_back(0);
                    }
                    continue;
                }

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
                // are handled as general single bit gates.

                // The same gate is not applied twice consecutively in sequence.

                if (d == 0) {
                    // For the first iteration, we can pick any gate.

                    gateRand = 3 * qReg->Rand();
                    if (gateRand < ONE_R1) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(0);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(1);
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates.push_back(2);
                    }
                } else {
                    // For all subsequent iterations after the first, we eliminate the choice of the same gate applied
                    // on the immediately previous iteration.

                    gateChoice = (int)(2 * qReg->Rand());
                    if (gateChoice >= 2) {
                        gateChoice = 1;
                    }
                    if (gateChoice >= lastSingleBitGates[i]) {
                        ++gateChoice;
                    }

                    if (gateChoice == 0) {
                        qReg->SqrtX(i);
                        // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 0;
                    } else if (gateChoice == 1) {
                        qReg->SqrtY(i);
                        // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 1;
                    } else {
                        qReg->SqrtW(i);
                        // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                        lastSingleBitGates[i] = 2;
                    }
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose")
                // the representation without changing the logical state of the QUnit, up to float error:
                // qReg->TrySeparate(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            for (row = 1; row < rowLen; row += 2) {
                for (col = 0; col < colLen; col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);
                    tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                        continue;
                    }

                    b1 = row * colLen + col;
                    b2 = tempRow * colLen + tempCol;

                    if ((n == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                        continue;
                    }

                    // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
                    // std::endl;

                    if (((row < patchBound) && (tempRow >= patchBound)) ||
                        ((tempRow < patchBound) && (row >= patchBound))) {
                        // This is our version of ("semi-classical") gate "elision":
                        QRACK_CONST real1_f phase_fac = 3 * PI_R1 / 2;
                        // FSim controlled phase
                        real1_f prob1 = qReg->Prob(b1);
                        qReg->U(b2, 0, 0, -prob1 * PI_R1 / 6);
                        // CNOT(b1, b2)^x
                        qReg->H(b2);
                        qReg->U(b2, 0, 0, prob1 * phase_fac);
                        qReg->H(b2);
                        // CNOT(b2, b1)^x
                        real1_f prob2 = qReg->Prob(b2);
                        qReg->H(b1);
                        qReg->U(b1, 0, 0, prob2 * phase_fac);
                        qReg->H(b1);
                        // CNOT(b1, b2)^x
                        prob1 = qReg->Prob(b1);
                        qReg->H(b2);
                        qReg->U(b2, 0, 0, prob1 * phase_fac);
                        qReg->H(b2);
                        // CZ(b1, b2)^-x
                        qReg->U(b2, 0, 0, -prob1 * phase_fac);
                        // T(b1)
                        qReg->T(b1);
                        // T(b2)
                        qReg->T(b2);
                    } else {
                        qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                    }
                }
            }
            // std::cout<<"Depth++"<<std::endl;
        }
    });
}

TEST_CASE("test_random_circuit_sampling", "[speed]")
{
    // This is a "quantum volume" circuit, but we're measuring execution time, not "heavy-output probability."
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;

        std::set<bitLenInt> unusedBitSet;
        for (bitLenInt i = 0; i < n; ++i) {
            unusedBitSet.insert(i);
        }

        for (bitLenInt d = 0U; d < depth; ++d) {
            // Single-qubit gate layer
            for (bitLenInt i = 0U; i < n; ++i) {
                real1_f theta = 2 * M_PI * qReg->Rand();
                real1_f phi = 2 * M_PI * qReg->Rand();
                real1_f lambda = 2 * M_PI * qReg->Rand();

                qReg->U(i, theta, phi, lambda);
            }

            // Two-qubit gate layer
            std::set<bitLenInt> unusedBits(unusedBitSet);
            while (unusedBits.size() > 1) {
                const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                qReg->CNOT(b1, b2);
            }
        }
    });
}

TEST_CASE("test_random_circuit_sampling_nn", "[speed]")
{
    // "nn" stands for "nearest-neighbor (coupler gates)"
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        std::vector<int> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };
        const int rowLen = std::ceil(std::sqrt(n));
        const int GateCount2Qb = 12;

        for (bitLenInt d = 0U; d < depth; ++d) {
            for (bitLenInt i = 0U; i < n; ++i) {
                // This effectively covers x-z-x Euler angles, every 3 layers:
                qReg->H(i);
                qReg->RZ(qReg->Rand() * 2 * PI_R1, i);
            }

            int gate = gateSequence.front();
            gateSequence.erase(gateSequence.begin());
            gateSequence.push_back(gate);
            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < rowLen; col++) {
                    int tempRow = row;
                    int tempCol = col;
                    tempRow += (gate & 2) ? 1 : -1;
                    tempCol += (gate & 1) ? 1 : 0;
                    if (tempRow < 0 || tempCol < 0 || tempRow >= rowLen || tempCol >= rowLen) {
                        continue;
                    }
                    int b1 = row * rowLen + col;
                    int b2 = tempRow * rowLen + tempCol;
                    if (b1 >= n || b2 >= n) {
                        continue;
                    }
                    if ((2 * qReg->Rand()) < ONE_R1_F) {
                        std::swap(b1, b2);
                    }
                    const real1_f gateId = GateCount2Qb * qReg->Rand();
                    if (gateId < ONE_R1_F) {
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 2)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 3)) {
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 4)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 5)) {
                        qReg->ISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 6)) {
                        qReg->IISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 7)) {
                        qReg->CNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 8)) {
                        qReg->CY(b1, b2);
                    } else if (gateId < (ONE_R1_F * 9)) {
                        qReg->CZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 10)) {
                        qReg->AntiCNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 11)) {
                        qReg->AntiCY(b1, b2);
                    } else {
                        qReg->AntiCZ(b1, b2);
                    }
                }
            }
        }
    });
}

TEST_CASE("test_random_circuit_sampling_nn_orbifold", "[speed]")
{
    // "nn" stands for "nearest-neighbor (coupler gates)"
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
        std::vector<int> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };
        const int rowLen = std::ceil(std::sqrt(n));
        const int GateCount2Qb = 12;

        for (bitLenInt d = 0U; d < depth; ++d) {
            for (bitLenInt i = 0U; i < n; ++i) {
                // This effectively covers x-z-x Euler angles, every 3 layers:
                qReg->H(i);
                qReg->RZ(qReg->Rand() * 2 * PI_R1, i);
            }

            int gate = gateSequence.front();
            gateSequence.erase(gateSequence.begin());
            gateSequence.push_back(gate);
            for (int row = 1; row < (rowLen & ~1); row += 2) {
                for (int col = 0; col < (rowLen & ~1); col++) {
                    int tempRow = row;
                    int tempCol = col;
                    tempRow += (gate & 2) ? 1 : -1;
                    tempCol += (gate & 1) ? 1 : 0;

                    // Periodic ("orbifold") boundary conditions
                    if (tempRow < 0) {
                        tempRow += rowLen;
                    }
                    if (tempCol < 0) {
                        tempCol += rowLen;
                    }
                    if (tempRow >= rowLen) {
                        tempRow -= rowLen;
                    }
                    if (tempCol >= rowLen) {
                        tempCol -= rowLen;
                    }

                    int b1 = row * rowLen + col;
                    int b2 = tempRow * rowLen + tempCol;
                    if (b1 >= n || b2 >= n) {
                        continue;
                    }
                    if ((2 * qReg->Rand()) < ONE_R1_F) {
                        std::swap(b1, b2);
                    }
                    const real1_f gateId = GateCount2Qb * qReg->Rand();
                    if (gateId < ONE_R1_F) {
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 2)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 3)) {
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 4)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 5)) {
                        qReg->ISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 6)) {
                        qReg->IISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 7)) {
                        qReg->CNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 8)) {
                        qReg->CY(b1, b2);
                    } else if (gateId < (ONE_R1_F * 9)) {
                        qReg->CZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 10)) {
                        qReg->AntiCNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 11)) {
                        qReg->AntiCY(b1, b2);
                    } else {
                        qReg->AntiCZ(b1, b2);
                    }
                }
            }
        }
    });
}

TEST_CASE("test_cosmology", "[cosmos]")
{
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // We assume that the treatment of that work is valid for a bipartite system that has a pure state, entire
    // between interior and (event horizon) boundary degrees of freedom for the Hilbert space. We start with each
    // qubit region subsystem with only internal entanglement between its two internal degrees of freedom,
    // (effectively such that one is interior and the other is boundary, in a totally random basis). We do not
    // explicitly partition between boundary and interior, in part because entanglement can occur internally. We
    // assume the DFT or its inverse is the maximally entangling operation across the ensemble of initially Planck
    // scale separable subsystems. The finite number of subsystems is due to resource limit for our model, but it
    // might effectively represent an entanglement or "entropy" budget for a closed universe; the time to maximum
    // entanglement for "n" available qubits should be "n" Planck time steps on average. (The von Neumann entropy
    // actually remains 0, in this entire simulation, as the state is pure and evolves in a unitary fashion, but, if
    // unitary evolution holds for the entire real physical cosmological system of our universe, then this
    // entangling action gives rise to the appearance of non-zero von Neumann entropy of a mixed state.)  We limit
    // to the 1 spatial + 1 time dimension case.
    //
    // If the (inverse) DFT is truly maximally entangling, it might not be appropriate to iterate the full-width,
    // monotonic DFT as a time-step, because this then consumes the entire entropy budget of the Hubble sphere in
    // one step. Further, deterministic progression toward higher entanglement, and therefore higher effective
    // entropy, assumes a fixed direction for the "arrow of time." Given the time symmetry of unitary evolution,
    // hopefully, the thermodynamic arrow of time would be emergent in a very-early-universe model, rather than
    // assumed to be fixed. As such, suppose that there is locally a 0.5/0.5 of 1.0 probability for either direction
    // of apparent time in a step, represented by randomly choosing QFT or inverse on a local region. Further,
    // initially indepedent regions cannot be causally influenced by distant regions faster than the speed of light,
    // where the light cone grows at a rate of one Planck distance per Planck time. Locality implies that, in one
    // Planck time step, a 2 qubit (inverse) DFT can be acted between each nearest-neighbor pair. We also assume
    // that causally disconnected regions develop local entanglement in parallel. However, if we took a longer time
    // step, an integer multiple of the Planck time, then higher order QFTs would be needed to simulate the step.
    // Probably, the most accurate simulation would take the "squarest" possible time step by space step, but then
    // this is simply a single QFT or its inverse for the entire entropy budget of the space. (We must acknowledge,
    // it is apparent to us that this simulation we choose is a problem that can be made relatively easy for
    // Qrack::QUnit.)

    // "RandInit" -
    // true - initialize all qubits with completely random (single qubit, separable) states
    // false - initialize entire register as |0>
    //
    // Setting a totally random eigenstate for each bit simulates the limits of causality, since qubits have not had
    // time to interact with each other and reach homogeneity. However, if the initial state of each region is an
    // eigenstate, then maybe we can call each initial state the local |0> state, by convention. (This might not
    // actually be self-consistent; the limitation on causality and homogeneity might preempt the validity of this
    // initialization. It might still be an interesting case to consider, and to debug with.)
    const bool RandInit = true;

    // "UseTDepth"
    // true - for "n" qubits, simulate time to depth "n"
    // false - simulate to "depth" time steps
    const bool UseTDepth = true;
    const int TDepth = 8;
    // Time step of simulation, (in "Planck times")
    const bitLenInt TStep = 1;
    // If true, loop the parallel local evolution back around on the boundaries of the qubit array.
    const bool DoOrbifold = true;

    benchmarkLoop(
        [&](QInterfacePtr qUniverse, bitLenInt n) {
            int t, x;
            int tMax = UseTDepth ? TDepth : n;

            for (t = 1; t < tMax; t += TStep) {
                for (x = 0; x < (int)(n - TStep); x++) {
                    if (qUniverse->Rand() < (ONE_R1 / 2)) {
                        qUniverse->QFT(x, TStep + 1U);
                    } else {
                        qUniverse->IQFT(x, TStep + 1U);
                    }
                }

                if (!DoOrbifold) {
                    continue;
                }

                // Orbifold the last and first bits.
                qUniverse->ROL(TStep, 0, n);
                for (x = 0; x < (int)TStep; x++) {
                    if (qUniverse->Rand() < (ONE_R1 / 2)) {
                        qUniverse->QFT(x, TStep + 1U);
                    } else {
                        qUniverse->IQFT(x, TStep + 1U);
                    }
                }
                qUniverse->ROR(TStep, 0, n);
            }
        },
        false, false, false, RandInit);
}

TEST_CASE("test_qft_cosmology", "[cosmos]")
{
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // Per the notes in the previous test, this is probably our most accurate possible simulation of a cosmos: one
    // QFT (or inverse) to consume the entire "entropy" budget.
    //
    // Note that, when choosing between QFT and inverse QFT, AKA inverse DFT and DFT respectively, the choice of the
    // QFT over the IQFT is not entirely arbitrary: we are mapping from a single phase in the phase space of
    // potential universes to a single configuration. Remember that we initialize as a collection of entirely
    // random, single, separable qubits.

    benchmarkLoop([&](QInterfacePtr qUniverse, bitLenInt n) { qUniverse->QFT(0, n); }, false, false, false, true);
}

TEST_CASE("test_iqft_cosmology", "[cosmos]")
{
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // Per the notes of the previous test, we give the option to consider the inverse as better motivated.

    benchmarkLoop([&](QInterfacePtr qUniverse, bitLenInt n) { qUniverse->IQFT(0, n); }, false, false, false, true);
}

TEST_CASE("test_qft_cosmology_inverse", "[cosmos]")
{
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // Per the notes in the previous tests, this is probably our most accurate possible simulation of a cosmos: one
    // QFT (or inverse) to consume the entire "entropy" budget.
    //
    // For the time reversal, say we "know the ultimate basis of measurement, at the end of the universe." It is trivial
    // to reverse to statistically compatible initial state. (This is simply the "uncomputation" of the forward-in-time
    // simulation.)

    benchmarkLoop(
        [&](QInterfacePtr qUniverse, bitLenInt n) {
            qUniverse->IQFT(0, n);

            for (bitLenInt i = 0; i < qUniverse->GetQubitCount(); i++) {
                real1_f theta = -2 * M_PI * qUniverse->Rand();
                real1_f phi = -2 * M_PI * qUniverse->Rand();
                real1_f lambda = -2 * M_PI * qUniverse->Rand();

                qUniverse->U(i, theta, phi, lambda);
            }
        },
        true, false, false, false);
}

TEST_CASE("test_bq_comparison", "[metriq]")
{
    constexpr int GateCount1Qb = 6;
    constexpr int GateCountMultiQb = 2;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {
        std::set<bitLenInt> allBits;
        for (bitLenInt i = 0; i < n; ++i) {
            allBits.insert(allBits.end(), i);
        }
        for (bitLenInt d = 0; d < n; ++d) {
            bitCapInt zMask = ZERO_BCI;
            bitCapInt xMask = ZERO_BCI;
            for (bitLenInt i = 0; i < n; ++i) {
                const real1_f gateRand = GateCount1Qb * qReg->Rand();
                if (gateRand < 1) {
                    qReg->H(i);
                } else if (gateRand < 2) {
                    bi_or_ip(&zMask, pow2(i));
                } else if (gateRand < 3) {
                    bi_or_ip(&xMask, pow2(i));
                } else if (gateRand < 4) {
                    qReg->Y(i);
                } else if (gateRand < 5) {
                    qReg->S(i);
                } else {
                    qReg->T(i);
                }
            }
            qReg->ZMask(zMask);
            qReg->XMask(xMask);

            std::set<bitLenInt> unusedBits(allBits);
            while (unusedBits.size() > 1) {
                const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
                const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
                const real1_f gateRand = GateCountMultiQb * qReg->Rand();
                if (gateRand < ONE_R1) {
                    qReg->CZ(b1, b2);
                } else {
                    qReg->CNOT(b1, b2);
                }
            }
        }
    });
}

TEST_CASE("test_n_bell", "[stabilizer]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        qftReg->H(0);
        for (bitLenInt i = 0; i < (n - 1); i++) {
            qftReg->CNOT(i, i + 1U);
        }
    });
}

TEST_CASE("test_repeat_h_cnot", "[stabilizer]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        for (bitLenInt i = 0; i < (n - 1); i++) {
            qftReg->H(i);
            qftReg->CNOT(i, i + 1U);
        }
    });
}

struct MultiQubitGate {
    int gate;
    bitLenInt b1;
    bitLenInt b2;
    bitLenInt b3;
};

TEST_CASE("test_universal_circuit_digital_cross_entropy", "[supreme]")
{
    std::cout << ">>> 'test_universal_circuit_digital_cross_entropy':" << std::endl;

    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 4;
    const int Depth = 3;

    const int ITERATIONS = 20000;
    const int n = 8;
    bitCapInt permCount = pow2(n);
    bitCapInt perm;

    std::cout << "Width: " << n << " qubits" << std::endl;
    std::cout << "Depth: " << Depth << " layers of 1 qubit then multi-qubit gates" << std::endl;
    std::cout << "samples collected: " << ITERATIONS << std::endl;

    int d;
    size_t i;
    std::vector<std::vector<int>> gate1QbRands(Depth);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(Depth);
    int maxGates;

    QInterfacePtr goldStandard =
        CreateQuantumInterface({ testSubEngineType, testSubSubEngineType, testSubSubSubEngineType }, n, ZERO_BCI, rng,
            ONE_CMPLX, enable_normalization, true, use_host_dma, device_id, !disable_hardware_rng);
    if (disable_t_injection) {
        goldStandard->SetTInjection(false);
    }
    if (disable_reactive_separation) {
        goldStandard->SetReactiveSeparate(false);
    }

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < n; i++) {
            layer1QbRands.push_back((int)(goldStandard->Rand() * GateCount1Qb));
        }

        std::set<bitLenInt> unusedBits;
        for (i = 0; i < n; i++) {
            // In the past, "goldStandard->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
            // the method can give performance advantages, under opportune conditions, but it does not, here.
            unusedBits.insert(unusedBits.end(), i);
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        while (unusedBits.size() > 1) {
            MultiQubitGate multiGate;
            multiGate.b1 = pickRandomBit(goldStandard->Rand(), &unusedBits);
            multiGate.b2 = pickRandomBit(goldStandard->Rand(), &unusedBits);

            if (unusedBits.size() > 0) {
                maxGates = GateCountMultiQb;
            } else {
                maxGates = GateCountMultiQb - 1U;
            }

            multiGate.gate = (bitLenInt)(maxGates * goldStandard->Rand());

            if (multiGate.gate > 2) {
                multiGate.b3 = pickRandomBit(goldStandard->Rand(), &unusedBits);
            }

            layerMultiQbRands.push_back(multiGate);
        }
    }

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        bitCapInt xMask = ZERO_BCI;
        bitCapInt yMask = ZERO_BCI;
        for (i = 0; i < layer1QbRands.size(); i++) {
            int gate1Qb = layer1QbRands[i];
            if (gate1Qb == 0) {
                goldStandard->H(i);
            } else if (gate1Qb == 1) {
                bi_or_ip(&xMask, pow2(i));
            } else if (gate1Qb == 2) {
                bi_or_ip(&yMask, pow2(i));
            } else {
                goldStandard->T(i);
            }
        }
        goldStandard->XMask(xMask);
        goldStandard->YMask(yMask);

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                goldStandard->Swap(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 1) {
                goldStandard->CZ(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 2) {
                goldStandard->CNOT(multiGate.b1, multiGate.b2);
            } else {
                goldStandard->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
            }
        }
    }

    std::vector<bitCapInt> qPowers(n);
    for (i = 0; i < n; i++) {
        qPowers[i] = pow2(i);
    }

    std::map<bitCapInt, int> goldStandardResult = goldStandard->MultiShotMeasureMask(qPowers, ITERATIONS);

    std::map<bitCapInt, int>::iterator measurementBin;

    real1_f uniformRandomCount = (real1_f)(ITERATIONS / bi_to_double(permCount));
    int goldBinResult;
    real1_f crossEntropy = ZERO_R1_F;
    for (perm = ZERO_BCI; bi_compare(perm, permCount) < 0; bi_increment(&perm, 1U)) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }
        crossEntropy += (uniformRandomCount - goldBinResult) * (uniformRandomCount - goldBinResult);
    }
    if (crossEntropy < ZERO_R1_F) {
        crossEntropy = ZERO_R1_F;
    }
    crossEntropy = ONE_R1_F - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. uniform random cross entropy (out of 1.0): " << crossEntropy << std::endl;

    std::map<bitCapInt, int> goldStandardResult2 = goldStandard->MultiShotMeasureMask(qPowers, ITERATIONS);

    int testBinResult;
    crossEntropy = ZERO_R1_F;
    for (perm = ZERO_BCI; bi_compare(perm, permCount) < 0; bi_increment(&perm, 1U)) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = goldStandardResult2.find(perm);
        if (measurementBin == goldStandardResult2.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1_F) {
        crossEntropy = ZERO_R1_F;
    }
    crossEntropy = ONE_R1_F - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. gold standard cross entropy (out of 1.0): " << crossEntropy << std::endl;

    QInterfacePtr testCase = CreateQuantumInterface({ testEngineType, testSubEngineType }, n, ZERO_BCI, rng, ONE_CMPLX,
        enable_normalization, true, use_host_dma, device_id, !disable_hardware_rng, sparse);
    if (disable_t_injection) {
        testCase->SetTInjection(false);
    }
    if (disable_reactive_separation) {
        testCase->SetReactiveSeparate(false);
    }

    std::map<bitCapInt, int> testCaseResult;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        testCase->SetPermutation(ZERO_BCI);
        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            bitCapInt xMask = ZERO_BCI;
            bitCapInt yMask = ZERO_BCI;
            for (i = 0; i < layer1QbRands.size(); i++) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    bi_or_ip(&xMask, pow2(i));
                } else if (gate1Qb == 2) {
                    bi_or_ip(&yMask, pow2(i));
                } else {
                    testCase->T(i);
                }
            }
            testCase->XMask(xMask);
            testCase->YMask(yMask);

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }
        }

        perm = testCase->MReg(0, n);
        if (testCaseResult.find(perm) == testCaseResult.end()) {
            testCaseResult[perm] = 1;
        } else {
            testCaseResult[perm] += 1;
        }
    }
    // Comment out the ITERATIONS loop and testCaseResult[perm] update above, and uncomment this line below, for a
    // faster benchmark. This will not test the effect of the MReg() method.
    // testCaseResult = testCase->MultiShotMeasureMask(qPowers, n, ITERATIONS);

    crossEntropy = ZERO_R1_F;
    for (perm = ZERO_BCI; bi_compare(perm, permCount) < 0; bi_increment(&perm, 1U)) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = testCaseResult.find(perm);
        if (measurementBin == testCaseResult.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1_F) {
        crossEntropy = ZERO_R1_F;
    }
    crossEntropy = ONE_R1_F - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. test case cross entropy (out of 1.0): " << crossEntropy << std::endl;

    std::map<bitCapInt, int> testCaseResult2;

    testCase->SetPermutation(ZERO_BCI);

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        bitCapInt xMask = ZERO_BCI;
        bitCapInt yMask = ZERO_BCI;
        for (i = 0; i < layer1QbRands.size(); i++) {
            int gate1Qb = layer1QbRands[i];
            if (gate1Qb == 0) {
                testCase->H(i);
            } else if (gate1Qb == 1) {
                bi_or_ip(&xMask, pow2(i));
            } else if (gate1Qb == 2) {
                bi_or_ip(&yMask, pow2(i));
            } else {
                testCase->T(i);
            }
        }
        testCase->XMask(xMask);
        testCase->YMask(yMask);

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                testCase->Swap(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 1) {
                testCase->CZ(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 2) {
                testCase->CNOT(multiGate.b1, multiGate.b2);
            } else {
                testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
            }
        }
    }
    testCaseResult2 = testCase->MultiShotMeasureMask(qPowers, ITERATIONS);

    crossEntropy = ZERO_R1_F;
    for (perm = ZERO_BCI; bi_compare(perm, permCount) < 0; bi_increment(&perm, 1U)) {
        measurementBin = testCaseResult.find(perm);
        if (measurementBin == testCaseResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = testCaseResult2.find(perm);
        if (measurementBin == testCaseResult2.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1_F) {
        crossEntropy = ZERO_R1_F;
    }
    crossEntropy = ONE_R1_F - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Test case vs. (duplicate) test case cross entropy (out of 1.0): " << crossEntropy << std::endl;
}

struct SingleQubitGate {
    real1_f th;
    real1_f ph;
    real1_f lm;
};

TEST_CASE("test_noisy_fidelity", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth: " << n << std::endl;

    int d;
    int i;
    int maxGates;

    int gate;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    std::vector<std::vector<SingleQubitGate>> gate1QbRands(w);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(w);

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < w; i++) {
            SingleQubitGate gate1qb;
            gate1qb.th = 4 * PI_R1 * rng->Rand();
            gate1qb.ph = 4 * PI_R1 * rng->Rand();
            gate1qb.lm = 4 * PI_R1 * rng->Rand();
            layer1QbRands.push_back(gate1qb);
        }

        std::set<bitLenInt> unusedBits;
        for (i = 0; i < w; i++) {
            unusedBits.insert(i);
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        while (unusedBits.size() > 1) {
            MultiQubitGate multiGate;
            multiGate.b1 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b2 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b3 = 0;

            if (unusedBits.size() > 0) {
                maxGates = GateCountMultiQb;
            } else {
                maxGates = GateCount2Qb;
            }

            gate = (int)(rng->Rand() * maxGates);
            if (gate >= maxGates) {
                gate = (maxGates - 1U);
            }

            multiGate.gate = gate;

            if (multiGate.gate >= GateCount2Qb) {
                multiGate.b3 = pickRandomBit(rng->Rand(), &unusedBits);
            }

            layerMultiQbRands.push_back(multiGate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

#if defined(_WIN32) && !defined(__CYGWIN__)
    std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
    _putenv(envVar.c_str());
#else
    unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
#endif

    QInterfacePtr goldStandard = CreateQuantumInterface(engineStack, w, randPerm);

    std::cout << "Dispatching \"gold standard\" (noiseless) simulation...";

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < (int)layer1QbRands.size(); i++) {
            SingleQubitGate gate1Qb = layer1QbRands[i];
            goldStandard->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
            // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
            //           << ");" << std::endl;
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                goldStandard->ISwap(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 1) {
                goldStandard->IISwap(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 2) {
                goldStandard->CNOT(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 3) {
                goldStandard->CY(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 4) {
                goldStandard->CZ(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 5) {
                goldStandard->AntiCNOT(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                //           << std::endl;
            } else if (multiGate.gate == 6) {
                goldStandard->AntiCY(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 7) {
                goldStandard->AntiCZ(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 8) {
                goldStandard->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->CCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 9) {
                goldStandard->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->CCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 10) {
                goldStandard->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->CCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 11) {
                goldStandard->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->AntiCCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 12) {
                goldStandard->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->AntiCCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            } else {
                goldStandard->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                // std::cout << "qReg->AntiCCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                //           << (int)multiGate.b3 << ");" << std::endl;
            }
        }
    }

    std::cout
        << "Done. ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s)" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                SingleQubitGate gate1Qb = layer1QbRands[i];
                testCase->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
                // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
                //           << ");" << std::endl;
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->ISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 1) {
                    testCase->IISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 3) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                    //           << std::endl;
                } else if (multiGate.gate == 6) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 7) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 8) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 9) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 10) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 12) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                }
            }
        }

        testCase->Finish();

        // We mirrored for half, hence the "gold standard" is identically |randPerm>.
        std::cout << "For SDRP=" << sdrp << ": " << std::endl;

        std::cout << "\"Gold standard\" fidelity: " << (ONE_R1 - goldStandard->SumSqrDiff(testCase)) << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;

        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

real1_f diophantine_fidelity_correction(real1_f sigmoid, real1_f sdrp)
{
    // Nonlinear transform for normalized variance of sigmoid levels:
    sigmoid = pow(sigmoid, 1 - sqrt(sdrp));

    // Found in guess-and-check regression (R^2 = ~94.3%)
    sigmoid +=
        -pow(sdrp, 6) + pow(sdrp, 5) / 4 - 3 * pow(sdrp, 4) + 4 * pow(sdrp, 3) - 3 * pow(sdrp, 2) + (5 * sdrp) / 6;

    // Empirical overall bias correction:
    sigmoid -= (real1_f)0.00741342570942599;

    // Reverse variance normalization:
    sigmoid = pow(sigmoid, 1 / (1 - sqrt(sdrp)));

    if (std::isnan((real1_s)sigmoid)) {
        return 0;
    }

    if (sigmoid > 1) {
        return 1;
    }

    return sigmoid;
}

TEST_CASE("test_noisy_fidelity_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_fidelity_estimate':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    int d;
    int i;
    int maxGates;

    int gate;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    std::vector<std::vector<SingleQubitGate>> gate1QbRands(w);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(w);

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < w; i++) {
            SingleQubitGate gate1qb;
            gate1qb.th = 4 * PI_R1 * rng->Rand();
            gate1qb.ph = 4 * PI_R1 * rng->Rand();
            gate1qb.lm = 4 * PI_R1 * rng->Rand();
            layer1QbRands.push_back(gate1qb);
        }

        std::set<bitLenInt> unusedBits;
        for (i = 0; i < w; i++) {
            unusedBits.insert(i);
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        while (unusedBits.size() > 1) {
            MultiQubitGate multiGate;
            multiGate.b1 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b2 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b3 = 0;

            if (unusedBits.size() > 0) {
                maxGates = GateCountMultiQb;
            } else {
                maxGates = GateCount2Qb;
            }

            gate = (int)(rng->Rand() * maxGates);
            if (gate >= maxGates) {
                gate = (maxGates - 1U);
            }

            multiGate.gate = gate;

            if (multiGate.gate >= GateCount2Qb) {
                multiGate.b3 = pickRandomBit(rng->Rand(), &unusedBits);
            }

            layerMultiQbRands.push_back(multiGate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                SingleQubitGate gate1Qb = layer1QbRands[i];
                testCase->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
                // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
                //           << ");" << std::endl;
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->ISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 1) {
                    testCase->IISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 3) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                    //           << std::endl;
                } else if (multiGate.gate == 6) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 7) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 8) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 9) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 10) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 12) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                }
            }
        }

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_validation", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_validation':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "WARNING: These outputs are meant to be piped to a file." << std::endl;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    int d;
    int i;
    int maxGates;

    int gate;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    std::vector<std::vector<SingleQubitGate>> gate1QbRands(w);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(w);

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < w; i++) {
            SingleQubitGate gate1Qb;
            gate1Qb.th = 4 * PI_R1 * rng->Rand();
            gate1Qb.ph = 4 * PI_R1 * rng->Rand();
            gate1Qb.lm = 4 * PI_R1 * rng->Rand();
            layer1QbRands.push_back(gate1Qb);

            std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm << ");"
                      << std::endl;
        }

        std::set<bitLenInt> unusedBits;
        for (i = 0; i < w; i++) {
            unusedBits.insert(i);
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        while (unusedBits.size() > 1) {
            MultiQubitGate multiGate;
            multiGate.b1 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b2 = pickRandomBit(rng->Rand(), &unusedBits);
            multiGate.b3 = 0;

            if (unusedBits.size() > 0) {
                maxGates = GateCountMultiQb;
            } else {
                maxGates = GateCount2Qb;
            }

            gate = (int)(rng->Rand() * maxGates);
            if (gate >= maxGates) {
                gate = (maxGates - 1U);
            }

            multiGate.gate = gate;

            if (multiGate.gate >= GateCount2Qb) {
                multiGate.b3 = pickRandomBit(rng->Rand(), &unusedBits);
            }

            layerMultiQbRands.push_back(multiGate);

            if (multiGate.gate == 0) {
                std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 1) {
                std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 2) {
                std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 3) {
                std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 4) {
                std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 5) {
                std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 6) {
                std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 7) {
                std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 8) {
                std::cout << "qReg->CCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                          << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 9) {
                std::cout << "qReg->CCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", " << (int)multiGate.b3
                          << ");" << std::endl;
            } else if (multiGate.gate == 10) {
                std::cout << "qReg->CCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", " << (int)multiGate.b3
                          << ");" << std::endl;
            } else if (multiGate.gate == 11) {
                std::cout << "qReg->AntiCCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                          << (int)multiGate.b3 << ");" << std::endl;
            } else if (multiGate.gate == 12) {
                std::cout << "qReg->AntiCCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                          << (int)multiGate.b3 << ");" << std::endl;
            } else {
                std::cout << "qReg->AntiCCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                          << (int)multiGate.b3 << ");" << std::endl;
            }
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    std::vector<bitCapInt> qPowers;
    for (bitLenInt i = 0U; i < w; ++i) {
        qPowers.push_back(pow2(i));
    }
    std::unique_ptr<unsigned long long[]> results(new unsigned long long[1000000U]);

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                SingleQubitGate gate1Qb = layer1QbRands[i];
                testCase->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
                // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
                //           << ");" << std::endl;
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->ISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 1) {
                    testCase->IISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 3) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                    //           << std::endl;
                } else if (multiGate.gate == 6) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 7) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 8) {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 9) {
                    testCase->CCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 10) {
                    testCase->CCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->CCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 11) {
                    testCase->AntiCCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else if (multiGate.gate == 12) {
                    testCase->AntiCCY(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                } else {
                    testCase->AntiCCZ(multiGate.b1, multiGate.b2, multiGate.b3);
                    // std::cout << "qReg->AntiCCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ", "
                    //           << (int)multiGate.b3 << ");" << std::endl;
                }
            }
        }

        const int exeTime =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Circuit execution time: " << exeTime << "s" << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        start = std::chrono::high_resolution_clock::now();
        if (exeTime > 360) {
            std::cout << n << " depth layer random circuit measurement samples:" << std::endl;
            testCase->MultiShotMeasureMask(qPowers, 1000000U, results.get());
            for (size_t i = 0U; i < 1000000U; ++i) {
                std::cout << results.get()[i] << std::endl;
            }
            std::cout << "(You should apply XEB against ideal simulation measurements, to find the true fidelity...)"
                      << std::endl;
            std::cout << "Measurement sampling time: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << "s" << std::endl;
        }

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_nn", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_nn':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth: " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                // Try to pack 3-qubit gates as "greedily" as we can:
                int tempGate = 0;
                int b3 = 0;

                const bool canBe3Qubit = (d & 1U);

                if (canBe3Qubit) {
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));
                }

                const bool is3Qubit = canBe3Qubit && (tempGate < 4);
                if (is3Qubit) {
                    usedBits.push_back(b3);
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b3);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b2, b3);
                    }
                }

                if (is3Qubit) {
                    gate = (int)(rng->Rand() * (GateCountMultiQb - GateCount2Qb)) + GateCount2Qb;
                    if (gate >= GateCountMultiQb) {
                        gate = GateCountMultiQb - 1U;
                    }
                } else {
                    gate = (int)(rng->Rand() * GateCount2Qb);
                    if (gate >= GateCount2Qb) {
                        gate = GateCount2Qb - 1U;
                    }
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                const std::set<bitLenInt> controls{ (bitLenInt)b1, (bitLenInt)b2 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else if (gate == 7) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                } else if (gate == 8) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, 3U));
                } else if (gate == 9) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, 3U));
                } else if (gate == 10) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, 3U));
                } else if (gate == 11) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, ZERO_BCI));
                } else if (gate == 12) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

#if defined(_WIN32) && !defined(__CYGWIN__)
    std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
    _putenv(envVar.c_str());
#else
    unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
#endif

    QInterfacePtr goldStandard = CreateQuantumInterface(engineStack, w, randPerm);

    std::cout << "Dispatching \"gold standard\" (noiseless) simulation...";
    circuit->Run(goldStandard);
    goldStandard->Finish();

    std::cout
        << "Done. ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s)" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;

        std::cout << "\"Gold standard\" fidelity: " << (ONE_R1 - goldStandard->SumSqrDiff(testCase)) << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;

        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_nn_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_fidelity_nn_estimate':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                // Try to pack 3-qubit gates as "greedily" as we can:
                int tempGate = 0;
                int b3 = 0;

                const bool canBe3Qubit = (d & 1U);

                if (canBe3Qubit) {
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));
                }

                const bool is3Qubit = canBe3Qubit && (tempGate < 4);
                if (is3Qubit) {
                    usedBits.push_back(b3);
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b3);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b2, b3);
                    }
                }

                if (is3Qubit) {
                    gate = (int)(rng->Rand() * (GateCountMultiQb - GateCount2Qb)) + GateCount2Qb;
                    if (gate >= GateCountMultiQb) {
                        gate = GateCountMultiQb - 1U;
                    }
                } else {
                    gate = (int)(rng->Rand() * GateCount2Qb);
                    if (gate >= GateCount2Qb) {
                        gate = GateCount2Qb - 1U;
                    }
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                const std::set<bitLenInt> controls{ (bitLenInt)b1, (bitLenInt)b2 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else if (gate == 7) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                } else if (gate == 8) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, 3U));
                } else if (gate == 9) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, 3U));
                } else if (gate == 10) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, 3U));
                } else if (gate == 11) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, ZERO_BCI));
                } else if (gate == 12) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_nn_mirror", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_nn_mirror':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                // Try to pack 3-qubit gates as "greedily" as we can:
                int tempGate = 0;
                int b3 = 0;

                const bool canBe3Qubit = (d & 1U);

                if (canBe3Qubit) {
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));
                }

                const bool is3Qubit = canBe3Qubit && (tempGate < 4);
                if (is3Qubit) {
                    usedBits.push_back(b3);
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b3);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b2, b3);
                    }
                }

                if (is3Qubit) {
                    gate = (int)(rng->Rand() * (GateCountMultiQb - GateCount2Qb)) + GateCount2Qb;
                    if (gate >= GateCountMultiQb) {
                        gate = GateCountMultiQb - 1U;
                    }
                } else {
                    gate = (int)(rng->Rand() * GateCount2Qb);
                    if (gate >= GateCount2Qb) {
                        gate = GateCount2Qb - 1U;
                    }
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                const std::set<bitLenInt> controls{ (bitLenInt)b1, (bitLenInt)b2 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else if (gate == 7) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                } else if (gate == 8) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, 3U));
                } else if (gate == 9) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, 3U));
                } else if (gate == 10) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, 3U));
                } else if (gate == 11) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, ZERO_BCI));
                } else if (gate == 12) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();

    QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
    circuit->Run(testCase);
    circuit->Inverse()->Run(testCase);
    testCase->Finish();

    std::cout << "Mirror circuit fidelity: " << testCase->ProbAll(randPerm) << std::endl;
    std::cout
        << "Execution time: "
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s" << std::endl;
}

TEST_CASE("test_noisy_fidelity_nn_validation", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_nn_validation':" << std::endl;

    const int GateCountMultiQb = 14;
    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "WARNING: These outputs are meant to be piped to a file." << std::endl;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                // Try to pack 3-qubit gates as "greedily" as we can:
                int tempGate = 0;
                int b3 = 0;

                const bool canBe3Qubit = (d & 1U);

                if (canBe3Qubit) {
                    do {
                        tempRow = row;
                        tempCol = col;

                        tempRow += ((tempGate & 2) ? 1 : -1);
                        tempCol += (colLen == 1) ? 0 : ((tempGate & 1) ? 1 : 0);

                        b3 = tempRow * colLen + tempCol;

                        ++tempGate;
                    } while ((tempGate < 4) &&
                        ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                            (std::find(usedBits.begin(), usedBits.end(), b3) != usedBits.end())));
                }

                const bool is3Qubit = canBe3Qubit && (tempGate < 4);
                if (is3Qubit) {
                    usedBits.push_back(b3);
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b2);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b1, b3);
                    }
                    if ((rng->Rand() * 2) >= ONE_R1) {
                        std::swap(b2, b3);
                    }
                }

                if (is3Qubit) {
                    gate = (int)(rng->Rand() * (GateCountMultiQb - GateCount2Qb)) + GateCount2Qb;
                    if (gate >= GateCountMultiQb) {
                        gate = GateCountMultiQb - 1U;
                    }
                } else {
                    gate = (int)(rng->Rand() * GateCount2Qb);
                    if (gate >= GateCount2Qb) {
                        gate = GateCount2Qb - 1U;
                    }
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                const std::set<bitLenInt> controls{ (bitLenInt)b1, (bitLenInt)b2 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else if (gate == 7) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                } else if (gate == 8) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, 3U));
                } else if (gate == 9) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, 3U));
                } else if (gate == 10) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, 3U));
                } else if (gate == 11) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, x, controls, ZERO_BCI));
                } else if (gate == 12) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, y, controls, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b3, z, controls, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    std::vector<bitCapInt> qPowers;
    for (bitLenInt i = 0U; i < w; ++i) {
        qPowers.push_back(pow2(i));
    }
    std::unique_ptr<unsigned long long[]> results(new unsigned long long[1000000U]);

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        const int exeTime =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Circuit execution time: " << exeTime << "s" << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        start = std::chrono::high_resolution_clock::now();
        if (exeTime > 360) {
            std::cout << n << " depth layer random circuit measurement samples:" << std::endl;
            testCase->MultiShotMeasureMask(qPowers, 1000000U, results.get());
            for (size_t i = 0U; i < 1000000U; ++i) {
                std::cout << results.get()[i] << std::endl;
            }
            std::cout << "(You should apply XEB against ideal simulation measurements, to find the true fidelity...)"
                      << std::endl;
            std::cout << "Measurement sampling time: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << "s" << std::endl;
        }

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_2qb_nn", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_2qb_nn':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth: " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

#if defined(_WIN32) && !defined(__CYGWIN__)
    std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
    _putenv(envVar.c_str());
#else
    unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
#endif

    QInterfacePtr goldStandard = CreateQuantumInterface(engineStack, w, randPerm);

    std::cout << "Dispatching \"gold standard\" (noiseless) simulation...";
    circuit->Run(goldStandard);
    goldStandard->Finish();

    std::cout
        << "Done. ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s)" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;

        std::cout << "\"Gold standard\" fidelity: " << (ONE_R1 - goldStandard->SumSqrDiff(testCase)) << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;

        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_2qb_nn_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_fidelity_2qb_nn_estimate':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_2qb_nn_validation", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_2qb_nn_validation':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "WARNING: These outputs are meant to be piped to a file." << std::endl;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            const real1_f theta = 4 * PI_R1 * rng->Rand();
            const real1_f phi = 4 * PI_R1 * rng->Rand();
            const real1_f lambda = 4 * PI_R1 * rng->Rand();
            const real1 cos0 = (real1)cos(theta / 2);
            const real1 sin0 = (real1)sin(theta / 2);
            const complex uGate[4]{ complex(cos0, ZERO_R1),
                sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
                sin0 * complex((real1)cos(phi), (real1)sin(phi)),
                cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
            circuit->AppendGate(std::make_shared<QCircuitGate>(i, uGate));
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    std::vector<bitCapInt> qPowers;
    for (bitLenInt i = 0U; i < w; ++i) {
        qPowers.push_back(pow2(i));
    }
    std::unique_ptr<unsigned long long[]> results(new unsigned long long[1000000U]);

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
        circuit->Run(testCase);
        testCase->Finish();

        const int exeTime =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Circuit execution time: " << exeTime << "s" << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        start = std::chrono::high_resolution_clock::now();
        if (exeTime > 360) {
            std::cout << n << " depth layer random circuit measurement samples:" << std::endl;
            testCase->MultiShotMeasureMask(qPowers, 1000000U, results.get());
            for (size_t i = 0U; i < 1000000U; ++i) {
                std::cout << results.get()[i] << std::endl;
            }
            std::cout << "(You should apply XEB against ideal simulation measurements, to find the true fidelity...)"
                      << std::endl;
            std::cout << "Measurement sampling time: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << "s" << std::endl;
        }

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_fidelity_2qb_nn_comparison", "[supreme]")
{
    std::cout << ">>> 'test_noisy_fidelity_2qb_nn_comparison':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "WARNING: These outputs are meant to be piped to a file." << std::endl;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    std::vector<std::vector<SingleQubitGate>> gate1QbRands(w);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(w);

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < w; i++) {
            SingleQubitGate gate1Qb;
            gate1Qb.th = 4 * PI_R1 * rng->Rand();
            gate1Qb.ph = 4 * PI_R1 * rng->Rand();
            gate1Qb.lm = 4 * PI_R1 * rng->Rand();
            layer1QbRands.push_back(gate1Qb);

            std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm << ");"
                      << std::endl;
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                MultiQubitGate multiGate;
                multiGate.b1 = b1;
                multiGate.b2 = b2;
                multiGate.b3 = 0;

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                multiGate.gate = gate;

                layerMultiQbRands.push_back(multiGate);

                if (multiGate.gate == 0) {
                    std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 1) {
                    std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 2) {
                    std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 3) {
                    std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                              << std::endl;
                } else if (multiGate.gate == 6) {
                    std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 7) {
                    std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                }
            }
        }

        gateSequence.pop_front();
        gateSequence.push_back(gate);
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

#if defined(_WIN32) && !defined(__CYGWIN__)
    std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
    _putenv(envVar.c_str());
#else
    unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
#endif

    QInterfacePtr goldStandard = CreateQuantumInterface(engineStack, w, randPerm);

    std::cout << "Dispatching \"gold standard\" (noiseless) simulation...";

    auto start = std::chrono::high_resolution_clock::now();

    for (d = 0; d < n; d++) {
        std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < (int)layer1QbRands.size(); i++) {
            SingleQubitGate gate1Qb = layer1QbRands[i];
            goldStandard->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
            // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
            //           << ");" << std::endl;
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                goldStandard->ISwap(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 1) {
                goldStandard->IISwap(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 2) {
                goldStandard->CNOT(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 3) {
                goldStandard->CY(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 4) {
                goldStandard->CZ(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
            } else if (multiGate.gate == 5) {
                goldStandard->AntiCNOT(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                //           << std::endl;
            } else if (multiGate.gate == 6) {
                goldStandard->AntiCY(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            } else if (multiGate.gate == 7) {
                goldStandard->AntiCZ(multiGate.b1, multiGate.b2);
                // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                // std::endl;
            }
        }
    }

    std::cout
        << "Done. ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s)" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                SingleQubitGate gate1Qb = layer1QbRands[i];
                testCase->U(i, gate1Qb.th, gate1Qb.ph, gate1Qb.lm);
                // std::cout << "qReg->U(" << (int)i << ", " << gate1Qb.th << ", " << gate1Qb.ph << ", " << gate1Qb.lm
                //           << ");" << std::endl;
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->ISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 1) {
                    testCase->IISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 3) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                    //           << std::endl;
                } else if (multiGate.gate == 6) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 7) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                }
            }
        }

        testCase->Finish();

        const int exeTime =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Circuit execution time: " << exeTime << "s" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        // We mirrored for half, hence the "gold standard" is identically |randPerm>.
        std::cout << "Fidelity against \"gold standard\" for SDRP=" << sdrp << ": "
                  << (ONE_R1 - goldStandard->SumSqrDiff(testCase)) << ", Time:"
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        // Mirror the circuit
        start = std::chrono::high_resolution_clock::now();
        for (d = n - 1U; d >= 0; d--) {
            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = (layerMultiQbRands.size() - 1U); i >= 0; i--) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->IISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->IISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 1) {
                    testCase->ISwap(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->ISwap(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 3) {
                    testCase->CY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 4) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->CZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" << std::endl;
                } else if (multiGate.gate == 5) {
                    testCase->AntiCNOT(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCNOT(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");"
                    //           << std::endl;
                } else if (multiGate.gate == 6) {
                    testCase->AntiCY(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCY(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                } else if (multiGate.gate == 7) {
                    testCase->AntiCZ(multiGate.b1, multiGate.b2);
                    // std::cout << "qReg->AntiCZ(" << (int)multiGate.b1 << ", " << (int)multiGate.b2 << ");" <<
                    // std::endl;
                }
            }

            std::vector<SingleQubitGate>& layer1QbRands = gate1QbRands[d];
            for (i = (layer1QbRands.size() - 1U); i >= 0; i--) {
                SingleQubitGate gate1Qb = layer1QbRands[i];
                // Order reversal is intentional.
                testCase->U(i, -gate1Qb.th, -gate1Qb.lm, -gate1Qb.ph);
                // std::cout << "qReg->U(" << (int)i << ", " << -gate1Qb.th << ", " << -gate1Qb.lm << ", " <<
                // -gate1Qb.ph
                //           << ");" << std::endl;
            }
        }

        testCase->Finish();

        // We mirrored for half, hence the "gold standard" is identically |randPerm>.
        const real1_f rawFidelity = testCase->ProbAll(randPerm);
        const real1_f signalFraction = ONE_R1_F / (ONE_R1_F + exp(-tan(PI_R1 * (ONE_R1_F / 2 - sdrp))));
        const real1_f fidelity = diophantine_fidelity_correction(signalFraction * rawFidelity, sdrp);

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;

        std::cout << "\"Gold standard\" fidelity (estimated): " << fidelity << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;

        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_rcs_nn", "[speed]")
{
    // "nn" stands for "nearest-neighbor (coupler gates)"
    const int n = max_qubits;
    const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    std::cout << "Circuit layer depth: " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::vector<int> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };
    const int rowLen = std::ceil(std::sqrt(n));
    const int GateCount2Qb = 12;

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0 - 0.025;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);
    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(n));
    if (randPerm >= pow2Ocl(n)) {
        randPerm = pow2Ocl(n) - 1U;
    }

    while (sdrp >= 0) {

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        start = std::chrono::high_resolution_clock::now();
        QInterfacePtr qReg = CreateQuantumInterface(engineStack, n, randPerm);

        for (bitLenInt d = 0U; d < depth; ++d) {
            for (bitLenInt i = 0U; i < n; ++i) {
                // This effectively covers x-z-x Euler angles, every 3 layers:
                qReg->H(i);
                qReg->RZ(qReg->Rand() * 2 * PI_R1, i);
            }

            int gate = gateSequence.front();
            gateSequence.erase(gateSequence.begin());
            gateSequence.push_back(gate);
            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < rowLen; col++) {
                    int tempRow = row;
                    int tempCol = col;
                    tempRow += (gate & 2) ? 1 : -1;
                    tempCol += (gate & 1) ? 1 : 0;
                    if (tempRow < 0 || tempCol < 0 || tempRow >= rowLen || tempCol >= rowLen) {
                        continue;
                    }
                    int b1 = row * rowLen + col;
                    int b2 = tempRow * rowLen + tempCol;
                    if (b1 >= n || b2 >= n) {
                        continue;
                    }
                    if ((2 * qReg->Rand()) < ONE_R1_F) {
                        std::swap(b1, b2);
                    }
                    const real1_f gateId = GateCount2Qb * qReg->Rand();
                    if (gateId < ONE_R1_F) {
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 2)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 3)) {
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 4)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 5)) {
                        qReg->ISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 6)) {
                        qReg->IISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 7)) {
                        qReg->CNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 8)) {
                        qReg->CY(b1, b2);
                    } else if (gateId < (ONE_R1_F * 9)) {
                        qReg->CZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 10)) {
                        qReg->AntiCNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 11)) {
                        qReg->AntiCY(b1, b2);
                    } else {
                        qReg->AntiCZ(b1, b2);
                    }
                }
            }
        }

        qReg->MAll();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << qReg->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        if (sdrp == 0) {
            return;
        }

        sdrp -= 0.025;
        if (abs(sdrp) <= FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_rcs_u3_nn", "[speed]")
{
    // "nn" stands for "nearest-neighbor (coupler gates)"
    const int n = max_qubits;
    const bitLenInt depth = (benchmarkDepth <= 0) ? n : benchmarkDepth;
    if (benchmarkDepth <= 0) {
        std::cout << "(random circuit depth: square)" << std::endl;
    } else {
        std::cout << "(random circuit depth: " << benchmarkDepth << ")" << std::endl;
    }
    std::cout << "Circuit layer depth: " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::vector<int> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };
    const int rowLen = std::ceil(std::sqrt(n));
    const int GateCount2Qb = 12;

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0 - 0.025;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);
    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(n));
    if (randPerm >= pow2Ocl(n)) {
        randPerm = pow2Ocl(n) - 1U;
    }

    while (sdrp >= 0) {

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        start = std::chrono::high_resolution_clock::now();
        QInterfacePtr qReg = CreateQuantumInterface(engineStack, n, randPerm);

        for (bitLenInt d = 0U; d < depth; ++d) {
            for (bitLenInt i = 0U; i < n; ++i) {
                qReg->U(i, 4 * M_PI * qReg->Rand(), 4 * M_PI * qReg->Rand(), 4 * M_PI * qReg->Rand());
            }

            int gate = gateSequence.front();
            gateSequence.erase(gateSequence.begin());
            gateSequence.push_back(gate);
            for (int row = 1; row < rowLen; row += 2) {
                for (int col = 0; col < rowLen; col++) {
                    int tempRow = row;
                    int tempCol = col;
                    tempRow += (gate & 2) ? 1 : -1;
                    tempCol += (gate & 1) ? 1 : 0;
                    if (tempRow < 0 || tempCol < 0 || tempRow >= rowLen || tempCol >= rowLen) {
                        continue;
                    }
                    int b1 = row * rowLen + col;
                    int b2 = tempRow * rowLen + tempCol;
                    if (b1 >= n || b2 >= n) {
                        continue;
                    }
                    if ((2 * qReg->Rand()) < ONE_R1_F) {
                        std::swap(b1, b2);
                    }
                    const real1_f gateId = GateCount2Qb * qReg->Rand();
                    if (gateId < ONE_R1_F) {
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 2)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 3)) {
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 4)) {
                        qReg->AntiCZ(b1, b2);
                        qReg->Swap(b1, b2);
                        qReg->AntiCZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 5)) {
                        qReg->ISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 6)) {
                        qReg->IISwap(b1, b2);
                    } else if (gateId < (ONE_R1_F * 7)) {
                        qReg->CNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 8)) {
                        qReg->CY(b1, b2);
                    } else if (gateId < (ONE_R1_F * 9)) {
                        qReg->CZ(b1, b2);
                    } else if (gateId < (ONE_R1_F * 10)) {
                        qReg->AntiCNOT(b1, b2);
                    } else if (gateId < (ONE_R1_F * 11)) {
                        qReg->AntiCY(b1, b2);
                    } else {
                        qReg->AntiCZ(b1, b2);
                    }
                }
            }
        }

        qReg->MAll();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << qReg->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        if (sdrp == 0) {
            return;
        }

        sdrp -= 0.025;
        if (abs(sdrp) <= FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_sycamore", "[supreme]")
{
    std::cout << ">>> 'test_noisy_sycamore':" << std::endl;
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit." << std::endl;

    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth: " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    const bitLenInt deadQubit = 3U;
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    // std::cout<<"n="<<(int)n<<std::endl;
    // std::cout<<"rowLen="<<(int)rowLen<<std::endl;
    // std::cout<<"colLen="<<(int)colLen<<std::endl;

    int d;
    int i;

    int gate;

    int row, col;

    std::vector<std::vector<int>> gate1QbRands(n);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(n);
    std::vector<int> lastSingleBitGates;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        std::vector<int> layer1QbRands;
        std::vector<MultiQubitGate> layerMultiQbRands;
        for (i = 0; i < w; ++i) {
            // Each individual bit has one of these 3 gates applied at random.
            // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
            // are handled as general single bit gates.

            // The same gate is not applied twice consecutively in sequence.

            if (d == 0) {
                // For the first iteration, we can pick any gate.

                int gate = (int)(3 * rng->Rand());
                if (gate > 2) {
                    gate = 2;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates.push_back(gate);
            } else {
                // For all subsequent iterations after the first, we eliminate the choice of the same gate
                // applied on the immediately previous iteration.

                int gate = (int)(2 * rng->Rand());
                if (gate > 1) {
                    gate = 1;
                }
                if (gate >= lastSingleBitGates[i]) {
                    ++gate;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates[i] = gate;
            }
        }

        gate1QbRands[d] = layer1QbRands;

        gate = gateSequence.front();
        gateSequence.pop_front();
        gateSequence.push_back(gate);

        for (row = 1; row < rowLen; row += 2) {
            for (col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                    continue;
                }

                int b1 = row * colLen + col;
                int b2 = tempRow * colLen + tempCol;

                MultiQubitGate multiGate;
                multiGate.b1 = b1;
                multiGate.b2 = b2;

                layerMultiQbRands.push_back(multiGate);
            }
        }

        gateMultiQbRands[d] = layerMultiQbRands;
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

#if defined(_WIN32) && !defined(__CYGWIN__)
    std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
    _putenv(envVar.c_str());
#else
    unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
#endif

    QInterfacePtr goldStandard = CreateQuantumInterface(engineStack, w, randPerm);

    std::cout << "Dispatching \"gold standard\" (noiseless) simulation...";

    for (d = 0; d < n; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < (int)layer1QbRands.size(); i++) {
            if ((w == 54U) && (i == deadQubit)) {
                continue;
            }

            int gate1Qb = layer1QbRands[i];
            if (!gate1Qb) {
                goldStandard->SqrtX(i);
                // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
            } else if (gate1Qb == 1U) {
                goldStandard->SqrtY(i);
                // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
            } else {
                goldStandard->SqrtW(i);
                // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
            }
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            const bitLenInt b1 = multiGate.b1;
            const bitLenInt b2 = multiGate.b2;

            if ((w == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                continue;
            }

            // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
            // std::endl;

            if (d == (n - 1)) {
                // For the last layer of couplers, the immediately next operation is measurement, and the phase effects
                // make no observable difference.
                goldStandard->Swap(b1, b2);

                continue;
            }

            goldStandard->TrySeparate(b1, b2);
            goldStandard->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
            goldStandard->TrySeparate(b1, b2);
        }
    }

    std::cout
        << "Done. ("
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s)" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                if ((w == 54U) && (i == deadQubit)) {
                    continue;
                }

                int gate1Qb = layer1QbRands[i];
                if (!gate1Qb) {
                    testCase->SqrtX(i);
                    // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                } else if (gate1Qb == 1U) {
                    testCase->SqrtY(i);
                    // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                } else {
                    testCase->SqrtW(i);
                    // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                const bitLenInt b1 = multiGate.b1;
                const bitLenInt b2 = multiGate.b2;

                if ((w == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                    continue;
                }

                // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
                // std::endl;

                if (d == (n - 1)) {
                    // For the last layer of couplers, the immediately next operation is measurement, and the phase
                    // effects make no observable difference.
                    testCase->Swap(b1, b2);

                    continue;
                }

                testCase->TrySeparate(b1, b2);
                testCase->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                testCase->TrySeparate(b1, b2);
            }
        }

        testCase->Finish();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;

        std::cout << "\"Gold standard\" fidelity: " << (ONE_R1 - goldStandard->SumSqrDiff(testCase)) << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;

        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_sycamore_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_sycamore_estimate':" << std::endl;
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit." << std::endl;

    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    const bitLenInt deadQubit = 3U;
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    // std::cout<<"n="<<(int)n<<std::endl;
    // std::cout<<"rowLen="<<(int)rowLen<<std::endl;
    // std::cout<<"colLen="<<(int)colLen<<std::endl;

    int d;
    int i;

    int gate;

    int row, col;

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    std::vector<std::vector<int>> gate1QbRands(n);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(n);
    std::vector<int> lastSingleBitGates;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        std::vector<int> layer1QbRands;
        std::vector<MultiQubitGate> layerMultiQbRands;
        for (i = 0; i < w; ++i) {
            // Each individual bit has one of these 3 gates applied at random.
            // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
            // are handled as general single bit gates.

            // The same gate is not applied twice consecutively in sequence.

            if (d == 0) {
                // For the first iteration, we can pick any gate.

                int gate = (int)(3 * rng->Rand());
                if (gate > 2) {
                    gate = 2;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates.push_back(gate);
            } else {
                // For all subsequent iterations after the first, we eliminate the choice of the same gate
                // applied on the immediately previous iteration.

                int gate = (int)(2 * rng->Rand());
                if (gate > 1) {
                    gate = 1;
                }
                if (gate >= lastSingleBitGates[i]) {
                    ++gate;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates[i] = gate;
            }
        }

        gate1QbRands[d] = layer1QbRands;

        gate = gateSequence.front();
        gateSequence.pop_front();
        gateSequence.push_back(gate);

        for (row = 1; row < rowLen; row += 2) {
            for (col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                    continue;
                }

                int b1 = row * colLen + col;
                int b2 = tempRow * colLen + tempCol;

                MultiQubitGate multiGate;
                multiGate.b1 = b1;
                multiGate.b2 = b2;

                layerMultiQbRands.push_back(multiGate);
            }
        }

        gateMultiQbRands[d] = layerMultiQbRands;
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                if ((w == 54U) && (i == deadQubit)) {
                    continue;
                }

                int gate1Qb = layer1QbRands[i];
                if (!gate1Qb) {
                    testCase->SqrtX(i);
                    // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                } else if (gate1Qb == 1U) {
                    testCase->SqrtY(i);
                    // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                } else {
                    testCase->SqrtW(i);
                    // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                const bitLenInt b1 = multiGate.b1;
                const bitLenInt b2 = multiGate.b2;

                if ((w == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                    continue;
                }

                // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" <<
                // std::endl;

                if (d == (n - 1)) {
                    // For the last layer of couplers, the immediately next operation is measurement, and the phase
                    // effects make no observable difference.
                    testCase->Swap(b1, b2);

                    continue;
                }

                testCase->TrySeparate(b1, b2);
                testCase->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                testCase->TrySeparate(b1, b2);
            }
        }

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_sycamore_validation", "[supreme]")
{
    std::cout << ">>> 'test_noisy_sycamore_validation':" << std::endl;
    std::cout << "WARNING: These outputs are meant to be piped to a file." << std::endl;
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit." << std::endl;

    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    const bitLenInt deadQubit = 3U;
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    int gate;

    int row, col;

    double sdrp = 1.0;

    std::vector<bitCapInt> qPowers;
    for (bitLenInt i = 0U; i < w; ++i) {
        qPowers.push_back(pow2(i));
    }
    std::unique_ptr<unsigned long long[]> results(new unsigned long long[1000000U]);

    std::vector<std::vector<int>> gate1QbRands(n);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(n);
    std::vector<int> lastSingleBitGates;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        std::vector<int> layer1QbRands;
        std::vector<MultiQubitGate> layerMultiQbRands;
        for (i = 0; i < w; ++i) {
            // Each individual bit has one of these 3 gates applied at random.
            // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
            // are handled as general single bit gates.

            // The same gate is not applied twice consecutively in sequence.

            int gate;
            if (d == 0) {
                // For the first iteration, we can pick any gate.

                gate = (int)(3 * rng->Rand());
                if (gate > 2) {
                    gate = 2;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates.push_back(gate);
            } else {
                // For all subsequent iterations after the first, we eliminate the choice of the same gate
                // applied on the immediately previous iteration.

                gate = (int)(2 * rng->Rand());
                if (gate > 1) {
                    gate = 1;
                }
                if (gate >= lastSingleBitGates[i]) {
                    ++gate;
                }
                layer1QbRands.push_back(gate);
                lastSingleBitGates[i] = gate;
            }

            if (!gate) {
                std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
            } else if (gate == 1U) {
                std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
            } else {
                std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
            }
        }

        gate1QbRands[d] = layer1QbRands;

        gate = gateSequence.front();
        gateSequence.pop_front();
        gateSequence.push_back(gate);

        for (row = 1; row < rowLen; row += 2) {
            for (col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                    continue;
                }

                int b1 = row * colLen + col;
                int b2 = tempRow * colLen + tempCol;

                MultiQubitGate multiGate;
                multiGate.b1 = b1;
                multiGate.b2 = b2;

                layerMultiQbRands.push_back(multiGate);

                std::cout << "qReg->FSim(PI_R1 / 2, -PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");" << std::endl;
            }
        }

        gateMultiQbRands[d] = layerMultiQbRands;
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    while (sdrp >= 0) {
        auto start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);

        for (d = 0; d < n; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < (int)layer1QbRands.size(); i++) {
                if ((w == 54U) && (i == deadQubit)) {
                    continue;
                }

                int gate1Qb = layer1QbRands[i];
                if (!gate1Qb) {
                    testCase->SqrtX(i);
                    // std::cout << "qReg->SqrtX(" << (int)i << ");" << std::endl;
                } else if (gate1Qb == 1U) {
                    testCase->SqrtY(i);
                    // std::cout << "qReg->SqrtY(" << (int)i << ");" << std::endl;
                } else {
                    testCase->SqrtW(i);
                    // std::cout << "qReg->SqrtW(" << (int)i << ");" << std::endl;
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < (int)layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                const bitLenInt b1 = multiGate.b1;
                const bitLenInt b2 = multiGate.b2;

                if ((w == 54U) && ((b1 == deadQubit) || (b2 == deadQubit))) {
                    continue;
                }

                // std::cout << "qReg->FSim((3 * PI_R1) / 2, PI_R1 / 6, " << (int)b1 << ", " << (int)b2 << ");"
                //           << std::endl;

                if (d == (n - 1)) {
                    // For the last layer of couplers, the immediately next operation is measurement, and the phase
                    // effects make no observable difference.
                    testCase->Swap(b1, b2);

                    continue;
                }

                testCase->TrySeparate(b1, b2);
                testCase->FSim((3 * PI_R1) / 2, PI_R1 / 6, b1, b2);
                testCase->TrySeparate(b1, b2);
            }
        }

        const int exeTime =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Circuit execution time: " << exeTime << "s" << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        start = std::chrono::high_resolution_clock::now();
        if (exeTime > 360) {
            std::cout << n << " depth layer random circuit measurement samples:" << std::endl;
            testCase->MultiShotMeasureMask(qPowers, 1000000U, results.get());
            for (size_t i = 0U; i < 1000000U; ++i) {
                std::cout << results.get()[i] << std::endl;
            }
            std::cout << "(You should apply XEB against ideal simulation measurements, to find the true fidelity...)"
                      << std::endl;
            std::cout << "Measurement sampling time: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << "s" << std::endl;
        }

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_stabilizer_rz_mirror", "[supreme]")
{
    std::cout << ">>> 'test_stabilizer_rz_mirror':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex h[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    const complex x[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4U]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    std::set<bitLenInt> qubitSet;
    for (bitLenInt i = 0; i < w; ++i) {
        qubitSet.insert(i);
    }

    QCircuitPtr circuit = std::make_shared<QCircuit>(false);

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (int d = 0; d < n; d++) {
        const bitLenInt layerMagicQubit = std::max((real1_s)(w - 1), (real1_s)(w * rng->Rand()));
        const bitLenInt layerMagicAxis = std::max((real1_s)2, (real1_s)(3 * rng->Rand()));
        for (int i = 0; i < w; i++) {
            // Random general 3-parameter unitary gate via "x-z-x" Euler angles:
            for (int p = 0; p < 3; ++p) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));

                // Clifford rotation
                if ((2 * rng->Rand()) <= ONE_R1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, z));
                }
                if ((2 * rng->Rand()) <= ONE_R1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                }

                if ((i == layerMagicQubit) && (p == layerMagicAxis)) {
                    // Non-Clifford rotation
                    const real1 gateRand = (real1)(PI_R1 * rng->Rand() / 2);
                    const complex mtrx[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, std::polar(ONE_R1, gateRand) };
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, mtrx));
                }
            }
        }

        std::set<bitLenInt> unusedBits = qubitSet;
        while (unusedBits.size() > 1) {
            const bitLenInt b1 = pickRandomBit(rng->Rand(), &unusedBits);
            const bitLenInt b2 = pickRandomBit(rng->Rand(), &unusedBits);
            int gate = (int)(rng->Rand() * GateCount2Qb);
            if (gate >= GateCount2Qb) {
                gate = GateCount2Qb - 1U;
            }

            const std::set<bitLenInt> control{ (bitLenInt)b1 };
            if (gate == 0) {
                circuit->Swap(b1, b2);
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
            } else if (gate == 1) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                circuit->Swap(b1, b2);
            } else if (gate == 2) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
            } else if (gate == 3) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
            } else if (gate == 4) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
            } else if (gate == 5) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
            } else if (gate == 6) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
            } else {
                circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
            }
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();

    QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
    circuit->Run(testCase);
    circuit->Inverse()->Run(testCase);
    testCase->Finish();

    std::cout << "Mirror circuit fidelity: " << testCase->ProbAll(randPerm) << std::endl;
    std::cout
        << "Execution time: "
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s" << std::endl;
}

TEST_CASE("test_stabilizer_rz_nn_mirror", "[supreme]")
{
    std::cout << ">>> 'test_stabilizer_rz_nn_mirror':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex h[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    const complex x[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4U]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    QCircuitPtr circuit = std::make_shared<QCircuit>(false);

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        const bitLenInt layerMagicQubit = std::max((real1_s)(w - 1), (real1_s)(w * rng->Rand()));
        const bitLenInt layerMagicAxis = std::max((real1_s)2, (real1_s)(3 * rng->Rand()));
        for (i = 0; i < w; i++) {
            // Random general 3-parameter unitary gate via "x-z-x" Euler angles:
            for (int p = 0; p < 3; ++p) {
                circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));

                // Clifford rotation
                if ((2 * rng->Rand()) <= ONE_R1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, z));
                }
                if ((2 * rng->Rand()) <= ONE_R1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, s));
                }

                if ((i == layerMagicQubit) && (p == layerMagicAxis)) {
                    // Non-Clifford rotation
                    const real1 gateRand = (real1)(PI_R1 * rng->Rand() / 2);
                    const complex mtrx[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, std::polar(ONE_R1, gateRand) };
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, mtrx));
                }
            }
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };

                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();

    QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
    circuit->Run(testCase);
    circuit->Inverse()->Run(testCase);
    testCase->Finish();

    std::cout << "Mirror circuit fidelity: " << testCase->ProbAll(randPerm) << std::endl;
    std::cout
        << "Execution time: "
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s" << std::endl;
}

TEST_CASE("test_stabilizer_rz_hard_nn_mirror", "[supreme]")
{
    std::cout << ">>> 'test_stabilizer_rz_hard_nn_mirror':" << std::endl;

    const int GateCount2Qb = 8;
    const int w = max_qubits;
    const int n = (benchmarkDepth <= 0) ? w : benchmarkDepth;
    std::cout << "Circuit width: " << w << std::endl;
    std::cout << "Circuit layer depth (excluding factor of x2 for mirror validation): " << n << std::endl;
    std::cout << "WARNING: 54 qubit reading is rather 53 qubits with Sycamore's excluded qubit." << std::endl;

    // The test runs 2 bit gates according to a tiling sequence.
    // The 1 bit indicates +/- column offset.
    // The 2 bit indicates +/- row offset.
    // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
    // paper.
    std::list<bitLenInt> gateSequence{ 0, 3, 2, 1, 2, 1, 0, 3 };
    const bitLenInt deadQubit = 3U;

    // We factor the qubit count into two integers, as close to a perfect square as we can.
    int colLen = std::sqrt(w);
    while (((w / colLen) * colLen) != w) {
        colLen--;
    }
    int rowLen = w / colLen;

    int d;
    int i;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    const complex ONE_PLUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2));
    const complex ONE_MINUS_I_DIV_2 = complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2));
    const complex sqrtx[4]{ ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2 };
    const complex sqrty[4]{ ONE_PLUS_I_DIV_2, -ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2 };
    const complex wconj[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, exp(I_CMPLX * (real1)(PI_R1 / 8)) };
    const complex h[4U]{ SQRT1_2_R1, SQRT1_2_R1, SQRT1_2_R1, -SQRT1_2_R1 };
    const complex x[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const complex y[4U]{ ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
    const complex z[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const complex s[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    const complex is[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };

    std::vector<int> lastSingleBitGates;

    QCircuitPtr circuit = std::make_shared<QCircuit>(false);

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    for (d = 0; d < n; d++) {
        for (i = 0; i < w; i++) {
            if ((n == 54U) && (i == deadQubit)) {
                if (d == 0) {
                    lastSingleBitGates.push_back(0);
                }
                continue;
            }

            // Each individual bit has one of these 3 gates applied at random.
            // "SqrtX" and "SqrtY" are Clifford.
            // "SqrtW" requires one non-Clifford RZ gate, (beside H gates).
            // The same gate is not applied twice consecutively in sequence.

            if (d == 0) {
                // For the first iteration, we can pick any gate.

                const real1_f gateRand = 3 * rng->Rand();
                if (gateRand < ONE_R1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, sqrtx));
                    lastSingleBitGates.push_back(0);
                } else if (gateRand < (2 * ONE_R1)) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, sqrty));
                    lastSingleBitGates.push_back(1);
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, wconj));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    lastSingleBitGates.push_back(2);
                }
            } else {
                // For all subsequent iterations after the first, we eliminate the choice of the same gate applied
                // on the immediately previous iteration.

                int gateChoice = std::max(1, (int)(2 * rng->Rand()));
                if (gateChoice >= lastSingleBitGates[i]) {
                    ++gateChoice;
                }

                if (gateChoice == 0) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, sqrtx));
                    lastSingleBitGates[i] = 0;
                } else if (gateChoice == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, sqrty));
                    lastSingleBitGates[i] = 1;
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, wconj));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(i, h));
                    lastSingleBitGates[i] = 2;
                }
            }
        }

        int gate = gateSequence.front();
        std::vector<bitLenInt> usedBits;

        for (int row = 1; row < rowLen; row += 2) {
            for (int col = 0; col < colLen; col++) {
                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                // awkwardly.)

                int b1 = row * colLen + col;

                if (std::find(usedBits.begin(), usedBits.end(), b1) != usedBits.end()) {
                    continue;
                }

                int tempRow = row;
                int tempCol = col;

                tempRow += ((gate & 2U) ? 1 : -1);
                tempCol += (colLen == 1) ? 0 : ((gate & 1U) ? 1 : 0);

                int b2 = tempRow * colLen + tempCol;

                if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen) ||
                    (std::find(usedBits.begin(), usedBits.end(), b2) != usedBits.end())) {
                    continue;
                }

                usedBits.push_back(b1);
                usedBits.push_back(b2);

                gate = (int)(rng->Rand() * GateCount2Qb);
                if (gate >= GateCount2Qb) {
                    gate = GateCount2Qb - 1U;
                }

                const std::set<bitLenInt> control{ (bitLenInt)b1 };
                if (gate == 0) {
                    circuit->Swap(b1, b2);
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, s));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, s));
                } else if (gate == 1) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b1, is));
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                    circuit->Swap(b1, b2);
                } else if (gate == 2) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ONE_BCI));
                } else if (gate == 3) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ONE_BCI));
                } else if (gate == 4) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ONE_BCI));
                } else if (gate == 5) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, x, control, ZERO_BCI));
                } else if (gate == 6) {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, y, control, ZERO_BCI));
                } else {
                    circuit->AppendGate(std::make_shared<QCircuitGate>(b2, z, control, ZERO_BCI));
                }
            }
        }

        if (d & 1) {
            gateSequence.pop_front();
            gateSequence.push_back(gate);
        }
    }

    bitCapIntOcl randPerm = (bitCapIntOcl)(rng->Rand() * pow2Ocl(w));
    if (randPerm >= pow2Ocl(w)) {
        randPerm = pow2Ocl(w) - 1U;
    }

    auto start = std::chrono::high_resolution_clock::now();

    QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, randPerm);
    circuit->Run(testCase);
    circuit->Inverse()->Run(testCase);
    testCase->Finish();

    std::cout << "Mirror circuit fidelity: " << testCase->ProbAll(randPerm) << std::endl;
    std::cout
        << "Execution time: "
        << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()
        << "s" << std::endl;
}

TEST_CASE("test_noisy_qft_cosmology_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_qft_cosmology_estimate':" << std::endl;

    const int w = max_qubits;
    std::cout << "Circuit width: " << w << std::endl;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, ZERO_BCI);
        for (bitLenInt i = 0; i < w; i++) {
            RandomInitQubit(testCase, i);
        }
        const bitLenInt end = w - 1U;
        for (bitLenInt i = 0U; i < w; ++i) {
            const bitLenInt hBit = end - i;
            for (bitLenInt j = 0U; j < i; ++j) {
                const bitLenInt c = hBit;
                const bitLenInt t = hBit + 1U + j;
                testCase->CPhaseRootN(j + 2U, c, t);
                testCase->TrySeparate(c, t);
            }
            testCase->H(hBit);
        }
        testCase->MAll();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}

TEST_CASE("test_noisy_qft_ghz_estimate", "[supreme_estimate]")
{
    std::cout << ">>> 'test_noisy_qft_ghz_estimate':" << std::endl;

    const int w = max_qubits;
    std::cout << "Circuit width: " << w << std::endl;

    const std::vector<QInterfaceEngine> engineStack = BuildEngineStack();

    QInterfacePtr rng = CreateQuantumInterface(engineStack, 1, ZERO_BCI);

    auto start = std::chrono::high_resolution_clock::now();
    double sdrp = 1.0;

    while (sdrp >= 0) {
        start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32) && !defined(__CYGWIN__)
        if (sdrp <= FP_NORM_EPSILON) {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=";
            _putenv(envVar.c_str());
        } else {
            std::string envVar = "QRACK_QUNIT_SEPARABILITY_THRESHOLD=" + std::to_string(sdrp);
            _putenv(envVar.c_str());
        }
#else
        if (sdrp <= FP_NORM_EPSILON) {
            unsetenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD");
        } else {
            setenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD", std::to_string(sdrp).c_str(), 1);
        }
#endif

        QInterfacePtr testCase = CreateQuantumInterface(engineStack, w, ZERO_BCI);
        testCase->H(0U);
        const bitLenInt end = w - 1U;
        for (bitLenInt i = 0; i < end; i++) {
            testCase->CNOT(i, i + 1U);
        }
        testCase->QFT(0, w);
        testCase->MAll();

        std::cout << "For SDRP=" << sdrp << ": " << std::endl;
        std::cout << "Unitary fidelity: " << testCase->GetUnitaryFidelity() << std::endl;
        std::cout << "Execution time: "
                  << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start)
                         .count()
                  << "s" << std::endl;

        sdrp -= 0.025;
        if (abs(sdrp) < FP_NORM_EPSILON) {
            sdrp = 0;
        }
    }
}
