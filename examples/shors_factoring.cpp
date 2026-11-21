//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates Shor's algorithm for integer factoring. (This file was heavily adapted from
// https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/shor.py, with thanks to ProjectQ!)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#include <chrono>
#include <cmath>
#include <iomanip> // For setw
#include <iostream> // For cout
#include <random>
#include <stdlib.h>
#include <time.h>

using namespace Qrack;

bitCapInt gcd(bitCapInt n1, bitCapInt n2)
{
    if (bi_compare_0(n2) == 0)
        return n1;

    bitCapInt rem;
    bi_div_mod(n1, n2, NULL, &rem);

    return gcd(n2, rem);
}

bitCapInt continued_fraction_step(bitCapInt* numerator, bitCapInt* denominator)
{
    bitCapInt intPart;
    bi_div_mod(*numerator, *denominator, &intPart, NULL);
    bitCapInt partDenominator = (*numerator) - intPart * (*denominator);
    bitCapInt partNumerator = (*denominator);

    (*numerator) = partNumerator;
    (*denominator) = partDenominator;
    return intPart;
}

real1_f calc_continued_fraction(std::vector<bitCapInt> denominators, bitCapInt* numerator, bitCapInt* denominator)
{
    bitCapInt approxNumer = ONE_BCI;
    bitCapInt approxDenom = denominators.size() ? denominators.back() : ONE_BCI;
    bitCapInt temp;

    for (int i = (denominators.size() - 1); i > 0; i--) {
        temp = denominators[i] * approxDenom + approxNumer;
        approxNumer = approxDenom;
        approxDenom = temp;
    }

    (*numerator) = approxNumer;
    (*denominator) = approxDenom;
    return ((real1_f)bi_to_double(approxNumer)) / (bitCapIntOcl)approxDenom;
}

int main()
{
    uint64_t toFactor;
    bitCapInt base;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

    const double clockFactor = 1.0 / 1000.0; // Report in ms
    auto iterClock = std::chrono::high_resolution_clock::now();

    bitLenInt qubitCount = ceil(Qrack::log2Ocl(toFactor));

    // Choose a base at random:
    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());
    std::uniform_int_distribution<> rand_dist(2, toFactor - 1);
    base = rand_dist(rand_gen);

    bitCapInt testFactor = gcd(toFactor, base);
    if (bi_compare_1(testFactor) != 0) {
        bitCapInt quo;
        bi_div_mod(toFactor, testFactor, &quo, NULL);
        std::cout << "Chose non- relative prime: " << testFactor << " * " << quo << std::endl;
        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - iterClock);
        std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;
        return 0;
    }

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, qubitCount * 2, ZERO_BCI);
    QAluPtr qAlu = std::dynamic_pointer_cast<QAlu>(qReg);
    // TODO: This shouldn't be necessary:
    qReg->Finish();

    // This is the period-finding subroutine of Shor's algorithm.
    qReg->H(0, qubitCount);
    qAlu->POWModNOut(base, toFactor, 0, qubitCount, qubitCount);
    qReg->IQFT(0, qubitCount);

    bitCapInt y = qReg->MAll() & (pow2(qubitCount) - ONE_BCI);
    if (bi_compare_0(y) == 0) {
        std::cout << "Failed: y = 0 in period estimation subroutine." << std::endl;
        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - iterClock);
        std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;
        return 0;
    }
    bitCapInt qubitPower = Qrack::pow2(qubitCount);

    // Value is always fractional, so skip first step, by flipping numerator and denominator:
    bitCapInt numerator = qubitPower;
    bitCapInt denominator = y;

    std::vector<bitCapInt> denominators;
    bitCapInt approxNumer;
    bitCapInt approxDenom;
    do {
        denominators.push_back(continued_fraction_step(&numerator, &denominator));
        calc_continued_fraction(denominators, &approxNumer, &approxDenom);
    } while ((bi_compare_0(denominator) > 0) && (bi_compare(approxDenom, toFactor) < 0));
    denominators.pop_back();

    bitCapInt r;
    if (denominators.size() == 0) {
        r = y;
    } else {
        calc_continued_fraction(denominators, &approxNumer, &r);
    }

    // Try to determine the factors
    if (bi_and_1(r)) {
        bi_lshift_ip(&r, 1U);
    }
    bitCapInt apowrhalf = ((bitCapIntOcl)pow(bi_to_double(base), bi_to_double(r >> 1U))) % toFactor;
    bitCapIntOcl f1 = (bitCapIntOcl)gcd(apowrhalf + ONE_BCI, toFactor);
    bitCapIntOcl f2 = (bitCapIntOcl)gcd(apowrhalf - ONE_BCI, toFactor);
    bitCapIntOcl res1 = f1;
    bitCapIntOcl res2 = f2;
    if (((f1 * f2) != toFactor) && ((f1 * f2) > 1) &&
        (((uint64_t)((ONE_R1 * toFactor) / (f1 * f2)) * f1 * f2) == toFactor)) {
        res1 = f1 * f2;
        res2 = toFactor / (f1 * f2);
    }
    if (((res1 * res2) == toFactor) && (res1 > 1) && (res2 > 1)) {
        std::cout << "Success: Found " << res1 << " * " << res2 << " = " << toFactor << std::endl;
    } else {
        std::cout << "Failure: Found " << res1 << " and " << res2 << std::endl;
    }
    auto tClock =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - iterClock);
    std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;

    return 0;
}
