//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// This example demonstrates the quantum volume random unitary circuit generation
// protocol. It also "hashes" a "determinant" from the random circuit generated.
// This was developed in collaboration between Dan Strano and (Anthropic) Claude.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#include <cmath>
#include <numeric>
#include <set>
#include <vector>

using namespace Qrack;

bitLenInt pickRandomBit(real1_f rand, std::vector<bitLenInt>* unusedBitsPtr)
{
    bitLenInt bitRand = (bitLenInt)(unusedBitsPtr->size() * rand);
    if (bitRand >= unusedBitsPtr->size()) {
        bitRand = unusedBitsPtr->size() - 1U;
    }
    bitLenInt result = (*unusedBitsPtr)[bitRand];
    unusedBitsPtr->erase(unusedBitsPtr->begin() + bitRand);

    return result;
}

real1_f fix_range(real1_f theta)
{
    while (theta <= -PI_R1) {
        theta += 2 * PI_R1;
    }
    while (theta > PI_R1) {
        theta -= 2 * PI_R1;
    }

    return theta;
}

int main()
{
    const bitLenInt n = 16;
    const int depth = n; // QV uses square circuits

    std::cout << "Qubit width: " << (int)n << std::endl;
    std::cout << std::endl;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_TENSOR_NETWORK, n, ZERO_BCI);

    std::vector<bitLenInt> allBits(n);
    std::iota(allBits.begin(), allBits.end(), 0U);

    // Two parallel running products, one per X/Z pair
    std::vector<real1_f> det_th(n, 0.0);
    std::vector<real1_f> det_ph(n, 0.0);

    // Per-layer sin values, kept separate
    std::vector<real1_f> s_th(n), s_ph(n);
    std::vector<real1_f> th_last(n, 0.0);

    for (int d = 0; d < depth; ++d) {
        // Single-qubit layer
        for (bitLenInt i = 0U; i < n; ++i) {
            const real1_f th = 2 * PI_R1 * qReg->Rand() - PI_R1;
            const real1_f ph = 2 * PI_R1 * qReg->Rand() - PI_R1;
            qReg->AI(i, th, ph);

            th_last[i] = th;
            s_th[i] = th;
            s_ph[i] = ph;

            // Accumulate own contribution
            det_th[i] = fix_range(det_th[i] + th);
            det_ph[i] = fix_range(det_ph[i] + ph);
        }

        // Two-qubit layer
        std::vector<bitLenInt> unusedBits(allBits);
        while (unusedBits.size() > 1U) {
            const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
            const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
            qReg->CNOT(b1, b2);

            // Forward: propagate bit-flip state onto target
            det_ph[b2] = fix_range(det_ph[b2] + s_ph[b1]);

            // Reverse kickback: propagate phase-flip state onto contro
            det_th[b1] = fix_range(det_th[b1] + s_th[b2]);
        }

        // Unpaired qubit — already accumulated in single-qubit pass
        // nothing extra needed
    }

    // Output pair — depth independent
    std::cout << "Determinant pairs, bit probability:" << std::endl;
    for (bitLenInt i = 0U; i < n; ++i) {
        std::cout << (int)i << ": (" << det_th[i] << ", " << det_ph[i] << "), ";
        std::cout << qReg->Prob(i) << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
