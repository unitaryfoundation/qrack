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

    // Three parallel running products, one per U3 parameter
    std::vector<real1_f> det_theta(n, 0.0);
    std::vector<real1_f> det_phi(n, 0.0);
    std::vector<real1_f> det_lambda(n, 0.0);

    // Per-layer sin values, kept separate
    std::vector<real1_f> s_theta(n), s_phi(n), s_lambda(n);
    std::vector<real1_f> theta_last(n, 0.0);

    for (int d = 0; d < depth; ++d) {
        // Single-qubit layer
        for (bitLenInt i = 0U; i < n; ++i) {
            const real1_f theta = 2 * PI_R1 * qReg->Rand() - PI_R1;
            const real1_f phi = 2 * PI_R1 * qReg->Rand() - PI_R1;
            const real1_f lambda = 2 * PI_R1 * qReg->Rand() - PI_R1;
            qReg->U(i, theta, phi, lambda);

            theta_last[i] = theta;
            s_theta[i] = theta;
            s_phi[i] = phi;
            s_lambda[i] = lambda;

            // Accumulate own contribution
            det_theta[i] = fix_range(det_theta[i] + theta);
            det_phi[i] = fix_range(det_phi[i] + phi);
            det_lambda[i] = fix_range(det_lambda[i] + lambda);
        }

        // Two-qubit layer
        std::vector<bitLenInt> unusedBits(allBits);
        while (unusedBits.size() > 1U) {
            const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
            const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
            qReg->CNOT(b1, b2);

            // Forward: propagate each parameter of control into target
            det_theta[b2] = fix_range(det_theta[b2] + s_theta[b1]);
            det_phi[b2] = fix_range(det_phi[b2] + s_phi[b1]);
            det_lambda[b2] = fix_range(det_lambda[b2] + s_lambda[b1]);

            // Reverse kickback: cos(theta_b2/2) scales back into control
            // Applied to all three parameters of the control qubit
            const real1_f kickback = (PI_R1 + theta_last[b2]) / 2;
            det_theta[b1] = fix_range(det_theta[b1] + kickback);
            det_phi[b1] = fix_range(det_phi[b1] + kickback);
            det_lambda[b1] = fix_range(det_lambda[b1] + kickback);
        }

        // Unpaired qubit — already accumulated in single-qubit pass
        // nothing extra needed
    }

    // Collapse each parameter determinant across all qubits
    real1_f D_theta = 0.0, D_phi = 0.0, D_lambda = 0.0;
    for (bitLenInt i = 0U; i < n; ++i) {
        D_theta += det_theta[i];
        D_phi += det_phi[i];
        D_lambda += det_lambda[i];
    }

    // Output triplet — width and depth independent
    std::cout << "Determinant triplet:" << std::endl;
    std::cout << D_theta << " " << D_phi << " " << D_lambda << std::endl;
    std::cout << std::endl;

    std::vector<bitCapInt> allPowers(n);
    for (size_t i = 0U; i < n; ++i) {
        allPowers[i] = bitCapInt(1) << allBits[i];
    }

    // Sample the circuit
    std::map<bitCapInt, int> counter = qReg->MultiShotMeasureMask(allPowers, 100U);

    std::cout << "Measurement shot counts:" << std::endl;
    for (const auto& pair : counter) {
        std::cout << (uint64_t)(pair.first) << ": " << pair.second << std::endl;
    }

    return 0;
}
