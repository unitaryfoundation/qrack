//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates a quantum cosmology simulation of interest to the author.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream> // std::cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

void StatePrep(QInterfacePtr qReg)
{
    real1_f theta = 2 * PI_R1 * qReg->Rand();
    real1_f phi = 2 * PI_R1 * qReg->Rand();
    real1_f lambda = 2 * PI_R1 * qReg->Rand();
    qReg->U(0, theta, phi, lambda);
}

void AddBit(QInterfacePtr qReg)
{
    QInterfacePtr nBit = CreateQuantumInterface(QINTERFACE_OPTIMAL, 1, 0);
    StatePrep(nBit);
    qReg->Compose(nBit);
}

int main()
{
    const bitLenInt maxLength = 28U;
    std::vector<bitLenInt> bits;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 1, 0);
    StatePrep(qReg);
    bits.push_back(0);

    bitLenInt i, j;
    bitLenInt c, t;
    for (i = 0; i < maxLength; i++) {
        for (j = 0; j < i; j++) {
            c = i;
            t = i - (1U + j);
            qReg->CPhaseRootN(j + 2U, c, t);
        }
        qReg->H(i);

        std::vector<bitLenInt> expBits(bits);
        std::reverse(expBits.begin(), expBits.end());

        std::cout << "Folds=" << (real1)(i + 1U)
                  << ", Manifold size=" << qReg->ExpectationBitsAll(&(expBits[0]), expBits.size()) << std::endl;

        if (i < (maxLength - 1U)) {
            AddBit(qReg);
            bits.push_back(bits.back() + 1U);
        }
    }
}
