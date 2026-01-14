//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates an example of a "quantum associative memory" network with the Qrack::QNeuron
// class. QNeuron is a type of "neuron" that can learn and predict in superposition, for general machine learning
// purposes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"
// "qneuron.hpp" defines the QNeuron class.
#include "qneuron.hpp"

#include <iostream> // For cout

using namespace Qrack;

int main()
{
    const bitLenInt InputCount = 4;
    const bitLenInt OutputCount = 4;
    const bitCapInt InputPower = 1U << InputCount;
    // const bitCapInt OutputPower = 1U << OutputCount;
    const real1 eta = ONE_R1 / (real1)2.0f;
    std::unique_ptr<real1_s[]> angles(new real1_s[pow2Ocl(InputCount)]());

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, InputCount + OutputCount, ZERO_BCI);

    std::vector<bitLenInt> inputIndices(InputCount);
    for (bitLenInt i = 0; i < InputCount; i++) {
        inputIndices[i] = i;
    }

    std::vector<QNeuronPtr> outputLayer;
    for (bitLenInt i = 0; i < OutputCount; i++) {
        outputLayer.push_back(std::make_shared<QNeuron>(qReg, inputIndices, InputCount + i));
    }

    // Train the network to associate powers of 2 with their log2()
    std::cout << "Learning (Two's complement)..." << std::endl;
    for (bitCapInt perm = ZERO_BCI; bi_compare(perm, InputPower) < 0; bi_increment(&perm, 1U)) {
        std::cout << "Epoch " << (perm + ONE_BCI) << " out of " << InputPower << std::endl;
        const bitCapInt comp = (~perm) + ONE_BCI;
        qReg->SetPermutation(perm);
        for (bitLenInt i = 0; i < OutputCount; i++) {
            outputLayer[i]->LearnPermutation(angles.get(), (real1_f)eta, bi_compare_0(comp & pow2(i)) != 0);
        }
    }

    std::cout << "Should associate each input with its two's complement as output..." << std::endl;
    for (bitCapInt perm = ZERO_BCI; bi_compare(perm, InputPower) < 0; bi_increment(&perm, 1U)) {
        qReg->SetPermutation(perm);
        for (bitLenInt i = 0; i < OutputCount; i++) {
            outputLayer[i]->Predict(angles.get());
        }
        const bitCapInt comp = qReg->MReg(InputCount, OutputCount);
        std::cout << "Input: " << perm << ", Output: " << comp << std::endl;
    }
}
