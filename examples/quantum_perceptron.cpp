//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates a (trivial) example of "quantum neuron" or a "quantum perceptron" with the Qrack::QNeuron
// class. This is a type of "neuron" that can learn and predict in superposition, for general machine learning purposes.
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
    const bitLenInt ControlCount = 4;
    const bitCapInt ControlPower = 1U << ControlCount;
    const bitLenInt ControlLog = 2;
    const real1 eta = ONE_R1 / (real1)2.0f;
    std::unique_ptr<real1[]> angles(new real1[pow2Ocl(ControlCount)]());

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, ControlCount + 1, ZERO_BCI);

    std::vector<bitLenInt> inputIndices(ControlCount);
    for (bitLenInt i = 0; i < ControlCount; i++) {
        inputIndices[i] = i;
    }

    QNeuronPtr qPerceptron = std::make_shared<QNeuron>(qReg, inputIndices, ControlCount);

    // Train the network to recognize powers of 2
    bool isPowerOf2;
    bitCapInt perm;
    std::cout << "Learning (to recognize powers of 2)..." << std::endl;
    for (perm = ZERO_BCI; bi_compare(perm, ControlPower) < 0; bi_increment(&perm, 1U)) {
        std::cout << "Epoch " << (perm + ONE_BCI) << " out of " << ControlPower << std::endl;
        qReg->SetPermutation(perm);
        isPowerOf2 = (bi_compare_0(perm) != 0) && (bi_compare_0(perm & (perm - ONE_BCI)) == 0);
        qPerceptron->LearnPermutation(angles.get(), (real1_f)eta, isPowerOf2);
    }

    std::cout << "Should be close to 1 for powers of two, and close to 0 for all else..." << std::endl;
    for (perm = ZERO_BCI; bi_compare(perm, ControlPower) < 0; bi_increment(&perm, 1U)) {
        qReg->SetPermutation(perm);
        std::cout << "Permutation: " << perm << ", Probability: " << qPerceptron->Predict(angles.get()) << std::endl;
    }

    // Now, we prepare a superposition of all available powers of 2, to predict.
    std::unique_ptr<unsigned char[]> powersOf2(new unsigned char[ControlCount]);
    for (bitLenInt i = 0; i < ControlCount; i++) {
        powersOf2.get()[i] = 1U << i;
    }

    QInterfacePtr qReg2 = CreateQuantumInterface(QINTERFACE_OPTIMAL, ControlLog, ZERO_BCI);

    qReg->Compose(qReg2);
    qReg->SetPermutation(Qrack::pow2(ControlCount + 1));
    qReg->H(ControlCount + 1, ControlLog);
    std::dynamic_pointer_cast<QAlu>(qReg)->IndexedLDA(ControlCount + 1, ControlLog, 0, ControlCount, powersOf2.get());
    qReg->H(ControlCount + 1, ControlLog);
    qReg->Dispose(ControlCount + 1, ControlLog);

    std::cout << "(Superposition of all powers of 2) Probability: " << qPerceptron->Predict(angles.get()) << std::endl;
}
