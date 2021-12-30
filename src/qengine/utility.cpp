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

namespace Qrack {

QInterfacePtr QEngineCPU::Clone()
{
    Finish();

    QEngineCPUPtr clone = std::dynamic_pointer_cast<QEngineCPU>(
        CreateQuantumInterface(QINTERFACE_CPU, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
            false, -1, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor));
    if (stateVec) {
        clone->stateVec->copy(stateVec);
    } else {
        clone->ZeroAmplitudes();
    }
    return clone;
}

real1_f QEngineCPU::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    for (bitCapIntOcl i = 0; i < maxQPower; i++) {
        bitCapIntOcl outputInt = (i & outputMask) >> valueStart;
        real1 prob = norm(stateVec->read(i));
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return average;
}

} // namespace Qrack
