//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

namespace Qrack {

QInterfacePtr QEngineCPU::Clone()
{
    if (!stateVec) {
        return CloneEmpty();
    }

    QEngineCPUPtr clone = std::make_shared<QEngineCPU>(qubitCount, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, !hardware_rand_generator ? false : true, isSparse, (real1_f)amplitudeFloor);

    Finish();
    clone->Finish();
    clone->runningNorm = runningNorm;
    clone->fidelity = fidelity;
    clone->stateVec->copy(stateVec);

    return clone;
}

QEnginePtr QEngineCPU::CloneEmpty()
{
    QEngineCPUPtr clone = std::make_shared<QEngineCPU>(0U, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, !hardware_rand_generator ? false : true, isSparse, (real1_f)amplitudeFloor);

    clone->SetQubitCount(qubitCount);

    return clone;
}

QInterfacePtr QEngineCPU::Copy()
{
    QEngineCPUPtr clone = std::dynamic_pointer_cast<QEngineCPU>(CloneEmpty());
    clone->stateVec = stateVec;
    clone->fidelity = fidelity;

    return clone;
}

bitLenInt QEngineCPU::Allocate(bitLenInt start, bitLenInt length)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QEngineCPU::Allocate argument is out-of-bounds!");
    }

    if (!length) {
        return start;
    }

    QEngineCPUPtr nQubits = std::make_shared<QEngineCPU>(length, ZERO_BCI, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, !hardware_rand_generator ? false : true, isSparse, (real1_f)amplitudeFloor);

    return Compose(nQubits, start);
}

real1_f QEngineCPU::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    for (bitCapIntOcl i = 0U; i < maxQPowerOcl; ++i) {
        bitCapIntOcl outputInt = (i & outputMask) >> valueStart;
        real1 prob = norm(stateVec->read(i));
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > ZERO_R1) {
        average /= totProb;
    }
    return (real1_f)average;
}

} // namespace Qrack
