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

#include "qinterface.hpp"

namespace Qrack {

/// General unitary gate
void QInterface::U(bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4]{ complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    Mtrx(uGate, target);
}

/// Controlled general unitary gate
void QInterface::CU(
    const std::vector<bitLenInt>& controls, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4]{ complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    MCMtrx(controls, uGate, target);
}

/// (Anti-)Controlled general unitary gate
void QInterface::AntiCU(
    const std::vector<bitLenInt>& controls, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4]{ complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    MACMtrx(controls, uGate, target);
}

/// "Azimuth, Inclination"
void QInterface::AI(bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    Mtrx(mtrx, target);
}

/// Inverse "Azimuth, Inclination"
void QInterface::IAI(bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    complex invMtrx[4];
    inv2x2(mtrx, invMtrx);
    Mtrx(invMtrx, target);
}

/// Controlled "Azimuth, Inclination"
void QInterface::CAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    const std::vector<bitLenInt> controls{ control };
    MCMtrx(controls, mtrx, target);
}

/// Controlled inverse "Azimuth, Inclination"
void QInterface::CIAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    const std::vector<bitLenInt> controls{ control };
    complex invMtrx[4];
    inv2x2(mtrx, invMtrx);
    MCMtrx(controls, invMtrx, target);
}

/// Controlled "Azimuth, Inclination"
void QInterface::AntiCAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    const std::vector<bitLenInt> controls{ control };
    MACMtrx(controls, mtrx, target);
}

/// Controlled inverse "Azimuth, Inclination"
void QInterface::AntiCIAI(bitLenInt control, bitLenInt target, real1_f azimuth, real1_f inclination)
{
    const real1 cosineA = (real1)cos(azimuth);
    const real1 sineA = (real1)sin(azimuth);
    const real1 cosineI = (real1)cos(inclination / 2);
    const real1 sineI = (real1)sin(inclination / 2);
    const complex mtrx[4]{ cosineI, complex(-cosineA, sineA) * sineI, complex(cosineA, sineA) * sineI, cosineI };
    const std::vector<bitLenInt> controls{ control };
    complex invMtrx[4];
    inv2x2(mtrx, invMtrx);
    MACMtrx(controls, invMtrx, target);
}

/// Uniformly controlled y axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli y axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRY(
    const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, const real1* angles)
{
    const bitCapIntOcl permCount = pow2Ocl(controls.size());
    std::unique_ptr<complex[]> pauliRYs(new complex[4U * permCount]);

    for (bitCapIntOcl i = 0U; i < permCount; ++i) {
        const real1 cosine = (real1)cos(angles[i] / 2);
        const real1 sine = (real1)sin(angles[i] / 2);

        pauliRYs[0U + 4U * i] = complex(cosine, ZERO_R1);
        pauliRYs[1U + 4U * i] = complex(-sine, ZERO_R1);
        pauliRYs[2U + 4U * i] = complex(sine, ZERO_R1);
        pauliRYs[3U + 4U * i] = complex(cosine, ZERO_R1);
    }

    UniformlyControlledSingleBit(controls, qubitIndex, pauliRYs.get());
}

/// Uniformly controlled z axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli z axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRZ(
    const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, const real1* angles)
{
    const bitCapIntOcl permCount = pow2Ocl(controls.size());
    std::unique_ptr<complex[]> pauliRZs(new complex[4U * permCount]);

    for (bitCapIntOcl i = 0U; i < permCount; ++i) {
        const real1 cosine = (real1)cos(angles[i] / 2);
        const real1 sine = (real1)sin(angles[i] / 2);

        pauliRZs[0U + 4U * i] = complex(cosine, -sine);
        pauliRZs[1U + 4U * i] = ZERO_CMPLX;
        pauliRZs[2U + 4U * i] = ZERO_CMPLX;
        pauliRZs[3U + 4U * i] = complex(cosine, sine);
    }

    UniformlyControlledSingleBit(controls, qubitIndex, pauliRZs.get());
}

/// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
void QInterface::RT(real1_f radians, bitLenInt qubit)
{
    Phase(ONE_CMPLX, complex((real1)cos(radians / 2), (real1)sin(radians / 2)), qubit);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::RX(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRX[4]{ complex(cosine, ZERO_R1), complex(ZERO_R1, -sine), complex(ZERO_R1, -sine),
        complex(cosine, ZERO_R1) };
    Mtrx(pauliRX, qubit);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void QInterface::RY(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRY[4]{ cosine, -sine, sine, cosine };
    Mtrx(pauliRY, qubit);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void QInterface::RZ(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    Phase(complex(cosine, -sine), complex(cosine, sine), qubit);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void QInterface::CRZ(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const std::vector<bitLenInt> controls{ control };
    MCPhase(controls, complex(cosine, -sine), complex(cosine, sine), target);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void QInterface::CRY(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRY[4]{ complex(cosine, ZERO_R1), complex(-sine, ZERO_R1), complex(sine, ZERO_R1),
        complex(cosine, ZERO_R1) };
    const std::vector<bitLenInt> controls{ control };
    MCMtrx(controls, pauliRY, target);
}

#if ENABLE_ROT_API
/// Exponentiate identity operator
void QInterface::Exp(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Phase(phaseFac, phaseFac, qubit);
}

/// Imaginary exponentiate of arbitrary single bit gate
void QInterface::Exp(
    const std::vector<bitLenInt>& controls, bitLenInt qubit, const complex matrix2x2[4U], bool antiCtrled)
{
    complex timesI[4U];
    for (bitLenInt i = 0U; i < 4U; ++i) {
        timesI[i] = I_CMPLX * matrix2x2[i];
    }
    complex toApply[4U];
    exp2x2(timesI, toApply);
    if (antiCtrled) {
        MACMtrx(controls, toApply, qubit);
    } else {
        MCMtrx(controls, toApply, qubit);
    }
}

/// Exponentiate Pauli X operator
void QInterface::ExpX(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Invert(phaseFac, phaseFac, qubit);
}

/// Exponentiate Pauli Y operator
void QInterface::ExpY(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Invert(phaseFac * -I_CMPLX, phaseFac * I_CMPLX, qubit);
}

/// Exponentiate Pauli Z operator
void QInterface::ExpZ(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Phase(phaseFac, -phaseFac, qubit);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QInterface::CRT(real1_f radians, bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1]{ control };
    MCPhase(controls, 1, ONE_CMPLX, complex((real1)cos(radians / 2), (real1)sin(radians / 2)), target);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::CRX(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRX[4]{ complex(cosine, ZERO_R1), complex(ZERO_R1, sine), complex(ZERO_R1, sine),
        complex(cosine, ZERO_R1) };
    const bitLenInt controls[1]{ control };
    MCMtrx(controls, 1, pauliRX, target);
}
#endif
} // namespace Qrack
