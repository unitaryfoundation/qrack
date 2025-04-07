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

QRACK_CONST complex C_SQRT1_2 = complex(SQRT1_2_R1, ZERO_R1);
QRACK_CONST complex C_SQRT_I = complex(SQRT1_2_R1, SQRT1_2_R1);
QRACK_CONST complex C_SQRT_N_I = complex(SQRT1_2_R1, -SQRT1_2_R1);
QRACK_CONST complex I_CMPLX_NEG = complex(ZERO_R1, -ONE_R1);
QRACK_CONST complex C_SQRT1_2_NEG = complex(-SQRT1_2_R1, ZERO_R1);

void QInterface::UCMtrx(
    const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target, const bitCapInt& controlPerm)
{
    size_t setCount = 0U;
    for (size_t i = 0U; i < controls.size(); ++i) {
        if (bi_and_1(controlPerm >> i)) {
            ++setCount;
        }
    }

    if ((setCount << 1U) > controls.size()) {
        for (size_t i = 0U; i < controls.size(); ++i) {
            if (!bi_and_1(controlPerm >> i)) {
                X(controls[i]);
            }
        }
        MCMtrx(controls, mtrx, target);
        for (size_t i = 0U; i < controls.size(); ++i) {
            if (!bi_and_1(controlPerm >> i)) {
                X(controls[i]);
            }
        }

        return;
    }

    for (size_t i = 0U; i < controls.size(); ++i) {
        if (bi_and_1(controlPerm >> i)) {
            X(controls[i]);
        }
    }
    MACMtrx(controls, mtrx, target);
    for (size_t i = 0U; i < controls.size(); ++i) {
        if (bi_and_1(controlPerm >> i)) {
            X(controls[i]);
        }
    }
}

void QInterface::UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
    const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, const bitCapInt& mtrxSkipValueMask)
{
    for (const bitLenInt& control : controls) {
        X(control);
    }
    const bitCapInt maxI = pow2Ocl(controls.size()) - 1U;
    for (bitCapInt lcv = ZERO_BCI; bi_compare(lcv, maxI) < 0; bi_increment(&lcv, 1U)) {
        const bitCapInt index = pushApartBits(lcv, mtrxSkipPowers) | mtrxSkipValueMask;
        MCMtrx(controls, mtrxs + (((bitCapIntOcl)index) << 2U), qubitIndex);

        const bitCapInt lcvDiff = lcv ^ (lcv + ONE_BCI);
        for (size_t bit_pos = 0U; bit_pos < controls.size(); ++bit_pos) {
            if (bi_and_1(lcvDiff >> bit_pos)) {
                X(controls[bit_pos]);
            }
        }
    }
    const bitCapInt index = pushApartBits(maxI, mtrxSkipPowers) | mtrxSkipValueMask;
    MCMtrx(controls, mtrxs + (((bitCapIntOcl)index) << 2U), qubitIndex);
}

void QInterface::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        return Phase(-ONE_CMPLX, ONE_CMPLX, start);
    }

    std::vector<bitLenInt> controls(length - 1U);
    for (size_t i = 0U; i < controls.size(); ++i) {
        controls[i] = start + i;
    }
    MACPhase(controls, -ONE_CMPLX, ONE_CMPLX, start + controls.size());
}

void QInterface::XMask(const bitCapInt& _mask)
{
    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        X(log2(mask ^ v));
        mask = v;
    }
}

void QInterface::YMask(const bitCapInt& _mask)
{
    bitLenInt bit = log2(_mask);

    if (bi_compare(pow2(bit), _mask) == 0) {
        return Y(bit);
    }

    ZMask(_mask);
    XMask(_mask);

    if (randGlobalPhase) {
        return;
    }

    int parity = 0;
    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(v) != 0) {
        v = v & (v - ONE_BCI);
        parity = (parity + 1) & 3;
    }

    if (parity == 1) {
        Phase(I_CMPLX, I_CMPLX, 0U);
    } else if (parity == 2) {
        PhaseFlip();
    } else if (parity == 3) {
        Phase(-I_CMPLX, -I_CMPLX, 0U);
    }
}

void QInterface::ZMask(const bitCapInt& _mask)
{
    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        Z(log2(mask ^ v));
        mask = v;
    }
}

void QInterface::PhaseRootNMask(bitLenInt n, const bitCapInt& _mask)
{
    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        PhaseRootN(n, log2(mask ^ v));
        mask = v;
    }
}

void QInterface::Swap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    CNOT(q1, q2);
    CNOT(q2, q1);
    CNOT(q1, q2);
}

void QInterface::ISwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    Swap(q1, q2);
    CZ(q1, q2);
    S(q1);
    S(q2);
}

void QInterface::IISwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    IS(q2);
    IS(q1);
    CZ(q1, q2);
    Swap(q1, q2);
}

void QInterface::SqrtSwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    CNOT(q1, q2);
    H(q1);
    IT(q2);
    T(q1);
    H(q2);
    H(q1);
    CNOT(q1, q2);
    H(q1);
    H(q2);
    IT(q1);
    H(q1);
    CNOT(q1, q2);
    IS(q1);
    S(q2);
}

void QInterface::ISqrtSwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    IS(q2);
    S(q1);
    CNOT(q1, q2);
    H(q1);
    T(q1);
    H(q2);
    H(q1);
    CNOT(q1, q2);
    H(q1);
    H(q2);
    IT(q1);
    T(q2);
    H(q1);
    CNOT(q1, q2);
}

void QInterface::CSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (!controls.size()) {
        return Swap(q1, q2);
    }

    if (q1 == q2) {
        return;
    }

    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());

    lControls[controls.size()] = q1;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    lControls[controls.size()] = q2;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q1);

    lControls[controls.size()] = q1;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = ZERO_BCI;
    for (const bitLenInt& control : controls) {
        bi_or_ip(&m, pow2(control));
    }

    XMask(m);
    CSwap(controls, q1, q2);
    XMask(m);
}

void QInterface::CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (!controls.size()) {
        return SqrtSwap(q1, q2);
    }

    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    lControls[controls.size()] = q1;

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    QRACK_CONST complex had[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG };
    MCMtrx(controls, had, q1);

    QRACK_CONST complex it[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(controls, it, q2);

    QRACK_CONST complex t[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(controls, t, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(controls, had, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, it, q1);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    QRACK_CONST complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX_NEG };
    MCMtrx(controls, is, q1);

    QRACK_CONST complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(controls, s, q2);
}

void QInterface::CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    lControls[controls.size()] = q1;

    QRACK_CONST complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX_NEG };
    MCMtrx(controls, is, q2);

    QRACK_CONST complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(controls, s, q1);
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    QRACK_CONST complex had[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2_NEG };
    MCMtrx(controls, had, q1);

    QRACK_CONST complex t[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(controls, t, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(controls, had, q1);

    MCMtrx(controls, had, q2);

    QRACK_CONST complex it[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(controls, it, q1);

    MCMtrx(controls, t, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = ZERO_BCI;
    for (const bitLenInt& control : controls) {
        bi_or_ip(&m, pow2(control));
    }

    XMask(m);
    CSqrtSwap(controls, q1, q2);
    XMask(m);
}

void QInterface::AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = ZERO_BCI;
    for (const bitLenInt& control : controls) {
        bi_or_ip(&m, pow2(control));
    }

    XMask(m);
    CISqrtSwap(controls, q1, q2);
    XMask(m);
}

void QInterface::PhaseParity(real1_f radians, const bitCapInt& _mask)
{
    if (bi_compare_0(_mask) == 0) {
        return;
    }

    std::vector<bitLenInt> qubits;
    bitCapInt mask = _mask;
    bitCapInt v = mask;
    while (bi_compare_0(mask) != 0) {
        v = v & (v - ONE_BCI);
        qubits.push_back(log2(mask ^ v));
        mask = v;
    }

    const bitLenInt end = qubits.size() - 1U;
    for (bitLenInt i = 0; i < end; ++i) {
        CNOT(qubits[i], qubits[i + 1U]);
    }
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    Phase(cosine - I_CMPLX * sine, cosine + I_CMPLX * sine, qubits[end]);
    for (bitLenInt i = 0; i < end; ++i) {
        CNOT(qubits[end - (i + 1U)], qubits[end - i]);
    }
}

void QInterface::TimeEvolve(Hamiltonian h, real1_f timeDiff_f)
{
    real1 timeDiff = (real1)timeDiff_f;

    if (abs(timeDiff) <= REAL1_EPSILON) {
        return;
    }

    // Exponentiation of an arbitrary serial string of gates, each HamiltonianOp component times timeDiff, e^(-i * H *
    // t) as e^(-i * H_(N - 1) * t) * e^(-i * H_(N - 2) * t) * ... e^(-i * H_0 * t)

    for (const HamiltonianOpPtr& op : h) {
        complex* opMtrx = op->matrix.get();

        bitCapIntOcl maxJ = 4U;
        if (op->uniform) {
            maxJ *= pow2Ocl(op->controls.size());
        }
        std::unique_ptr<complex[]> mtrx(new complex[maxJ]);

        for (bitCapIntOcl j = 0U; j < maxJ; ++j) {
            mtrx[j] = opMtrx[j] * (-timeDiff);
        }

        if (op->toggles.size()) {
            for (size_t j = 0U; j < op->controls.size(); ++j) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        if (op->uniform) {
            std::unique_ptr<complex[]> expMtrx(new complex[maxJ]);
            for (bitCapIntOcl j = 0U; j < pow2(op->controls.size()); ++j) {
                exp2x2(mtrx.get() + (j * 4U), expMtrx.get() + (j * 4U));
            }
            UniformlyControlledSingleBit(op->controls, op->targetBit, expMtrx.get());
        } else {
            complex timesI[4U]{ I_CMPLX * mtrx[0U], I_CMPLX * mtrx[1U], I_CMPLX * mtrx[2U], I_CMPLX * mtrx[3U] };
            complex toApply[4U];
            exp2x2(timesI, toApply);
            if (op->controls.size() == 0U) {
                Mtrx(toApply, op->targetBit);
            } else if (op->anti) {
                MACMtrx(op->controls, toApply, op->targetBit);
            } else {
                MCMtrx(op->controls, toApply, op->targetBit);
            }
        }

        if (op->toggles.size()) {
            for (size_t j = 0U; j < op->controls.size(); ++j) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }
    }
}

void QInterface::DepolarizingChannelWeak1Qb(bitLenInt qubit, real1_f lambda)
{
    if (lambda <= ZERO_R1) {
        return;
    }

    const real1_f thirdLambda = lambda / 3;
    if (Rand() < thirdLambda) {
        X(qb);
    }
    if (Rand() < thirdLambda) {
        Y(qb);
    }
    if (Rand() < thirdLambda) {
        Z(qb);
    }
}

bitLenInt QInterface::DepolarizingChannelStrong1Qb(bitLenInt qubit, real1_f lambda)
{
    // Original qubit, Z->X basis
    H(qubit);

    // Allocate an ancilla
    const bitLenInt ancilla = Allocate(1U);
    // Partially entangle with the ancilla
    CRY(2 * asin(std::pow((real1_s)lambda, (real1_s)(1.0f / 4.0f))), qubit, ancilla);

    // Uncompute
    H(qubit);

    return ancilla;
}

} // namespace Qrack
