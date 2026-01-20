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
#pragma once

#include "common/qrack_types.hpp"

struct _QrackTimeEvolveOpHeader {
    unsigned target;
    unsigned controlLen;
    unsigned controls[32U];
};

namespace Qrack {

typedef std::shared_ptr<complex> BitOp;

/**
 * A Hamiltonian can be specified in terms of arbitrary controlled single bit gates, each one an "HamiltonianOp."
 */
struct HamiltonianOp {
    bitLenInt targetBit;
    bool anti;
    bool uniform;
    BitOp matrix;
    std::vector<bitLenInt> controls;
    std::vector<bool> toggles;

    HamiltonianOp()
        : targetBit(0U)
        , anti(false)
        , uniform(false)
        , matrix{ nullptr }
        , controls()
        , toggles()
    {
    }

    HamiltonianOp(bitLenInt target, BitOp mtrx)
        : targetBit(target)
        , anti(false)
        , uniform(false)
        , matrix(mtrx)
        , controls()
        , toggles()
    {
    }

    HamiltonianOp(const std::vector<bitLenInt>& ctrls, bitLenInt target, BitOp mtrx, bool antiCtrled = false,
        const std::vector<bool>& ctrlToggles = std::vector<bool>())
        : targetBit(target)
        , anti(antiCtrled)
        , uniform(false)
        , matrix(mtrx)
        , controls(ctrls)
        , toggles(ctrlToggles)
    {
    }
};

struct UniformHamiltonianOp : HamiltonianOp {
    UniformHamiltonianOp(const std::vector<bitLenInt>& ctrls, bitLenInt target, BitOp mtrx)
        : HamiltonianOp(ctrls, target, mtrx)
    {
        uniform = true;
    }

    UniformHamiltonianOp(const _QrackTimeEvolveOpHeader& teoh, double* mtrx)
        : HamiltonianOp()
    {
        targetBit = (bitLenInt)(teoh.target);

        const bitLenInt controlLen = (bitLenInt)teoh.controlLen;
        controls = std::vector<bitLenInt>(teoh.controls, teoh.controls + controlLen);

        uniform = true;

        bitCapIntOcl mtrxTermCount = ((bitCapIntOcl)1U << controlLen) * 4U;
        BitOp m(new complex[mtrxTermCount], std::default_delete<complex[]>());
        matrix = std::move(m);
        for (bitCapIntOcl i = 0U; i < mtrxTermCount; ++i) {
            matrix.get()[i] = complex((real1)mtrx[i * 2U], (real1)mtrx[(i * 2U) + 1U]);
        }
    }
};

/**
 * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are
 * applied by left-multiplication in low-to-high vector index order on the state vector.
 *
 * To specify Hamiltonians with interaction terms, arbitrary sets of control bits may be specified for each term, in
 * which case the term is acted only if the (superposable) control bits are true. The "antiCtrled" bool flips the
 * overall convention, such that the term is acted only if all control bits are false. Additionally, for a combination
 * of control bits and "anti-control" bits, an array of booleans, "ctrlToggles," of length "ctrlLen" may be specified
 * that flips the activation state for each control bit, (relative the global anti- on/off convention,) without altering
 * the state of the control bits.
 *
 * The point of this "toggle" behavior is to allow enumeration of arbitrary local Hamiltonian terms with permutations of
 * a set of control bits. For example, a Hamiltonian might represent an array of local electromagnetic potential wells.
 * If there are 4 wells, each with independent potentials, control "toggles" could be used on two control bits, to
 * enumerate all four permutations of two control bits with four different local Hamiltonian terms.
 *
 * \warning Hamiltonian components might not commute.
 *
 * As a general point of linear algebra, where A and B are linear operators, e^{i * (A + B) * t} = e^{i * A * t} *
 * e^{i * B * t} might NOT hold, if the operators A and B do not commute. As a rule of thumb, A will commute with B
 * at least in the case that A and B act on entirely different sets of qubits. However, for defining the intended
 * Hamiltonian, the programmer can be guaranteed that the exponential factors will be applied right-to-left, by left
 * multiplication, in the order e^(i * H_(N - 1) * t) * e^(i * H_(N - 2) * t) * ... e^(i * H_0 * t) * |psi>. (For
 * example, if A and B are single bit gates acting on the same bit, form their composition into one gate by the intended
 * right-to-left fusion and apply them as a single HamiltonianOp.)
 */
typedef std::shared_ptr<HamiltonianOp> HamiltonianOpPtr;
typedef std::vector<HamiltonianOpPtr> Hamiltonian;
} // namespace Qrack
