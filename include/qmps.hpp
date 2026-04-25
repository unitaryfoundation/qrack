//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// The initial draft of qmps.hpp and qmps.cpp was produced by (Anthropic) Claude.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "qinterface.hpp"

namespace Qrack {

// MPS tensor for qubit i: shape [chi_left, 2, chi_right]
// chi = bond dimension (max_bond controls approximation quality)
struct MPSTensor {
    size_t chi_l, chi_r;
    std::vector<complex> data; // chi_l * 2 * chi_r, row-major

    MPSTensor(size_t cl, size_t cr)
        : chi_l(cl)
        , chi_r(cr)
        , data(cl * 2 * cr, 0.0)
    {
    }

    complex& at(size_t il, size_t s, size_t ir) { return data[il * 2 * chi_r + s * chi_r + ir]; }
};

class QMPS;
typedef std::shared_ptr<QMPS> QMPSPtr;

class QMPS : public QInterface {
protected:
    std::vector<MPSTensor> tensors;
    size_t max_bond;
    QInterfacePtr fallback; // delegate for unimplemented ops

    // SVD truncation to max_bond after two-site gate
    void truncate_svd(size_t site);
    void decompose_bond_after_measurement(bitLenInt qubit);
    void apply_two_site_gate(bitLenInt site_l, bitLenInt site_r, const complex mtrx[4U], bool is_controlled);
    void apply_mcmtrx_via_swap(const std::vector<bitLenInt>& controls, const complex mtrx[4U], bitLenInt target);

public:
    QMPS(bitLenInt n, size_t max_bond = 0, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false,
        bool useHardwareRNG = true, bool randomGlobalPhase = true, real1_f norm_thresh = REAL1_EPSILON);

    // ---- Core state ops (pure virtual in QInterface) ----

    void SetPermutation(const bitCapInt& perm, const complex& phaseFac = CMPLX_DEFAULT_ARG) override;
    void SetQuantumState(const complex* inputState) override;
    void GetQuantumState(complex* outputState) override;
    void GetProbs(real1* outputProbs) override;
    complex GetAmplitude(const bitCapInt& perm) override;
    void SetAmplitude(const bitCapInt& perm, const complex& amp) override;

    // ---- Single-qubit gates (native MPS ops) ----
    void Mtrx(const complex* mtrx, bitLenInt qubit) override;
    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true) override;

    // ---- Two-qubit gates (native for nearest-neighbor) ----
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target) override;

    // ---- Fallback for everything else ----
    // Compose/Decompose needed for QUnit interop
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start) override;
    void Decompose(bitLenInt start, QInterfacePtr dest) override;
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length) override
    {
        QMPSPtr dest = std::make_shared<QMPS>(engines, length, ZERO_BCI, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
            separabilityThreshold);

        Decompose(start, dest);

        return dest;
    }
    void Dispose(bitLenInt start, bitLenInt length) override;
    void Dispose(bitLenInt start, bitLenInt length, const bitCapInt& disposedPerm) override
    {
        // Permutation-eigenstate dispose: same as above since we know the
        // state of the disposed qubits exactly — no probability accumulation needed.
        Dispose(start, length);
    }
    bitLenInt Allocate(bitLenInt start, bitLenInt length) override;

    // Clone
    QInterfacePtr Clone() override;

    real1_f SumSqrDiff(QInterfacePtr toCompare) {
        
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) {};
    void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F) {};

};

} // namespace Qrack
