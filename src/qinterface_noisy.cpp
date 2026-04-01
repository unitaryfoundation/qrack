//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

namespace Qrack {

QInterfaceNoisy::QInterfaceNoisy(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, const bitCapInt& initState,
    qrack_rand_gen_ptr rgp, const complex& phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem,
    int64_t deviceId, bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh, bool useTurbo)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , logFidelity(0.0)
    , noiseParam(ONE_R1_F / 100)
    , engines(eng)
{
    engine = CreateQuantumInterface(engines, qBitCount, initState, rgp, phaseFac, doNorm, randGlobalPhase, useHostMem,
        deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, sep_thresh, useTurbo);
}
} // namespace Qrack
