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
#include "common/qrack_functions.hpp"

namespace Qrack {

struct MpsShard;
typedef std::shared_ptr<MpsShard> MpsShardPtr;

struct MpsShard {
    complex gate[4U];

    MpsShard()
        : gate{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX }
    {
        // Intentionally left blank
    }

    MpsShard(const complex g[4U]) { std::copy(g, g + 4U, gate); }

    MpsShardPtr Clone() { return std::make_shared<MpsShard>(gate); }

    void Compose(const complex g[4U])
    {
        complex o[4U];
        std::copy(gate, gate + 4U, o);
        mul2x2((complex*)g, o, gate);
        if (IsPhase()) {
            gate[1U] = ZERO_R1;
            gate[2U] = ZERO_R1;
            gate[0U] /= abs(gate[0U]);
            gate[3U] /= abs(gate[3U]);
        } else if (IsInvert()) {
            gate[0U] = ZERO_R1;
            gate[3U] = ZERO_R1;
            gate[1U] /= abs(gate[1U]);
            gate[2U] /= abs(gate[2U]);
        }
    }

    bool IsPhase() { return (norm(gate[1U]) <= FP_NORM_EPSILON) && (norm(gate[2U]) <= FP_NORM_EPSILON); }

    bool IsInvert() { return (norm(gate[0U]) <= FP_NORM_EPSILON) && (norm(gate[3U]) <= FP_NORM_EPSILON); }

    bool IsHPhase()
    {
        return ((norm(gate[0U] - gate[1U]) <= FP_NORM_EPSILON) && (norm(gate[2U] + gate[3U]) <= FP_NORM_EPSILON));
    }

    bool IsHInvert()
    {
        return ((norm(gate[0U] + gate[1U]) <= FP_NORM_EPSILON) && (norm(gate[2U] - gate[3U]) <= FP_NORM_EPSILON));
    }

    bool IsIdentity() { return IsPhase() && (norm(gate[0U] - gate[3U]) <= FP_NORM_EPSILON); }

    bool IsX(bool randGlobalPhase = true)
    {
        return IsInvert() && (norm(gate[1U] - gate[2U]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX - gate[1U]) <= FP_NORM_EPSILON));
    }

    bool IsY(bool randGlobalPhase = true)
    {
        return IsInvert() && (norm(gate[1U] + gate[2U]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX + gate[1U]) <= FP_NORM_EPSILON));
    }

    bool IsZ(bool randGlobalPhase = true)
    {
        return IsPhase() && (norm(gate[0U] + gate[3U]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX - gate[0U]) <= FP_NORM_EPSILON));
    }

    bool IsH()
    {
        return (norm(SQRT1_2_R1 - gate[0U]) <= FP_NORM_EPSILON) && (norm(SQRT1_2_R1 - gate[1U]) <= FP_NORM_EPSILON) &&
            (norm(SQRT1_2_R1 - gate[2U]) <= FP_NORM_EPSILON) && (norm(SQRT1_2_R1 + gate[3U]) <= FP_NORM_EPSILON);
    }
};

} // namespace Qrack
