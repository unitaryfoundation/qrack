//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbinary_decision_tree_node.hpp"

#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

void QBinaryDecisionTreeNode::Prune(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Prune recursively to depth.
    branches[0]->Prune(depth - 1U);
    if (b0 != b1) {
        branches[1]->Prune(depth - 1U);
    }

    complex phaseFac = std::polar(ONE_R1, (real1)(IS_NORM_0(b0->scale) ? std::arg(b1->scale) : std::arg(b0->scale)));
    scale *= phaseFac;
    b0->scale /= phaseFac;
    if (b0 == b1) {
        // Combining branches is the only other thing we try, below.
        return;
    }
    b1->scale /= phaseFac;

    // Now, we try to combine pointers to equivalent branches.

    bool isSameAtTop = true;
    size_t bit;
    bitLenInt j, k;
    bitCapInt depthPow = ONE_BCI << depth;
    complex scale0, scale1, prevScale0, prevScale1;
    QBinaryDecisionTreeNodePtr leaf0, leaf1;
    std::vector<complex> branch1Scales(depth);
    for (bitCapInt i = 0; i < depthPow; i++) {
        leaf0 = b0;
        leaf1 = b1;

        prevScale0 = leaf0->scale;
        prevScale1 = leaf1->scale;

        scale0 = ONE_CMPLX;
        scale1 = ONE_CMPLX;

        for (j = 0; j < depth; j++) {
            bit = SelectBit(i, j);

            if (leaf0) {
                prevScale0 = leaf0->scale;
                scale0 *= prevScale0;
                leaf0 = leaf0->branches[bit];
            }

            if (leaf1) {
                prevScale1 = leaf1->scale;
                scale1 *= prevScale1;
                leaf1 = leaf1->branches[bit];
            }

            if (leaf0 == leaf1) {
                break;
            }
        }

        if ((leaf0 != leaf1)) {
            // We can't combine our immediate children within depth.
            isSameAtTop = false;
            continue;
        }

        if (leaf0) {
            scale0 *= leaf0->scale;
            scale1 *= leaf1->scale;
        }

        isSameAtTop &= IS_NORM_0(scale0 - scale1);

        if (!IS_NORM_0(prevScale0 - prevScale1) || IS_NORM_0(prevScale0) || IS_NORM_0(prevScale1)) {
            continue;
        }

        if (!j) {
            if ((b0->branches[0] == b1->branches[0]) && (b0->branches[1] == b1->branches[1])) {
                b1 = b0;
                return;
            }

            continue;
        }

        if (!leaf0) {
            continue;
        }

        leaf0 = b0;
        leaf1 = b1;
        for (k = 0; k < (j - 1U); k++) {
            bit = SelectBit(i, k);
            leaf0 = leaf0->branches[bit];
            leaf1 = leaf1->branches[bit];
        }

        bit = SelectBit(i, k);
        QBinaryDecisionTreeNodePtr& sb0 = leaf0->branches[bit];
        QBinaryDecisionTreeNodePtr& sb1 = leaf1->branches[bit];

        bit = SelectBit(i, (k + 1U));
        bit ^= 1U;
        if (IS_NORM_0(sb0->scale - sb1->scale) && (sb0->branches[bit] == sb1->branches[bit])) {
            sb1 = sb0;
        }
    }

    if (isSameAtTop) {
        // The branches terminate equal, within depth.
        b1 = b0;
    }
}

void QBinaryDecisionTreeNode::Branch(bitLenInt depth, bool isZeroBranch)
{
    if (!depth) {
        return;
    }
    if (!isZeroBranch && IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    if (!b0) {
        b0 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
        b1 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
    } else {
        // Split all clones.
        b0 = b0->ShallowClone();
        b1 = b1->ShallowClone();
    }

    b0->Branch(depth - 1U, isZeroBranch);
    b1->Branch(depth - 1U, isZeroBranch);
}

void QBinaryDecisionTreeNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    b0->Normalize(depth - 1U);
    if (b0 != b1) {
        b1->Normalize(depth - 1U);
    }

    real1 nrm = (real1)sqrt(norm(b0->scale) + norm(b1->scale));
    b0->scale *= ONE_R1 / nrm;
    if (b0 != b1) {
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBinaryDecisionTreeNode::ConvertStateVector(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Depth-first
    depth--;
    b0->ConvertStateVector(depth);
    if (b0 != b1) {
        b1->ConvertStateVector(depth);
    }

    real1 nrm0 = norm(b0->scale);
    real1 nrm1 = norm(b1->scale);

    if ((nrm0 + nrm1) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (nrm0 <= FP_NORM_EPSILON) {
        scale = b1->scale;
        b0->SetZero();
        b1->scale = ONE_CMPLX;
        return;
    }

    if (nrm1 <= FP_NORM_EPSILON) {
        scale = b0->scale;
        b0->scale = ONE_CMPLX;
        b1->SetZero();
        return;
    }

    scale = std::polar((real1)sqrt(nrm0 + nrm1), (real1)std::arg(b0->scale));
    b0->scale /= scale;
    b1->scale /= scale;
}

} // namespace Qrack
