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

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateVec) {                                                                                                   \
        return;                                                                                                        \
    }

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineCPU::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::ROL range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    shift %= length;

    if (!shift) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ regMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl regInt = (lcv & regMask) >> start;
        const bitCapIntOcl outInt = (regInt >> (length - shift)) | ((regInt << shift) & lengthMask);
        nStateVec->write((outInt << start) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

#if ENABLE_ALU
/// Add integer (without sign)
void QEngineCPU::INC(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INC range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthMask = pow2MaskOcl(length);
    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd & lengthMask;

    if (!toAddOcl) {
        return;
    }

    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = (inOutInt + toAddOcl) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with controls)
void QEngineCPU::CINC(
    const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    CHECK_ZERO_SKIP();

    if (controls.empty()) {
        return INC(toAdd, inOutStart, length);
    }

    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::CINC range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineCPU::CINC control is out-of-bounds!");

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd & lengthMask;

    if (!toAddOcl) {
        return;
    }

    std::vector<bitCapIntOcl> controlPowers(controls.size());
    bitCapIntOcl controlMask = 0;
    for (size_t i = 0; i < controls.size(); ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.begin(), controlPowers.end());

    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->copy(stateVec);

    par_for_mask(0, maxQPowerOcl, controlPowers, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = (inOutInt + toAddOcl) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes | controlMask, stateVec->read(lcv | controlMask));
    });

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with carry)
void QEngineCPU::INCDECC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INCDECC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCDECC carryIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod & lengthMask;

    if (!toModOcl) {
        return;
    }

    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryIndex), 1U, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        const bitCapIntOcl outRes = (outInt < lengthPower)
            ? ((outInt << inOutStart) | otherRes)
            : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        nStateVec->write(outRes, stateVec->read(lcv));
    });
    ResetStateVec(nStateVec);
}

/**
 * Add an integer to the register, with sign and without carry. Because the
 * register length is an arbitrary number of bits, the sign bit position on the
 * integer to add is variable. Hence, the integer to add is specified as cast
 * to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QEngineCPU::INCS(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INCS range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCS overflowIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd & lengthMask;

    if (!toAddOcl) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl signMask = pow2Ocl(length - 1U);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAddOcl;
        const bitCapIntOcl outRes = (outInt < lengthPower) ? ((outInt << inOutStart) | otherRes)
                                                           : (((outInt - lengthPower) << inOutStart) | otherRes);
        const bool isOverflow = isOverflowAdd(inOutInt, toAddOcl, signMask, lengthPower);
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INCDECSC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCDECSC carryIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod & lengthMask;

    if (!toModOcl) {
        return;
    }

    const bitCapIntOcl signMask = pow2Ocl(length - 1U);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, carryMask, 1U, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl inInt = toModOcl;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        const bitCapIntOcl outRes = (outInt < lengthPower)
            ? ((outInt << inOutStart) | otherRes)
            : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        const bool isOverflow = isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        if (isOverflow) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(
    const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INCDECSC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCDECSC carryIndex is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCDECSC overflowIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - 1U;
    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod & lengthMask;

    if (!toModOcl) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl signMask = pow2Ocl(length - 1U);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, carryMask, 1U, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl inInt = toModOcl;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        const bitCapIntOcl outRes = (outInt < lengthPower)
            ? ((outInt << inOutStart) | otherRes)
            : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        const bool isOverflow = isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::MULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::MULDIV range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::MULDIV range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl toMulOcl = (bitCapIntOcl)toMul;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl highMask = lowMask << length;
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryStart), length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl mulInt = ((lcv & inOutMask) >> inOutStart) * toMulOcl;
        const bitCapIntOcl mulRes =
            ((mulInt & lowMask) << inOutStart) | (((mulInt & highMask) >> length) << carryStart) | otherRes;
        nStateVec->write(outFn(lcv, mulRes), stateVec->read(inFn(lcv, mulRes)));
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    SetReg(carryStart, length, ZERO_BCI);

    if (bi_compare_0(toMul) == 0) {
        return SetReg(inOutStart, length, ZERO_BCI);
    }

    if (bi_compare_1(toMul) == 0) {
        return;
    }

    MULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; }, toMul, inOutStart, carryStart, length);
}

void QEngineCPU::DIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (bi_compare_0(toDiv) == 0) {
        throw std::invalid_argument("DIV by zero");
    }

    if (bi_compare_1(toDiv) == 0) {
        return;
    }

    MULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; }, toDiv, inOutStart, carryStart, length);
}

void QEngineCPU::CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length, const std::vector<bitLenInt>& controls)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::CMULDIV range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::CMULDIV range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineCPU::CMULDIV control is out-of-bounds!");

    CHECK_ZERO_SKIP();

    const bitCapIntOcl toMulOcl = (bitCapIntOcl)toMul;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl highMask = lowMask << length;
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::vector<bitCapIntOcl> skipPowers(controls.size() + length);
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controls.size()]);
    bitCapIntOcl controlMask = 0;
    for (size_t i = 0; i < controls.size(); ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; ++i) {
        skipPowers[i + controls.size()] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.begin(), skipPowers.end());

    bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inOutMask | carryMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_mask(0, maxQPowerOcl, skipPowers, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl mulInt = ((lcv & inOutMask) >> inOutStart) * toMulOcl;
        const bitCapIntOcl mulRes = ((mulInt & lowMask) << inOutStart) |
            (((mulInt & highMask) >> length) << carryStart) | otherRes | controlMask;
        const bitCapIntOcl origRes = lcv | controlMask;
        nStateVec->write(outFn(origRes, mulRes), stateVec->read(inFn(origRes, mulRes)));

        nStateVec->write(lcv, stateVec->read(lcv));
        bitCapIntOcl partControlMask;
        for (bitCapIntOcl j = 1U; j < pow2MaskOcl(controls.size()); ++j) {
            partControlMask = 0;
            for (size_t k = 0; k < controls.size(); ++k) {
                if ((j >> k) & 1U) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
        }
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMUL(const bitCapInt& toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (controls.empty()) {
        return MUL(toMul, inOutStart, carryStart, length);
    }

    SetReg(carryStart, length, ZERO_BCI);

    if (bi_compare_0(toMul) == 0) {
        return SetReg(inOutStart, length, ZERO_BCI);
    }

    if (bi_compare_1(toMul) == 0) {
        return;
    }

    CMULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; }, toMul, inOutStart, carryStart, length,
        controls);
}

void QEngineCPU::CDIV(const bitCapInt& toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    if (controls.empty()) {
        return DIV(toDiv, inOutStart, carryStart, length);
    }

    if (bi_compare_0(toDiv) == 0) {
        throw std::invalid_argument("CDIV by zero");
    }

    if (bi_compare_1(toDiv) == 0) {
        return;
    }

    CMULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; }, toDiv, inOutStart, carryStart, length,
        controls);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bool& inverse)
{
    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::ModNOut range is out-of-bounds!");
    }

    if (isBadBitRange(outStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::ModNOut range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl modNOcl = (bitCapIntOcl)modN;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inMask = lowMask << inStart;
    const bitCapIntOcl modMask = (isPowerOfTwo(modN) ? modNOcl : pow2Ocl(log2Ocl(modNOcl) + 1U)) - 1U;
    const bitCapIntOcl outMask = modMask << outStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inMask | outMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, pow2Ocl(outStart), length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inRes = lcv & inMask;
        const bitCapIntOcl outRes = (kernelFn(inRes >> inStart) % modNOcl) << outStart;
        if (inverse) {
            nStateVec->write(lcv, stateVec->read(inRes | outRes | otherRes));
        } else {
            nStateVec->write(inRes | outRes | otherRes, stateVec->read(lcv));
        }
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MULModNOut(
    const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    SetReg(outStart, length, ZERO_BCI);

    if (bi_compare_0(toMod) == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    ModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length);
}

void QEngineCPU::IMULModNOut(
    const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (bi_compare_0(toMod) == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    ModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length, true);
}

void QEngineCPU::POWModNOut(
    const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (bi_compare_1(toMod) == 0) {
        return SetReg(outStart, length, ONE_BCI);
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    ModNOut(
        [&toModOcl](const bitCapIntOcl& inInt) { return intPowOcl(toModOcl, inInt); }, modN, inStart, outStart, length);
}

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const std::vector<bitLenInt>& controls, const bool& inverse)
{
    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::ModNOut range is out-of-bounds!");
    }

    if (isBadBitRange(outStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::ModNOut range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineCPU::ModNOut control is out-of-bounds!");

    CHECK_ZERO_SKIP();

    const bitCapIntOcl modNOcl = (bitCapIntOcl)modN;
    const bitCapIntOcl lowPower = pow2Ocl(length);
    const bitCapIntOcl lowMask = lowPower - 1U;
    const bitCapIntOcl inMask = lowMask << inStart;
    const bitCapIntOcl outMask = lowMask << outStart;

    std::vector<bitCapIntOcl> skipPowers(controls.size() + length);
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controls.size()]);
    bitCapIntOcl controlMask = 0;
    for (size_t i = 0; i < controls.size(); ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; ++i) {
        skipPowers[i + controls.size()] = pow2Ocl(outStart + i);
    }
    std::sort(skipPowers.begin(), skipPowers.end());

    bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ (inMask | outMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_mask(0, maxQPowerOcl, skipPowers, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inRes = lcv & inMask;
        const bitCapIntOcl outRes = (kernelFn(inRes >> inStart) % modNOcl) << outStart;

        if (inverse) {
            nStateVec->write(lcv | controlMask, stateVec->read(inRes | outRes | otherRes | controlMask));
        } else {
            nStateVec->write(inRes | outRes | otherRes | controlMask, stateVec->read(lcv | controlMask));
        }
        nStateVec->write(lcv, stateVec->read(lcv));

        for (bitCapIntOcl j = 1U; j < pow2MaskOcl(controls.size()); ++j) {
            bitCapIntOcl partControlMask = 0;
            for (size_t k = 0; k < controls.size(); ++k) {
                if ((j >> k) & 1U) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
        }
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMULModNOut(const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
    bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        return MULModNOut(toMod, modN, inStart, outStart, length);
    }

    SetReg(outStart, length, ZERO_BCI);

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    CModNOut(
        [&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length, controls);
}

void QEngineCPU::CIMULModNOut(const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
    bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (controls.empty()) {
        return IMULModNOut(toMod, modN, inStart, outStart, length);
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    CModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length,
        controls, true);
}

void QEngineCPU::CPOWModNOut(const bitCapInt& toMod, const bitCapInt& modN, bitLenInt inStart, bitLenInt outStart,
    bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (controls.empty()) {
        return POWModNOut(toMod, modN, inStart, outStart, length);
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    CModNOut([&toModOcl](const bitCapIntOcl& inInt) { return intPowOcl(toModOcl, inInt); }, modN, inStart, outStart,
        length, controls);
}

#if ENABLE_BCD
/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(const bitCapInt& toAdd, bitLenInt inOutStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INC range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length >> 2U;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toAdd %= maxPow;

    if (!toAdd) {
        return;
    }

    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd;
    const bitCapIntOcl inOutMask = bitRegMaskOcl(inOutStart, length);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl partToAdd = toAddOcl;
        bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        std::unique_ptr<int8_t[]> nibbles(new int8_t[nibbleCount]);
        bool isValid = true;
        for (bitLenInt j = 0; j < nibbleCount; ++j) {
            int8_t test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4UL;
            int8_t test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapIntOcl outInt = 0;
            for (bitLenInt j = 0; j < nibbleCount; ++j) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        ++(nibbles[j + 1]);
                    }
                }
                outInt |= (bitCapIntOcl)nibbles[j] << (j * 4U * ONE_BCI);
            }
            nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add BCD integer (without sign, with carry)
void QEngineCPU::INCDECBCDC(const bitCapInt& toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::INCDECBCDC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::INCDECBCDC carryIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length >> 2U;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;

    if (!toMod) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    const bitCapIntOcl inOutMask = bitRegMaskOcl(inOutStart, length);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryIndex), ONE_BCI, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl partToAdd = toModOcl;
        bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        int test1, test2;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;

        test1 = (int)(inOutInt & 15UL);
        inOutInt >>= 4U * ONE_BCI;
        test2 = (int)(partToAdd % 10);
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (bitLenInt j = 1; j < nibbleCount; ++j) {
            test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4U * ONE_BCI;
            test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapIntOcl outInt = 0;
            bitCapIntOcl outRes = 0;
            bitCapIntOcl carryRes = 0;
            for (bitLenInt j = 0; j < nibbleCount; ++j) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        ++(nibbles[j + 1]);
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= (bitCapIntOcl)nibbles[j] << (j * 4U * ONE_BCI);
            }
            outRes = (outInt << inOutStart) | otherRes | carryRes;
            nStateVec->write(outRes, stateVec->read(lcv));
            outRes ^= carryMask;
            nStateVec->write(outRes, stateVec->read(lcv | carryMask));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
            nStateVec->write(lcv | carryMask, stateVec->read(lcv | carryMask));
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}
#endif

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values, bool resetValue)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (!stateVec) {
        return ZERO_BCI;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, ZERO_BCI);
    }

    const bitLenInt valueBytes = (valueLength + 7U) >> 3U;
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl skipPower = pow2Ocl(valueStart);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    ParallelFunc fn;
    if (valueBytes == 1) {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)values[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else if (valueBytes == 2) {
        uint16_t* inputIntPtr = (uint16_t*)values;
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)inputIntPtr[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else if (valueBytes == 4) {
        uint32_t* inputIntPtr = (uint32_t*)values;
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)inputIntPtr[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl inputInt = (lcv & inputMask) >> indexStart;
            bitCapIntOcl outputInt = 0;
            for (bitCapIntOcl j = 0; j < valueBytes; ++j) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
            bitCapIntOcl outputRes = outputInt << valueStart;
            nStateVec->write(outputRes | lcv, stateVec->read(lcv));
        };
    }

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, skipPower, 0), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, valueLength, fn);
    }

    ResetStateVec(nStateVec);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return ZERO_BCI;
#endif
}

/// Add based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IndexedADC carryIndex is out-of-bounds!");
    }

    if (!stateVec) {
        return ZERO_BCI;
    }

    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are ADded with Carry (ADC) to values entangled in the "outputStart" register with the "inputStart" register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry has to first to be measured for its input value.
    bitCapIntOcl carryIn = 0;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 1;
        X(carryIndex);
    }

    Finish();

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    const bitLenInt valueBytes = (valueLength + 7U) >> 3U;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl skipPower = pow2Ocl(carryIndex);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        const bitCapIntOcl otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        const bitCapIntOcl inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        const bitCapIntOcl inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapIntOcl outputRes = lcv & outputMask;

        // Maintaining the entanglement, we add the classical input value
        // corresponding with the state of the "inputStart" register to
        // "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapIntOcl outputInt = 0;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (valueBytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < valueBytes; ++j) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt += (outputRes >> valueStart) + carryIn;

        // If we exceed max char, we subtract 256 and entangle the carry as
        // set.
        bitCapIntOcl carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }
        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << valueStart;

        nStateVec->write(outputRes | inputRes | otherRes | carryRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, skipPower, 0), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, 1, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return ZERO_BCI;
#endif
}

/// Subtract based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::IndexedLDA range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IndexedSBC carryIndex is out-of-bounds!");
    }

    if (!stateVec) {
        return ZERO_BCI;
    }

    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are SuBtracted with Carry (SBC) from values entangled in the "outputStart" register with the "inputStart"
    // register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry (or "borrow") has to first to be measured for its input value.
    bitCapIntOcl carryIn = 1;
    if (M(carryIndex)) {
        // If the carry is set, we borrow 1 going in. We always initially clear the carry after testing for borrow in.
        carryIn = 0;
        X(carryIndex);
    }

    Finish();

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    const bitLenInt valueBytes = (valueLength + 7U) >> 3U;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl skipPower = pow2Ocl(carryIndex);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        const bitCapIntOcl otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        const bitCapIntOcl inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        const bitCapIntOcl inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapIntOcl outputRes = lcv & outputMask;

        // Maintaining the entanglement, we subtract the classical input
        // value corresponding with the state of the "inputStart" register
        // from "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapIntOcl outputInt = 0;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (valueBytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < valueBytes; ++j) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt = (outputRes >> valueStart) + (lengthPower - (outputInt + carryIn));

        // If our subtractions results in less than 0, we add 256 and
        // entangle the carry as set.  (Since we're using unsigned types,
        // we start by adding 256 with the carry, and then subtract 256 and
        // clear the carry if we don't have a borrow-out.)
        bitCapIntOcl carryRes = 0;

        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << valueStart;

        nStateVec->write(outputRes | inputRes | otherRes | carryRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, skipPower, 0), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, valueLength, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return ZERO_BCI;
#endif
}

/// Transform a length of qubit register via lookup through a hash table.
void QEngineCPU::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::Hash range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitLenInt bytes = (length + 7U) >> 3U;
    const bitCapIntOcl inputMask = bitRegMaskOcl(start, length);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl inputRes = lcv & inputMask;
        const bitCapIntOcl inputInt = inputRes >> start;
        bitCapIntOcl outputInt = 0;
        if (bytes == 1) {
            outputInt = values[inputInt];
        } else if (bytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (bytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < bytes; ++j) {
                outputInt |= (bitCapIntOcl)values[inputInt * bytes + j] << (8U * j);
            }
        }
        bitCapIntOcl outputRes = outputInt << start;
        nStateVec->write(outputRes | (lcv & ~inputRes), stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    if (inputBit1 >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::FullAdd inputBit1 is out-of-bounds!");
    }

    if (inputBit2 >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::FullAdd inputBit2 is out-of-bounds!");
    }

    if (carryInSumOut >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::FullAdd carryInSumOut is out-of-bounds!");
    }

    if (carryOut >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::FullAdd carryOut is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl input1Mask = pow2Ocl(inputBit1);
    const bitCapIntOcl input2Mask = pow2Ocl(inputBit2);
    const bitCapIntOcl carryInSumOutMask = pow2Ocl(carryInSumOut);
    const bitCapIntOcl carryOutMask = pow2Ocl(carryOut);

    std::vector<bitCapIntOcl> qPowers{ carryInSumOutMask, carryOutMask };
    std::sort(qPowers.begin(), qPowers.end());

    Finish();

    par_for_mask(0, maxQPowerOcl, qPowers, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // Carry-in, sum bit in
        const complex ins0c0 = stateVec->read(lcv);
        const complex ins0c1 = stateVec->read(lcv | carryInSumOutMask);
        const complex ins1c0 = stateVec->read(lcv | carryOutMask);
        const complex ins1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        const bool aVal = (lcv & input1Mask) != 0;
        const bool bVal = (lcv & input2Mask) != 0;

        // Carry-out, sum bit out
        complex outs0c0, outs0c1, outs1c0, outs1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                outs0c0 = ins0c0;
                outs1c0 = ins0c1;
                // Non-coding:
                outs0c1 = ins1c0;
                outs1c1 = ins1c1;
            } else {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            }
        } else {
            if (!bVal) {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            } else {
                // Coding:
                outs0c1 = ins0c0;
                outs1c1 = ins0c1;
                // Non-coding:
                outs0c0 = ins1c0;
                outs1c0 = ins1c1;
            }
        }

        stateVec->write(lcv, outs0c0);
        stateVec->write(lcv | carryOutMask, outs0c1);
        stateVec->write(lcv | carryInSumOutMask, outs1c0);
        stateVec->write(lcv | carryInSumOutMask | carryOutMask, outs1c1);
    });
}

void QEngineCPU::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    if (inputBit1 >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IFullAdd inputBit1 is out-of-bounds!");
    }

    if (inputBit2 >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IFullAdd inputBit2 is out-of-bounds!");
    }

    if (carryInSumOut >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IFullAdd carryInSumOut is out-of-bounds!");
    }

    if (carryOut >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::IFullAdd carryOut is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl input1Mask = pow2Ocl(inputBit1);
    const bitCapIntOcl input2Mask = pow2Ocl(inputBit2);
    const bitCapIntOcl carryInSumOutMask = pow2Ocl(carryInSumOut);
    const bitCapIntOcl carryOutMask = pow2Ocl(carryOut);

    std::vector<bitCapIntOcl> qPowers{ carryInSumOutMask, carryOutMask };
    std::sort(qPowers.begin(), qPowers.end());

    Finish();

    par_for_mask(0, maxQPowerOcl, qPowers, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // Carry-in, sum bit out
        const complex outs0c0 = stateVec->read(lcv);
        const complex outs0c1 = stateVec->read(lcv | carryOutMask);
        const complex outs1c0 = stateVec->read(lcv | carryInSumOutMask);
        const complex outs1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        const bool aVal = (lcv & input1Mask) != 0;
        const bool bVal = (lcv & input2Mask) != 0;

        // Carry-out, sum bit in
        complex ins0c0, ins0c1, ins1c0, ins1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                ins0c0 = outs0c0;
                ins0c1 = outs1c0;
                // Non-coding:
                ins1c0 = outs0c1;
                ins1c1 = outs1c1;
            } else {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            }
        } else {
            if (!bVal) {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            } else {
                // Coding:
                ins0c0 = outs0c1;
                ins0c1 = outs1c1;
                // Non-coding:
                ins1c0 = outs0c0;
                ins1c1 = outs1c0;
            }
        }

        stateVec->write(lcv, ins0c0);
        stateVec->write(lcv | carryInSumOutMask, ins0c1);
        stateVec->write(lcv | carryOutMask, ins1c0);
        stateVec->write(lcv | carryInSumOutMask | carryOutMask, ins1c1);
    });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::CPhaseFlipIfLess range is out-of-bounds!");
    }

    if (flagIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::CPhaseFlipIfLess flagIndex is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    Dispatch(maxQPowerOcl, [this, greaterPerm, start, length, flagIndex] {
        const bitCapIntOcl regMask = bitRegMaskOcl(start, length);
        const bitCapIntOcl flagMask = pow2Ocl(flagIndex);
        const bitCapIntOcl greaterPermOcl = (bitCapIntOcl)greaterPerm;

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if ((((lcv & regMask) >> start) < greaterPermOcl) & ((lcv & flagMask) == flagMask))
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(const bitCapInt& greaterPerm, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::CPhaseFlipIfLess range is out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    Dispatch(maxQPowerOcl, [this, greaterPerm, start, length] {
        const bitCapIntOcl regMask = bitRegMaskOcl(start, length);
        const bitCapIntOcl greaterPermOcl = (bitCapIntOcl)greaterPerm;

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if (((lcv & regMask) >> start) < greaterPermOcl)
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}
#endif
}; // namespace Qrack
