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

#pragma once

#include "qfactory.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

/* A quick-and-dirty epsilon for clamping floating point values. */
#define QRACK_TEST_EPSILON 0.5

/*
 * Default engine type to run the tests with. Global because catch doesn't
 * support parameterization.
 */
extern enum Qrack::QInterfaceEngine testEngineType;
extern enum Qrack::QInterfaceEngine testSubEngineType;
extern enum Qrack::QInterfaceEngine testSubSubEngineType;
extern enum Qrack::QInterfaceEngine testSubSubSubEngineType;
extern qrack_rand_gen_ptr rng;
extern bool enable_normalization;
extern bool disable_t_injection;
extern bool disable_reactive_separation;
extern bool disable_terminal_measurement;
extern bool use_host_dma;
extern bool disable_hardware_rng;
extern bool async_time;
extern int device_id;
extern bitLenInt max_qubits;
extern bitLenInt min_qubits;
extern bool single_qubit_run;
extern std::string mOutputFileName;
extern std::ofstream mOutputFile;
extern bool isBinaryOutput;
extern int benchmarkSamples;
extern int benchmarkDepth;
extern int benchmarkMaxMagic;
extern int benchmarkShots;
extern int timeout;
extern std::vector<int64_t> devList;
extern bool optimal;
extern bool optimal_single;

/* Declare the stream-to-probability prior to including catch.hpp. */
namespace Qrack {

inline std::ostream& outputPerBitProbs(std::ostream& os, Qrack::QInterfacePtr qftReg);
inline std::ostream& outputProbableResult(std::ostream& os, Qrack::QInterfacePtr qftReg);
inline std::ostream& outputIndependentBits(std::ostream& os, Qrack::QInterfacePtr qftReg);

inline std::ostream& operator<<(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    if (os.flags() & std::ios_base::showpoint) {
        os.unsetf(std::ios_base::showpoint);
        return outputPerBitProbs(os, qftReg);
    }
    if (os.flags() & std::ios_base::showbase) {
        os.unsetf(std::ios_base::showbase);
        return outputIndependentBits(os, qftReg);
    }
    return outputProbableResult(os, qftReg);
}

inline std::ostream& outputPerBitProbs(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    os << "[\n";

    for (int i = qftReg->GetQubitCount() - 1; i >= 0; i--) {
        os << "\t " << std::setw(2) << i << "]: " << qftReg->Prob(i) << std::endl;
    }
    return os;
}

inline std::ostream& outputProbableResult(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    double maxProb = 0.0;
    bitCapInt maxProbIdx = ZERO_BCI;

    // Iterate through all possible values of the bit array
    for (bitCapInt i = ZERO_BCI; bi_compare(i, qftReg->GetMaxQPower()) < 0; bi_increment(&i, 1U)) {
        double prob = (double)qftReg->ProbAll(i);
        if (prob > maxProb) {
            maxProb = prob;
            maxProbIdx = i;
        }
    }

    os << qftReg->GetQubitCount() << "/";

    // Print the resulting maximum probability bit pattern.
    for (bitCapInt i = (qftReg->GetMaxQPower() >> 1U); bi_compare_0(i) != 0; bi_rshift_ip(&i, 1U)) {
        os << ((bi_compare_0(i & maxProbIdx) == 0) ? "0" : "1");
    }

    // And print the probability, for interest.
    os << ":" << maxProb;

    return os;
}

inline std::ostream& outputIndependentBits(std::ostream& os, Qrack::QInterfacePtr qftReg)
{
    os << "" << qftReg->GetQubitCount() << "/";

    for (int j = qftReg->GetQubitCount() - 1; j >= 0; j--) {
        os << (int)(qftReg->Prob(j) > QRACK_TEST_EPSILON);
    }

    return os;
}

} // namespace Qrack

#include "catch.hpp"

/*
 * A fixture to create a unique QInterface test, of the appropriate type, for
 * each executing test case.
 */
class QInterfaceTestFixture {
protected:
    Qrack::QInterfacePtr qftReg;

public:
    QInterfaceTestFixture();
};

class ProbPattern : public Catch::MatcherBase<Qrack::QInterfacePtr> {
    bitLenInt start;
    bitLenInt length;
    bitCapInt mask;

public:
    ProbPattern(bitLenInt s, bitLenInt l, bitCapInt m)
        : start(s)
        , length(l)
        , mask(m)
    {
    }

    virtual bool match(Qrack::QInterfacePtr const& qftReg) const override
    {
        if (length == 0U) {
            ((ProbPattern*)this)->length = qftReg->GetQubitCount();
        }

        if (length > sizeof(mask) * 8U) {
            WARN("requested length " << length << " larger than possible bitmap " << sizeof(mask) * 8U);
            return false;
        }

        for (bitLenInt j = 0U; j < length; j++) {
            /* Consider anything more than a 50% probability as a '1'. */
            const bool bit = qftReg->Prob(j + start) > QRACK_TEST_EPSILON;
            if (bit == (bi_and_1(mask >> j) == 0U)) {
                return false;
            }
        }
        return true;
    }

    virtual std::string describe() const override
    {
        std::ostringstream ss;
        ss << "matches bit pattern [" << (int)start << "," << start + length << "]: " << (int)length << "/";
        for (int j = (length - 1); j >= 0; j--) {
            ss << bi_and_1(mask >> j);
        }
        return ss.str();
    }
};

inline ProbPattern HasProbability(bitLenInt s, bitLenInt l, bitCapInt m) { return ProbPattern(s, l, m); }
inline ProbPattern HasProbability(bitCapInt m) { return ProbPattern(0, 0, m); }
