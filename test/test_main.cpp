//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "qinterface.hpp"

#include "qfactory.hpp"

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

using namespace Qrack;

/*
 * Default engine type to run the tests with. Global because catch doesn't
 * support parameterization.
 */
enum QInterfaceEngine testEngineType = QENGINE_FIRST;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool disable_opencl = false;
    bool disable_cpu = false;

    using namespace Catch::clara;

    /*
     * Allow disabling running OpenCL tests on the command line, even if
     * supported.
     */
    auto cli = session.cli() |
            Opt(disable_opencl)["--disable-opencl"]("Disable OpenCL even if supported") |
            Opt(disable_cpu)["--disable-cpu"]("Disable the CPU-based implementation tests");

    session.cli(cli);

    /* Set some defaults for convenience. */
    session.configData().useColour = Catch::UseColour::No;
    session.configData().reporterNames = { "compact" };
    session.configData().rngSeed = std::time(0);

    // session.configData().abortAfter = 1;

    /* Parse the command line. */
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    session.config().stream() << "Random Seed: " << session.configData().rngSeed << std::endl;

    int num_failed = 0;

    /* Perform the run against the default (software) variant. */
    if (!disable_cpu) {
        session.config().stream() << "Executing test suite using the CPU Implementation" << std::endl;
        num_failed = session.run();
    }

#if ENABLE_OPENCL
    if (num_failed == 0 && !disable_opencl) {
        session.config().stream() << "Executing test suite using the OpenCL Implementation" << std::endl;
        testEngineType = QENGINE_OPENCL;
        CreateQuantumInterface(testEngineType, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
        num_failed = session.run();
    }
#endif

    return num_failed;
}

QInterfaceTestFixture::QInterfaceTestFixture()
{
    uint32_t rngSeed = Catch::getCurrentContext().getConfig()->rngSeed();

    if (rngSeed == 0) {
        rngSeed = std::time(0);
    }

    std::shared_ptr<std::default_random_engine> rng = std::make_shared<std::default_random_engine>();
    rng->seed(rngSeed);


    qftReg = CreateQuantumInterface(testEngineType, 20, 0, rng);
}
