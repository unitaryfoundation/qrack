//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This class allows access to on-chip RNG capabilities. The class is adapted from these two sources:
// https://codereview.stackexchange.com/questions/147656/checking-if-cpu-supports-rdrand/150230
// https://stackoverflow.com/questions/45460146/how-to-use-intels-rdrand-using-inline-assembly-with-net
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "rdrandwrapper.hpp"

#if ENABLE_DEVRAND
#include <sys/random.h>
#elif ENABLE_RNDFILE
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <sys/types.h>
#include <thread>
#endif

namespace Qrack {

bool getRdRand(unsigned* pv)
{
#if ENABLE_RDRAND || ENABLE_DEVRAND
    const int max_rdrand_tries = 10;
    for (int i = 0; i < max_rdrand_tries; ++i) {
#if ENABLE_DEVRAND
        if (sizeof(unsigned) == getrandom(reinterpret_cast<char*>(pv), sizeof(unsigned), 0))
#else
        if (_rdrand32_step(pv))
#endif
            return true;
    }
#endif
    return false;
}

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
// From http://www.cplusplus.com/forum/unices/3548/
std::vector<std::string> _readDirectoryFileNames(const std::string& path)
{
    std::vector<std::string> result;
    errno = 0;
    DIR* dp = opendir(path.empty() ? "." : path.c_str());
    if (dp) {
        while (true) {
            errno = 0;
            dirent* de = readdir(dp);
            if (de == NULL) {
                break;
            }
            if (std::string(de->d_name) != "." && std::string(de->d_name) != "..") {
                result.push_back(path + "/" + std::string(de->d_name));
            }
        }
        closedir(dp);
        std::sort(result.begin(), result.end());
    }
    return result;
}

std::string _getDefaultRandomNumberFilePath()
{
#if ENABLE_ENV_VARS
    if (getenv("QRACK_RNG_PATH")) {
        std::string toRet = std::string(getenv("QRACK_RNG_PATH"));
        if ((toRet.back() != '/') && (toRet.back() != '\\')) {
#if defined(_WIN32) && !defined(__CYGWIN__)
            toRet += "\\";
#else
            toRet += "/";
#endif
        }
        return toRet;
    }
#endif
#if defined(_WIN32) && !defined(__CYGWIN__)
    return std::string(getenv("HOMEDRIVE") ? getenv("HOMEDRIVE") : "") +
        std::string(getenv("HOMEPATH") ? getenv("HOMEPATH") : "") + "\\.qrack\\rng\\";
#else
    return std::string(getenv("HOME") ? getenv("HOME") : "") + "/.qrack/rng/";
#endif
}

void RandFile::_readNextRandDataFile()
{
    if (dataFile) {
        fclose(dataFile);
    }

    std::string path = _getDefaultRandomNumberFilePath();
    std::vector<std::string> fileNames = _readDirectoryFileNames(path);
    if (fileNames.size() <= fileOffset) {
        throw std::runtime_error("Out of RNG files!");
    }

    while (!(dataFile = fopen(fileNames[fileOffset].c_str(), "r"))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    fileOffset++;
}
#endif

bool RdRandom::SupportsRDRAND()
{
#if ENABLE_RDRAND
    const unsigned flag_RDRAND = (1 << 30);

#if _MSC_VER
    int ex[4];
    __cpuid(ex, 1);

    return ((ex[2] & flag_RDRAND) == flag_RDRAND);
#else
    unsigned eax, ebx, ecx, edx;
    ecx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return ((ecx & flag_RDRAND) == flag_RDRAND);
#endif

#else
    return false;
#endif
}

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
unsigned RandFile::NextRaw()
{
    size_t fSize = 0;
    unsigned v;
    while (fSize < 1) {
        fSize = fread(&v, sizeof(unsigned), 1, dataFile);
        if (fSize < 1) {
            _readNextRandDataFile();
        }
    }

    return v;
}
unsigned RdRandom::NextRaw() { return RandFile::getInstance().NextRaw(); }
#else
unsigned RdRandom::NextRaw()
{
    unsigned v;
    if (!getRdRand(&v)) {
        throw std::runtime_error("Random number generator failed up to retry limit.");
    }

    return v;
}
#endif

real1_f RdRandom::Next()
{
    unsigned v = NextRaw();

    real1_f res = ZERO_R1_F;
    real1_f part = ONE_R1_F;
    for (unsigned i = 0U; i < 32U; i++) {
        part /= 2;
        if ((v >> i) & 1U) {
            res += part;
        }
    }

#if FPPOW > 5
    v = NextRaw();

    for (unsigned i = 0U; i < 32U; i++) {
        part /= 2;
        if ((v >> i) & 1U) {
            res += part;
        }
    }
#endif

    return res;
}
} // namespace Qrack
