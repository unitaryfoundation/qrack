//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// QPager::LossySaveStateVector / LossyLoadStateVector
//
// By (Anthropic) Claude
//
// TurboQuant save/load for QPager, the multi-GPU paging layer.
//
// ARCHITECTURAL NOTE, stated explicitly because it's the whole reason this
// format differs from QUnit's: QPager's qPages are NOT separable subsystems.
// They are equal-length, index-ordered, contiguous slices of ONE fully
// coherent state vector, distributed across devices purely for memory/
// parallelism reasons (verified directly against QPager::GetQuantumState /
// SetQuantumState: page i always holds amplitudes
// [i*pageMaxQPower, (i+1)*pageMaxQPower), with no subsystem structure to
// preserve). So, unlike qunit_turboquant.cpp, there is no separability
// reasoning here, no logical-qubit-to-subsystem map to reconstruct on load --
// page order alone determines amplitude range. What IS specific to QPager
// and worth preserving is multi-GPU device placement (deviceIDs), which
// QUnit's single-device-oriented version never needed to think about.
//
// Format on disk (binary, in order):
//
//   Header:
//     uint64_t  magic          = 0x5150474554510000  ("QPGETQ\0\0")
//     size_t    qubitCount
//     int       p_default      (TurboQuant block power)
//     int       b_default      (TurboQuant bits)
//     size_t    num_pages
//     size_t    qubits_per_page (uniform across all pages, by construction)
//     size_t    num_device_ids
//     int64_t[num_device_ids]  (original deviceIDs list, for SetDeviceList on load)
//
//   Per page (num_pages records):
//     bool      is_compressed
//     if is_compressed:
//       StateVectorTurboQuant stream (via existing save())
//     else (page too small for one TurboQuant block):
//       complex[pageMaxQPower]  (raw amplitudes, uncompressed)
//
// SCOPE NOTE, stated honestly: on load, this requires the target QPager to
// already have the correct qubitCount. Resizing via Allocate()/Dispose() the
// way qunit_turboquant.cpp does is NOT attempted here -- I have confirmed
// QPager::Allocate/Dispose exist as QPager-specific overrides (not just
// inherited generic behavior), but I have not read their implementation
// bodies, so I'm not confident enough in that path to build on it silently.
// A qubit-count mismatch throws a clear, explicit error instead of
// attempting an unverified resize.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qpager.hpp"
#include "statevector_turboquant.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace Qrack {

// Magic number: "QPGETQ\0\0"
static constexpr uint64_t QPAGER_TQ_MAGIC = 0x5150474554510000ULL;

// ── helpers (identical wire format primitives to qunit_turboquant.cpp,
//    duplicated rather than shared, to keep each file's on-disk format
//    fully self-contained and independently readable) ──────────────────────

static void _write_u64(std::ostream& os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_sz(std::ostream& os, size_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_int(std::ostream& os, int v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_bool(std::ostream& os, bool v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_i64(std::ostream& os, int64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_cmplx(std::ostream& os, const complex& c) { os.write(reinterpret_cast<const char*>(&c), sizeof(c)); }

static uint64_t _read_u64(std::istream& is)
{
    uint64_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static size_t _read_sz(std::istream& is)
{
    size_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static int _read_int(std::istream& is)
{
    int v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static bool _read_bool(std::istream& is)
{
    bool v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static int64_t _read_i64(std::istream& is)
{
    int64_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static complex _read_cmplx(std::istream& is)
{
    complex c;
    is.read(reinterpret_cast<char*>(&c), sizeof(c));
    return c;
}

// ── LossySaveStateVector ─────────────────────────────────────────────────────

void QPager::LossySaveStateVector(std::string f, int p, int b)
{
    const int p_eff = p ? p : (int)qubitCount;
    const int b_eff = b ? b : QRACK_TURBO_BITS;

    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    const size_t BLOCK = (size_t)(1U) << p_eff;
    const bool compress = ((size_t)pagePower >= BLOCK);

    std::ofstream os(f, std::ios::binary);
    if (!os)
        throw std::runtime_error("QPager::LossySaveStateVector: cannot open " + f);

    _write_u64(os, QPAGER_TQ_MAGIC);
    _write_sz(os, (size_t)qubitCount);
    _write_int(os, p_eff);
    _write_int(os, b_eff);
    _write_sz(os, qPages.size());
    _write_sz(os, (size_t)qubitsPerPage());
    _write_sz(os, deviceIDs.size());
    for (const int64_t& did : deviceIDs) {
        _write_i64(os, did);
    }

    // Per-page extraction, using the SAME verified pattern
    // QPager::GetQuantumState itself uses: qPages[i]->GetQuantumState(buffer)
    // gives exactly that page's pagePower amplitudes, in order. Compression
    // decision (compress vs raw) is uniform across all pages here, because
    // pages are uniform size by construction -- unlike QUnit's subsystems,
    // which vary in size and so are decided per-subsystem.
    std::unique_ptr<complex[]> pageBuf(new complex[pagePower]);
    for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
        qPages[i]->GetQuantumState(pageBuf.get());

        _write_bool(os, compress);

        if (compress) {
            StateVectorTurboQuant tq(pagePower, p_eff, b_eff, pageBuf.get());
            tq.save(os);
        } else {
            for (bitCapIntOcl j = 0U; j < pagePower; ++j) {
                _write_cmplx(os, pageBuf[j]);
            }
        }
    }

    os.close();
}

// ── LossyLoadStateVector ─────────────────────────────────────────────────────

void QPager::LossyLoadStateVector(std::string f)
{
    std::ifstream is(f, std::ios::binary);
    if (!is)
        throw std::runtime_error("QPager::LossyLoadStateVector: cannot open " + f);

    const uint64_t magic = _read_u64(is);
    if (magic != QPAGER_TQ_MAGIC) {
        // Not our format -- fall back to the base QEngine implementation,
        // which handles a flat TurboQuant stream (matching
        // qunit_turboquant.cpp's own fallback behavior for consistency).
        is.close();
        QEngine::LossyLoadStateVector(f);
        return;
    }

    const size_t saved_qubits = _read_sz(is);
    _read_int(is); // p_saved (unused directly -- each page records its own
                    // is_compressed flag, and StateVectorTurboQuant::load
                    // reads its own parameters from its own stream)
    _read_int(is); // b_saved (unused, same reason)
    const size_t num_pages = _read_sz(is);
    const size_t saved_qubits_per_page = _read_sz(is);
    const size_t num_device_ids = _read_sz(is);
    std::vector<int64_t> saved_device_ids(num_device_ids);
    for (size_t i = 0U; i < num_device_ids; ++i) {
        saved_device_ids[i] = _read_i64(is);
    }

    // SCOPE NOTE (see file header): require exact qubit-count match rather
    // than attempt an unverified Allocate()/Dispose() resize path.
    if ((size_t)qubitCount != saved_qubits) {
        throw std::runtime_error(
            "QPager::LossyLoadStateVector: qubit count mismatch (this QPager has " + std::to_string(qubitCount) +
            ", file has " + std::to_string(saved_qubits) +
            "). Resizing on load is not implemented in this version -- "
            "construct a QPager with the correct qubitCount before loading.");
    }

    if (num_pages != qPages.size()) {
        throw std::runtime_error(
            "QPager::LossyLoadStateVector: page count mismatch (this QPager has " + std::to_string(qPages.size()) +
            " pages, file has " + std::to_string(num_pages) +
            "). This can happen if CombineEngines()/SeparateEngines() was called at a different threshold "
            "than when the file was saved -- call SeparateEngines() to match saved_qubits_per_page="
            + std::to_string(saved_qubits_per_page) + " before loading.");
    }

    // Restore device placement BEFORE reconstructing page contents, so any
    // freshly-created engines land on the originally-saved devices.
    if (num_device_ids > 0U) {
        SetDeviceList(saved_device_ids);
    }

    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    std::unique_ptr<complex[]> pageBuf(new complex[pagePower]);

    for (size_t i = 0U; i < num_pages; ++i) {
        const bool compressed = _read_bool(is);

        if (compressed) {
            StateVectorTurboQuantPtr tq = StateVectorTurboQuant::load(is);
            tq->copy_out(pageBuf.get());
        } else {
            for (bitCapIntOcl j = 0U; j < pagePower; ++j) {
                pageBuf[j] = _read_cmplx(is);
            }
        }

        qPages[i]->SetQuantumState(pageBuf.get());
    }

    is.close();
}

} // namespace Qrack
