//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// QUnit::LossySaveStateVector / LossyLoadStateVector
//
// By (Anthropic) Claude
//
// Overrides QInterface's flat state-vector TurboQuant save/load with a
// subsystem-aware version that respects QUnit's separability structure.
//
// Format on disk (binary, in order):
//
//   Header:
//     uint64_t  magic        = 0x5155544E54510000  ("QUNTQ\0\0\0")
//     size_t    qubitCount                          (total logical qubits)
//     int       p_default                           (default TurboQuant block power)
//     int       b_default                           (default TurboQuant bits)
//     size_t    num_subsystems
//
//   Per subsystem (num_subsystems records):
//     size_t    sub_qubit_count                     (qubits in this subsystem)
//     bitCapInt sub_perm                            (permutation if single-qubit clean shard)
//     bool      is_compressed                       (true = TurboQuant; false = raw complex[])
//     if is_compressed:
//       StateVectorTurboQuant stream (via existing save())
//     else (sub too small for TurboQuant block):
//       complex[1 << sub_qubit_count]               (raw amplitudes, uncompressed)
//
//   Logical qubit map:
//     size_t    qubitCount   (repeated for alignment check)
//     For each logical qubit i in [0, qubitCount):
//       size_t  subsystem_index
//       size_t  mapped_index_within_subsystem
//
// Rationale for subsystem-aware format:
//   - Separable subsystem boundaries (different shard.unit pointers) are the
//     natural compression units: each subsystem's state is independent and can
//     be compressed/decompressed independently without entanglement error.
//   - On load, each subsystem is restored as its own QInterface shard, and the
//     QUnit mapping is rebuilt exactly — no spurious entanglement introduced.
//   - Small subsystems (2^n < BLOCK) are stored uncompressed to avoid the
//     TurboQuant rotation overhead dominating at tiny size.
//   - ACE-produced artificial separability boundaries are preserved exactly,
//     which is the whole point of ACE.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qunit.hpp"
#include "statevector_turboquant.hpp"

#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace Qrack {

// Magic number: "QUNTQ\0\0\0"
static constexpr uint64_t QUNIT_TQ_MAGIC = 0x5155544E54510000ULL;

// ── helpers ──────────────────────────────────────────────────────────────────

static void _write_u64(std::ostream& os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_sz(std::ostream& os, size_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_int(std::ostream& os, int v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void _write_bool(std::ostream& os, bool v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
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
static complex _read_cmplx(std::istream& is)
{
    complex c;
    is.read(reinterpret_cast<char*>(&c), sizeof(c));
    return c;
}

// ── LossySaveStateVector ─────────────────────────────────────────────────────

void QUnit::LossySaveStateVector(std::string f, int p, int b)
{
    // Default block power: use caller's p if given, else qubitCount (matching
    // QInterface base behaviour), clamped to at least 1.
    const int p_eff = p ? p : (int)qubitCount;
    const int b_eff = b ? b : QRACK_TURBO_BITS;
    const size_t BLOCK = (size_t)(1U) << p_eff; // amplitudes per TurboQuant block

    // ── 1. Collect unique subsystems in logical qubit order ───────────────
    //
    // Walk shards in logical index order.  Group contiguous shards that share
    // the same unit pointer into one subsystem record.  We do NOT rely on
    // pointer identity alone for ordering — we enumerate in logical order so
    // that the on-disk layout is deterministic and reproducible.

    struct SubsystemRecord {
        QInterfacePtr unit;
        size_t sub_qubit_count; // qubits in this subsystem (= unit->GetQubitCount())
        std::vector<bitLenInt> logical_qubits; // logical indices of qubits in this sub
    };

    // Map from unit pointer to subsystem index (insertion order)
    std::map<QInterfacePtr, size_t> unit_to_sub;
    std::vector<SubsystemRecord> subsystems;

    // Logical qubit → (subsystem_index, mapped_index_within_subsystem)
    std::vector<std::pair<size_t, size_t>> qubit_map(qubitCount);

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        QEngineShard& shard = shards[i];

        if (!shard.unit) {
            // Single-qubit shard not yet entangled — treat as its own 1-qubit subsystem.
            // Create a temporary 1-qubit engine representing this shard's state.
            // We encode it as a raw 2-amplitude record (always uncompressed).
            SubsystemRecord rec;
            rec.unit = nullptr; // signals single-shard-raw
            rec.sub_qubit_count = 1U;
            rec.logical_qubits = { i };
            const size_t sub_idx = subsystems.size();
            subsystems.push_back(std::move(rec));
            qubit_map[i] = { sub_idx, 0U };
        } else {
            auto it = unit_to_sub.find(shard.unit);
            if (it == unit_to_sub.end()) {
                // First time we see this unit
                SubsystemRecord rec;
                rec.unit = shard.unit;
                rec.sub_qubit_count = (size_t)shard.unit->GetQubitCount();
                rec.logical_qubits = { i };
                const size_t sub_idx = subsystems.size();
                unit_to_sub[shard.unit] = sub_idx;
                subsystems.push_back(std::move(rec));
                qubit_map[i] = { sub_idx, (size_t)shard.mapped };
            } else {
                const size_t sub_idx = it->second;
                subsystems[sub_idx].logical_qubits.push_back(i);
                qubit_map[i] = { sub_idx, (size_t)shard.mapped };
            }
        }
    }

    // ── 2. Open file and write header ─────────────────────────────────────
    std::ofstream os(f, std::ios::binary);
    if (!os)
        throw std::runtime_error("QUnit::LossySaveStateVector: cannot open " + f);

    _write_u64(os, QUNIT_TQ_MAGIC);
    _write_sz(os, (size_t)qubitCount);
    _write_int(os, p_eff);
    _write_int(os, b_eff);
    _write_sz(os, subsystems.size());

    // ── 3. Write each subsystem ───────────────────────────────────────────
    for (const SubsystemRecord& rec : subsystems) {
        _write_sz(os, rec.sub_qubit_count);

        if (!rec.unit) {
            // Single-qubit un-entangled shard: write raw amplitudes (2 complex values).
            // Retrieve amp0/amp1 from the shard directly.
            const bitLenInt lq = rec.logical_qubits[0];
            const QEngineShard& sh = shards[lq];

            _write_bool(os, false); // is_compressed = false
            _write_cmplx(os, sh.amp0);
            _write_cmplx(os, sh.amp1);
            continue;
        }

        const bitCapIntOcl sub_pow = (bitCapIntOcl)(1ULL << rec.sub_qubit_count);
        const bool compress = (sub_pow >= BLOCK);

        _write_bool(os, compress);

        if (compress) {
            // Extract state vector and compress with TurboQuant
            std::unique_ptr<complex[]> sv(new complex[sub_pow]);
            rec.unit->GetQuantumState(sv.get());

            StateVectorTurboQuant tq(sub_pow, p_eff, b_eff, sv.get());
            tq.save(os);
        } else {
            // Subsystem smaller than one TurboQuant block — store raw
            std::unique_ptr<complex[]> sv(new complex[sub_pow]);
            rec.unit->GetQuantumState(sv.get());
            for (bitCapIntOcl j = 0U; j < sub_pow; ++j)
                _write_cmplx(os, sv[j]);
        }
    }

    // ── 4. Write logical qubit map ────────────────────────────────────────
    _write_sz(os, (size_t)qubitCount); // alignment sentinel
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        _write_sz(os, qubit_map[i].first); // subsystem index
        _write_sz(os, qubit_map[i].second); // mapped index within subsystem
    }

    os.close();
}

// ── LossyLoadStateVector ─────────────────────────────────────────────────────

void QUnit::LossyLoadStateVector(std::string f)
{
    std::ifstream is(f, std::ios::binary);
    if (!is)
        throw std::runtime_error("QUnit::LossyLoadStateVector: cannot open " + f);

    // ── 1. Validate header ────────────────────────────────────────────────
    const uint64_t magic = _read_u64(is);
    if (magic != QUNIT_TQ_MAGIC) {
        // Not our format — fall back to the base QInterface implementation
        // which handles a flat TurboQuant stream.
        is.close();
        QInterface::LossyLoadStateVector(f);
        return;
    }

    const size_t saved_qubits = _read_sz(is);
    _read_int(is); // p_saved (unused)
    _read_int(is); // b_saved (unused)
    const size_t num_subsystems = _read_sz(is);

    // Resize QUnit to match saved qubit count if necessary
    if ((size_t)qubitCount < saved_qubits) {
        Allocate(qubitCount, (bitLenInt)(saved_qubits - qubitCount));
    } else if ((size_t)qubitCount > saved_qubits) {
        Dispose(0U, (bitLenInt)(qubitCount - saved_qubits));
    }

    // ── 2. Load each subsystem and build new shard map ────────────────────
    //
    // We rebuild the QUnit shard structure from scratch:
    //   - Each compressed subsystem becomes a fresh QInterface engine.
    //   - Each raw single-qubit shard is restored directly into shard amp0/amp1.
    //   - The logical qubit map written at save time is replayed to wire
    //     shards[i].unit and shards[i].mapped correctly.

    struct LoadedSub {
        bool is_single_raw; // true = single-qubit un-entangled shard
        complex amp0, amp1; // valid if is_single_raw
        QInterfacePtr unit; // valid if !is_single_raw
        size_t sub_qubit_count;
    };
    std::vector<LoadedSub> loaded(num_subsystems);

    for (size_t si = 0U; si < num_subsystems; ++si) {
        const size_t sub_qc = _read_sz(is);
        const bool compressed = _read_bool(is);
        loaded[si].sub_qubit_count = sub_qc;

        if (sub_qc == 1U && !compressed) {
            // Single-qubit raw shard
            loaded[si].is_single_raw = true;
            loaded[si].amp0 = _read_cmplx(is);
            loaded[si].amp1 = _read_cmplx(is);
            loaded[si].unit = nullptr;
        } else if (compressed) {
            // TurboQuant-compressed subsystem
            loaded[si].is_single_raw = false;
            StateVectorTurboQuantPtr tq = StateVectorTurboQuant::load(is);
            const bitCapIntOcl sub_pow = (bitCapIntOcl)tq->get_size();

            // Reconstruct state vector
            std::unique_ptr<complex[]> sv(new complex[sub_pow]);
            tq->copy_out(sv.get());

            // Create a fresh engine of the appropriate type and load into it
            QInterfacePtr eng = MakeEngine((bitLenInt)sub_qc, ZERO_BCI);
            eng->SetQuantumState(sv.get());
            loaded[si].unit = eng;
        } else {
            // Raw uncompressed subsystem (sub_pow < BLOCK)
            loaded[si].is_single_raw = false;
            const bitCapIntOcl sub_pow = (bitCapIntOcl)(1ULL << sub_qc);
            std::unique_ptr<complex[]> sv(new complex[sub_pow]);
            for (bitCapIntOcl j = 0U; j < sub_pow; ++j)
                sv[j] = _read_cmplx(is);

            QInterfacePtr eng = MakeEngine((bitLenInt)sub_qc, ZERO_BCI);
            eng->SetQuantumState(sv.get());
            loaded[si].unit = eng;
        }
    }

    // ── 3. Read logical qubit map and rebuild shards ──────────────────────
    const size_t sentinel = _read_sz(is);
    if (sentinel != saved_qubits)
        throw std::runtime_error("QUnit::LossyLoadStateVector: map sentinel mismatch");

    for (bitLenInt i = 0U; i < (bitLenInt)saved_qubits; ++i) {
        const size_t sub_idx = _read_sz(is);
        const size_t mapped_idx = _read_sz(is);
        const LoadedSub& sub = loaded[sub_idx];

        QEngineShard& shard = shards[i];

        // Clear any existing phase/entanglement buffers on this shard
        shard.DumpPhaseBuffers();
        shard.MakeDirty();

        if (sub.is_single_raw) {
            // Restore directly into shard cached amplitudes — no unit needed
            shard.unit = nullptr;
            shard.mapped = 0U;
            shard.amp0 = sub.amp0;
            shard.amp1 = sub.amp1;
            shard.isProbDirty = false;
            shard.isPhaseDirty = false;
        } else {
            shard.unit = sub.unit;
            shard.mapped = (bitLenInt)mapped_idx;
            // Mark dirty so Prob() and phase queries re-read from the engine
            shard.isProbDirty = true;
            shard.isPhaseDirty = true;
        }
    }

    is.close();
}

} // namespace Qrack
