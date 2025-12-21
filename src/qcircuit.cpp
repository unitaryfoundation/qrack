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

#include "qcircuit.hpp"

namespace Qrack {

std::ostream& operator<<(std::ostream& os, const QCircuitGatePtr g)
{
    os << (size_t)g->target << " ";

    os << g->controls.size() << " ";
    for (const bitLenInt& c : g->controls) {
        os << (size_t)c << " ";
    }

    os << g->payloads.size() << " ";
    for (const auto& p : g->payloads) {
        os << (uint64_t)p.first << " ";
        const complex* mtrx = p.second.get();
        for (size_t i = 0U; i < 4U; ++i) {
            os << mtrx[i] << " ";
        }
    }

    return os;
}

std::istream& operator>>(std::istream& is, QCircuitGatePtr& g)
{
    g->payloads.clear();

    size_t target;
    is >> target;
    g->target = (bitLenInt)target;

    size_t cSize;
    is >> cSize;
    for (size_t i = 0U; i < cSize; ++i) {
        size_t c;
        is >> c;
        g->controls.insert((bitLenInt)c);
    }

    size_t pSize;
    is >> pSize;
    for (size_t i = 0U; i < pSize; ++i) {
        bitCapInt k;
        is >> k;

        g->payloads[k] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        for (size_t j = 0U; j < 4U; ++j) {
            is >> g->payloads[k].get()[j];
        }
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, const QCircuitPtr c)
{
    os << (size_t)c->GetQubitCount() << " ";

    std::list<QCircuitGatePtr> gates = c->GetGateList();
    os << gates.size() << " ";
    for (auto g = gates.begin(); g != gates.end(); ++g) {
        os << *g;
    }

    return os;
}

std::istream& operator>>(std::istream& is, QCircuitPtr& c)
{
    size_t qubitCount;
    is >> qubitCount;
    c->SetQubitCount((bitLenInt)qubitCount);

    size_t gSize;
    is >> gSize;
    std::list<QCircuitGatePtr> gl;
    for (size_t i = 0U; i < gSize; ++i) {
        QCircuitGatePtr g = std::make_shared<QCircuitGate>();
        is >> g;
        gl.push_back(g);
    }
    c->SetGateList(gl);

    return is;
}

bool QCircuit::AppendGate(QCircuitGatePtr nGate)
{
    if (!isCollapsed) {
        gates.push_back(nGate);

        return false;
    }

    if (nGate->IsIdentity()) {
        return true;
    }

    if ((nGate->target + 1U) > qubitCount) {
        qubitCount = nGate->target + 1U;
    }
    if (!(nGate->controls.empty())) {
        const bitLenInt q = *(nGate->controls.rbegin());
        if ((q + 1U) > qubitCount) {
            qubitCount = (q + 1U);
        }
    }

    std::set<bitLenInt> nQubits(nGate->controls);
    nQubits.insert(nGate->target);
    bool didCommute = false;
    for (std::list<QCircuitGatePtr>::reverse_iterator gate = gates.rbegin(); gate != gates.rend(); ++gate) {
        if ((*gate)->TryCombine(nGate, isNearClifford)) {
            if ((*gate)->IsIdentity()) {
                std::set<bitLenInt> gQubits((*gate)->controls);
                gQubits.insert((*gate)->target);
                std::list<QCircuitGatePtr>::reverse_iterator _gate = gate++;
                std::list<QCircuitGatePtr> head(_gate.base(), gates.end());
                gates.erase(gate.base(), gates.end());
                for (; head.size() && gQubits.size(); head.erase(head.begin())) {
                    std::set<bitLenInt> hQubits(head.front()->controls);
                    hQubits.insert((*head.begin())->target);
                    if (!std::any_of(hQubits.begin(), hQubits.end(),
                            [&gQubits](bitLenInt element) { return gQubits.count(element) > 0; })) {
                        gates.push_back(head.front());
                        continue;
                    }
                    if (AppendGate(head.front())) {
                        gQubits.insert(hQubits.begin(), hQubits.end());
                    } else {
                        for (const auto& hq : hQubits) {
                            gQubits.erase(hq);
                        }
                    }
                }
                gates.insert(gates.end(), head.begin(), head.end());
            }

            return true;
        }

        if (!(*gate)->CanPass(nGate)) {
            gates.insert(gate.base(), { nGate });

            return didCommute;
        }

        std::set<bitLenInt> gQubits((*gate)->controls);
        gQubits.insert((*gate)->target);
        didCommute |= std::any_of(
            nQubits.begin(), nQubits.end(), [&gQubits](bitLenInt element) { return gQubits.count(element) > 0; });
    }

    gates.push_front(nGate);

    return didCommute;
}

void QCircuit::Run(QInterfacePtr qsim)
{
    if (qsim->GetQubitCount() < qubitCount) {
        qsim->Allocate(qubitCount - qsim->GetQubitCount());
    }

    // Peephole rewrite provided by Elara (the OpenAI custom GPT)
    // Peephole: CNOT(a->b); CNOT(b->a); CNOT(a->b) == SWAP(a,b)
    // Also valid for "AntiCNOT" (control-on-|0>) when all three are AntiCNOT.
    std::list<QCircuitGatePtr> nGates;
    if (gates.size() < 3U) {
        nGates = gates;
    } else {
        auto end = gates.begin();
        std::advance(end, gates.size() - 2U); // we will look ahead 2 gates

        auto gate = gates.begin();
        for (; gate != end; ++gate) {
            const bool isAnti = (*gate)->IsAntiCnot();
            const bool isCnotType = (*gate)->IsCnot() || isAnti;

            if (!isCnotType) {
                nGates.push_back(*gate);
                continue;
            }

            // Look ahead to next two gates
            auto g2 = std::next(gate);
            auto g3 = std::next(g2);

            // All three must be the same type: either all CNOT or all AntiCNOT
            const bool g2ok = (isAnti && (*g2)->IsAntiCnot()) || (!isAnti && (*g2)->IsCnot());
            const bool g3ok = (isAnti && (*g3)->IsAntiCnot()) || (!isAnti && (*g3)->IsCnot());
            if (!g2ok || !g3ok) {
                nGates.push_back(*gate);
                continue;
            }

            // Each must be a single-control CNOT/AntiCNOT by construction of IsCnot/IsAntiCnot,
            // but we still assume controls.size()==1 and read the only control.
            const bitLenInt a = *((*gate)->controls.begin()); // control of gate1
            const bitLenInt b = (*gate)->target;              // target of gate1

            const bitLenInt g2c = *((*g2)->controls.begin());
            const bitLenInt g2t = (*g2)->target;

            const bitLenInt g3c = *((*g3)->controls.begin());
            const bitLenInt g3t = (*g3)->target;

            // Match canonical SWAP decomposition:
            // gate1: CNOT(a->b)
            // gate2: CNOT(b->a)
            // gate3: CNOT(a->b)
            if (!(g2c == b && g2t == a && g3c == a && g3t == b)) {
                nGates.push_back(*gate);
                continue;
            }

            // Replace with SWAP(a,b). The Swap gate ctor is QCircuitGate(q1,q2) with target=q1 and controls={q2}.
            // In Run(), payloads.empty() triggers Swap(controls[0], target), i.e. Swap(q2,q1).
            // So pass (b,a) here to produce Swap(a,b) (order doesn't matter for swap, but keep canonical).
            nGates.push_back(std::make_shared<QCircuitGate>(b, a));

            // Skip the next two gates that we consumed (g2 and g3).
            gate = g3;

            // If fewer than 3 gates remain ahead, break and copy the tail below.
            if (std::distance(gate, gates.end()) < 3) {
                ++gate; // move to the first unprocessed gate
                break;
            }
        }

        // Copy any remaining gates (including tail after break)
        for (; gate != gates.end(); ++gate) {
            nGates.push_back(*gate);
        }
    }

    for (auto gIt = nGates.begin(); gIt != nGates.end(); ++gIt) {
        const QCircuitGatePtr& gate = *gIt;
        const bitLenInt& t = gate->target;

        if (gate->controls.empty()) {
            qsim->Mtrx(gate->payloads[ZERO_BCI].get(), t);

            continue;
        }

        std::vector<bitLenInt> controls = gate->GetControlsVector();

        if (gate->payloads.empty()) {
            qsim->Swap(controls[0U], t);

            continue;
        }

        if (gate->payloads.size() == 1U) {
            const auto& payload = gate->payloads.begin();
            qsim->UCMtrx(controls, payload->second.get(), t, payload->first);

            continue;
        }

        std::unique_ptr<complex[]> payload = gate->MakeUniformlyControlledPayload();
        qsim->UniformlyControlledSingleBit(controls, t, payload.get());
    }
}
} // namespace Qrack
