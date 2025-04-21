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

    std::list<QCircuitGatePtr> nGates;
    if (gates.size() < 3U) {
        nGates = gates;
    } else {
        std::list<QCircuitGatePtr>::iterator end = gates.begin();
        std::advance(end, gates.size() - 2U);
        std::list<QCircuitGatePtr>::iterator gate;
        for (gate = gates.begin(); gate != end; ++gate) {
            bool isAnti = (*gate)->IsAntiCnot();
            if (!((*gate)->IsCnot() || isAnti)) {
                nGates.push_back(*gate);
                continue;
            }
            std::list<QCircuitGatePtr>::iterator adv = gate;
            ++adv;
            if (!((isAnti && (*adv)->IsAntiCnot()) || (!isAnti && (*adv)->IsCnot())) ||
                ((*adv)->target != *((*gate)->controls.begin())) || ((*gate)->target != *((*adv)->controls.begin()))) {
                nGates.push_back(*gate);
                continue;
            }
            ++adv;
            if (!((isAnti && (*adv)->IsAntiCnot()) || (!isAnti && (*adv)->IsCnot())) ||
                ((*adv)->target != (*gate)->target) || (*((*gate)->controls.begin()) != *((*adv)->controls.begin()))) {
                nGates.push_back(*gate);
                continue;
            }
            nGates.push_back(std::make_shared<QCircuitGate>((*gate)->target, *((*gate)->controls.begin())));
            gate = adv;
            if (std::distance(gate, gates.end()) < 3) {
                ++gate;
                break;
            }
        }
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
