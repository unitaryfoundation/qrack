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

void QCircuit::AppendGate(QCircuitGatePtr nGate)
{
    if (nGate->IsIdentity()) {
        return;
    }

    if (nGate->target > maxQubit) {
        maxQubit = nGate->target;
    }
    if (!(nGate->controls.empty())) {
        const bitLenInt q = *(nGate->controls.rbegin());
        if (q > maxQubit) {
            maxQubit = q;
        }
    }

    for (size_t i = 0U; i < gates.size(); ++i) {
        QCircuitGatePtr gate = gates[gates.size() - (i + 1U)];
        if (gate->TryCombine(nGate)) {
            if (gate->IsIdentity()) {
                gates.erase(gates.end() - (i + 1U));
            }
            return;
        }
        if (!gate->CanPass(nGate)) {
            break;
        }
    }

    gates.push_back(nGate);
}

void QCircuit::Run(QInterfacePtr qsim)
{
    if (qsim->GetQubitCount() < maxQubit) {
        qsim->Allocate(maxQubit - qsim->GetQubitCount());
    }

    for (const QCircuitGatePtr& gate : gates) {
        const bitLenInt& t = gate->target;

        if (!gate->controls.size()) {
            qsim->Mtrx(gate->payloads[0].get(), t);

            continue;
        }

        std::vector<bitLenInt> controls = gate->GetControlsVector();

        if ((gate->payloads.size() == (1U << controls.size())) || (gate->payloads.size() >= 8)) {
            std::unique_ptr<complex[]> payload = gate->MakeUniformlyControlledPayload();
            qsim->UniformlyControlledSingleBit(controls, gate->target, payload.get());

            continue;
        }

        std::vector<bool> controlStates(maxQubit, false);
        for (const auto& payload : gate->payloads) {
            std::map<bitLenInt, bool> controlMismatch;
            size_t mismatchCount = 0;
            for (const auto& c : controls) {
                controlMismatch[c] = (((bool)((payload.first >> c) & 1)) != controlStates[c]);
                if (controlMismatch[c]) {
                    ++mismatchCount;
                }
            }

            if (controlStates[t]) {
                qsim->X(t);
                controlStates[t] = false;
            }

            if ((mismatchCount << 1U) > controls.size()) {
                for (const auto& c : controls) {
                    if (!controlMismatch[c]) {
                        qsim->X(c);
                        controlStates[c] = !controlStates[c];
                    }
                }

                qsim->MCMtrx(controls, payload.second.get(), t);

                continue;
            }

            for (const auto& c : controls) {
                if (controlMismatch[c]) {
                    qsim->X(c);
                    controlStates[c] = !controlStates[c];
                }
            }

            qsim->MACMtrx(controls, payload.second.get(), t);
        }

        for (size_t i = 0U; i < controlStates.size(); ++i) {
            if (controlStates[i]) {
                qsim->X(i);
            }
        }
    }
}
} // namespace Qrack
