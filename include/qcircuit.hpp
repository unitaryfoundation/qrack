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

#pragma once

#include "qinterface.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>

#define amp_leq_0(x) (norm(x) <= FP_NORM_EPSILON)

namespace Qrack {

/**
 * Single gate in `QCircuit` definition
 */
struct QCircuitGate;
typedef std::shared_ptr<QCircuitGate> QCircuitGatePtr;

struct QCircuitGate {
    bitLenInt target;
    std::map<bitCapInt, std::shared_ptr<complex>> payloads;
    std::set<bitLenInt> controls;

    /**
     * Identity gate constructor
     */
    QCircuitGate()
        : target(0)
        , payloads()
        , controls()

    {
        Clear();
    }

    /**
     * `Swap` gate constructor
     */
    QCircuitGate(bitLenInt q1, bitLenInt q2)
        : target(q1)
        , payloads()
        , controls({ q2 })

    {
        // Swap gate constructor.
    }

    /**
     * Single-qubit gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[])
        : target(trgt)
    {
        payloads[0] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, payloads[0].get());
    }

    /**
     * Controlled gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[], const std::set<bitLenInt>& ctrls, bitCapInt perm)
        : target(trgt)
        , controls(ctrls)
    {
        const std::shared_ptr<complex>& p = payloads[perm] =
            std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, p.get());
    }

    /**
     * Uniformly controlled gate constructor (that only accepts control qubits is ascending order)
     */
    QCircuitGate(
        bitLenInt trgt, const std::map<bitCapInt, std::shared_ptr<complex>>& pylds, const std::set<bitLenInt>& ctrls)
        : target(trgt)
        , controls(ctrls)
    {
        for (const auto& payload : pylds) {
            payloads[payload.first] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4, payloads[payload.first].get());
        }
    }

    QCircuitGatePtr Clone() { return std::make_shared<QCircuitGate>(target, payloads, controls); }

    /**
     * Can I combine myself with gate `other`?
     */
    bool CanCombine(QCircuitGatePtr other)
    {
        if (target != other->target) {
            return false;
        }

        if (std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end()) ||
            std::includes(controls.begin(), controls.end(), other->controls.begin(), other->controls.end())) {
            return true;
        }

        return false;
    }

    /**
     * Set this gate to the identity operator.
     */
    void Clear()
    {
        controls.clear();
        payloads.clear();

        payloads[0] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        complex* p = payloads[0].get();
        p[0] = ONE_CMPLX;
        p[1] = ZERO_CMPLX;
        p[2] = ZERO_CMPLX;
        p[3] = ONE_CMPLX;
    }

    /**
     * Add control qubit.
     */
    void AddControl(bitLenInt c)
    {
        if (controls.find(c) != controls.end()) {
            return;
        }

        controls.insert(c);

        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);
        const bitCapInt lowMask = midPow - 1U;
        const bitCapInt highMask = ~lowMask;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            const bitCapInt nKey = (payload.first & lowMask) | ((payload.first & highMask) << 1U);

            nPayloads.emplace(nKey, payload.second);

            std::shared_ptr<complex> np = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4U, np.get());
            nPayloads.emplace(nKey | midPow, np);
        }

        payloads = nPayloads;
    }

    /**
     * Check if a control qubit can be removed.
     */
    bool CanRemoveControl(bitLenInt c)
    {
        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);

        for (const auto& payload : payloads) {
            const bitCapInt nKey = payload.first & ~midPow;

            if (nKey == payload.first) {
                if (payloads.find(nKey | midPow) == payloads.end()) {
                    return false;
                }
            } else {
                if (payloads.find(nKey) == payloads.end()) {
                    return false;
                }
            }

            const complex* l = payloads[nKey].get();
            const complex* h = payloads[nKey | midPow].get();
            if (amp_leq_0(l[0] - h[0]) && amp_leq_0(l[1] - h[1]) && amp_leq_0(l[2] - h[2]) && amp_leq_0(l[3] - h[3])) {
                continue;
            }

            return false;
        }

        return true;
    }

    /**
     * Remove control qubit.
     */
    void RemoveControl(bitLenInt c)
    {
        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);
        const bitCapInt lowMask = midPow - 1U;
        const bitCapInt highMask = ~(lowMask | midPow);

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            const bitCapInt nKey = (payload.first & lowMask) | ((payload.first & highMask) >> 1U);
            nPayloads.emplace(nKey, payload.second);
        }

        payloads = nPayloads;
        controls.erase(c);
    }

    /**
     * Check if I can remove control, and do so, if possible
     */
    bool TryRemoveControl(bitLenInt c)
    {
        if (!CanRemoveControl(c)) {
            return false;
        }
        RemoveControl(c);

        return true;
    }

    /**
     * Combine myself with gate `other`
     */
    void Combine(QCircuitGatePtr other)
    {
        std::set<bitLenInt> ctrlsToTest;
        std::set_intersection(controls.begin(), controls.end(), other->controls.begin(), other->controls.end(),
            std::inserter(ctrlsToTest, ctrlsToTest.begin()));

        if (controls.size() < other->controls.size()) {
            for (const bitLenInt& oc : other->controls) {
                AddControl(oc);
            }
        } else if (controls.size() > other->controls.size()) {
            for (const bitLenInt& c : controls) {
                other->AddControl(c);
            }
        }

        for (const auto& payload : other->payloads) {
            const auto& pit = payloads.find(payload.first);
            if (pit == payloads.end()) {
                const std::shared_ptr<complex>& p = payloads[payload.first] =
                    std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            complex* p = pit->second.get();
            complex out[4];
            mul2x2(payload.second.get(), p, out);

            if (amp_leq_0(out[1]) && amp_leq_0(out[2]) && amp_leq_0(ONE_CMPLX - out[0]) &&
                amp_leq_0(ONE_CMPLX - out[3])) {
                payloads.erase(pit);

                continue;
            }

            std::copy(out, out + 4U, p);
        }

        if (!payloads.size()) {
            Clear();
            return;
        }

        for (const bitLenInt& c : ctrlsToTest) {
            TryRemoveControl(c);
        }
    }

    /**
     * Check if I can combine with gate `other`, and do so, if possible
     */
    bool TryCombine(QCircuitGatePtr other)
    {
        if (!CanCombine(other)) {
            return false;
        }
        Combine(other);

        return true;
    }

    /**
     * Am I an identity gate?
     */
    bool IsIdentity()
    {
        if (controls.size()) {
            return false;
        }

        if (payloads.size() != 1U) {
            return false;
        }

        complex* p = payloads.begin()->second.get();

        if (amp_leq_0(p[1]) && amp_leq_0(p[2]) && amp_leq_0(ONE_CMPLX - p[0]) && amp_leq_0(ONE_CMPLX - p[3])) {
            return true;
        }

        return false;
    }

    /**
     * Am I a phase gate?
     */
    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1]) > FP_NORM_EPSILON) || (norm(p[2]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a Pauli X plus a phase gate?
     */
    bool IsInvert()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a combination of "phase" and "invert" payloads?
     */
    bool IsPhaseInvert()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if (((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON)) &&
                ((norm(p[1]) > FP_NORM_EPSILON) || (norm(p[2]) > FP_NORM_EPSILON))) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a CNOT gate?
     */
    bool IsCnot()
    {
        if ((controls.size() != 1U) || (payloads.size() != 1U) || (payloads.find(1U) == payloads.end())) {
            return false;
        }
        complex* p = payloads[1U].get();
        if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON) ||
            (norm(ONE_CMPLX - p[1]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[2]) > FP_NORM_EPSILON)) {
            return false;
        }

        return true;
    }

    /**
     * Do I commute with gate `other`?
     */
    bool CanPass(QCircuitGatePtr other)
    {
        std::set<bitLenInt>::iterator c = other->controls.find(target);
        if (c != other->controls.end()) {
            if (controls.find(other->target) != controls.end()) {
                return IsPhase() && other->IsPhase();
            }
            if (IsPhase()) {
                return true;
            }
            if (!IsPhaseInvert() ||
                !std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end())) {
                return false;
            }

            std::vector<bitCapInt> opfPows;
            opfPows.reserve(controls.size());
            for (const bitLenInt& ctrl : controls) {
                opfPows.emplace_back(pow2(std::distance(other->controls.begin(), other->controls.find(ctrl))));
            }
            const bitCapInt p = pow2(std::distance(other->controls.begin(), c));
            std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
            for (const auto& payload : other->payloads) {
                bitCapInt pf = 0U;
                for (size_t i = 0U; i < opfPows.size(); ++i) {
                    if (payload.first & opfPows[i]) {
                        pf |= pow2(i);
                    }
                }
                const auto& poi = payloads.find(pf);
                if ((poi == payloads.end()) || (norm(poi->second.get()[0]) > FP_NORM_EPSILON)) {
                    nPayloads[payload.first] = payload.second;
                } else {
                    nPayloads[payload.first ^ p] = payload.second;
                }
            }
            other->payloads = nPayloads;

            return true;
        }

        if (controls.find(other->target) != controls.end()) {
            return other->IsPhase();
        }

        return (target != other->target) || (IsPhase() && other->IsPhase());
    }

    /**
     * To run as a uniformly controlled gate, generate my payload array.
     */
    std::unique_ptr<complex[]> MakeUniformlyControlledPayload()
    {
        const bitCapIntOcl maxQPower = pow2Ocl(controls.size());
        std::unique_ptr<complex[]> toRet(new complex[maxQPower << 2U]);
        constexpr complex identity[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
        for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
            complex* mtrx = toRet.get() + (i << 2U);
            const auto& p = payloads.find(i);
            if (p == payloads.end()) {
                std::copy(identity, identity + 4, mtrx);
                continue;
            }

            const complex* oMtrx = p->second.get();
            std::copy(oMtrx, oMtrx + 4U, mtrx);
        }

        return toRet;
    }

    /**
     * Convert my set of qubit indices to a vector
     */
    std::vector<bitLenInt> GetControlsVector() { return std::vector<bitLenInt>(controls.begin(), controls.end()); }

    /**
     * Erase a control index, if it exists, (via post selection).
     */
    void PostSelectControl(bitLenInt c, bool eigen)
    {
        const auto controlIt = controls.find(c);
        if (controlIt == controls.end()) {
            return;
        }

        const size_t cpos = std::distance(controls.begin(), controlIt);
        const bitCapInt midPow = pow2(cpos);
        const bitCapInt lowMask = midPow - 1U;
        const bitCapInt highMask = ~(lowMask | midPow);
        const bitCapInt qubitPow = pow2(cpos);
        const bitCapInt eigenPow = eigen ? qubitPow : 0U;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            if ((payload.first & qubitPow) != eigenPow) {
                continue;
            }
            const bitCapInt nKey = (payload.first & lowMask) | ((payload.first & highMask) >> 1U);
            nPayloads.emplace(nKey, payload.second);
        }

        payloads = nPayloads;
        controls.erase(c);
    }
};

std::ostream& operator<<(std::ostream& os, const QCircuitGatePtr g);
std::istream& operator>>(std::istream& os, QCircuitGatePtr& g);

/**
 * Define and optimize a circuit, before running on a `QInterface`.
 */
class QCircuit;
typedef std::shared_ptr<QCircuit> QCircuitPtr;

class QCircuit {
protected:
    bool isCollapsed;
    bitLenInt qubitCount;
    std::list<QCircuitGatePtr> gates;

public:
    /**
     * Default constructor
     */
    QCircuit(bool collapse = true)
        : isCollapsed(collapse)
        , qubitCount(0)
        , gates()
    {
        // Intentionally left blank
    }

    /**
     * Manual constructor
     */
    QCircuit(bitLenInt qbCount, const std::list<QCircuitGatePtr>& g, bool collapse = true)
        : isCollapsed(collapse)
        , qubitCount(qbCount)
    {
        for (const QCircuitGatePtr& gate : g) {
            gates.push_back(gate->Clone());
        }
    }

    QCircuitPtr Clone() { return std::make_shared<QCircuit>(qubitCount, gates, isCollapsed); }

    QCircuitPtr Inverse()
    {
        QCircuitPtr clone = Clone();
        for (QCircuitGatePtr& gate : clone->gates) {
            for (auto& p : gate->payloads) {
                const complex* m = p.second.get();
                complex inv[4U]{ conj(m[0U]), conj(m[2U]), conj(m[1U]), conj(m[3U]) };
                std::copy(inv, inv + 4U, p.second.get());
            }
        }
        clone->gates.reverse();

        return clone;
    }

    /**
     * Get the (automatically calculated) count of qubits in this circuit, so far.
     */
    bitLenInt GetQubitCount() { return qubitCount; }

    /**
     * Set the count of qubits in this circuit, so far.
     */
    void SetQubitCount(bitLenInt n) { qubitCount = n; }

    /**
     * Return the raw list of gates.
     */
    std::list<QCircuitGatePtr> GetGateList() { return gates; }

    /**
     * Set the raw list of gates.
     */
    void SetGateList(std::list<QCircuitGatePtr> gl) { gates = gl; }

    /**
     * Add a `Swap` gate to the gate sequence.
     */
    void Swap(bitLenInt q1, bitLenInt q2)
    {
        if (q1 == q2) {
            return;
        }

        // If all swap gates are constructed in the same order, between high and low qubits, then the chances of
        // combining them might be higher.
        if (q1 > q2) {
            std::swap(q1, q2);
        }

        constexpr complex m[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        const std::set<bitLenInt> s1 = { q1 };
        const std::set<bitLenInt> s2 = { q2 };
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, 1U));
        AppendGate(std::make_shared<QCircuitGate>(q2, m, s1, 1U));
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, 1U));
    }

    /**
     * Append circuit (with identical qubit index mappings) at the end of this circuit.
     */
    void Append(QCircuitPtr circuit)
    {
        if (circuit->qubitCount > qubitCount) {
            qubitCount = circuit->qubitCount;
        }
        gates.insert(gates.end(), circuit->gates.begin(), circuit->gates.end());
    }

    /**
     * Combine circuit (with identical qubit index mappings) at the end of this circuit, by acting all additional gates
     * in sequence.
     */
    void Combine(QCircuitPtr circuit)
    {
        if (circuit->qubitCount > qubitCount) {
            qubitCount = circuit->qubitCount;
        }
        for (const QCircuitGatePtr& g : circuit->gates) {
            AppendGate(g);
        }
    }

    /**
     * Add a gate to the gate sequence.
     */
    void AppendGate(QCircuitGatePtr nGate);
    /**
     * Run this circuit.
     */
    void Run(QInterfacePtr qsim);

    /**
     * Check if an index is any target qubit of this circuit.
     */
    bool IsNonPhaseTarget(bitLenInt qubit)
    {
        for (const QCircuitGatePtr& gate : gates) {
            if ((gate->target == qubit) && !(gate->IsPhase())) {
                return true;
            }
        }

        return false;
    }

    /**
     * (If the qubit is not a target of a non-phase gate...) Delete this qubits' controls and phase targets.
     */
    void DeletePhaseTarget(bitLenInt qubit, bool eigen)
    {
        std::list<QCircuitGatePtr> nGates;
        gates.reverse();
        for (const QCircuitGatePtr& gate : gates) {
            if (gate->target == qubit) {
                continue;
            }
            QCircuitGatePtr nGate = gate->Clone();
            nGate->PostSelectControl(qubit, eigen);
            nGates.insert(nGates.begin(), nGate);
        }
        gates = nGates;
    }

#if ENABLE_ALU
    /** Add integer (without sign) */
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
#endif
};

std::ostream& operator<<(std::ostream& os, const QCircuitPtr g);
std::istream& operator>>(std::istream& os, QCircuitPtr& g);
} // namespace Qrack
