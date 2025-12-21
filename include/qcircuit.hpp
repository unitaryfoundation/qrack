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
#define __IS_REAL_1(r) (abs(ONE_R1 - r) <= FP_NORM_EPSILON)
#define __IS_SAME(a, b) (norm((a) - (b)) <= FP_NORM_EPSILON)
#define __IS_CTRLED_CLIFFORD(top, bottom)                                                                              \
    ((__IS_REAL_1(std::real(top)) || __IS_REAL_1(std::imag(bottom))) &&                                                \
        (__IS_SAME(top, bottom) || __IS_SAME(top, -bottom)))
#define __IS_CLIFFORD_PHASE_INVERT(top, bottom)                                                                        \
    (__IS_SAME(top, bottom) || __IS_SAME(top, -bottom) || __IS_SAME(top, I_CMPLX * bottom) ||                          \
        __IS_SAME(top, -I_CMPLX * bottom))
#define __IS_CLIFFORD(mtrx)                                                                                            \
    ((__IS_PHASE(mtrx) && __IS_CLIFFORD_PHASE_INVERT(mtrx[0], mtrx[3])) ||                                             \
        (__IS_INVERT(mtrx) && __IS_CLIFFORD_PHASE_INVERT(mtrx[1], mtrx[2])) ||                                         \
        ((__IS_SAME(mtrx[0U], mtrx[1U]) || __IS_SAME(mtrx[0U], -mtrx[1U]) ||                                           \
             __IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) || __IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                   \
            (__IS_SAME(mtrx[0U], mtrx[2U]) || __IS_SAME(mtrx[0U], -mtrx[2U]) ||                                        \
                __IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) || __IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                \
            (__IS_SAME(mtrx[0U], mtrx[3U]) || __IS_SAME(mtrx[0U], -mtrx[3U]) ||                                        \
                __IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) || IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U]))))
#define __IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define __IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

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
        : target(0U)
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
        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4U, payloads[ZERO_BCI].get());
    }

    /**
     * Controlled gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[], const std::set<bitLenInt>& ctrls, const bitCapInt& perm)
        : target(trgt)
        , controls(ctrls)
    {
        const std::shared_ptr<complex>& p = payloads[perm] =
            std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4U, p.get());
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
            payloads[payload.first] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4U, payloads[payload.first].get());
        }
    }

    QCircuitGatePtr Clone() { return std::make_shared<QCircuitGate>(target, payloads, controls); }

    /**
     * Can I combine myself with gate `other`?
     */
    bool CanCombine(QCircuitGatePtr other, bool clifford = false)
    {
        if (target != other->target) {
            return false;
        }

        if (clifford) {
            if (controls.empty() && other->controls.empty()) {
                return true;
            }

            const bool mc = IsClifford();
            const bool oc = other->IsClifford();

            if (mc != oc) {
                return false;
            }

            if (mc) {
                return controls.empty() || other->controls.empty() ||
                    (*(controls.begin()) == *(other->controls.begin()));
            }
        }

        if (controls.empty() || other->controls.empty()) {
            return true;
        }

        return std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end()) ||
            std::includes(controls.begin(), controls.end(), other->controls.begin(), other->controls.end());
    }

    /**
     * Set this gate to the identity operator.
     */
    void Clear()
    {
        controls.clear();
        payloads.clear();

        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        complex* p = payloads[ZERO_BCI].get();
        p[0U] = ONE_CMPLX;
        p[1U] = ZERO_CMPLX;
        p[2U] = ZERO_CMPLX;
        p[3U] = ONE_CMPLX;
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
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~lowMask;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            bitCapInt nKey = (payload.first & lowMask) | ((payload.first & highMask) << 1U);

            nPayloads.emplace(nKey, payload.second);

            std::shared_ptr<complex> np = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4U, np.get());
            bi_or_ip(&nKey, midPow);
            nPayloads.emplace(nKey, np);
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
            bitCapInt nKey = ~midPow & payload.first;

            if (bi_compare(nKey, payload.first) == 0) {
                if (payloads.find(nKey | midPow) == payloads.end()) {
                    return false;
                }
            } else {
                if (payloads.find(nKey) == payloads.end()) {
                    return false;
                }
            }

            const complex* l = payloads[nKey].get();
            bi_or_ip(&nKey, midPow);
            const complex* h = payloads[nKey].get();
            if (amp_leq_0(l[0U] - h[0U]) && amp_leq_0(l[1U] - h[1U]) && amp_leq_0(l[2U] - h[2U]) &&
                amp_leq_0(l[3U] - h[3U])) {
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
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~(lowMask | midPow);

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            nPayloads.emplace((payload.first & lowMask) | ((payload.first & highMask) >> 1U), payload.second);
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
                    std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            complex* p = pit->second.get();
            complex out[4];
            mul2x2(payload.second.get(), p, out);

            if (amp_leq_0(out[1U]) && amp_leq_0(out[2U]) && amp_leq_0(ONE_CMPLX - out[0U]) &&
                amp_leq_0(ONE_CMPLX - out[3U])) {
                payloads.erase(pit);

                continue;
            }

            std::copy(out, out + 4U, p);
        }

        if (payloads.empty()) {
            return Clear();
        }

        for (const bitLenInt& c : ctrlsToTest) {
            TryRemoveControl(c);
        }
    }

    /**
     * Check if I can combine with gate `other`, and do so, if possible
     */
    bool TryCombine(QCircuitGatePtr other, bool clifford = false)
    {
        if (!CanCombine(other, clifford)) {
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
        const bitCapInt controlPow = pow2(controls.size());
        if (payloads.size() == controlPow) {
            const complex* refP = payloads.begin()->second.get();
            if (norm(refP[0U] - refP[3U]) > FP_NORM_EPSILON) {
                return false;
            }

            const complex phaseFac = refP[0U];
            for (const auto& payload : payloads) {
                complex* p = payload.second.get();
                if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON) ||
                    (norm(phaseFac - p[0U]) > FP_NORM_EPSILON) || (norm(phaseFac - p[3U]) > FP_NORM_EPSILON)) {
                    return false;
                }
            }

            return true;
        }

        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON) ||
                (norm(ONE_CMPLX - p[0U]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[3U]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a phase gate?
     */
    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON)) {
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
            if ((norm(p[0U]) > FP_NORM_EPSILON) || (norm(p[3U]) > FP_NORM_EPSILON)) {
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
        if ((controls.size() != 1U) || (payloads.size() != 1U) || (payloads.find(ONE_BCI) == payloads.end())) {
            return false;
        }

        complex* p = payloads[ONE_BCI].get();
        if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON) ||
            (norm(ONE_CMPLX - p[1]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[2]) > FP_NORM_EPSILON)) {
            return false;
        }

        return true;
    }

    /**
     * Am I a CNOT gate?
     */
    bool IsAntiCnot()
    {
        if ((controls.size() != 1U) || (payloads.size() != 1U) || (payloads.find(ZERO_BCI) == payloads.end())) {
            return false;
        }

        complex* p = payloads[ZERO_BCI].get();
        if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON) ||
            (norm(ONE_CMPLX - p[1]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[2]) > FP_NORM_EPSILON)) {
            return false;
        }

        return true;
    }

    /**
     * Am I a Clifford gate?
     */
    bool IsClifford()
    {
        if (payloads.empty()) {
            // Swap gate is Clifford
            return true;
        }

        if (controls.size() > 1U) {
            return false;
        }

        if (controls.empty()) {
            return __IS_CLIFFORD(payloads[ZERO_BCI].get());
        }

        for (const auto& kvPair : payloads) {
            const complex* p = kvPair.second.get();
            if ((norm(p[1U]) <= FP_NORM_EPSILON) && (norm(p[2U]) <= FP_NORM_EPSILON)) {
                // Phase payload
                if (!__IS_CLIFFORD_PHASE_INVERT(p[0U], p[3U])) {
                    return false;
                }
            } else if ((norm(p[0U]) <= FP_NORM_EPSILON) && (norm(p[3U]) <= FP_NORM_EPSILON)) {
                // Negation payload
                if (!__IS_CLIFFORD_PHASE_INVERT(p[1U], p[2U])) {
                    return false;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    /**
     * Do I commute with gate `other`?
     */
    bool CanPass(QCircuitGatePtr other)
    {
        const std::set<bitLenInt>::iterator c = other->controls.find(target);
        if (c == other->controls.end()) {
            if (controls.find(other->target) != controls.end()) {
                return other->IsPhase();
            }

            return (target != other->target) || (IsPhase() && other->IsPhase());
        }

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
            bitCapInt pf = ZERO_BCI;
            for (size_t i = 0U; i < opfPows.size(); ++i) {
                if (bi_compare_0(payload.first & opfPows[i]) != 0) {
                    bi_or_ip(&pf, pow2(i));
                }
            }
            const auto& poi = payloads.find(pf);
            if ((poi == payloads.end()) || (norm(poi->second.get()[0U]) > FP_NORM_EPSILON)) {
                nPayloads[payload.first] = payload.second;
            } else {
                nPayloads[payload.first ^ p] = payload.second;
            }
        }
        other->payloads = nPayloads;

        return true;
    }

    /**
     * To run as a uniformly controlled gate, generate my payload array.
     */
    std::unique_ptr<complex[]> MakeUniformlyControlledPayload()
    {
        const bitCapIntOcl maxQPower = pow2Ocl(controls.size());
        std::unique_ptr<complex[]> toRet(new complex[maxQPower << 2U]);
        QRACK_CONST complex identity[4U]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
        for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
            complex* mtrx = toRet.get() + (i << 2U);
            const auto& p = payloads.find(i);
            if (p == payloads.end()) {
                std::copy(identity, identity + 4U, mtrx);
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
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~(lowMask | midPow);
        const bitCapInt qubitPow = pow2(cpos);
        const bitCapInt eigenPow = eigen ? qubitPow : ZERO_BCI;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            if (bi_compare(payload.first & qubitPow, eigenPow) != 0) {
                continue;
            }
            nPayloads.emplace((payload.first & lowMask) | ((payload.first & highMask) >> 1U), payload.second);
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
    bool isNearClifford;
    bitLenInt qubitCount;
    std::list<QCircuitGatePtr> gates;

public:
    /**
     * Default constructor
     */
    QCircuit(bool collapse = true, bool clifford = false)
        : isCollapsed(collapse)
        , isNearClifford(clifford)
        , qubitCount(0U)
        , gates()
    {
        // Intentionally left blank
    }

    /**
     * Manual constructor
     */
    QCircuit(bitLenInt qbCount, const std::list<QCircuitGatePtr>& g, bool collapse = true, bool clifford = false)
        : isCollapsed(collapse)
        , isNearClifford(clifford)
        , qubitCount(qbCount)
    {
        for (auto gIt = g.begin(); gIt != g.end(); ++gIt) {
            gates.push_back((*gIt)->Clone());
        }
    }

    QCircuitPtr Clone() { return std::make_shared<QCircuit>(qubitCount, gates, isCollapsed, isNearClifford); }

    QCircuitPtr Inverse()
    {
        QCircuitPtr clone = Clone();
        for (auto gIt = clone->gates.begin(); gIt != clone->gates.end(); ++gIt) {
            for (auto& p : (*gIt)->payloads) {
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

        QRACK_CONST complex m[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        const std::set<bitLenInt> s1{ q1 };
        const std::set<bitLenInt> s2{ q2 };
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, ONE_BCI));
        AppendGate(std::make_shared<QCircuitGate>(q2, m, s1, ONE_BCI));
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, ONE_BCI));
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
     * Combine circuit (with identical qubit index mappings) at the end of this circuit, by acting all additional
     * gates in sequence.
     */
    void Combine(QCircuitPtr circuit)
    {
        if (circuit->qubitCount > qubitCount) {
            qubitCount = circuit->qubitCount;
        }
        for (auto gIt = circuit->gates.begin(); gIt != circuit->gates.end(); ++gIt) {
            AppendGate(*gIt);
        }
    }

    /**
     * Add a gate to the gate sequence.
     */
    bool AppendGate(QCircuitGatePtr nGate);
    /**
     * Run this circuit.
     */
    void Run(QInterfacePtr qsim);

    /**
     * Check if an index is any target qubit of a nonclassical gate in this circuit.
     */
    bool IsNonClassicalTarget(bitLenInt qubit)
    {
        for (const QCircuitGatePtr& gate : gates) {
            if (gate->target != qubit) {
                continue;
            }
            if (gate->IsInvert() && gate->controls.empty()) {
                continue;
            }
            if (!(gate->IsPhase())) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check if an index is any target qubit of a nonclassical gate in this circuit.
     */
    bool IsNonClassicalTarget(bitLenInt qubit, bool* eigen)
    {
        for (const QCircuitGatePtr& gate : gates) {
            if (gate->target != qubit) {
                continue;
            }
            if (gate->IsInvert() && gate->controls.empty()) {
                *eigen = !*eigen;
                continue;
            }
            if (!(gate->IsPhase())) {
                return true;
            }
        }

        return false;
    }

    /**
     * (If the qubit is not a target of a non-classical gate...) Delete this qubits' controls and phase targets.
     */
    bool DeleteClassicalTarget(bitLenInt qubit, bool eigen)
    {
        std::list<QCircuitGatePtr> nGates;
        gates.reverse();
        for (const QCircuitGatePtr& gate : gates) {
            if (gate->target == qubit) {
                if (gate->IsInvert()) {
                    eigen = !eigen;
                    QCircuitGatePtr nGate = gate->Clone();
                    nGates.insert(nGates.begin(), nGate);
                }
                continue;
            }
            QCircuitGatePtr nGate = gate->Clone();
            nGate->PostSelectControl(qubit, eigen);
            nGates.insert(nGates.begin(), nGate);
        }
        gates = nGates;
        return eigen;
    }

    /**
     * Return (as a new QCircuit) just the gates on the past light cone of a set of qubit indices.
     */
    QCircuitPtr PastLightCone(std::set<bitLenInt>& qubits)
    {
        // We're working from latest gate to earliest gate.
        gates.reverse();

        std::list<QCircuitGatePtr> nGates;
        for (auto gIt = gates.begin(); gIt != gates.end(); ++gIt) {
            QCircuitGatePtr& gate = *gIt;
            // Is the target qubit on the light cone?
            if (qubits.find(gate->target) == qubits.end()) {
                // The target isn't on the light cone, but the controls might be.
                bool isNonCausal = true;
                for (const bitLenInt& c : gate->controls) {
                    if (qubits.find(c) != qubits.end()) {
                        isNonCausal = false;
                        break;
                    }
                }
                if (isNonCausal) {
                    // This gate is not on the past light cone.
                    continue;
                }
            }

            // This gate is on the past light cone.
            nGates.insert(nGates.begin(), gate->Clone());

            // Every qubit involved in this gate is now considered to be part of the past light cone.
            qubits.insert(gate->target);
            qubits.insert(gate->controls.begin(), gate->controls.end());
        }

        // Restore the original order of this QCircuit's gates.
        gates.reverse();

        return std::make_shared<QCircuit>(qubitCount, nGates, isCollapsed, isNearClifford);
    }

    /**
     * Return (as a new QCircuit) just the gates on the past light cone of a set of qubit indices, while removing these gates from the original circuit.
     */
    QCircuitPtr RemovePastLightCone(std::set<bitLenInt>& qubits)
    {
        // We're working from latest gate to earliest gate.
        gates.reverse();

        std::list<QCircuitGatePtr> oGates;
        std::list<QCircuitGatePtr> nGates;
        for (auto gIt = gates.begin(); gIt != gates.end(); ++gIt) {
            QCircuitGatePtr& gate = *gIt;
            // Is the target qubit on the light cone?
            if (qubits.find(gate->target) != qubits.end()) {
                // This gate is not on the past light cone.
                oGates.insert(oGates.begin(), gate);
                // Every qubit involved in this gate is now considered to be part of the past light cone.
                qubits.insert(gate->controls.begin(), gate->controls.end());
                continue;
            }
            // The target isn't on the light cone, but the controls might be.
            bool isNonCausal = true;
            for (const bitLenInt& c : gate->controls) {
                if (qubits.find(c) != qubits.end()) {
                    isNonCausal = false;
                    break;
                }
            }
            if (isNonCausal) {
                // This gate is not on the past light cone.
                oGates.insert(oGates.begin(), gate);
                continue;
            }

            // This gate is on the past light cone.
            nGates.insert(nGates.begin(), gate);
            // Every qubit involved in this gate is now considered to be part of the past light cone.
            qubits.insert(gate->target);
            qubits.insert(gate->controls.begin(), gate->controls.end());
        }

        // Internally retain just the gates off the light cone.
        gates = oGates;

        return std::make_shared<QCircuit>(qubitCount, nGates, isCollapsed, isNearClifford);
    }

#if ENABLE_ALU
    /** Add integer (without sign) */
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
#endif
};

std::ostream& operator<<(std::ostream& os, const QCircuitPtr g);
std::istream& operator>>(std::istream& os, QCircuitPtr& g);
} // namespace Qrack
