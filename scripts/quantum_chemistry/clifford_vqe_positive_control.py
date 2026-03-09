# Positive control: VQE implementation validation
# Developed with help from Claude (Anthropic)
#
# Purpose: Demonstrate that the VQE implementation in clifford_vqe_ig_entangled.py
# is capable of finding energy improvements when the bootstrap starting point is
# deliberately wrong. This addresses the referee question:
# "How do we know VQE would ever return a non-null variational update?"
#
# Method: Run VQE on H2 at equilibrium (STO-3G, 4 qubits) starting from:
#   (a) all-zeros state (vacuum, definitely wrong)
#   (b) random state
#   (c) correct bootstrap state (should already be near ground state)
#
# Expected result:
#   (a) and (b): VQE finds substantial improvement toward known ground state energy
#   (c): VQE finds little or no improvement (consistent with main paper results)
#
# Reference: H2 STO-3G FCI ground state energy ~ -1.1373 Ha
#            Hartree-Fock energy               ~ -1.1175 Ha

from openfermion import MolecularData, FermionOperator, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

import itertools
import multiprocessing
import numpy as np
import os
import random

import pennylane as qml
from pennylane import numpy as nppl


# ── molecule: H2 at equilibrium ──────────────────────────────────────────────

basis        = "sto-3g"
multiplicity = 1
charge       = 0

geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 equilibrium

print("H2 STO-3G positive control for VQE implementation validation")
print("Reference FCI ground state energy: ~ -1.1373 Ha")
print()


def geometry_to_atom_str(geometry):
    return "; ".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in geometry
    )


# ── build Hamiltonians ────────────────────────────────────────────────────────

molecule_of = MolecularData(geometry, basis, multiplicity=multiplicity, charge=charge)
molecule_of = run_pyscf(molecule_of, run_scf=True, run_mp2=False, run_cisd=False,
                        run_ccsd=False, run_fci=False)
fermion_ham = get_fermion_operator(molecule_of.get_molecular_hamiltonian())
n_electrons = molecule_of.n_electrons
n_qubits    = molecule_of.n_qubits

print(f"Hartree-Fock energy: {molecule_of.hf_energy:.10f} Ha")
print(f"{n_electrons} electrons, {n_qubits} qubits")

# Interaction graph
edges = set()
for term, coeff in fermion_ham.terms.items():
    jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
    for pauli_string, _ in jw_term.terms.items():
        qubits = [q for q, op in pauli_string if op != 'I']
        for a, b in itertools.combinations(sorted(qubits), 2):
            edges.add((a, b))
ig_edges = sorted(edges)
n_edges  = len(ig_edges)
print(f"Interaction graph: {n_edges} undirected edges")

# Z-only Hamiltonian for bootstrap energy evaluation
z_hamiltonian = []
for term, coeff in fermion_ham.terms.items():
    jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
    for pauli_string, jw_coeff in jw_term.terms.items():
        if any(op in ('X', 'Y') for _, op in pauli_string):
            continue
        q = [qubit for qubit, op in pauli_string if op == 'Z']
        z_hamiltonian.append((q, jw_coeff.real))

def z_energy(theta):
    energy = 0.0
    for qubits, coeff in z_hamiltonian:
        c = coeff
        for q in qubits:
            if theta[q]:
                c *= -1
        energy += c
    return energy

# PennyLane Hamiltonian
coeffs      = []
observables = []
for term, coeff in fermion_ham.terms.items():
    jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
    for pauli_string, jw_coeff in jw_term.terms.items():
        ops = []
        for qubit_idx, pauli in pauli_string:
            if   pauli == 'X': ops.append(qml.PauliX(qubit_idx))
            elif pauli == 'Y': ops.append(qml.PauliY(qubit_idx))
            elif pauli == 'Z': ops.append(qml.PauliZ(qubit_idx))
        observables.append(ops[0] if ops else qml.Identity(0))
        if len(ops) > 1:
            for op in ops[1:]:
                observables[-1] = observables[-1] @ op
        coeffs.append(nppl.array(jw_coeff.real, requires_grad=False))
pl_hamiltonian = qml.Hamiltonian(coeffs, observables)


# ── VQE runner ────────────────────────────────────────────────────────────────

def run_vqe(label, bootstrap_theta, n_steps=200, stepsize=np.pi/1800):
    print(f"\n── {label} ──")
    print(f"  Initial θ: {bootstrap_theta.astype(int)}")
    initial_energy = z_energy(bootstrap_theta)
    print(f"  Initial Z-basis energy: {initial_energy:.10f} Ha")

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(delta, gamma):
        for i in range(n_qubits):
            if bootstrap_theta[i]:
                qml.X(wires=i)
        for i in range(n_qubits):
            qml.RY(delta[i], wires=i)
        for idx, (a, b) in enumerate(ig_edges):
            qml.IsingZZ(gamma[idx], wires=[a, b])
        return qml.expval(pl_hamiltonian)

    delta = nppl.array(np.random.uniform(-0.1, 0.1, n_qubits), requires_grad=True)
    gamma = nppl.array(np.random.uniform(-0.1, 0.1, n_edges), requires_grad=True)

    opt        = qml.AdamOptimizer(stepsize=stepsize)
    min_energy = initial_energy
    best_delta = delta.copy()
    best_gamma = gamma.copy()

    for step in range(n_steps):
        (delta, gamma), energy = opt.step_and_cost(circuit, delta, gamma)
        if float(energy) < min_energy:
            min_energy = float(energy)
            best_delta = delta.copy()
            best_gamma = gamma.copy()
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:3d}: Energy = {float(energy):.10f} Ha")

    print(f"  Initial energy:  {initial_energy:.10f} Ha")
    print(f"  Final energy:    {min_energy:.10f} Ha")
    print(f"  Improvement:     {initial_energy - min_energy:.10f} Ha")
    print(f"  δ (RY):  {np.array(best_delta).round(4)}")
    print(f"  γ (RZZ): {np.array(best_gamma).round(4)}")

    return min_energy


# ── Bootstrap state (correct priors) ─────────────────────────────────────────
# H2 STO-3G: 2 electrons in 4 spin-orbitals
# Correct HF state: qubits 0,1 occupied (alpha/beta of bonding orbital)

bootstrap_theta = np.array([True, True, False, False])

# ── Run three cases ───────────────────────────────────────────────────────────

e_zeros    = run_vqe("Case (a): all-zeros initial state (vacuum)",
                     np.array([False, False, False, False]))

e_random   = run_vqe("Case (b): random initial state",
                     np.array([bool(random.randint(0,1)) for _ in range(n_qubits)]))

e_bootstrap = run_vqe("Case (c): correct bootstrap state",
                      bootstrap_theta)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"Hartree-Fock energy:              {molecule_of.hf_energy:.10f} Ha")
print(f"Reference FCI energy:             ~ -1.1373000000 Ha")
print(f"{'─'*60}")
print(f"Case (a) vacuum  → VQE final:     {e_zeros:.10f} Ha")
print(f"Case (b) random  → VQE final:     {e_random:.10f} Ha")
print(f"Case (c) bootstrap → VQE final:   {e_bootstrap:.10f} Ha")
print(f"{'─'*60}")
print()
print("Interpretation:")
print("  Cases (a) and (b) should show substantial VQE improvement,")
print("  demonstrating the optimizer is functional.")
print("  Case (c) should show little or no improvement,")
print("  consistent with the bootstrap state being a convex minimum.")
