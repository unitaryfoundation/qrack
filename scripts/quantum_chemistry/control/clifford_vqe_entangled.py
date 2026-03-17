# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# Adapted to parse interaction graph for VQE by (Anthropic) Claude
# (Requires OpenFermion)

from openfermion import MolecularData, FermionOperator, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

import itertools
import multiprocessing
import numpy as np
import os
import random

import pennylane as qml
from pennylane import numpy as nppl


# Step 1: Define the molecule (Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen)

basis = "sto-3g"  # Minimal Basis Set
# basis = '6-31g'  # Larger basis set
# basis = 'cc-pVDZ' # Even larger basis set!
multiplicity = 1  # singlet, closed shell, all electrons are paired (neutral molecules with full valence)
# multiplicity = 2  # doublet, one unpaired electron (ex.: OH- radical)
# multiplicity = 3  # triplet, two unpaired electrons (ex.: O2)
charge = 0  # Excess +/- elementary charge, beyond multiplicity

print(f"charge = {charge}")
print(f"multiplicity = {multiplicity}")

# Hydrogen (and lighter):

# geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 Molecule

# geometry = [
#     ("H", (-1.0, 0.0, -1.0)), ("H", (-1.0, 0.0, 1.00)),
#     ("H", (1.0, 0.0, -1.0)), ("H", (1.0, 0.0, 1.00))
# ]  # H4 Dissociation (hard for Hartree-Fock)

# Helium (and lighter):

# geometry = [('He', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.774))]  # HeH Molecule

# Lithium (and lighter):

# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.596))]  # equilibrium
# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 2.5))]   # stretched
geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 4.0))]   # near dissociation

# Beryllium (and lighter):

# geometry = [('H', (0.0, 0.0, -1.335)), ('Be', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.335))]  # equilibrium
# geometry = [('H', (0.0, 0.0, -2.5)), ('Be', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 2.5))]  # stretched
# geometry = [('H', (0.0, 0.0, -4.0)), ('Be', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 4.0))]  # near dissociation

# Carbon (and lighter):

# Methane (CH4):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Central carbon
#     ('H', (1.0900, 0.0000, 0.0000)),  # Hydrogen 1
#     ('H', (-0.3630, 1.0270, 0.0000)),  # Hydrogen 2
#     ('H', (-0.3630, -0.5130, 0.8890)),  # Hydrogen 3
#     ('H', (-0.3630, -0.5130, -0.8890))  # Hydrogen 4
# ]

# Nitrogen (and lighter):

# geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.095))]  # N2 Molecule
# geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 3.0))]  # stretched

# Ammonia:
# geometry = [
#     ('N', (0.0000, 0.0000, 0.0000)),  # Nitrogen at center
#     ('H', (0.9400, 0.0000, -0.3200)),  # Hydrogen 1
#     ('H', (-0.4700, 0.8130, -0.3200)), # Hydrogen 2
#     ('H', (-0.4700, -0.8130, -0.3200)) # Hydrogen 3
# ]

# Oxygen (and lighter):

# geometry = [('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.97))]  # OH- Radical
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.128))]  # CO Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.16)), ('O', (0.0000, 0.0000, -1.16))]  # CO2 Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.55))]  # NO Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 11.5))]  # NO+ Radical
# geometry = [('O', (0.0, 0.0, 0.0)), ('O', (0.0, 0.0, 1.21))]  # O2 Molecule

# geometry = [
#     ('O', (0.0000, 0.0000, 0.1173)),
#     ('H', (0.0000, 0.7572, -0.4692)),
#     ('H', (0.0000, -0.7572, -0.4692))
# ]  # H2O equilibrium, bond length ~0.957 Å, angle ~104.5°

# geometry = [
#     ('O', (0.0000, 0.0000, 0.0000)),
#     ('H', (0.0000, 0.7572, -0.4692)),      # fixed
#     ('H', (0.0000, -2.5000, -0.4692))      # stretched
# ]

# Nitrogen dioxide (toxic pollutant)
# geometry = [
#     ('N', (0.0000,  0.0000,  0.0000)),
#     ('O', (1.2000,  0.0000,  0.0000)),
#     ('O', (-1.2000 * np.cos(np.deg2rad(134)), 1.2000 * np.sin(np.deg2rad(134)), 0.0000))
# ]
# geometry = [('N', (0.0000, 0.0000, 0.0000)), ('N', (1.128, 0.0000, 0.0000)), ('O', (2.241, 0.0000, 0.0000))]  # N2O Molecule

# Ozone:
# geometry = [
#     ('O', (0.0000,  0.0000,  0.0000)),  # Central oxygen
#     ('O', (1.2780,  0.0000,  0.0000)),  # One oxygen
#     ('O', (-1.2780 * np.cos(np.deg2rad(117)), 1.2780 * np.sin(np.deg2rad(117)), 0.0000))  # The other oxygen
# ]

# H3O+ (Hydronium Ion):
# bond_length = 10.3  # Scaled bond length (1.03 Å in script units)
# bond_angle = 113.0  # Approximate bond angle of H3O+
# Convert angles to radians
# theta = np.deg2rad(bond_angle)
# Final geometry list
# geometry = [
#     ('O', (0.0, 0.0, 0.0)),
#     ('H', (bond_length, 0.0, 0.0)),
#     ('H', (-bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0)),
#     ('H', (-bond_length * np.cos(theta), -bond_length * np.sin(theta), 0.0))
# ]

# Carbonate ion (CO3--):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon at center
#     ('O', (1.2900, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-0.6450, 1.1180, 0.0000)),  # Oxygen 2
#     ('O', (-0.6450, -1.1180, 0.0000))  # Oxygen 3
# ]

# Bicarbonate ion (HCO3-):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon center
#     ('O', (1.2200, 0.0000, 0.0000)),  # Oxygen (C=O)
#     ('O', (-0.6100, 1.0550, 0.0000)),  # Oxygen (-O⁻)
#     ('O', (-0.6100, -1.0550, 0.0000)),  # Oxygen (OH)
#     ('H', (-1.2200, -1.0550, 0.0000))  # Hydrogen in OH group
# ]

# Sulfur chemistry:

# Hydrogen sulfide (rotten egg smell, major biologic sulfur compound):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('H', (1.3400, 0.0000, 0.0000)),
#     ('H', (-1.3400 * np.cos(np.deg2rad(92)), 1.3400 * np.sin(np.deg2rad(92)), 0.0000))
# ]

# Sulfur dioxide (major volcanic gas and pollutant):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),  # Sulfur at center
#     ('O', (1.4300, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.4300 * np.cos(np.deg2rad(119)), 1.4300 * np.sin(np.deg2rad(119)), 0.0000))  # Oxygen 2
# ]

# Sulfur trioxide (key in acid rain, forms H₂SO₄):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4200, 0.0000, 0.0000)),
#     ('O', (-0.7100, 1.2290, 0.0000)),
#     ('O', (-0.7100, -1.2290, 0.0000))
# ]

# Sulfate ion (SO4--, major oceanic anion, ionically bonds to Mg++):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4900, 0.0000, 0.0000)),
#     ('O', (-0.7450, 1.2900, 0.0000)),
#     ('O', (-0.7450, -1.2900, 0.0000)),
#     ('O', (0.0000, 0.0000, 1.4900))
# ]

# Oceanic electrolytes (consider isolating cations and anions as single atoms with excess charge):

# Sodium chloride:
# geometry = [
#     ('Na', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.3600, 0.0000, 0.0000))
# ]

# Potassium chloride (biologically important):
# geometry = [
#     ('K', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.6700, 0.0000, 0.0000))
# ]

# Calcium chloride:
# geometry = [
#     ('Ca', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.7800, 0.0000, 0.0000)),
#     ('Cl', (-2.7800, 0.0000, 0.0000))
# ]

# Diatomic halogens:

# Fluorine gas (F2)
# geometry = [('F', (0.0000, 0.0000, 0.0000)), ('F', (1.4200, 0.0000, 0.0000))]

# Chlorine gas (Cl2)
# geometry = [('Cl', (0.0000, 0.0000, 0.0000)), ('Cl', (1.9900, 0.0000, 0.0000))]

# Silicon dioxide (quartz, sand, granite):
# geometry = [
#     ('Si', (0.0000, 0.0000, 0.0000)),  # Silicon at center
#     ('O', (1.6200, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.6200, 0.0000, 0.0000))  # Oxygen 2
# ]

# The above are major atmospheric, oceanic, and soil components on Earth.
# Proper organic chemistry is beyond the scope of this script,
# but we give a memorable token example of a carbon ring.

# Benzene (C6H6)

# Define bond lengths (in angstroms, converted to script units)
# C_C = 13.9  # Carbon-carbon bond (1.39 Å)
# C_H = 10.9  # Carbon-hydrogen bond (1.09 Å)

# Angle of 120° between C-C bonds in the hexagonal ring
# theta = np.deg2rad(120)

# Define carbon positions (hexagonal ring)
# geometry = [
#     ('C', (C_C, 0.0, 0.0)),  # First carbon at x-axis
#     ('C', (C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C, 0.0, 0.0)),
#     ('C', (-C_C * np.cos(theta), -C_C * np.sin(theta), 0.0)),
#     ('C', (C_C * np.cos(theta), -C_C * np.sin(theta), 0.0))
# ]

# Define hydrogen positions (bonded to carbons)
# for i in range(6):
#     x, y, z = geometry[i][1]  # Get carbon position
#     hydrogen_x = x + (C_H * (x / C_C))  # Extend outward along C-C axis
#     hydrogen_y = y + (C_H * (y / C_C))
#     hydrogen_z = z  # Planar
#     geometry.append(('H', (hydrogen_x, hydrogen_y, hydrogen_z)))

# Now, `geometry` contains all 6 carbons and 6 hydrogens!

# Step 2: Create OpenFermion molecule
def geometry_to_atom_str(geometry):
    return "; ".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in geometry
    )


# ── interaction graph ────────────────────────────────────────────────────────

def build_interaction_graph(fermion_ham, n_qubits):
    """
    Extract undirected edges (i, j) with i < j where qubits i and j
    appear together in at least one JW Pauli string.
    """
    edges = set()
    for term, coeff in fermion_ham.terms.items():
        jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
        for pauli_string, _ in jw_term.terms.items():
            qubits = [q for q, op in pauli_string if op != 'I']
            for a, b in itertools.combinations(sorted(qubits), 2):
                edges.add((a, b))
    return sorted(edges)


# ── Phase 1: Z-basis bootstrap ───────────────────────────────────────────────
#
# Retain only pure-Z Hamiltonian terms (drop X/Y); find the computational-basis
# state that minimises this diagonal part.  This gives a physically grounded
# starting point for the variational phase.

def initial_energy(theta, z_hamiltonian):
    energy = 0.0
    for qubits, coeff in z_hamiltonian:
        sign = 1
        for q in qubits:
            if theta[q]:
                sign *= -1
        energy += sign * coeff
    return energy


def bootstrap_worker(theta, z_hamiltonian, indices):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    return initial_energy(local_theta, z_hamiltonian)


def bootstrap_round(theta, z_hamiltonian, k, combos):
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        args = [(theta, z_hamiltonian, combos[j*k:(j+1)*k])
                for j in range(len(combos) // k)]
        energies = pool.starmap(bootstrap_worker, args)
    return energies


def phase1_bootstrap(z_hamiltonian, z_qubits, n_qubits, reheat_tries=1):
    best_theta        = np.random.randint(2, size=n_qubits).astype(bool)
    min_energy        = initial_energy(best_theta, z_hamiltonian)
    combos_list       = []
    reheat_theta      = best_theta.copy()
    reheat_min_energy = min_energy
    nz                = len(z_qubits)

    for reheat_round in range(reheat_tries + 1):
        improved = True
        quality  = 1
        while improved:
            improved = False
            k = 1
            while k <= quality:
                if nz < k:
                    break
                if len(combos_list) < k:
                    combos = list(
                        item
                        for sublist in itertools.combinations(z_qubits, k)
                        for item in sublist
                    )
                    combos_list.append(combos)
                else:
                    combos = combos_list[k - 1]

                energies    = bootstrap_round(reheat_theta, z_hamiltonian, k, combos)
                energy      = min(energies)
                idx         = energies.index(energy)
                indices     = combos[idx*k:(idx+1)*k]

                if energy < reheat_min_energy:
                    reheat_min_energy = energy
                    for i in indices:
                        reheat_theta[i] = not reheat_theta[i]
                    improved = True
                    if quality < k + 1:
                        quality = k + 1
                    if reheat_min_energy < min_energy:
                        print(f"  [P1] Qubits {indices} flipped. Energy: {reheat_min_energy:.10f}")
                    break
                k += 1
                print("  [P1] All flips rejected.")

        if min_energy < reheat_min_energy:
            reheat_theta      = best_theta.copy()
            reheat_min_energy = min_energy
        else:
            best_theta = reheat_theta.copy()
            min_energy = reheat_min_energy

        if reheat_round < reheat_tries:
            print("  [P1] Reheating...")
            for bit in random.sample(z_qubits, max(1, int(np.round(np.log2(nz))))):
                reheat_theta[bit] = not reheat_theta[bit]
            reheat_min_energy = initial_energy(reheat_theta, z_hamiltonian)

    return best_theta, min_energy


# ── Phase 2: interaction-graph weak-entanglement VQE ─────────────────────────
#
# Ansatz (one layer):
#   1. X(i)          if bootstrap theta[i] == 1   (initialise to bootstrap state)
#   2. RY(delta[i])  for each qubit i              (single-qubit rotation)
#   3. RZZ(gamma[e]) for each edge e in the        (weak entanglement on interaction
#                    interaction graph              graph edges only)
#
# Parameters: delta (n_qubits,) + gamma (n_edges,), all initialised near zero.
# The RZZ gate is exp(-i gamma/2 Z⊗Z), a natural "Ising-coupling" perturbation
# around the bootstrap product state.  gamma=0 recovers the pure product state.

def fit_ig_entanglement(pl_hamiltonian, bootstrap_theta, n_qubits, ig_edges,
                        bootstrap_energy, n_steps=100, stepsize=np.pi/180):

    n_edges = len(ig_edges)
    print(f"  [P2] Interaction graph: {n_edges} edges.")
    print(f"  [P2] Variational parameters: {n_qubits} (RY) + {n_edges} (RZZ) = "
          f"{n_qubits + n_edges} total.")

    # Fast low-width simulation:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    # Ideal simulation with "automatic circuit elision" approximation for large circuits:
    # dev = qml.device("qrack.simulator", wires=n_qubits)
    # Does heuristic Clifford+RZ gate set approximation:
    # dev = qml.device("qrack.stabilizer", wires=n_qubits)

    # Alternative near-Clifford version requires "weak simulation condition"
    # @qml.set_shots(100)
    # @qml.qnode(dev, mcm_method="one-shot")
    # State vector option
    @qml.qnode(dev)
    def circuit(delta, gamma):
        # Initialise to bootstrap computational-basis state
        for i in range(n_qubits):
            if bootstrap_theta[i]:
                qml.X(wires=i)

        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RY(delta[i], wires=i)

        # Alternative near-Clifford single-qubit rotations
        # for i in range(n_qubits):
        #     qml.Hadamard(wires=i)
        #     qml.RZ(delta[i], wires=i)
        #     qml.Hadamard(wires=i)

        # Weak entanglement: one RZZ per interaction-graph edge
        for idx, (a, b) in enumerate(ig_edges):
            qml.IsingZZ(gamma[idx], wires=[a, b])

        # Alternative near-Clifford weak entanglement
        # for idx, (a, b) in enumerate(ig_edges):
        #     qml.CNOT(wires=[a, b])
        #     qml.RZ(gamma[idx], wires=b)
        #     qml.CNOT(wires=[a, b])

        return qml.expval(pl_hamiltonian)

    delta = nppl.array(np.random.uniform(-0.1, 0.1, n_qubits), requires_grad=True)
    gamma = nppl.array(np.random.uniform(-0.1, 0.1, n_edges), requires_grad=True)

    opt        = qml.AdamOptimizer(stepsize=stepsize)
    min_energy = bootstrap_energy
    best_delta = delta.copy()
    best_gamma = gamma.copy()

    for step in range(n_steps):
        (delta, gamma), energy = opt.step_and_cost(circuit, delta, gamma)
        print(f"  Step {step+1:3d}: Energy = {float(energy):.10f} Ha")
        if float(energy) < min_energy:
            min_energy = float(energy)
            best_delta = delta.copy()
            best_gamma = gamma.copy()

    return best_delta, best_gamma, min_energy


# ── main loop ────────────────────────────────────────────────────────────────

is_charge_update = True
while is_charge_update:
    is_charge_update = False

    molecule_of = MolecularData(geometry, basis, multiplicity=multiplicity, charge=charge)
    molecule_of = run_pyscf(molecule_of, run_scf=True, run_mp2=False, run_cisd=False,
                            run_ccsd=False, run_fci=False)
    fermion_ham = get_fermion_operator(molecule_of.get_molecular_hamiltonian())
    n_electrons = molecule_of.n_electrons
    n_qubits    = molecule_of.n_qubits
    print(f"Hartree-Fock energy:  {molecule_of.hf_energy:.10f} Ha")
    print(f"{n_electrons} electrons, {n_qubits} qubits")

    # ── build interaction graph (from full JW Hamiltonian) ───────────────────
    ig_edges = build_interaction_graph(fermion_ham, n_qubits)
    print(f"Interaction graph: {len(ig_edges)} undirected edges")

    # ── build Z-only Hamiltonian for bootstrap ───────────────────────────────
    z_hamiltonian = []
    z_qubits      = set()
    for term, coeff in fermion_ham.terms.items():
        jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
        for pauli_string, jw_coeff in jw_term.terms.items():
            if any(op in ('X', 'Y') for _, op in pauli_string):
                continue
            q = []
            for qubit, op in pauli_string:
                if op == 'Z':
                    q.append(qubit)
                    z_qubits.add(qubit)
            z_hamiltonian.append((q, jw_coeff.real))
    z_qubits = list(z_qubits)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("\n── Phase 1: Z-basis bootstrap ──")
    theta, energy_p1 = phase1_bootstrap(z_hamiltonian, z_qubits, n_qubits, reheat_tries=1)
    print(f"\nPhase 1 energy: {energy_p1:.10f} Ha")
    print(f"  θ: {theta.astype(int)}")

    # Charge/multiplicity self-consistency check
    r_electrons    = int(theta.sum())
    d_electrons    = r_electrons - n_electrons
    r_charge       = charge - d_electrons
    r_multiplicity = 1
    for i in range(0, len(theta), 2):
        if theta[i] != theta[i + 1]:
            r_multiplicity += 1

    if n_electrons != r_electrons or multiplicity != r_multiplicity:
        print("\nRegressed electron count or multiplicity doesn't match assumptions!")
        print(f"  charge = {r_charge},  multiplicity = {r_multiplicity}")
        charge           = r_charge
        multiplicity     = r_multiplicity
        is_charge_update = True
        continue

    # ── build PennyLane Hamiltonian (full, including X/Y terms) ─────────────
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
            if ops:
                obs = ops[0]
                for op in ops[1:]:
                    obs = obs @ op
                observables.append(obs)
            else:
                observables.append(qml.Identity(0))
            coeffs.append(nppl.array(jw_coeff.real, requires_grad=False))
    pl_hamiltonian = qml.Hamiltonian(coeffs, observables)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print("\n── Phase 2: interaction-graph weak-entanglement VQE ──")
    best_delta, best_gamma, energy_p2 = fit_ig_entanglement(
        pl_hamiltonian, theta, n_qubits, ig_edges, energy_p1,
        n_steps=100, stepsize=np.pi/180
    )

    print(f"\n{'─'*55}")
    print(f"Hartree-Fock energy:        {molecule_of.hf_energy:.10f} Ha")
    print(f"Phase 1 (Z bootstrap):      {energy_p1:.10f} Ha")
    print(f"Phase 2 (IG-VQE):           {energy_p2:.10f} Ha")
    print(f"{'─'*55}")
    print(f"Bootstrap θ:  {theta.astype(int)}")
    print(f"RY  δ:        {np.array(best_delta)}")
    if len(ig_edges) > 0:
        print(f"RZZ γ edges:  {ig_edges}")
        print(f"RZZ γ values: {np.array(best_gamma)}")
    else:
        print("No interaction-graph edges — product state is the full ansatz.")
