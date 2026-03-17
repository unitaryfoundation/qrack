# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# Adapted to atomic checks of Z/X/Y basis and CNOT interaction graph by (Anthropic) Claude
# (Requires OpenFermion)

from openfermion import MolecularData, FermionOperator, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

import itertools
import multiprocessing
import numpy as np
import os
import random


# Step 1: Define the molecule (Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen)

# basis = "sto-3g"  # Minimal Basis Set
basis = '6-31g'  # Larger basis set
# basis = 'cc-pVDZ' # Even larger basis set!
multiplicity = 4  # singlet, closed shell, all electrons are paired (neutral molecules with full valence)
# multiplicity = 2  # doublet, one unpaired electron (ex.: OH- radical)
# multiplicity = 3  # triplet, two unpaired electrons (ex.: O2)
charge = -1  # Excess +/- elementary charge, beyond multiplicity

print(f"charge = {charge}")
print(f"multiplicity = {multiplicity}")

# Hydrogen (and lighter):

# geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 Molecule

geometry = [
    ("H", (-1.0, 0.0, -1.0)), ("H", (-1.0, 0.0, 1.00)),
    ("H", (1.0, 0.0, -1.0)), ("H", (1.0, 0.0, 1.00))
]  # H4 Dissociation (hard for Hartree-Fock)

# Helium (and lighter):

# geometry = [('He', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.774))]  # HeH Molecule

# Lithium (and lighter):

# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.596))]  # equilibrium
# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 2.5))]   # stretched
# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 4.0))]   # near dissociation

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


# ── Hamiltonian representation ───────────────────────────────────────────────
#
# Each term: (qubit_list, basis_list, coeff)
# basis: 0=Z, 1=X, 2=Y
# Single-qubit state: theta[q] = False/True (|0>/|1>), phi[q] = 0/1/2 (Z/X/Y basis)
#
# Expectation value of a product Pauli on a product Clifford state:
#   - Term contributes only if every qubit in the term is measured in the matching basis.
#   - Sign: product of (-1)^theta[q] for each qubit in the term.

def compute_energy(theta, phi, hamiltonian):
    energy = 0.0
    for qubits, bases, coeff in hamiltonian:
        if not all(phi[qubits[i]] == bases[i] for i in range(len(qubits))):
            continue
        c = coeff
        for q in qubits:
            if theta[q]:
                c *= -1
        energy += c
    return energy


# ── CNOT conjugation ─────────────────────────────────────────────────────────
#
# CNOT(c, t) maps a two-qubit Pauli P_c ⊗ P_t to another Pauli (possibly with sign).
# We represent basis as integer: 0=I, 1=X, 2=Y, 3=Z  (internally for conjugation only).
# After conjugation we drop any I's and reconstruct the term.

# CNOT conjugation table: (P_c, P_t) -> (sign, P_c', P_t')
# Using: I=0, X=1, Y=2, Z=3
_CNOT_TABLE = {
    (0, 0): ( 1, 0, 0),  # II -> II
    (0, 1): ( 1, 0, 1),  # IX -> IX
    (0, 2): ( 1, 3, 2),  # IY -> ZY
    (0, 3): ( 1, 3, 3),  # IZ -> ZZ
    (1, 0): ( 1, 1, 1),  # XI -> XX
    (1, 1): ( 1, 1, 0),  # XX -> XI
    (1, 2): (-1, 2, 3),  # XY -> -YZ
    (1, 3): (-1, 2, 2),  # XZ -> -YY  (standard: XZ->iYY, but Hermitian obs: sign -1)
    (2, 0): ( 1, 2, 1),  # YI -> YX
    (2, 1): ( 1, 2, 0),  # YX -> YI
    (2, 2): (-1, 1, 3),  # YY -> -XZ
    (2, 3): (-1, 1, 2),  # YZ -> -XY
    (3, 0): ( 1, 3, 0),  # ZI -> ZI
    (3, 1): ( 1, 3, 1),  # ZX -> ZX  (wait: CNOT maps ZX->ZX? yes, control Z unaffected by target X)
    (3, 2): ( 1, 0, 2),  # ZY -> IY
    (3, 3): ( 1, 0, 3),  # ZZ -> IZ
}

# Map our basis encoding (0=Z,1=X,2=Y) to internal (0=I,1=X,2=Y,3=Z)
_TO_INTERNAL = {0: 3, 1: 1, 2: 2}  # Z->3, X->1, Y->2
_FROM_INTERNAL = {3: 0, 1: 1, 2: 2}  # 3->Z=0, 1->X=1, 2->Y=2


def apply_cnot_to_hamiltonian(hamiltonian, control, target):
    """
    Return a new Hamiltonian with CNOT(control, target) conjugated through every term.
    Terms not involving both control and target are unaffected except when one qubit
    appears and the other doesn't (handled by treating the absent qubit as I).
    """
    new_ham = []
    for qubits, bases, coeff in hamiltonian:
        qubit_set = dict(zip(qubits, bases))

        # Get Pauli on control and target (I=identity if absent)
        pc = _TO_INTERNAL.get(qubit_set.get(control, None), 0)  # 0=I if absent
        pt = _TO_INTERNAL.get(qubit_set.get(target, None), 0)

        sign, pc2, pt2 = _CNOT_TABLE[(pc, pt)]
        new_coeff = coeff * sign

        # Rebuild the qubit/basis lists
        new_qubit_set = {q: b for q, b in qubit_set.items()
                         if q != control and q != target}

        if pc2 != 0:  # not identity
            new_qubit_set[control] = _FROM_INTERNAL[pc2]
        if pt2 != 0:
            new_qubit_set[target] = _FROM_INTERNAL[pt2]

        if not new_qubit_set:
            # Scalar term — add to a running constant (include as empty-qubit term)
            new_ham.append(([], [], new_coeff))
        else:
            new_qubits = sorted(new_qubit_set.keys())
            new_bases  = [new_qubit_set[q] for q in new_qubits]
            new_ham.append((new_qubits, new_bases, new_coeff))

    return new_ham


# ── Phase 1: single-qubit Clifford bootstrap ─────────────────────────────────

def bootstrap_worker(theta, phi, hamiltonian, qubit_indices, new_states):
    local_theta = theta.copy()
    local_phi   = phi.copy()
    for idx, (t, p) in zip(qubit_indices, new_states):
        local_theta[idx] = t
        local_phi[idx]   = p
    return compute_energy(local_theta, local_phi, hamiltonian)


def bootstrap_single_qubit(theta, phi, hamiltonian, k, qubit_combos):
    clifford_states = [
        (False, 0), (True, 0),
        (False, 1), (True, 1),
        (False, 2), (True, 2),
    ]
    args = []
    meta = []
    for combo in qubit_combos:
        for new_states in itertools.product(clifford_states, repeat=k):
            if all(theta[combo[i]] == new_states[i][0] and
                   phi[combo[i]]   == new_states[i][1]
                   for i in range(k)):
                continue
            args.append((theta, phi, hamiltonian, list(combo), list(new_states)))
            meta.append((list(combo), list(new_states)))
    if not args:
        return []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        energies = pool.starmap(bootstrap_worker, args)
    return list(zip(energies, meta))


def phase1_bootstrap(hamiltonian, n_qubits, reheat_tries=0):
    best_theta = np.array([bool(random.randint(0, 1)) for _ in range(n_qubits)])
    best_phi   = np.array([random.randint(0, 2)       for _ in range(n_qubits)])
    min_energy = compute_energy(best_theta, best_phi, hamiltonian)

    combos_list       = []
    reheat_theta      = best_theta.copy()
    reheat_phi        = best_phi.copy()
    reheat_min_energy = min_energy

    for reheat_round in range(reheat_tries + 1):
        improved = True
        quality  = 1
        while improved:
            improved = False
            k = 1
            while k <= quality:
                if n_qubits < k:
                    break
                if len(combos_list) < k:
                    combos = list(itertools.combinations(range(n_qubits), k))
                    combos_list.append(combos)
                else:
                    combos = combos_list[k - 1]

                results = bootstrap_single_qubit(reheat_theta, reheat_phi, hamiltonian, k, combos)
                if not results:
                    k += 1
                    continue

                best_result = min(results, key=lambda x: x[0])
                energy, (qubit_indices, new_states) = best_result

                if energy < reheat_min_energy:
                    reheat_min_energy = energy
                    for idx, (t, p) in zip(qubit_indices, new_states):
                        reheat_theta[idx] = t
                        reheat_phi[idx]   = p
                    improved = True
                    if quality < (k + 1):
                        quality = k + 1
                    if reheat_min_energy < min_energy:
                        bnames = {0: 'Z', 1: 'X', 2: 'Y'}
                        ss = [f"|{'1' if t else '0'}>_{bnames[p]}" for t, p in new_states]
                        print(f"  [P1] Qubits {qubit_indices} -> {ss}. Energy: {reheat_min_energy}")
                    break

                k += 1
                print("  [P1] Qubit updates all rejected.")

        if min_energy < reheat_min_energy:
            reheat_theta      = best_theta.copy()
            reheat_phi        = best_phi.copy()
            reheat_min_energy = min_energy
        else:
            best_theta = reheat_theta.copy()
            best_phi   = reheat_phi.copy()
            min_energy = reheat_min_energy

        if reheat_round < reheat_tries:
            print("  [P1] Reheating...")
            num_to_flip = max(1, int(np.round(np.log2(n_qubits))))
            for bit in random.sample(range(n_qubits), num_to_flip):
                reheat_theta[bit] = bool(random.randint(0, 1))
                reheat_phi[bit]   = random.randint(0, 2)
            reheat_min_energy = compute_energy(reheat_theta, reheat_phi, hamiltonian)

    return best_theta, best_phi, min_energy


# ── Phase 2: interaction-graph CNOT search ───────────────────────────────────

def build_interaction_graph(hamiltonian, n_qubits):
    """
    Return sorted list of (control, target) edges where both qubits appear
    together in at least one Hamiltonian term. Directed: both (c,t) and (t,c).
    """
    edges = set()
    for qubits, bases, coeff in hamiltonian:
        if len(qubits) < 2:
            continue
        for c, t in itertools.combinations(qubits, 2):
            edges.add((c, t))
            edges.add((t, c))
    return sorted(edges)


def phase2_cnot_search(theta, phi, hamiltonian, n_qubits, max_rounds=10):
    """
    Greedily apply CNOT(c,t) gates on interaction-graph edges.
    A CNOT is accepted if it lowers the energy of the current state.
    Repeat until no improvement or max_rounds exhausted.

    Note: applying CNOT(c,t) transforms the Hamiltonian rather than the state,
    which is equivalent (conjugation). We track the accumulated CNOT circuit
    separately for reporting.
    """
    edges = build_interaction_graph(hamiltonian, n_qubits)
    print(f"  [P2] Interaction graph: {len(edges)} directed edges.")

    current_ham   = list(hamiltonian)
    current_theta = theta.copy()
    current_phi   = phi.copy()
    current_energy = compute_energy(current_theta, current_phi, current_ham)
    cnot_circuit  = []  # list of (control, target) applied so far

    for round_idx in range(max_rounds):
        improved = False
        for control, target in edges:
            candidate_ham = apply_cnot_to_hamiltonian(current_ham, control, target)
            energy = compute_energy(current_theta, current_phi, candidate_ham)
            if energy < current_energy - 1e-12:
                print(f"  [P2] Round {round_idx+1}: CNOT({control},{target}) accepted. "
                      f"Energy: {current_energy:.10f} -> {energy:.10f}")
                current_ham    = candidate_ham
                current_energy = energy
                cnot_circuit.append((control, target))
                improved = True
                break  # restart edge scan after any improvement

        if not improved:
            print(f"  [P2] Round {round_idx+1}: No improving CNOT found. Stopping.")
            break

    return current_theta, current_phi, current_energy, current_ham, cnot_circuit


# ── main ─────────────────────────────────────────────────────────────────────

is_charge_update = True
while is_charge_update:
    is_charge_update = False

    atom_str    = geometry_to_atom_str(geometry)
    molecule_of = MolecularData(geometry, basis, multiplicity=multiplicity, charge=charge)
    molecule_of = run_pyscf(molecule_of, run_scf=True, run_mp2=False, run_cisd=False,
                            run_ccsd=False, run_fci=False)
    fermion_ham = get_fermion_operator(molecule_of.get_molecular_hamiltonian())
    n_electrons = molecule_of.n_electrons
    n_qubits    = molecule_of.n_qubits
    print(f"Hartree-Fock energy: {molecule_of.hf_energy}")
    print(f"{n_electrons} electrons...")
    print(f"{n_qubits} qubits...")

    # Build ZXY Hamiltonian
    zxy_hamiltonian = []
    for term, coeff in fermion_ham.terms.items():
        jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))
        for pauli_string, jw_coeff in jw_term.terms.items():
            q, b = [], []
            for qubit, op in pauli_string:
                if op == 'I':
                    continue
                q.append(qubit)
                b.append({'Z': 0, 'X': 1, 'Y': 2}[op])
            zxy_hamiltonian.append((q, b, jw_coeff.real))

    # Phase 1: single-qubit Clifford bootstrap
    print("\n── Phase 1: single-qubit Clifford bootstrap ──")
    theta, phi, energy_p1 = phase1_bootstrap(zxy_hamiltonian, n_qubits, reheat_tries=1)
    bnames = {0: 'Z', 1: 'X', 2: 'Y'}
    print(f"\nPhase 1 ground state energy: {energy_p1} Ha")
    print(f"  θ: {theta.astype(int)}")
    print(f"  φ: {[bnames[p] for p in phi]}")

    # Phase 2: interaction-graph CNOT search
    print("\n── Phase 2: interaction-graph CNOT search ──")
    theta2, phi2, energy_p2, final_ham, cnot_circuit = phase2_cnot_search(
        theta, phi, zxy_hamiltonian, n_qubits, max_rounds=20
    )

    print(f"\nPhase 2 ground state energy: {energy_p2} Ha")
    if cnot_circuit:
        print(f"CNOT circuit applied ({len(cnot_circuit)} gates):")
        for c, t in cnot_circuit:
            print(f"  CNOT({c}, {t})")
    else:
        print("No CNOT gates accepted — single-qubit Clifford state is locally optimal.")

    print(f"\nFinal ground state energy: {energy_p2} Ha")
    print(f"  θ: {theta2.astype(int)}")
    print(f"  φ: {[bnames[p] for p in phi2]}")

    # Electron counting (same as before)
    r_electrons = 0
    for i in range(n_qubits):
        if theta2[i]:
            r_electrons += 1 if phi2[i] == 0 else 0.5
    if int(r_electrons) != r_electrons:
        print("Whoops! Non-integer electron count after CNOT phase.")
        break
    r_electrons = int(r_electrons)

    d_electrons    = r_electrons - n_electrons
    r_charge       = charge - d_electrons
    r_multiplicity = 1
    for i in range(0, len(theta2), 2):
        if theta2[i] != theta2[i + 1]:
            r_multiplicity += 1

    if n_electrons != r_electrons or multiplicity != r_multiplicity:
        print()
        print("Regressed electron count or multiplicity doesn't match assumptions!")
        print(f"charge = {r_charge}")
        print(f"multiplicity = {r_multiplicity}")
        charge         = r_charge
        multiplicity   = r_multiplicity
        is_charge_update = True
