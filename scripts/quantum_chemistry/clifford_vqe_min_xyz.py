# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# Adapted to atomic checks of Z/X/Y basis by (Anthropic) Claude
# (Requires OpenFermion)

from openfermion import MolecularData, FermionOperator, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

import itertools
import multiprocessing
import numpy as np
import os
import random


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
    """Convert list of (symbol, (x,y,z)) to Pyscf atom string."""
    return "; ".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in geometry
    )


# phi encodes Pauli basis per qubit: 0=Z, 1=X, 2=Y
# theta encodes |0> vs |1> in that basis (False=|0>, True=|1>)
#
# For a single-qubit Clifford state |b>_P:
#   <b|_P sigma_P |b>_P = (-1)^b  (eigenvalue of the matching Pauli)
#   <b|_P sigma_Q |b>_P = 0       (orthogonal Pauli, no contribution)
#
# So a term contributes only when ALL its qubits are measured in the
# matching basis, and its sign is the product of (-1)^theta[qubit].

def compute_energy(theta, phi, zxy_hamiltonian):
    energy = 0.0
    for qubits, bases, coeff in zxy_hamiltonian:
        # bases[i] is 0/1/2 for Z/X/Y; phi[qubit] must match for term to contribute
        is_sat = all(phi[qubits[i]] == bases[i] for i in range(len(qubits)))
        if not is_sat:
            continue
        for qubit in qubits:
            if theta[qubit]:
                coeff *= -1
        energy += coeff
    return energy


# A candidate update is a pair (new_theta_slice, new_phi_slice) for a subset of qubits,
# tested atomically. We enumerate all 6 single-qubit Clifford states per qubit in the
# subset, and all combinations across qubits for k>1.

def bootstrap_worker(theta, phi, zxy_hamiltonian, qubit_indices, new_states):
    """new_states: list of (theta_val, phi_val) per qubit in qubit_indices."""
    local_theta = theta.copy()
    local_phi = phi.copy()
    for idx, (t, p) in zip(qubit_indices, new_states):
        local_theta[idx] = t
        local_phi[idx] = p
    return compute_energy(local_theta, local_phi, zxy_hamiltonian)


def bootstrap(theta, phi, zxy_hamiltonian, k, qubit_combos):
    """
    For each combination of k qubits, test all 6^k assignments of
    (theta, phi) per qubit atomically. Returns list of (energy, qubit_indices, new_states).
    """
    # 6 single-qubit Clifford states: (theta_val, phi_val)
    clifford_states = [
        (False, 0), (True, 0),   # |0>_Z, |1>_Z
        (False, 1), (True, 1),   # |0>_X, |1>_X
        (False, 2), (True, 2),   # |0>_Y, |1>_Y
    ]

    args = []
    meta = []
    for combo in qubit_combos:
        for new_states in itertools.product(clifford_states, repeat=k):
            # Skip if this assignment matches the current state (no change)
            if all(
                theta[combo[i]] == new_states[i][0] and phi[combo[i]] == new_states[i][1]
                for i in range(k)
            ):
                continue
            args.append((theta, phi, zxy_hamiltonian, list(combo), list(new_states)))
            meta.append((list(combo), list(new_states)))

    if not args:
        return []

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        energies = pool.starmap(bootstrap_worker, args)

    return list(zip(energies, meta))


def multiprocessing_bootstrap(zxy_hamiltonian, n_qubits, reheat_tries=0):
    # Initialise randomly across all 6 Clifford states
    best_theta = np.array([bool(random.randint(0, 1)) for _ in range(n_qubits)])
    best_phi   = np.array([random.randint(0, 2)       for _ in range(n_qubits)])
    min_energy = compute_energy(best_theta, best_phi, zxy_hamiltonian)

    combos_list = []
    reheat_theta = best_theta.copy()
    reheat_phi   = best_phi.copy()
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

                results = bootstrap(reheat_theta, reheat_phi, zxy_hamiltonian, k, combos)

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
                        basis_names = {0: 'Z', 1: 'X', 2: 'Y'}
                        state_strs = [f"|{'1' if t else '0'}>_{basis_names[p]}"
                                      for t, p in new_states]
                        print(f"  Qubits {qubit_indices} -> {state_strs}. New energy: {reheat_min_energy}")
                        print(f"  θ: {reheat_theta.astype(int)}")
                        print(f"  φ: {reheat_phi}")
                    break

                k += 1
                print("  Qubit updates all rejected.")

        if min_energy < reheat_min_energy:
            reheat_theta      = best_theta.copy()
            reheat_phi        = best_phi.copy()
            reheat_min_energy = min_energy
        else:
            best_theta = reheat_theta.copy()
            best_phi   = reheat_phi.copy()
            min_energy = reheat_min_energy

        if reheat_round < reheat_tries:
            print("  Reheating...")
            num_to_flip = int(np.round(np.log2(n_qubits)))
            for bit in random.sample(range(n_qubits), num_to_flip):
                reheat_theta[bit] = bool(random.randint(0, 1))
                reheat_phi[bit]   = random.randint(0, 2)
            reheat_min_energy = compute_energy(reheat_theta, reheat_phi, zxy_hamiltonian)

    return best_theta, best_phi, min_energy


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

    # Build ZXY Hamiltonian: retain Z, X, and Y terms.
    # A Pauli string contributes only when all its qubits are in the matching basis.
    # bases[i] = 0 (Z), 1 (X), or 2 (Y).
    zxy_hamiltonian = []
    for term, coeff in fermion_ham.terms.items():
        jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))

        for pauli_string, jw_coeff in jw_term.terms.items():
            q = []
            b = []
            for qubit, op in pauli_string:
                if op == 'I':
                    continue
                if op == 'Z':
                    b.append(0)
                elif op == 'X':
                    b.append(1)
                elif op == 'Y':
                    b.append(2)
                q.append(qubit)
            zxy_hamiltonian.append((q, b, jw_coeff.real))

    # Bootstrap!
    theta, phi, min_energy = multiprocessing_bootstrap(zxy_hamiltonian, n_qubits, 1)

    print(f"\nFinal Bootstrap Ground State Energy: {min_energy} Ha")
    print("Final Bootstrap Parameters:")
    basis_names = {0: 'Z', 1: 'X', 2: 'Y'}
    print(f"  θ: {theta.astype(int)}")
    print(f"  φ: {[basis_names[p] for p in phi]}")

    # Electron count: in Z basis a |1> is a full electron;
    # in X or Y basis a |1> contributes 1/2 (superposition of occupied/unoccupied).
    r_electrons = 0
    for i in range(n_qubits):
        if theta[i]:
            r_electrons += 1 if phi[i] == 0 else 0.5
    if int(r_electrons) != r_electrons:
        print("Whoops! We don't have an integer number of charges!")
        break
    r_electrons = int(r_electrons)

    d_electrons  = r_electrons - n_electrons
    r_charge     = charge - d_electrons
    r_multiplicity = 1
    for i in range(0, len(theta), 2):
        if theta[i] != theta[i + 1]:
            r_multiplicity += 1

    if n_electrons != r_electrons or multiplicity != r_multiplicity:
        print()
        print("Regressed electron count or multiplicity doesn't match the assumptions!")
        print("Running again with the natural parameters replacing your assumptions:")
        print(f"charge = {r_charge}")
        print(f"multiplicity = {r_multiplicity}")
        print()
        charge       = r_charge
        multiplicity = r_multiplicity
        is_charge_update = True
