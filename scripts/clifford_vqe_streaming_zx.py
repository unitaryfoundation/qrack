# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# (Requires OpenFermion)

from pyscf import gto, scf, ao2mo
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

# Hydrogen (and lighter):

# geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 Molecule

# Helium (and lighter):

# geometry = [('He', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.774))]  # HeH Molecule

# Lithium (and lighter):

# geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.595))]  # LiH Molecule

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

# geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.10))]  # N2 Molecule

# Ammonia:
geometry = [
    ('N', (0.0000, 0.0000, 0.0000)),  # Nitrogen at center
    ('H', (0.9400, 0.0000, -0.3200)),  # Hydrogen 1
    ('H', (-0.4700, 0.8130, -0.3200)), # Hydrogen 2
    ('H', (-0.4700, -0.8130, -0.3200)) # Hydrogen 3
]

# Oxygen (and lighter):

# geometry = [('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.97))]  # OH- Radical
# geometry = [('O', (0.0000, 0.0000, 0.0000)), ('H', (0.7586, 0.0000, 0.5043)),  ('H', (-0.7586, 0.0000, 0.5043))]  # H2O Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.128))]  # CO Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.16)), ('O', (0.0000, 0.0000, -1.16))]  # CO2 Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.55))]  # NO Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 11.5))]  # NO+ Radical
# geometry = [('O', (0.0, 0.0, 0.0)), ('O', (0.0, 0.0, 1.21))]  # O2 Molecule

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

# Step 2: Compute the Molecular Hamiltonian
def geometry_to_atom_str(geometry):
    """Convert list of (symbol, (x,y,z)) to Pyscf atom string."""
    return "; ".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in geometry
    )

# Convert and feed to gto.M()
atom_str = geometry_to_atom_str(geometry)

mol = gto.M(
    atom=atom_str,
    basis=basis
)

mf = scf.RHF(mol).run()

# Step 3: Create OpenFermion molecule
molecule_of = MolecularData(geometry, basis, multiplicity=1, charge=0)
molecule_of = run_pyscf(molecule_of, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)
fermion_ham = get_fermion_operator(molecule_of.get_molecular_hamiltonian())
# n_electrons = molecule_of.n_electrons
n_qubits = mol.nao << 1
print(f"{n_qubits} qubits...")

# Step 4: Bootstrap!
def compute_energy(theta_bits, phi_bits):
    energy = 0.0
    for term, coeff in fermion_ham.terms.items():
        jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))  # Transform single term

        for pauli_string, jw_coeff in jw_term.terms.items():
            is_sat = True
            for qubit, op in pauli_string:
                if (op != 'Z') != phi_bits[qubit]:
                    is_sat = False
                    break
            if not is_sat:
                continue

            term_value = jw_coeff.real
            for qubit, op in pauli_string:
                # Z/I terms: keep only Z
                if op == 'I':
                    continue
                if theta_bits[qubit]:
                    term_value *= -1
            energy += term_value

    return energy


def bootstrap_worker(theta, phi, d_theta, d_phi):
    local_theta = theta.copy()
    for i in d_theta:
        local_theta[i] = not local_theta[i]
    local_phi = phi.copy()
    for i in d_phi:
        local_phi[i] = not local_phi[i]
    energy = compute_energy(local_theta, local_phi)

    return energy


def bootstrap(theta, phi, k, indices_array):
    n = len(indices_array) // k
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        args = []
        for i in range(n):
            j = i * k
            args.append((theta, phi, indices_array[j : j + k], []))
        for i in range(n):
            j = i * k
            args.append((theta, phi, [], indices_array[j : j + k]))
        for i in range(n):
            j = i * k
            args.append((theta, phi, indices_array[j : j + k], indices_array[j : j + k]))
        energies = pool.starmap(bootstrap_worker, args)

    return energies


def multiprocessing_bootstrap(n_qubits, reheat_tries=0):
    best_theta = np.random.randint(2, size=n_qubits)
    best_phi = np.random.randint(2, size=n_qubits)
    min_energy = compute_energy(best_theta, best_phi)

    combos_list = []
    reheat_theta = best_theta.copy()
    reheat_phi = best_phi.copy()
    reheat_min_energy = min_energy
    for reheat_round in range(reheat_tries + 1):
        improved = True
        quality = 1
        while improved:
            improved = False
            k = 1
            while k <= quality:
                if n_qubits < k:
                    break

                if len(combos_list) < k:
                    combos = np.array(list(
                        item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
                    ))
                    combos_list.append(combos)
                else:
                    combos = combos_list[k - 1]

                energies = bootstrap(reheat_theta, reheat_phi, k, combos)

                energy = min(energies)

                if energy < reheat_min_energy:
                    reheat_min_energy = energy
                    index_match = energies.index(energy)

                    n_combos = len(combos)
                    is_theta = (index_match < n_combos) or (index_match >= (n_combos << 1))
                    is_phi = index_match >= n_combos
                    index_match %= n_combos
                    indices = combos[index_match * k : (index_match + 1) * k]
                    if is_theta:
                        for i in indices:
                            reheat_theta[i] = not reheat_theta[i]
                    if is_phi:
                        for i in indices:
                            reheat_phi[i] = not reheat_phi[i]

                    improved = True
                    if quality < (k + 1):
                        quality = k + 1
                    if reheat_min_energy < min_energy:
                        print(f"  Qubits {indices} flip accepted. New energy: {reheat_min_energy}")
                        print(f"  θ: {reheat_theta}")
                        print(f"  φ: {reheat_phi}")
                    break

                k = k + 1
                print("  Qubit flips all rejected.")

        if min_energy < reheat_min_energy:
            reheat_theta = best_theta.copy()
            reheat_phi = best_phi.copy()
            reheat_min_energy = min_energy
        else:
            best_theta = reheat_theta.copy()
            best_phi = reheat_phi.copy()
            min_energy = reheat_min_energy

        if reheat_round < reheat_tries:
            print("  Reheating...")
            num_to_flip = int(np.round(np.log2(len(reheat_theta))))
            z_bits_to_flip = random.sample(list(range(n_qubits)), num_to_flip)
            for bit in z_bits_to_flip:
                reheat_theta[bit] = not reheat_theta[bit]
            x_bits_to_flip = random.sample(list(range(n_qubits)), num_to_flip)
            for bit in x_bits_to_flip:
                reheat_phi[bit] = not reheat_phi[bit]
            reheat_min_energy = compute_energy(reheat_theta, reheat_phi)

    return best_theta, best_phi, min_energy


# Run threaded bootstrap
theta, phi, min_energy = multiprocessing_bootstrap(n_qubits, 2)

print(f"\nFinal Bootstrap Ground State Energy: {min_energy} Ha")
print("Final Bootstrap Parameters:")
print(f"  θ: {theta}")
print(f"  φ: {phi}")
