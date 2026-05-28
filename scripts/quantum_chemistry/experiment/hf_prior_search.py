# Exhaustive Hartree-Fock prior search over charge and multiplicity
#
# Systematically tries all physically plausible (charge, multiplicity)
# combinations for a given geometry and basis, runs PySCF for each,
# prints improvements as found, and returns the best priors and HF energy.
#
# Developed with help from (Anthropic) Claude.

from openfermion import MolecularData
from openfermionpyscf import run_pyscf

import numpy as np


# ---------------------------------------------------------------------------
# Molecule definition
# ---------------------------------------------------------------------------

# Search window: charges from -charge_radius to +charge_radius, inclusive.
charge_radius = 2

# Step 1: Define the molecule (Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen)

# basis = "sto-3g"  # Minimal Basis Set
# basis = '6-31g'  # Larger basis set
basis = 'cc-pVDZ' # Even larger basis set!

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
C_C = 1.39  # Carbon-carbon bond (1.39 Å)
C_H = 1.09  # Carbon-hydrogen bond (1.09 Å)

# Angle of 120° between C-C bonds in the hexagonal ring
theta = np.deg2rad(120)

# Define carbon positions (hexagonal ring)
geometry = [
    ('C', (C_C, 0.0, 0.0)),  # First carbon at x-axis
    ('C', (C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
    ('C', (-C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
    ('C', (-C_C, 0.0, 0.0)),
    ('C', (-C_C * np.cos(theta), -C_C * np.sin(theta), 0.0)),
    ('C', (C_C * np.cos(theta), -C_C * np.sin(theta), 0.0))
]

# Define hydrogen positions (bonded to carbons)
for i in range(6):
    x, y, z = geometry[i][1]  # Get carbon position
    hydrogen_x = x + (C_H * (x / C_C))  # Extend outward along C-C axis
    hydrogen_y = y + (C_H * (y / C_C))
    hydrogen_z = z  # Planar
    geometry.append(('H', (hydrogen_x, hydrogen_y, hydrogen_z)))

# Now, `geometry` contains all 6 carbons and 6 hydrogens!


# ---------------------------------------------------------------------------
# Enumerate valid (charge, multiplicity) pairs
# ---------------------------------------------------------------------------

def neutral_electron_count(geometry):
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
        'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
        'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20,
    }
    return sum(atomic_numbers[symbol] for symbol, _ in geometry)


def valid_charge_multiplicity_pairs(n_neutral, charge_radius):
    pairs = []
    for charge in range(-charge_radius, charge_radius + 1):
        n_electrons = n_neutral - charge
        if n_electrons < 1:
            continue
        min_mult = 1 if (n_electrons % 2 == 0) else 2
        for multiplicity in range(min_mult, n_electrons + 2, 2):
            pairs.append((charge, multiplicity))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_neutral = neutral_electron_count(geometry)
    pairs = valid_charge_multiplicity_pairs(n_neutral, charge_radius)

    print(f"Neutral electron count: {n_neutral}")
    print(f"Searching {len(pairs)} (charge, multiplicity) combinations...")
    print()

    best_energy = None
    best_result = None

    for charge, multiplicity in pairs:
        label = f"charge={charge:+d}, multiplicity={multiplicity}"
        try:
            molecule = MolecularData(geometry, basis,
                                     multiplicity=multiplicity, charge=charge)
            molecule = run_pyscf(molecule, run_scf=True, run_mp2=False,
                                 run_cisd=False, run_ccsd=False, run_fci=False)
        except Exception as e:
            print(f"  [{label}] PySCF failed: {e}")
            continue

        energy = molecule.hf_energy
        print(f"  [{label}] HF energy = {energy:.10f} Ha")

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_result = {'charge': charge, 'multiplicity': multiplicity, 'hf_energy': energy}
            print(f"    *** New best: {energy:.10f} Ha ***")

    print()
    print("=" * 60)
    print("Best Hartree-Fock priors:")
    print(f"  charge       = {best_result['charge']}")
    print(f"  multiplicity = {best_result['multiplicity']}")
    print(f"  HF energy    = {best_result['hf_energy']:.10f} Ha")
