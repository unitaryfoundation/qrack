# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

from collections import Counter

import numpy as np

from scipy.stats import distributions as dists

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator
from qiskit.providers.qrack import AceQasmSimulator


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    return circ


def main():
    n_qubits = 16
    depth = 20
    shots = 32768
    t1 = 0
    t2 = 1
    omega = 1.5

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        t1 = float(sys.argv[4])
    if len(sys.argv) > 5:
        shots = int(sys.argv[5])
    else:
        shots = max(65536, 1 << (n_qubits + 2))
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])
    else:
        trials = 8 if t1 > 0 else 1

    print("t1: " + str(t1))
    print("t2: " + str(t2))
    print("omega / pi: " + str(omega))

    omega *= math.pi
    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Coordination number for a square lattice:
    z = 4
    # Mean-field critical angle (in radians)
    theta_c = math.asin(h / (z * J))  # handles signs
    # Set theta relative to that:
    delta_theta = theta - theta_c

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    step = QuantumCircuit(n_qubits)
    trotter_step(step, qubits, (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    depths = list(range(0, depth + 1))
    min_sqr_mag = 1
    results = []
    magnetizations = []

    for trial in range(trials):
        magnetizations.append([])
        experiment = QrackSimulator(n_qubits)

        start = time.perf_counter()

        experiment.run_qiskit_circuit(qc)
        for d in depths:
            bias = []
            t = d * dt
            # Determine how to weight closed-form vs. conventional simulation contributions:
            model = (1 - 1 / math.exp(t / t1)) if (t1 > 0) else 1
            d_magnetization = 0
            d_sqr_magnetization = 0
            if np.isclose(h, 0):
                # This agrees with small perturbations away from h = 0.
                d_magnetization = 1
                d_sqr_magnetization = 1
                bias.append(1)
                bias += n_qubits * [0]
            elif np.isclose(J, 0):
                # This agrees with small perturbations away from J = 0.
                d_magnetization = 0
                d_sqr_magnetization = 0
                bias = (n_qubits + 1) * [1 / (n_qubits + 1)]
            else:
                # ChatGPT o3 suggested this cos_theta correction.
                sin_delta_theta = math.sin(delta_theta)
                # "p" is the exponent of the geometric series weighting, for (n+1) dimensions of Hamming weight.
                # Notice that the expected symmetries are respected under reversal of signs of J and/or h.
                p = (
                    (
                        (2 ** (abs(J / h) - 1))
                        * (
                            1
                            + sin_delta_theta
                            * math.cos(J * omega * t + theta)
                            / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                        )
                        - 1 / 2
                    )
                    if t2 > 0
                    else (2 ** abs(J / h))
                )
                if p >= 1024:
                    # This is approaching J / h -> infinity.
                    d_magnetization = 1
                    d_sqr_magnetization = 1
                    bias.append(1)
                    bias += n_qubits * [0]
                else:
                    # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
                    tot_n = 0
                    for q in range(n_qubits + 1):
                        n = 1 / ((n_qubits + 1) * (2 ** (p * q)))
                        if n == float("inf"):
                            tot_n = 1
                            bias = []
                            bias.append(1)
                            bias += n_qubits * [0]
                            break
                        bias.append(n)
                        tot_n += n
                    # Normalize the results for 1.0 total marginal probability.
                    for q in range(n_qubits + 1):
                        bias[q] /= tot_n
                        n = bias[q]
                        m = (n_qubits - (q << 1)) / n_qubits
                        d_magnetization += n * m
                        d_sqr_magnetization += n * m * m
            if J > 0:
                # This is antiferromagnetism.
                bias.reverse()
                d_magnetization = -d_magnetization

            if (d == 0) or (model < 0.99):
                experiment_samples = experiment.measure_shots(qubits, shots)
                magnetization = 0
                sqr_magnetization = 0
                for sample in experiment_samples:
                    m = 0
                    for _ in range(n_qubits):
                        m += -1 if (sample & 1) else 1
                        sample >>= 1
                    m /= n_qubits
                    magnetization += m
                    sqr_magnetization += m * m
                magnetization /= shots
                sqr_magnetization /= shots

                magnetization = model * d_magnetization + (1 - model) * magnetization
                sqr_magnetization = (
                    model * d_sqr_magnetization + (1 - model) * sqr_magnetization
                )
            else:
                magnetization = d_magnetization
                sqr_magnetization = d_sqr_magnetization

            if sqr_magnetization < min_sqr_mag:
                min_sqr_mag = sqr_magnetization

            seconds = time.perf_counter() - start

            results.append(
                {
                    "width": n_qubits,
                    "depth": d,
                    "trial": trial + 1,
                    "magnetization": magnetization,
                    "square_magnetization": sqr_magnetization,
                    "seconds": seconds,
                }
            )
            magnetizations[-1].append(sqr_magnetization)

            print(results[-1])

    if trials < 2:
        # Plotting (contributed by Elara, an OpenAI custom GPT)
        ylim = ((min_sqr_mag * 100) // 10) / 10

        plt.figure(figsize=(14, 14))
        plt.plot(depths, magnetizations[0], marker="o", linestyle="-")
        plt.title(
            "Square Magnetization vs Trotter Depth (" + str(n_qubits) + " Qubits)"
        )
        plt.xlabel("Trotter Depth")
        plt.ylabel("Square Magnetization")
        plt.grid(True)
        plt.xticks(depths)
        plt.ylim(ylim, 1.0)  # Adjusting y-axis for clearer resolution
        plt.show()

        return 0

    # Plot with error bands
    mean_magnetization = np.mean(magnetizations, axis=0)
    std_magnetization = np.std(magnetizations, axis=0, ddof=1)  # sample std dev
    sem_magnetization = std_magnetization / np.sqrt(trials)

    ylim = ((min(mean_magnetization) * 100) // 10) / 10

    # 95% confidence interval multiplier (two-tailed)
    confidence_level = 0.95
    degrees_freedom = trials - 1
    t_critical = dists.t.ppf((1 + confidence_level) / 2, df=degrees_freedom)
    ci95_magnetization = t_critical * sem_magnetization

    # Plot with 95% confidence intervals
    plt.figure(figsize=(14, 14))
    plt.errorbar(
        depths,
        mean_magnetization,
        yerr=ci95_magnetization,
        fmt="-o",
        capsize=5,
        label="Mean ± 95% CI",
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Square Magnetization")
    plt.title(
        "Square Magnetization vs Trotter Depth ("
        + str(n_qubits)
        + " Qubits, "
        + str(trials)
        + " Trials)\nWith Mean and 95% CI Error"
    )
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
