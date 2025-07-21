# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import math
import numpy as np
from pyqrack import Pauli
import matplotlib.pyplot as plt


def simulate_tfim(
    J_func,
    h_func,
    n_qubits=64,
    lrr=3,
    lrc=3,
    n_steps=20,
    delta_t=0.1,
    theta=2 * math.pi / 9,
    delta_theta=math.pi / 18,
    t2=1.0,
    omega=3 * math.pi / 2,
    shots=1024,
):
    qubits = list(range(n_qubits))
    magnetizations = []

    for step in range(n_steps):
        t = step * delta_t
        J_t = J_func(t)
        h_t = h_func(t)

        # compute effective ratio |J/h| for this step from local fields
        # aggregate all nonzero couplings
        J_vals = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if J_t[i, j] != 0.0:
                    J_vals.append(abs(J_t[i, j]))
        h_vals = [abs(hv) for hv in h_t if abs(hv) > 1e-12]

        if J_vals and h_vals:
            J_eff = np.mean(J_vals)
            h_eff = np.mean(h_vals)
        else:
            J_eff = 0.0
            h_eff = 0.0

        d_magnetization = 0
        d_sqr_magnetization = 0
        if np.isclose(h_eff, 0):
            d_magnetization = 1
            d_sqr_magnetization = 1
        elif np.isclose(J_eff, 0):
            d_magnetization = 0
            d_sqr_magnetization = 0
        else:
            # ChatGPT o3 suggested this cos_theta correction.
            sin_delta_theta = math.sin(delta_theta)
            p = (
                (
                    (2 ** (abs(J_eff / h_eff) - 1))
                    * (
                        1
                        + sin_delta_theta
                        * math.cos(J_eff * omega * t + theta)
                        / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                    )
                    - 1 / 2
                )
                if t2 > 0
                else (2 ** abs(J_eff / h_eff))
            )
            if p >= 1024:
                d_magnetization = 1
                d_sqr_magnetization = 1
            else:
                tot_n = 0
                for q in range(n_qubits + 1):
                    n = 1 / (n_qubits * (2 ** (p * q)))
                    if n == float("inf"):
                        d_magnetization = 1
                        d_sqr_magnetization = 1
                        tot_n = 1
                        break
                    m = (n_qubits - (q << 1)) / n_qubits
                    d_magnetization += n * m
                    d_sqr_magnetization += n * m * m
                    tot_n += n
                d_magnetization /= tot_n
                d_sqr_magnetization /= tot_n
        if J_eff > 0:
            d_magnetization = -d_magnetization

        magnetizations.append(d_magnetization)

    return magnetizations


# Dynamic J(t) generator
def generate_Jt(n_nodes, t):
    J = np.zeros((n_nodes, n_nodes))

    # Base ring topology
    for i in range(n_nodes):
        J[i, (i + 1) % n_nodes] = -1.0
        J[(i + 1) % n_nodes, i] = -1.0

    # Simulate disruption:
    if t >= 0.5 and t < 1.0:
        # "Port 3" temporarily fails → remove its coupling
        J[2, 3] = J[3, 2] = 0.0
    if t >= 1.0 and t < 1.5:
        # Alternate weak link opens between 1 and 4
        J[1, 4] = J[4, 1] = -0.3

    # Restoration: after step 15, port 3 recovers
    if t >= 1.5:
        J[2, 3] = J[3, 2] = -1.0

    return J


def generate_ht(n_nodes, t):
    # We can program h(q, t) for spatial-temporal locality.
    h = np.zeros(n_nodes)
    # Time-varying transverse field
    c = 0.5 * np.cos(t * math.pi / 10)
    # We can program for spatial locality, but we don't.
    #  n_sqrt = math.sqrt(n_nodes)
    for i in range(n_nodes):
        # "Longitude"-dependent severity (arbitrary)
        # h[i] = ((i % n_sqrt) / n_sqrt) * c
        h[i] = c

    return h


if __name__ == "__main__":
    # Example usage
    n_qubits = 64
    lrr = 3
    lrc = 3
    n_steps = 40
    delta_t = 0.1
    theta = math.pi / 18
    delta_theta = 2 * math.pi / 9
    omega = 3 * math.pi / 2
    shots = 1024
    J_func = lambda t: generate_Jt(n_qubits, t)
    h_func = lambda t: generate_ht(n_qubits, t)

    mag = simulate_tfim(
        J_func, h_func, n_qubits, lrr, lrc, n_steps, delta_t, theta, delta_theta, omega, shots
    )
    ylim = ((min(mag) * 100) // 10) / 10
    plt.figure(figsize=(14, 14))
    plt.plot(list(range(1, n_steps + 1)), mag, marker="o", linestyle="-")
    plt.title(
        "Supply Chain Resilience over Time (Magnetization vs Trotter Depth, "
        + str(n_qubits)
        + " Qubits)"
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Magnetization")
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
