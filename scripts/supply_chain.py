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
    n_steps=20,
    delta_t=0.1,
    theta=2 * math.pi / 9,
    delta_theta=math.pi / 18,
    t2=1.0,
    omega=3 * math.pi / 2,
):
    magnetizations = []

    for step in range(n_steps):
        t = step * delta_t
        J_t = J_func(t)
        h_t = h_func(t)

        # compute magnetization per qubit, then average
        mag_per_qubit = []

        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_vals = [J_t[q, j] for j in range(n_qubits) if (j != q) and (abs(J_t[q, j]) > 1e-12)]
            h_val = h_t[q] if abs(h_t[q]) > 1e-12 else None

            if not J_vals or h_val is None:
                # trivial cases
                if h_val is None:
                    mag_per_qubit.append(1.0)  # effectively pinned in Z
                else:
                    # J_vals is None
                    mag_per_qubit.append(0.0)  # no coupling
                continue

            J_eff = np.mean(J_vals)
            h_eff = h_val

            # compute p_i using your same formula
            sin_delta_theta = math.sin(delta_theta)
            if t2 > 0.0:
                p_i = (
                    (2 ** (abs(J_eff / h_eff) - 1))
                    * (
                        1
                        + sin_delta_theta
                        * math.cos(J_eff * omega * t + theta)
                        / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                    )
                    - 1 / 2
                )
            else:
                p_i = (2 ** (abs(J_eff / h_eff) - 1)) - 1 / 2

            # compute d_magnetization for this qubit
            if p_i >= 1024:
                m_i = 1.0
            else:
                tot_n = 0.0
                m_sum = 0.0
                for k in range(n_qubits + 1):
                    if (p_i * k) >= 1024:
                        m_i = 1.0
                        tot_n = 1.0
                        break
                    n_val = 1.0 / (n_qubits * (2 ** (p_i * k)))
                    if n_val == float("inf"):
                        m_i = 1.0
                        tot_n = 1.0
                        break
                    m = (n_qubits - (k << 1)) / n_qubits
                    m_sum += n_val * m
                    tot_n += n_val
                m_i = m_sum / tot_n if tot_n > 0 else 0.0

            # adjust for sign of local J_eff (antiferromagnetism)
            if J_eff > 0:
                m_i = -m_i

            mag_per_qubit.append(m_i)

        # combine per-qubit magnetizations (e.g., average)
        step_mag = float(np.mean(mag_per_qubit))
        magnetizations.append(step_mag)

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
        J[2, 3] = J[3, 2] = 1e-10
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
    n_steps = 40
    delta_t = 0.1
    theta = math.pi / 18
    delta_theta = 2 * math.pi / 9
    omega = 3 * math.pi / 2
    J_func = lambda t: generate_Jt(n_qubits, t)
    h_func = lambda t: generate_ht(n_qubits, t)

    mag = simulate_tfim(
        J_func, h_func, n_qubits, n_steps, delta_t, theta, delta_theta, omega
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
