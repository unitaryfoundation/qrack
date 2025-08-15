# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


def best_separation(adjacency, qubits, m):
    n_qubits = len(qubits)
    state_ints = [sum((1 << pos) for pos in combo) for combo in itertools.combinations(qubits, m)]
    bit_strings = [list(int_to_bitstring(x, n_qubits)) for x in state_ints]
    like_count = [0] * len(bit_strings)
    total_edges = 0
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j > i:
                for k in range(len(bit_strings)):
                    bit_string = bit_strings[k]
                    like_count[k] += -1 if bit_string[i] == bit_string[j] else 1
                total_edges += 1

    return state_ints[like_count.index(max(like_count))]


def get_hamming_probabilities(J, h, theta, z, t):
    t2 = 1
    omega = 3 * math.pi / 2
    bias = []
    if np.isclose(h, 0):
        # This agrees with small perturbations away from h = 0.
        bias.append(1)
        bias += n_qubits * [0]
    elif np.isclose(J, 0):
        # This agrees with small perturbations away from J = 0.
        bias = (n_qubits + 1) * [1 / (n_qubits + 1)]
    else:
        # compute p_i using formula for globally uniform J, h, and theta
        delta_theta = theta - math.asin(min(max(h / (z * J), -1), 1))
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
            bias.append(1)
            bias += n_qubits * [0]
        else:
            # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
            tot_n = 0
            for q in range(n_qubits + 1):
                if (p * q) >= 1024:
                    tot_n = 1
                    bias = []
                    bias.append(1)
                    bias += n_qubits * [0]
                    break
                n = 1 / ((n_qubits + 1) * (2 ** (p * q)))
                bias.append(n)
                tot_n += n
            # Normalize the results for 1.0 total marginal probability.
            for q in range(n_qubits + 1):
                bias[q] /= tot_n
    if J > 0:
        # This is antiferromagnetism.
        bias.reverse()

    return bias


def simulate_tfim(
    G,
    J_func,
    h_func,
    n_qubits,
    n_steps,
    delta_t,
    theta,
    z,
    n_rows = 0,
    n_cols = 0,
    shots=128,
):
    qubits = list(range(n_qubits))
    if n_rows == 0 or n_cols == 0:
        n_rows, n_cols = factor_width(n_qubits, False)
    hamming_probabilities = []
    measurements = []

    for step in range(n_steps):
        t = step * delta_t
        J_G = J_func(G)
        h_t = h_func(t)

        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_eff = sum(J_G[q, j] for j in range(n_qubits) if (j != q)) / z[q]

            bias = get_hamming_probabilities(J_eff, h_t, theta, z[q], t)
            if step == 0:
                hamming_probabilities = bias.copy()
            else:
                last_bias = get_hamming_probabilities(J_eff, h_t, theta, z[q], delta_t * (step - 1))
                tot_n = 0
                for i in range(len(bias)):
                    hamming_probabilities[i] += bias[i] - last_bias[i]
                    tot_n += hamming_probabilities[i]
                for i in range(len(bias)):
                    hamming_probabilities[i] /= tot_n
                last_bias = bias.copy()

    tot_prob = 0
    for q in range(1, n_qubits):
        tot_prob += hamming_probabilities[q]
    for q in range(1, n_qubits):
        hamming_probabilities[q] /= tot_prob
    thresholds = []
    tot_prob = 0
    for q in range(1, n_qubits):
        tot_prob += hamming_probabilities[q]
        thresholds.append(tot_prob)
    thresholds[-1] = 1

    samples = []
    G_dol = nx.to_dict_of_lists(G)
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1

        samples.append(best_separation(G_dol, qubits, m))

    return samples


def graph_to_J(G, n_nodes):
    """Convert networkx.Graph to J dictionary for TFIM."""
    J = np.zeros((n_nodes, n_nodes))
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight = 1.0
        J[u, v] = -weight

    return J


def generate_ht(t, max_t):
    # Time-varying transverse field
    return 2.0 * (max_t - t) / max_t


def evaluate_cut(G, bitstring_int):
    bitstring = list(map(int, int_to_bitstring(bitstring_int, G.number_of_nodes())))
    cut_edges = []
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
    return len(cut_edges), cut_edges


if __name__ == "__main__":
    # Example: Peterson graph
    G = nx.petersen_graph()
    # Qubit count
    n_qubits = G.number_of_nodes()
    # Trotter step count
    n_steps = 100
    # Simulated time per Trotter step
    delta_t = 0.1
    J_func = lambda G: graph_to_J(G, n_qubits)
    h_func = lambda t: generate_ht(t, n_steps * delta_t)
    # Number of nearest neighbors:
    z = [G.degree[i] for i in range(G.number_of_nodes())]
    # Initial temperature
    theta = 0

    meas = set(simulate_tfim(G, J_func, h_func, n_qubits, n_steps, delta_t, theta, z))
    meas.discard(0)
    meas.discard((1 << n_qubits) - 1)


    best_value = -1
    best_solution = None
    best_cut_edges = None

    for val in meas:
        cut_size, cut_edges = evaluate_cut(G, val)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = val
            best_cut_edges = cut_edges

    best_solution_bits = int_to_bitstring(best_solution, n_qubits) if best_solution is not None else None

    print((best_value, best_solution_bits, best_cut_edges))
