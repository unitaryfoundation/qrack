# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

import itertools
import math
import random
import multiprocessing
import numpy as np
import os
import networkx as nx
from numba import njit, prange


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


# By Elara (OpenAI custom GPT)
def separation_metric(adjacency, state_int, n_qubits):
    """
    Compute 'separation' metric for a given bitstring on an arbitrary graph.
    Rewards unlike bits across edges; penalizes like bits.
    Result is normalized to [-1, 1].
    """
    like_count = 0
    total_edges = 0
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j > i:
                like_count += -1 if ((state_int >> i) & 1) == ((state_int >> j) & 1) else 1
                total_edges += 1

    return like_count / total_edges if total_edges > 0 else 0.0


@njit(parallel=True)
def evaluate_cut_edges_numba(state, flat_edges):
    cut_edges = []
    for i in prange(len(flat_edges) // 2):
        i2 = i << 1
        u, v = flat_edges[i2], flat_edges[i2 + 1]
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_edges.append((u, v))

    return len(cut_edges), state, cut_edges


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


def maxcut_tfim(
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
    shots=0,
):
    qubits = list(range(n_qubits))
    if n_rows == 0 or n_cols == 0:
        n_rows, n_cols = factor_width(n_qubits, False)

    hamming_probabilities = []
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

    thresholds = []
    tot_prob = 0
    for q in range(n_qubits + 1):
        tot_prob += hamming_probabilities[q]
        thresholds.append(tot_prob)
    thresholds[-1] = 1

    if shots == 0:
        shots = n_qubits << 1
    G_dol = nx.to_dict_of_lists(G)
    separation_values = [0] * len(hamming_probabilities)
    separation_states = [0] * len(hamming_probabilities)
    samples = []
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1

        # Second dimension: permutation within Hamming weight
        state_int = 0
        is_caught_up = (separation_states[m] == 0)
        for combo in itertools.combinations(qubits, m + 1):
            state_int = sum((1 << pos) for pos in combo)
            if (not is_caught_up) and state_int != separation_states[m]:
                 continue
            is_caught_up = True
            separation_value = separation_metric(G_dol, state_int, n_qubits)
            if separation_value > separation_values[m]:
                separation_values[m] = separation_value
                separation_states[m] = state_int
                break

        samples.append(state_int)

    flat_edges = [int(item) for tup in G.edges() for item in tup]
    edge_count = len(flat_edges) >> 1
    best_value = -1
    best_solution = None
    best_cut_edges = None
    for state in samples:
        cut_size, state, cut_edges = evaluate_cut_edges_numba(state, flat_edges)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = state
            best_cut_edges = cut_edges
            if best_value == edge_count:
                break

    return best_value, int_to_bitstring(best_solution, n_qubits), best_cut_edges


def graph_to_J(G, n_nodes):
    """Convert networkx.Graph to J dictionary for TFIM."""
    J = np.zeros((n_nodes, n_nodes))
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight = 1.0
        J[u, v] = -weight

    return J


def generate_ht(t, max_t):
    # Time-varying transverse field
    return 2.0 * t / max_t


if __name__ == "__main__":
    # Example: Peterson graph
    # G = nx.petersen_graph()
    # Known MAXCUT size: 12

    # Example: Icosahedral graph
    G = nx.icosahedral_graph()
    # Known MAXCUT size: 20

    # Example: Complete bipartite K_{m, n}
    # m, n = 8, 8
    # G = nx.complete_bipartite_graph(m, n)
    # Known MAXCUT size: m * n

    # Qubit count
    n_qubits = G.number_of_nodes()
    # Trotter step count
    n_steps = 100
    # Simulated time per Trotter step
    delta_t = 0.01
    J_func = lambda G: graph_to_J(G, n_qubits)
    h_func = lambda t: generate_ht(t, n_steps * delta_t)
    # Number of nearest neighbors:
    z = [G.degree[i] for i in range(G.number_of_nodes())]
    # Initial temperature
    theta = 0

    print(maxcut_tfim(G, J_func, h_func, n_qubits, n_steps, delta_t, theta, z))
