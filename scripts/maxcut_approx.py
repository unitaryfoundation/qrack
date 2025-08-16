# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

import itertools
import math
import multiprocessing
import numpy as np
import os
import networkx as nx
from numba import njit, prange


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


@njit(parallel=True)
def evaluate_cut_edges_numba(state, flat_edges):
    cut_edges = []
    for i in prange(len(flat_edges) // 2):
        i2 = i << 1
        u, v = flat_edges[i2], flat_edges[i2 + 1]
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_edges.append((u, v))

    return len(cut_edges), state, cut_edges


# Made with help from Elara (OpenAI custom GPT)
@njit(parallel=True)
def random_shots(thresholds, n, shots):
    samples = [0] * shots
    for s in prange(shots):
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        samples[s] = random_bitmask(n, m)

    return samples


# Fisher-Yates subset sampler
@njit
def fisher_yates_sample(n, m):
    arr = np.arange(n)
    for i in range(m):
        j = np.random.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]
    return arr[:m]

# Build mask in chunks of 64 bits
@njit
def fisher_yates_mask(n, m):
    chunks = np.zeros(((n + 63) // 64,), dtype=np.uint64)
    positions = fisher_yates_sample(n, m)
    for pos in positions:
        chunks[pos // 64] |= np.uint64(1) << (pos % 64)
    return chunks

# Convert chunks -> Python int (arbitrary precision)
@njit
def chunks_to_int(chunks):
    result = 0
    for i, chunk in enumerate(chunks):
        result |= int(chunk) << (64 * i)
    return result

# Full random bitstring generator
@njit
def random_bitmask(n, m):
    chunks = fisher_yates_mask(n, m)
    return chunks_to_int(chunks)


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
                if ((p * q) + math.log2(n_qubits + 1)) >= 1024:
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
    shots,
):
    qubits = list(range(n_qubits))
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
    thresholds[-1] = 1.0

    samples = set(random_shots(thresholds, n_qubits, shots))

    flat_edges = [int(item) for tup in G.edges() for item in tup]
    best_value = -1
    best_solution = None
    best_cut_edges = None
    for state in samples:
        cut_size, state, cut_edges = evaluate_cut_edges_numba(state, flat_edges)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = state
            best_cut_edges = cut_edges

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
    # m, n = 16, 16
    # G = nx.complete_bipartite_graph(m, n)
    # Known MAXCUT size: m * n

    # Multiplicity (power of 2) of shots and steps
    mult_log2 = 6
    # Qubit count
    n_qubits = G.number_of_nodes()
    # Trotter step count
    n_steps = G.number_of_edges() << mult_log2
    # Simulated time per Trotter step
    delta_t = 1 / (n_steps << mult_log2)
    J_func = lambda G: graph_to_J(G, n_qubits)
    h_func = lambda t: generate_ht(t, n_steps * delta_t)
    # Number of nearest neighbors:
    z = [G.degree[i] for i in range(G.number_of_nodes())]
    # Initial temperature
    theta = 0

    print(maxcut_tfim(G, J_func, h_func, n_qubits, n_steps, delta_t, theta, z, n_steps))
