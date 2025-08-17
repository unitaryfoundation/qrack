# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# After tackling the case where parameters are uniform and independent of time, we generalize the model by averaging per-qubit behavior as if the static case and per-time-step behavior as finite difference. This provides the basis of a novel physics-inspired (adiabatic TFIM) MAXCUT approximate solver that often gives optimal or exact answers on a wide selection of graph types.

import itertools
import math
import multiprocessing
import numpy as np
import os
import networkx as nx
from numba import njit


@njit
def evaluate_cut_edges_numba(samples, flat_edges):
    best_value = -1
    best_solution = None
    best_cut_edges = None
    for state in samples:
        cut_edges = []
        for i in range(len(flat_edges) // 2):
            i2 = i << 1
            u, v = flat_edges[i2], flat_edges[i2 + 1]
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_edges.append((u, v))
        cut_size = len(cut_edges)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = state
            best_cut_edges = cut_edges

    return best_value, best_solution, best_cut_edges


@njit
def get_hamming_probabilities(n_qubits, J, h, theta, z, t):
    t2 = 1
    omega = 3 * math.pi / 2
    bias = (n_qubits + 1) * [0.0]
    if np.isclose(h, 0.0):
        # This agrees with small perturbations away from h = 0.
        bias[0] = 1.0
    elif np.isclose(J, 0.0):
        # This agrees with small perturbations away from J = 0.
        bias = (n_qubits + 1) * [1.0 / (n_qubits + 1.0)]
    else:
        # compute p_i using formula for globally uniform J, h, and theta
        delta_theta = theta - math.asin(min(max(h / (z * J), -1.0), 1.0))
        # ChatGPT o3 suggested this cos_theta correction.
        sin_delta_theta = math.sin(delta_theta)
        # "p" is the exponent of the geometric series weighting, for (n+1) dimensions of Hamming weight.
        # Notice that the expected symmetries are respected under reversal of signs of J and/or h.
        p = (
                (
                (
                    (2.0 ** (abs(J / h) - 1.0))
                    * (
                        1.0
                        + sin_delta_theta
                        * math.cos(J * omega * t + theta)
                        / ((1.0 + math.sqrt(t / t2)) if t2 > 0.0 else 1.0)
                    )
                    - 0.5
                )
                if t2 > 0.0
                else (2.0 ** abs(J / h))
            )
            if (abs(J / h) - 1.0) < 1024.0
            else 1024.0
        )
        if p >= 1024.0:
            # This is approaching J / h -> infinity.
            bias[0] = 1.0
        else:
            # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
            tot_n = 0.0
            for q in range(len(bias)):
                if ((p * q) + math.log2(n_qubits + 1)) >= 1024.0:
                    tot_n = 1
                    bias[0] = 1.0
                    for i in range(1, len(bias)):
                        bias[i] = 0.0
                    break
                n = 1.0 / ((n_qubits + 1.0) * (2.0 ** (p * q)))
                bias[q] = n
                tot_n += n
            # Normalize the results for 1.0 total marginal probability.
            for q in range(len(bias)):
                bias[q] /= tot_n
    if J > 0.0:
        # This is antiferromagnetism.
        bias.reverse()

    return bias


# Written by Elara (OpenAI custom GPT)
def local_repulsion_choice(adjacency, degrees, weights, n, m):
    """
    Pick m nodes (bit positions) out of n with repulsion bias:
    - High-degree nodes are already less likely
    - After choosing a node, its neighbors' probabilities are further reduced
    """

    # Base weights: inverse degree
    # degrees = np.array([len(adjacency.get(i, [])) for i in range(n)], dtype=np.float64)
    # weights = 1.0 / (degrees + 1.0)
    weights = weights.copy()

    chosen = []
    available = set(range(n))

    for _ in range(m):
        if not available:
            break

        # Normalize weights over remaining nodes
        sub_weights = np.array([weights[i] for i in available], dtype=np.float64)
        sub_weights /= sub_weights.sum()
        sub_nodes = list(available)

        # Sample one node
        idx = np.random.choice(len(sub_nodes), p=sub_weights)
        node = sub_nodes[idx]
        chosen.append(node)

        # Remove node from available
        available.remove(node)

        # Repulsion: penalize neighbors
        for nbr in adjacency.get(node, []):
            if nbr in available:
                weights[nbr] *= 0.5  # halve neighbor's weight (tunable!)

    # Build integer mask
    mask = 0
    for pos in chosen:
        mask |= (1 << pos)

    return mask


def graph_to_J(G, n_nodes):
    """Convert networkx.Graph to J dictionary for TFIM."""
    J = np.zeros((n_nodes, n_nodes))
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight = 1.0
        J[u, v] = -weight

    return J


def generate_ht(t, max_t, mult_log2):
    # Time-varying transverse field
    return (1 << (mult_log2 >> 1)) * t / max_t


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def maxcut_tfim(
    G,
    J_func = None,
    h_func = None,
    z = None,
    theta = None,
    n_steps = None,
    delta_t = None,
    shots = None,
):
    # Number of qubits/nodes
    n_qubits = G.number_of_nodes()
    # Multiplicity (power of 2) of shots and steps
    mult_log2 = 10
    if J_func is None:
        # Coupling interaction
        J_func = lambda G: graph_to_J(G, n_qubits)
    if h_func is None:
        # Transverse field
        h_func = lambda t: generate_ht(t, n_steps * delta_t, mult_log2)
    if z is None:
        # Number of nearest neighbors:
        z = [G.degree[i] for i in range(n_qubits)]
    if theta is None:
        # Initial temperature
        theta = [0] * n_qubits
    if n_steps is None:
        # Trotter step count
        n_steps = n_qubits << mult_log2
    if delta_t is None:
        # Simulated time per Trotter step
        delta_t = 1 / (n_steps << (mult_log2 >> 1))
    if shots is None:
        # Number of measurement shots
        shots = n_qubits << mult_log2

    qubits = list(range(n_qubits))
    hamming_probabilities = (n_qubits - 1) * [0.0]
    for step in range(n_steps):
        t = step * delta_t
        J_G = J_func(G)
        h_t = h_func(t)

        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_eff = sum(J_G[q, j] for j in range(n_qubits) if (j != q)) / z[q]
            bias = get_hamming_probabilities(n_qubits, J_eff, h_t, theta[q], z[q], t)

            if step == 0:
                for i in range(len(hamming_probabilities)):
                    hamming_probabilities[i] += bias[i + 1]
                continue

            last_bias = get_hamming_probabilities(n_qubits, J_eff, h_t, theta[q], z[q], delta_t * (step - 1))
            for i in range(len(hamming_probabilities)):
                hamming_probabilities[i] += bias[i + 1] - last_bias[i + 1]

        if step == 0:
            continue

        # Precision might suffer if we don't normalize, each step.
        tot_prob = sum(hamming_probabilities)

        if np.isclose(tot_prob, 0):
            continue

        for i in range(len(hamming_probabilities)):
            hamming_probabilities[i] /= tot_prob

    norm_prob = 0.0
    for i in range(len(hamming_probabilities)):
        norm_prob += hamming_probabilities[i]

    thresholds = len(hamming_probabilities) * [0.0]
    tot_prob = 0.0
    for i in range(len(hamming_probabilities)):
        tot_prob += hamming_probabilities[i] / norm_prob
        thresholds[i - 1] = tot_prob
    thresholds[-1] = 1.0

    G_dict = nx.to_dict_of_lists(G)
    degrees = np.array([sum(edge_attributes.get('weight', 1.0) for neighbor, edge_attributes in G.adj[i].items()) for i in range(n_qubits)], dtype=np.float64)
    weights = 1.0 / (degrees + 1.0)
    samples = []
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = np.random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        m += 1
        # Second dimension: permutation within Hamming weight
        samples.append(local_repulsion_choice(G_dict, degrees, weights, n_qubits, m))

    best_value, best_solution, best_cut_edges = evaluate_cut_edges_numba(samples, [int(item) for tup in G.edges() for item in tup])

    return best_value, int_to_bitstring(best_solution, n_qubits), best_cut_edges


if __name__ == "__main__":
    # We usually achieve the exact value
    # (or optimal, for Erdős–Rényi, with unknown exact value)
    # for each of the following examples.

    # Example: Peterson graph
    G = nx.petersen_graph()
    # Known MAXCUT size: 12

    # Example: Icosahedral graph
    #  G = nx.icosahedral_graph()
    # Known MAXCUT size: 20

    # Example: Complete bipartite K_{m, n}
    # m, n = 16, 16
    # G = nx.complete_bipartite_graph(m, n)
    # Known MAXCUT size: m * n

    # Generate a "harder" test case: Erdős–Rényi random graph with 20 nodes, edge probability 0.5
    # n_nodes = 20
    # edge_prob = 0.5
    # G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=42)
    # Cut value is approximately 63 for this example.

    # Create a Barabási–Albert (BA) graph with 20 nodes and 2 edges to attach from a new node to existing nodes
    # G = nx.barabasi_albert_graph(n=20, m=2, seed=42)

    # Non-uniform edge weights
    # G = nx.Graph()
    # G.add_edge(0, 1, weight=3.69)
    # G.add_edge(0, 2, weight=2.2)
    # G.add_edge(0, 3, weight=2.26)
    # G.add_edge(0, 4, weight=4.01)
    # G.add_edge(0, 5, weight=1.29)

    print(maxcut_tfim(G))
