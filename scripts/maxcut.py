# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

import itertools
import math
import random
import multiprocessing
import numpy as np
import os
import networkx as nx


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


def separation_metric(adjacency, state_int, n_qubits):
    like_count = 0
    total_edges = 0
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j > i:
                bit_i = (state_int >> (n_qubits - 1 - i)) & 1
                bit_j = (state_int >> (n_qubits - 1 - j)) & 1
                like_count += -1 if bit_i == bit_j else 1
                total_edges += 1

    return like_count / total_edges


def best_separation(args):
    adjacency, qubits, m = args
    n_qubits = len(qubits)
    combo_count = math.factorial(n_qubits) // (math.factorial(m) * math.factorial(n_qubits - m))

    if math.log2(combo_count) > 25:
        best_separation = 0
        best_state_int = 0
        for combo in itertools.combinations(qubits, m):
            state_int = sum((1 << pos) for pos in combo)
            sep = (1.0 + separation_metric(adjacency, state_int, n_qubits)) / 2.0
            if sep > best_separation:
                best_separation = sep
                best_state_int = state_int

        return best_state_int

    state_ints = [sum((1 << pos) for pos in combo) for combo in itertools.combinations(qubits, m)]
    like_count = [0] * len(state_ints)
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j > i:
                for k in range(len(state_ints)):
                    state_int = state_ints[k]
                    bit_i = (state_int >> (n_qubits - 1 - i)) & 1
                    bit_j = (state_int >> (n_qubits - 1 - j)) & 1
                    like_count[k] += -1 if bit_i == bit_j else 1

    return state_ints[like_count.index(max(like_count))]


def maxcut_by_hamming_weight(G, n_qubits):
    qubits = list(range(n_qubits))
    G_dol = nx.to_dict_of_lists(G)
    samples = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        args = []
        for m in range(1, n_qubits):
            args.append((G_dol, qubits, m))
        samples = pool.map(best_separation, args)

    return samples


def graph_to_J(G, n_nodes):
    """Convert networkx.Graph to J dictionary for TFIM."""
    J = np.zeros((n_nodes, n_nodes))
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight = 1.0
        J[u, v] = -weight

    return J


def evaluate_cut(G, bitstring_int):
    bitstring = list(map(int, int_to_bitstring(bitstring_int, G.number_of_nodes())))
    cut_edges = []
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
    return len(cut_edges), cut_edges


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

    meas = set(maxcut_by_hamming_weight(G, n_qubits))

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
