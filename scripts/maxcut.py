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
    return (bin(integer)[2:].zfill(length))[::-1]


def best_cut_in_weight(args):
    nodes, edges, m = args
    n = len(nodes)
    best_val = -1
    best_state = None

    for combo in itertools.combinations(nodes, m):
        # Represent state as an integer
        state = 0
        for pos in combo:
            state |= 1 << pos

        # Compute cut size using bitwise ops
        cut_val = 0
        for u, v in edges:
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_val += 1

        if cut_val > best_val:
            best_val = cut_val
            best_state = state

    return best_state


def maxcut_by_hamming_weight(G):
    nodes = G.nodes
    edges = G.edges()
    n_qubits = len(nodes)
    samples = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        args = []
        for m in range(1, n_qubits):
            args.append((nodes, edges, m))
        samples = pool.map(best_cut_in_weight, args)

    return samples


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

    meas = maxcut_by_hamming_weight(G)

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
