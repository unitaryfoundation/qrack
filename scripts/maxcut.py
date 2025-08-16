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


@njit(parallel=True)
def evaluate_cut_numba(combo, flat_edges):
    state = 0
    for pos in combo:
        state |= 1 << pos
    cut_size = 0
    for i in prange(len(flat_edges) // 2):
        i2 = i << 1
        if ((state >> flat_edges[i2]) & 1) != ((state >> flat_edges[i2 + 1]) & 1):
            cut_size += 1

    return cut_size, state


def best_cut_in_weight(nodes, flat_edges, m):
    n = len(nodes)
    edge_count = len(flat_edges)
    best_val = -1
    best_state = None
    for combo in itertools.combinations(nodes, m):
        # Compute cut size using bitwise ops with Numba JIT
        cut_val, state = evaluate_cut_numba(combo, flat_edges)
        if cut_val > best_val:
            best_val = cut_val
            best_state = state
            if best_val == edge_count:
                break

    return best_state


def maxcut(G):
    nodes = G.nodes
    flat_edges = [int(item) for tup in G.edges() for item in tup]
    edge_count = len(flat_edges) >> 1
    n_qubits = len(nodes)
    best_by_hamming = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        args = []
        for m in range(1, n_qubits):
            args.append((nodes, flat_edges, m))
        best_by_hamming = pool.starmap(best_cut_in_weight, args)

    best_value = -1
    best_solution = None
    best_cut_edges = None
    for state in best_by_hamming:
        cut_size, state, cut_edges = evaluate_cut_edges_numba(state, flat_edges)
        if cut_size > best_value:
            best_value = cut_size
            best_solution = state
            best_cut_edges = cut_edges
            if best_value == edge_count:
                break

    return best_value, int_to_bitstring(best_solution, n_qubits), best_cut_edges


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

    print(maxcut(G))
