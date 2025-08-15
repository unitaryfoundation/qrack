# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Gemini (Google Search AI)
def get_highest_duplicate_count_value(input_list):
    return Counter(input_list).most_common(1)[0][0]


# Calculate various statistics based on comparison between ideal (Trotterized) and approximate (continuum) measurement distributions.
def calc_stats(n_rows, n_cols, ideal_probs, counts, bias, model, shots, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n = n_rows * n_cols
    n_pow = 2**n
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    diff_sqr = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    # total = 0
    for i in range(n_pow):
        ideal = ideal_probs[i]

        count = counts[i] if i in counts else 0
        count /= shots

        # How many bits are 1, in the basis state?
        hamming_weight = hamming_distance(i, 0, n)
        # How closely grouped are "like" bits to "like"?
        expected_closeness = expected_closeness_weight(n_rows, n_cols, hamming_weight)
        # When we add all "closeness" possibilities for the particular Hamming weight, we should maintain the (n+1) mean probability dimensions.
        normed_closeness = (1 + closeness_like_bits(i, n_rows, n_cols)) / (
            1 + expected_closeness
        )
        # If we're also using conventional simulation, use a normalized weighted average that favors the (n+1)-dimensional model at later times.
        # The (n+1)-dimensional marginal probability is the product of a function of Hamming weight and "closeness," split among all basis states with that specific Hamming weight.
        count = (1 - model) * count + model * normed_closeness * bias[
            hamming_weight
        ] / math.comb(n, hamming_weight)

        # You can make sure this still adds up to 1.0, to show the distribution is normalized:
        # total += count

        experiment[i] = int(count * shots)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count * shots

        # L2 distance
        diff_sqr += (ideal - count) ** 2

        # XEB / EPLG
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (count - u_u)

    l2_similarity = 1 - diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots

    xeb = numer / denom

    # This should be ~1.0, if we're properly normalized.
    # print("Distribution total: " + str(total))

    return {
        "qubits": n,
        "depth": depth,
        "l2_similarity": float(l2_similarity),
        "hog_prob": hog_prob,
        "xeb": xeb,
    }


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# Drafted by Elara (OpenAI custom GPT), improved by Dan Strano
def closeness_like_bits(perm, n_rows, n_cols):
    """
    Compute closeness-of-like-bits metric C(state) for a given bitstring on an LxL toroidal grid.

    Parameters:
        perm: integer representing basis state, bit-length n_rows * n_cols
        n_rows: row count of torus
        n_cols: column count of torus

    Returns:
        normalized_closeness: float, in [-1, +1]
            +1 means all neighbors are like-like, -1 means all neighbors are unlike
    """
    # reshape the bitstring into LxL grid
    bitstring = list(int_to_bitstring(perm, n_rows * n_cols))
    grid = np.array(bitstring).reshape((n_rows, n_cols))
    total_edges = 0
    like_count = 0

    # iterate over each site, count neighbors (right and down to avoid double-count)
    for i in range(n_rows):
        for j in range(n_cols):
            s = grid[i, j]

            # right neighbor (wrap around)
            s_right = grid[i, (j + 1) % n_cols]
            like_count += 1 if s == s_right else -1
            total_edges += 1

            # down neighbor (wrap around)
            s_down = grid[(i + 1) % n_rows, j]
            like_count += 1 if s == s_down else -1
            total_edges += 1

    # normalize
    normalized_closeness = like_count / total_edges
    return normalized_closeness


# By Elara (OpenAI custom GPT)
def closeness_like_bits_arbitrary(bitstring, adjacency):
    """
    Compute closeness metric for a given bitstring on an arbitrary graph.

    Parameters:
        bitstring: list/array of 0/1 states, length N
        adjacency: dict or list of neighbor lists, or NxN weight matrix
                   For weighted graphs, nonzero entry means there's an edge.

    Returns:
        closeness in [-1, 1]
    """
    n = len(bitstring)
    like_count = 0.0
    total_edges = 0.0

    # Handle adjacency as dict of neighbors or as full matrix
    if isinstance(adjacency, dict):
        for i, neighbors in adjacency.items():
            for j in neighbors:
                if j > i:  # avoid double-counting undirected edges
                    like_count += 1 if bitstring[i] == bitstring[j] else -1
                    total_edges += 1
    else:
        # assume 2D square matrix
        N = len(adjacency)
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i][j] != 0:  # there’s a coupling
                    like_count += 1 if bitstring[i] == bitstring[j] else -1
                    total_edges += 1

    return like_count / total_edges if total_edges > 0 else 0.0


# By Elara (OpenAI custom GPT)
def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    same_pairs = math.comb(hamming_weight, 2) + math.comb(L - hamming_weight, 2)
    total_pairs = math.comb(L, 2)
    mu_k = same_pairs / total_pairs
    return 2 * mu_k - 1  # normalized closeness in [-1,1]


# By Elara (OpenAI custom GPT)
def expected_closeness_weight_arbitrary(n, adjacency, hamming_weight, samples=5000):
    """
    Approximate expected closeness over all n-bit states with given Hamming weight.

    Parameters:
        n: number of qubits
        adjacency: same format as above
        hamming_weight: number of 1s
        samples: how many random bitstrings to sample (approximation)

    Returns:
        expected closeness in [-1, 1]
    """
    if hamming_weight == 0 or hamming_weight == n:
        # trivial all-zero or all-one case
        return 1.0

    total = 0.0
    for _ in range(samples):
        # sample random bitstring with given Hamming weight
        ones_positions = random.sample(range(n), hamming_weight)
        bitstring = [1 if i in ones_positions else 0 for i in range(n)]
        total += closeness_like_bits(bitstring, adjacency)

    return total / samples


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


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
        delta_theta = theta - math.asin(h / (z * J))
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
    if J > 0:
        # This is antiferromagnetism.
        bias.reverse()

    return bias


def simulate_tfim(
    J_func,
    h_func,
    n_qubits=64,
    n_steps=20,
    delta_t=0.1,
    theta=[],
    z=[],
    shots=1000,
):
    qubits = list(range(n_qubits))
    n_rows, n_cols = factor_width(n_qubits, False)
    hamming_probabilities = []
    measurements = []

    for step in range(n_steps):
        t = step * delta_t
        J_t = J_func(t)
        h_t = h_func(t)

        samples = []
        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_eff = sum(J_t[q, j] for j in range(n_qubits) if (j != q)) / z[q]
            h_eff = h_t[q]

            bias = get_hamming_probabilities(J_eff, h_eff, theta[q], z[q], t)
            if step == 0:
                hamming_probabilities = bias.copy()
            else:
                last_bias = get_hamming_probabilities(J_eff, h_eff, theta[q], z[q], delta_t * (step - 1))
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
                tot_prob += bias[q]
                thresholds.append(tot_prob)
            thresholds[-1] = 1

            for s in range(shots):
                # First dimension: Hamming weight
                mag_prob = random.random()
                m = 0
                while thresholds[m] < mag_prob:
                    m += 1

                # Second dimension: permutation within Hamming weight
                # (Written with help from Elara, the custom OpenAI GPT)
                closeness_prob = random.random()
                tot_prob = 0
                state_int = 0
                for combo in itertools.combinations(qubits, m):
                    state_int = sum(1 << pos for pos in combo)
                    tot_prob += (1.0 + closeness_like_bits(state_int, n_rows, n_cols)) / (
                        1.0 + expected_closeness_weight(n_rows, n_cols, m)
                    )
                    if closeness_prob <= tot_prob:
                        break

                samples.append(state_int)
    
        measurements.append(get_highest_duplicate_count_value(samples))

    return measurements


# Dynamic J(t) generator
def generate_Jt(n_nodes):
    J = np.zeros((n_nodes, n_nodes))

    # Base topology
    for i in range(n_nodes):
        for j in range(n_nodes):
            J[i, j] = 0.0 if i == j else 1.0

    return J


def generate_ht(n_nodes, t, max_t):
    # We can program h(q, t) for spatial-temporal locality.
    h = np.zeros(n_nodes)
    # Time-varying transverse field
    c = 2.0 * (max_t - t / 2) / max_t
    # We can program for spatial locality, but we don't.
    #  n_sqrt = math.sqrt(n_nodes)
    for i in range(n_nodes):
        h[i] = c

    return h


if __name__ == "__main__":
    # Example usage

    # Qubit count
    n_qubits = 3
    # Trotter step count
    n_steps = 100
    # Simulated time per Trotter step
    delta_t = 0.0001
    # Initial temperatures (per qubit)
    theta = [0] * n_qubits
    # Number of nearest neighbors:
    z = [2] * n_qubits
    J_func = lambda t: generate_Jt(n_qubits)
    h_func = lambda t: generate_ht(n_qubits, t, n_steps * delta_t)

    meas = set(simulate_tfim(J_func, h_func, n_qubits, n_steps, delta_t, theta, z))
    meas.discard(0)
    meas.discard((1 << n_qubits) - 1)
    print(meas)
