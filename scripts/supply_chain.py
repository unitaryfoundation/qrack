# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt


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


def simulate_tfim(
    J_func,
    h_func,
    n_qubits=64,
    n_steps=20,
    delta_t=0.1,
    theta=[],
    z=[],
):
    t2 = 1
    omega = 3 * math.pi / 2

    magnetizations = []

    for step in range(n_steps):
        t = step * delta_t
        J_t = J_func(t)
        h_t = h_func(t)

        # compute magnetization per qubit, then average
        mag_per_qubit = []

        for q in range(n_qubits):
            # gather local couplings for qubit q
            J_eff = sum(J_t[q, j] for j in range(n_qubits) if (j != q)) / z[q]
            h_eff = h_t[q]
                
            if np.isclose(J_eff, 0):
                mag_per_qubit.append(0)
                continue
            if np.isclose(h_eff, 0):
                mag_per_qubit.append(1 if J_eff < 0 else -1)
                continue

            # compute p_i using formula for globally uniform J, h, and theta
            delta_theta = theta[q] - math.asin(h_eff / (z[q] * J_eff))
            sin_delta_theta = math.sin(delta_theta)
            if t2 > 0.0:
                p_i = (2 ** (abs(J_eff / h_eff) - 1)) * (
                    1
                    + sin_delta_theta
                    * math.cos(J_eff * omega * t + theta[q])
                    / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                ) - 1 / 2
            else:
                p_i = (2 ** (abs(J_eff / h_eff) - 1)) - 1 / 2

            # compute d_magnetization for this qubit
            m_i = 0.0
            if p_i >= 1024:
                m_i = 1.0
            else:
                tot_n = 0.0
                m_sum = 0.0
                for k in range(n_qubits + 1):
                    if (p_i * k) >= 1024:
                        m_sum = 1.0
                        tot_n = 1.0
                        break
                    n_val = 1.0 / (n_qubits * (2 ** (p_i * k)))
                    if n_val == float("inf"):
                        m_sum = 1.0
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
    if t >= 0.5 and t < 1.5:
        # "Port 3" temporarily fails → remove its coupling
        J[2, 3] = J[3, 2] = 0
        J[3, 4] = J[4, 3] = 0
    if t >= 1.0 and t < 1.5:
        # Alternate weak link opens between 1 and 4
        J[1, 4] = J[4, 1] = -0.3
    # Restoration: after step 15, port 3 recovers

    return J


def generate_ht(n_nodes, t):
    # We can program h(q, t) for spatial-temporal locality.
    h = np.zeros(n_nodes)
    # Time-varying transverse field
    c = 0.5 + 0.5 * np.cos(t * math.pi / 10)
    # We can program for spatial locality, but we don't.
    #  n_sqrt = math.sqrt(n_nodes)
    for i in range(n_nodes):
        # "Longitude"-dependent severity (arbitrary)
        # h[i] = ((i % n_sqrt) / n_sqrt) * c
        h[i] = c

    return h


if __name__ == "__main__":
    # Example usage

    # Qubit count
    n_qubits = 64
    # Trotter step count
    n_steps = 40
    # Simulated time per Trotter step
    delta_t = 0.1
    # Initial temperatures (per qubit)
    theta = [math.pi / 18] * n_qubits
    # Number of nearest neighbors:
    z = [2] * n_qubits
    J_func = lambda t: generate_Jt(n_qubits, t)
    h_func = lambda t: generate_ht(n_qubits, t)

    mag = simulate_tfim(J_func, h_func, n_qubits, n_steps, delta_t, theta, z)
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
