# Ising model Trotterization measurement sample generation

import itertools
import math
import random
import sys
import time

import numpy as np


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


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
def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    same_pairs = math.comb(hamming_weight, 2) + math.comb(L - hamming_weight, 2)
    total_pairs = math.comb(L, 2)
    mu_k = same_pairs / total_pairs
    return 2 * mu_k - 1  # normalized closeness in [-1,1]


def main():
    n_qubits = 16
    depth = 20
    shots = 100
    t2 = 1
    omega = 1.5

    # Quantinuum settings
    J, h = -1.0, 2.0
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h = -1.0, 0.0
    # theta = 0

    # Pure transverse field
    # J, h = 0.0, 2.0
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h = -1.0, 1.0
    # theta = -math.pi / 4

    t = 5
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        t = float(sys.argv[2])
    if len(sys.argv) > 4:
        shots = int(sys.argv[4])

    print("t2: " + str(t2))
    print("omega / pi: " + str(omega))

    omega *= math.pi
    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))
    depths = list(range(0, depth + 1))

    # Coordination number for a square lattice:
    z = 4
    # Mean-field critical angle (in radians)
    theta_c = math.asin(max(min(1, abs(h) / (z * J)) if np.isclose(z * J, 0) else (1 if J > 0 else -1), -1))
    # Set theta relative to that:
    delta_theta = theta - theta_c

    start = time.perf_counter()

    bias = []
    if np.isclose(h, 0):
        # This agrees with small perturbations away from h = 0.
        bias.append(1)
        bias += n_qubits * [0]
    elif np.isclose(J, 0):
        # This agrees with small perturbations away from J = 0.
        bias = (n_qubits + 1) * [1 / (n_qubits + 1)]
    else:
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

    thresholds = []
    tot_prob = 0
    for q in range(n_qubits + 1):
        tot_prob += bias[q]
        thresholds.append(tot_prob)
    thresholds[-1] = 1

    hamming_samples = [0] * len(thresholds)
    for s in range(shots):
        # First dimension: Hamming weight
        mag_prob = random.random()
        m = 0
        while thresholds[m] < mag_prob:
            m += 1
        hamming_samples[m] += 1

    samples = []
    for m in range(len(hamming_samples)):
        # Second dimension: permutation within Hamming weight
        # (Written with help from Elara, the custom OpenAI GPT)
        hs = hamming_samples[m]
        rands = []
        for s in range(hs):
            rands.append(random.random())
        rands.sort()
        state_int = 0
        tot_prob = 0
        s = 0
        for combo in itertools.combinations(qubits, m):
            state_int = sum(1 << pos for pos in combo)
            tot_prob += (1.0 + closeness_like_bits(state_int, n_rows, n_cols)) / (
                1.0 + expected_closeness_weight(n_rows, n_cols, m)
            )
            while (s < hs) and (rands[s] <= tot_prob):
                samples.append(state_int)
                s += 1
            if s == hs:
                break
        for r in range(hs - s):
            samples.append(state_int)

    random.shuffle(samples)

    seconds = time.perf_counter() - start

    print(samples)
    print("Seconds: " + str(seconds))


if __name__ == "__main__":
    sys.exit(main())
