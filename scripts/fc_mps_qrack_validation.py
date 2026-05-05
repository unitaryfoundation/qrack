# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import operator
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit

import jax
import jax.numpy as jnp


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


# Modified to use MPS by (Anthropic) Claude
def bench_qrack(width, depth, sdrp):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    retained = min(width ** 2, 1 << width)
    checked = min(width ** 2, 1 << width)

    # chi controls approximation quality vs. speed
    # chi = width is cheap; chi = width**2 is closer to exact
    # for QV circuits at modest width, chi = 2*width is a reasonable start
    chi = min(width ** 3, 1 << width)

    # CircuitMPS maintains state as MPS with bounded bond dimension
    # Gate application is O(chi^2 * width) per gate instead of exact
    quimb_rcs = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)

    rcs = QuantumCircuit(width)

    for d in range(depth):
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            rcs.u(th, ph, lm, i)
            quimb_rcs.apply_gate('U3', th, ph, lm, i)

        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            rcs.cx(c, t)
            quimb_rcs.apply_gate('CX', c, t)

    # Run Qrack for heavy output sieve
    experiment = QrackSimulator(width)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs, shots=0)
    highest_prob = experiment.highest_prob_perm()
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, checked)))
    experiment_counts[highest_prob] = checked
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1), reverse=True)
    experiment = None

    # Approximate amplitude estimation via MPS
    # amplitude() on CircuitMPS is O(chi^2 * width) per call
    # vs. O(exp(treewidth)) for exact contraction
    n_pow = 1 << width
    u_u = 1 / n_pow
    ideal_amps = {}
    sum_probs = 0

    for count_tuple in experiment_counts:
        if len(ideal_amps) >= retained and count_tuple[1] < 2:
            break
        key = count_tuple[0]
        bitstring = int_to_bitstring(key, width)
        # Approximate amplitude from MPS — cheap inner product
        amp = complex(quimb_rcs.amplitude(bitstring))
        prob = abs(amp) ** 2
        if prob <= u_u:
            continue
        ideal_amps[key] = amp
        sum_probs += prob

    # Normalize
    ideal_probs = {k: abs(v)**2 / sum_probs for k, v in ideal_amps.items()}

    # Qrack control for XEB reference
    control = QrackSimulator(width)
    control.run_qiskit_circuit(rcs, shots=0)
    control_probs = control.out_probs()

    return calc_stats(control_probs, ideal_probs)


def calc_stats(ideal_probs, exp_probs):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = 1 / 2
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    for i in range(n_pow):
        exp = (1 - model) * (exp_probs[i] if i in exp_probs else 0) + model * mean_guess
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # L2 norm
        sqr_diff += (ideal - exp) ** 2

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_diff": float(rss),
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [sdrp]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp = 0 # (1.0 - 1.0 / math.sqrt(2)) / 2.0
    if len(sys.argv) > 3:
        sdrp = float(sys.argv[3])

    # Run the benchmarks
    print(bench_qrack(width, depth, sdrp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
