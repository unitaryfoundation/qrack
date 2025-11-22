# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import operator
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def bench_qrack(width, depth, sdrp):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    retained = width * width
    shots = retained * width
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(width)

    rcs = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for b in range(3):
                rcs.h(i)
                rcs.rz(random.uniform(0, 2 * math.pi), i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = random.choice(two_bit_gates)
                g(rcs, b1, b2)

    experiment = QrackSimulator(width, isTensorNetwork=False, isSparse=True, isOpenCL=False)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(rcs)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1))

    quimb_rcs = quimb_circuit(rcs)
    n_pow = 1 << width
    u_u =  1 / n_pow
    idx = 0
    ideal_probs = {}
    sum_probs = 0
    for count_tuple in experiment_counts:
        key = count_tuple[0]
        prob = abs(quimb_rcs.amplitude(int_to_bitstring(key, width), backend="jax")) ** 2
        if prob <= u_u:
            continue
        val = count_tuple[1]
        ideal_probs[key] = val
        sum_probs += val
        if len(ideal_probs) >= retained:
            break

    for key in ideal_probs.keys():
        ideal_probs[key] = ideal_probs[key] / sum_probs

    rcs.save_statevector()
    control = AerSimulator(method="statevector")
    job = control.run(rcs)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    return calc_stats(control_probs, ideal_probs, shots, depth)


def calc_stats(ideal_probs, exp_probs, shots, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = 1 / math.sqrt(n)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
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
            sum_hog_counts += exp * shots

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (
        (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1
    )
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "depth": depth,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_diff": float(rss),
        "p-value": float(p_val),
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [trials]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp = 0
    if len(sys.argv) > 3:
        sdrp = float(sys.argv[3])

    # Run the benchmarks
    print(bench_qrack(width, depth, sdrp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
