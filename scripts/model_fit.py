# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from pyqrack import QrackSimulator


# Factor the qubit width for torus dimensions that are close as possible to square
def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Elara (the custom OpenAI GPT)
def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    return circ


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
def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    same_pairs = math.comb(hamming_weight, 2) + math.comb(L - hamming_weight, 2)
    total_pairs = math.comb(L, 2)
    mu_k = same_pairs / total_pairs
    return 2 * mu_k - 1  # normalized closeness in [-1,1]


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


def main():
    n_qubits = 8
    depth = 20
    t1 = 0
    t2 = 1
    omega = 1.5

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = math.pi / 18
    delta_theta = 2 * math.pi / 9

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0
    # delta_theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2
    # delta_theta = 0

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4
    # delta_theta = math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        t1 = float(sys.argv[4])
    if len(sys.argv) > 5:
        shots = int(sys.argv[5])
    else:
        shots = max(65536, 1 << (n_qubits + 2))
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])
    else:
        trials = 8 if t1 > 0 else 1

    print("t1: " + str(t1))
    print("t2: " + str(t2))
    print("omega / pi: " + str(omega))

    omega *= math.pi
    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    # The Aer circuit also starts with this initialization
    qc_aer = qc.copy()

    # Compile a single Trotter step for QrackSimulator.
    step = QuantumCircuit(n_qubits)
    trotter_step(step, qubits, (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    # If we're using conventional simulation in the approximation model, collect samples over the depth series.
    experiment_probs = [{}] * (depth + 1)
    if t1 > 0:
        for trial in range(trials):
            experiment = QrackSimulator(n_qubits)
            experiment.run_qiskit_circuit(qc)
            for d in range(depth + 1):
                if d > 0:
                    experiment.run_qiskit_circuit(step)

                counts = dict(Counter(experiment.measure_shots(qubits, shots)))

                for key, value in counts.items():
                    experiment_probs[d - 1][key] = (
                        experiment_probs[d - 1].get(key, 0) + value / shots
                    )

        for experiment in experiment_probs:
            for key in experiment.keys():
                experiment[key] /= trials

    r_squared = 0
    ss = 0
    ssr = 0
    for d in range(depth + 1):
        # For each depth step, we append an additional Trotter step to Aer's circuit.
        if d > 0:
            trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

        # Run the Trotterized simulation with Aer and get the marginal probabilities.
        control = AerSimulator(method="statevector")
        qc_aer = transpile(
            qc_aer,
            backend=control,
        )
        qc_aer_sv = qc_aer.copy()
        qc_aer_sv.save_statevector()
        job = control.run(qc_aer_sv)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        # This section calculates the geometric series weight per Hamming weight, with oscillating time dependence.
        # The mean-field ground state is encapsulated as a multiplier on the geometric series exponent.
        # Additionally, this same mean-field exponent is the amplitude of time-dependent oscillation (also in the geometric series exponent).
        bias = []
        t = d * dt
        # Determine how to weight closed-form vs. conventional simulation contributions:
        model = (1 - 1 / math.exp(t / t1)) if (t1 > 0) else 1
        d_magnetization = 0
        d_sqr_magnetization = 0
        if np.isclose(h, 0):
            # This agrees with small perturbations away from h = 0.
            d_magnetization = 1
            d_sqr_magnetization = 1
            bias.append(1)
            bias += n_qubits * [0]
        elif np.isclose(J, 0):
            # This agrees with small perturbations away from J = 0.
            d_magnetization = 0
            d_sqr_magnetization = 0
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
                        * math.cos(-J * omega * t)
                        / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                    )
                    - 1 / 2
                )
                if t2 > 0
                else 2 ** abs(J / h)
            )
            if p >= 1024:
                # This is approaching J / h -> infinity.
                d_magnetization = 1
                d_sqr_magnetization = 1
                bias.append(1)
                bias += n_qubits * [0]
            else:
                # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
                tot_n = 0
                for q in range(n_qubits + 1):
                    n = 1 / (n_qubits * (2 ** (p * q)))
                    if n == float("inf"):
                        d_magnetization = 1
                        d_sqr_magnetization = 1
                        tot_n = 1
                        bias.append(1)
                        bias += n_qubits * [0]
                        break
                    m = (n_qubits - (q << 1)) / n_qubits
                    d_magnetization += n * m
                    d_sqr_magnetization += n * m * m
                    bias.append(n)
                    tot_n += n
                # Normalize the results for 1.0 total marginal probability.
                d_magnetization /= tot_n
                d_sqr_magnetization /= tot_n
                for q in range(n_qubits + 1):
                    bias[q] /= tot_n
        if J > 0:
            # This is antiferromagnetism.
            bias.reverse()
            d_magnetization = -d_magnetization

        # The full 2^n marginal probabilities will be produced in the statistics calculation,
        # but notice that the global magnetization value only requires (n+1) dimensions of marginal probability,
        # the marginal probability per each Hilbert space basis dimension is trivial to calculate by closed form,
        # and we simply need to be thoughtful in how to extract expectation values to maximize similar symmetries.
        result = calc_stats(
            n_rows,
            n_cols,
            control_probs,
            experiment_probs[d],
            bias,
            model,
            shots,
            d,
        )

        # Add up the square residuals:
        r_squared += (1 - result["l2_similarity"]) ** 2

        if model < 0.99:
            # Mix in the conventional simulation component.
            magnetization = 0
            sqr_magnetization = 0
            for key, value in experiment_probs[d].items():
                m = 0
                for _ in range(n_qubits):
                    m += -1 if (key & 1) else 1
                    key >>= 1
                m /= n_qubits
                magnetization += value * m
                sqr_magnetization += value * m * m

            magnetization = model * d_magnetization + (1 - model) * magnetization
            sqr_magnetization = (
                model * d_sqr_magnetization + (1 - model) * sqr_magnetization
            )
        else:
            # Rely entirely on the (n+1)-dimensional model.
            magnetization = d_magnetization
            sqr_magnetization = d_sqr_magnetization

        # Calculate the "control-case" magnetization values, from Aer's samples.
        c_magnetization, c_sqr_magnetization = 0, 0
        for p in range(1 << n_qubits):
            perm = p
            m = 0
            for _ in range(n_qubits):
                m += -1 if (perm & 1) else 1
                perm >>= 1
            m /= n_qubits
            c_magnetization += control_probs[p] * m
            c_sqr_magnetization += control_probs[p] * m * m

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += c_sqr_magnetization**2
        ssr += (c_sqr_magnetization - sqr_magnetization) ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1 - r_squared / (depth + 1)
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1 - (ssr / ss)

    print("L2 norm R^2: " + str(r_squared))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    # Happy Qracking! You rock!

    return 0


if __name__ == "__main__":
    sys.exit(main())
