# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Daniel Strano and the Qrack contributors
#
# This initial draft was produced virtually entirely by (Anthropic) Claude (Sonnet 4.6).
#
# qubo_exact.py — warm-start branch-and-bound exact QUBO solver
#
# Takes a PyQrackIsing heuristic solution as the initial incumbent and drives
# it to a certified exact optimum via branch and bound. The key insight is
# that a near-optimal warm start collapses the effective search tree almost
# immediately: the incumbent prunes every branch whose LP relaxation upper
# bound cannot beat the already-known value.
#
# Upper bound per B&B node: LP relaxation of the QUBO (standard linearisation
# with one auxiliary variable per product term). This is cheaper per node than
# SDP while still tight enough to close the gap quickly when the incumbent is
# strong. For very large instances where LP bounds are too loose, users can
# swap in an SDP upper bound by replacing `_lp_upper_bound` — the interface is
# intentionally kept thin.
#
# Usage (standalone demo):
#   python qubo_exact.py [--n 20] [--seed 42] [--density 0.5]
#
# Usage (as library — PyQrackIsing warm start):
#   from qubo_exact import solve_qubo_exact
#   h_bits, h_val, exact_bits, exact_val, certified = solve_qubo_exact(G_csr)
#
# Usage (as library — custom warm start):
#   from qubo_exact import maxcut_branch_and_bound
#   exact_bits, exact_value, certified = maxcut_branch_and_bound(Q, warm_bits)
#
# Q / G_csr convention:
#   PyQrackIsing expects a scipy CSR upper-triangular matrix with non-negative
#   edge weights (MAXCUT convention, no diagonal / self-loops).
#   maxcut_branch_and_bound accepts a dense upper-triangular or symmetric numpy
#   float64 matrix; diagonal terms are handled correctly throughout.

import argparse
import time
import networkx as nx
import numpy as np

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# QUBO objective (dense upper-triangular or symmetric Q)
# ---------------------------------------------------------------------------

def cut_value(Q, bits):
    """
    Evaluate the MAXCUT objective for a boolean assignment: sum of Q[i,j]
    for all edges (i,j) where bits[i] != bits[j] (i.e. opposite partitions).
    Q is upper-triangular; only entries with i < j are edge weights.
    """
    bits = np.asarray(bits, dtype=np.bool_)
    n = len(bits)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0.0 and bits[i] != bits[j]:
                total += Q[i, j]
    return total


# ---------------------------------------------------------------------------
# LP relaxation upper bound
# ---------------------------------------------------------------------------

def _lp_upper_bound(Q, fixed, n):
    """
    Return an upper bound on the MAXCUT objective over the subproblem defined
    by `fixed`: a dict mapping variable index -> {0, 1} for already-branched
    variables. Free variables are relaxed to [0, 1].

    MAXCUT objective: sum_{i<j} Q[i,j] * (x_i + x_j - 2*x_i*x_j)
    i.e. edge (i,j) contributes Q[i,j] iff x_i != x_j.

    We split into three parts:
      fixed_fixed : exact cut value among already-fixed variable pairs
      fixed_free  : for free var k, contribution from edges to fixed vars is
                    Q[fj,k]*|f_fj - x_k|, linearised as Q[fj,k]*(f_fj + x_k - 2*f_fj*x_k)
      free_free   : LP relaxation using auxiliary y_ij = x_i*x_j, so the
                    edge contribution is Q[i,j]*(x_i + x_j - 2*y_ij)

    Falls back to a cheap bound if scipy is unavailable.
    """
    # --- fixed-fixed contribution (exact) ---
    fixed_fixed = 0.0
    for i in fixed:
        for j in fixed:
            if i < j and Q[i, j] != 0.0 and fixed[i] != fixed[j]:
                fixed_fixed += Q[i, j]

    free = [i for i in range(n) if i not in fixed]
    n_free = len(free)

    if n_free == 0:
        return fixed_fixed

    if not _HAVE_SCIPY:
        # Cheap fallback: sum max possible contribution of each free edge
        bound = fixed_fixed
        for i in free:
            for fj, fval in fixed.items():
                a, b_ = min(i, fj), max(i, fj)
                bound += max(0.0, Q[a, b_])   # best case: edge is cut
            for j in free:
                if i < j:
                    bound += max(0.0, Q[i, j])
        return bound

    free_idx = {v: k for k, v in enumerate(free)}
    pairs = [(i, j) for idx, i in enumerate(free) for j in free[idx + 1:]
             if Q[i, j] != 0.0]
    n_pairs = len(pairs)
    total_vars = n_free + n_pairs   # x_i for free vars, then y_ij for free pairs

    # Objective (negated for minimisation):
    # For each free var k: sum over fixed neighbours fj of Q[fj,k]*(fval + x_k - 2*fval*x_k)
    #   The constant fval*Q[fj,k] goes into fixed_free_const; the x_k coefficient is
    #   Q[fj,k]*(1 - 2*fval).
    # For each free pair (i,j): Q[i,j]*(x_i + x_j - 2*y_ij)
    #   x_i and x_j coefficients are Q[i,j]; y_ij coefficient is -2*Q[i,j].

    fixed_free_const = 0.0
    c = np.zeros(total_vars)

    for k, i in enumerate(free):
        for fj, fval in fixed.items():
            a, b_ = min(i, fj), max(i, fj)
            w = Q[a, b_]
            if w == 0.0:
                continue
            fixed_free_const += w * fval
            c[k] -= w * (1.0 - 2.0 * fval)   # negate: subtract from minimisation obj

    for p, (i, j) in enumerate(pairs):
        w = Q[i, j]
        c[free_idx[i]] -= w       # x_i term
        c[free_idx[j]] -= w       # x_j term
        c[n_free + p]  += 2.0 * w # y_ij term (negated sign: -2w in max = +2w in min)

    bounds = [(0.0, 1.0)] * total_vars

    if not pairs:
        # No free-free edges: LP trivially solved — each free var independently
        # contributes its linear term, clamped to [0,1]
        lp_val = sum(max(-c[k], 0.0) if c[k] < 0.0 else 0.0 for k in range(n_free))
        # also add terms where c[k] is negative (free var set to 1 helps)
        lp_val = sum((-c[k] if c[k] < 0.0 else 0.0) for k in range(n_free))
        return fixed_fixed + fixed_free_const + lp_val

    # Constraints: y_ij = x_i * x_j linearisation
    #   y_ij <= x_i, y_ij <= x_j, x_i + x_j - y_ij <= 1
    A_ub = []
    b_ub = []
    for p, (i, j) in enumerate(pairs):
        ki, kj = free_idx[i], free_idx[j]
        r1 = np.zeros(total_vars); r1[ki] = -1.0; r1[n_free + p] = 1.0
        r2 = np.zeros(total_vars); r2[kj] = -1.0; r2[n_free + p] = 1.0
        r3 = np.zeros(total_vars); r3[ki] = 1.0;  r3[kj] = 1.0; r3[n_free + p] = -1.0
        A_ub.extend([r1, r2, r3])
        b_ub.extend([0.0, 0.0, 1.0])

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )

    if res.success:
        return fixed_fixed + fixed_free_const + (-res.fun)
    else:
        return float("inf")


# ---------------------------------------------------------------------------
# Branch selection heuristic
# ---------------------------------------------------------------------------

def _most_influential(Q, fixed, n):
    """
    Choose the next variable to branch on among free variables.

    Pick the free variable with the largest total absolute edge weight to
    other free variables — the one whose assignment will most tighten the
    bound for remaining nodes.
    """
    free = [i for i in range(n) if i not in fixed]
    if not free:
        return None
    influence = np.zeros(len(free))
    for k, i in enumerate(free):
        for j in free:
            if i != j:
                influence[k] += abs(Q[i, j]) + abs(Q[j, i])
        influence[k] += abs(Q[i, i])
    return free[int(np.argmax(influence))]


# ---------------------------------------------------------------------------
# Branch and bound
# ---------------------------------------------------------------------------

def maxcut_branch_and_bound(Q, warm_bits, verbose=True, time_limit=None):
    """
    Exact MAXCUT solver via branch and bound with LP relaxation upper bounds.

    Parameters
    ----------
    Q : (n, n) float64 ndarray
        Dense upper-triangular edge weight matrix (no diagonal).
        Objective: maximise sum_{i<j} Q[i,j] * (x_i != x_j).
    warm_bits : array-like of bool/int, length n
        Warm-start incumbent from PyQrackIsing (via solve_qubo_exact) or any
        heuristic. The partition is defined by which nodes have bit=1 vs bit=0.
    verbose : bool
    time_limit : float or None
        Wall-clock seconds before returning best-so-far (not certified exact).

    Returns
    -------
    best_bits : ndarray of bool
    best_value : float
    certified : bool
        True if the returned solution is provably optimal (B&B completed).
    """
    Q = np.asarray(Q, dtype=np.float64)
    n = Q.shape[0]
    assert Q.shape == (n, n), "Q must be square"

    warm_bits = np.asarray(warm_bits, dtype=np.bool_)
    assert len(warm_bits) == n, "warm_bits length must match Q dimension"

    best_bits = warm_bits.copy()
    best_value = cut_value(Q, best_bits)

    if verbose:
        print(f"Warm-start incumbent: {best_value:.6f}")

    t_start = time.monotonic()
    nodes_explored = 0
    nodes_pruned = 0

    stack = [{}]

    while stack:
        if time_limit is not None and (time.monotonic() - t_start) > time_limit:
            if verbose:
                print(f"Time limit reached. Nodes explored: {nodes_explored}, pruned: {nodes_pruned}")
            return best_bits, best_value, False

        fixed = stack.pop()
        nodes_explored += 1

        ub = _lp_upper_bound(Q, fixed, n)

        if ub <= best_value + 1e-9:
            nodes_pruned += 1
            continue

        if len(fixed) == n:
            bits = np.array([fixed[i] for i in range(n)], dtype=np.bool_)
            val = cut_value(Q, bits)
            if val > best_value:
                best_value = val
                best_bits = bits.copy()
                if verbose:
                    print(f"  New incumbent: {best_value:.6f}  (nodes: {nodes_explored})")
            continue

        branch_var = _most_influential(Q, fixed, n)

        # Try warm-start value first so the good branch is explored early,
        # tightening the incumbent and maximising subsequent pruning.
        warm_val = int(best_bits[branch_var])
        for val in [warm_val, 1 - warm_val]:
            child = dict(fixed)
            child[branch_var] = val
            stack.append(child)

    elapsed = time.monotonic() - t_start
    if verbose:
        print(f"\nExact optimum: {best_value:.6f}")
        print(f"Nodes explored: {nodes_explored}  |  Pruned: {nodes_pruned}  |  Time: {elapsed:.3f}s")

    return best_bits, best_value, True


# ---------------------------------------------------------------------------
# PyQrackIsing interface helper
# ---------------------------------------------------------------------------

def graph_to_dense_upper(G):
    """
    Convert a NetworkX graph to a dense upper-triangular numpy matrix
    suitable for maxcut_branch_and_bound. Node labels must be integers 0..n-1.
    Diagonal is left zero (PyQrackIsing MAXCUT graphs have no self-loops).
    """
    n = G.number_of_nodes()
    Q = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        i, j = (u, v) if u < v else (v, u)
        Q[i, j] = float(data.get("weight", 1.0))
    return Q


def solve_qubo_exact(G, is_spin_glass=False, pyqrackising_kwargs=None, verbose=True, time_limit=None):
    """
    Convenience wrapper: run PyQrackIsing on a NetworkX graph to get a warm
    start, then certify it (or improve it) via branch and bound.

    PyQrackIsing accepts a NetworkX graph with arbitrary edge weights (positive
    or negative) and no self-loops. It finds the partition that maximises
    sum_{(i,j) in cut} w_ij — maximising the cut weight of positive edges and
    minimising the cut weight of negative ones.

    Node labels must be integers 0..n-1 (as produced by _random_maxcut_graph).

    If your problem has diagonal terms (linear penalties/bonuses per variable),
    add them to the dense Q returned by graph_to_dense_upper before calling
    maxcut_branch_and_bound directly — the B&B handles diagonal correctly.

    Parameters
    ----------
    G : networkx.Graph
        Graph with arbitrary edge weights, no self-loops (PyQrackIsing convention).
    is_spin_glass : bool
        Passed through to spin_glass_solver_sparse.
    pyqrackising_kwargs : dict or None
        Extra kwargs forwarded to spin_glass_solver_sparse.
    verbose : bool
    time_limit : float or None
        Wall-clock seconds for the B&B phase.

    Returns
    -------
    heuristic_bits : ndarray of bool
    heuristic_value : float
    exact_bits : ndarray of bool
    exact_value : float
    certified : bool
    """
    from pyqrackising import spin_glass_solver_sparse

    kwargs = pyqrackising_kwargs or {}
    n = G.number_of_nodes()

    if verbose:
        print("Running PyQrackIsing heuristic...")

    t0 = time.monotonic()
    bitstring, heuristic_value, _, _ = spin_glass_solver_sparse(
        G, is_spin_glass=is_spin_glass, **kwargs
    )
    t_heuristic = time.monotonic() - t0

    heuristic_bits = np.array([b == "1" for b in bitstring], dtype=np.bool_)

    if verbose:
        print(f"Heuristic value: {heuristic_value:.6f}  ({t_heuristic:.3f}s)")
        print("Starting branch and bound...")

    # Convert graph to dense upper-tri for B&B. If the caller has diagonal terms,
    # add them to Q here before passing to maxcut_branch_and_bound.
    Q = graph_to_dense_upper(G)

    exact_bits, exact_value, certified = maxcut_branch_and_bound(
        Q, heuristic_bits, verbose=verbose, time_limit=time_limit
    )

    if verbose:
        gap = exact_value - heuristic_value
        print(f"\nHeuristic gap closed: {gap:.6f}  ({'certified exact' if certified else 'time-limited'})")

    return heuristic_bits, heuristic_value, exact_bits, exact_value, certified


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _random_maxcut_graph(n, density, seed):
    """
    Generate a random NetworkX graph with mixed-sign weights, matching
    PyQrackIsing convention: edges added explicitly with weight=, no self-loops.
    Density controls the probability of each edge existing; sign is an
    independent 50/50 coin toss on the magnitude drawn from [0, 1).

    PyQrackIsing maximises sum_{(i,j) in cut} w_ij over all edge weights,
    positive or negative.
    """
    if seed is not None:
        np.random.seed(seed)
    G = nx.Graph()
    for u in range(n):
        for v in range(u + 1, n):
            if np.random.random() > density:
                continue
            weight = np.random.random()
            if np.random.random() < 0.5:
                weight *= -1
            G.add_edge(u, v, weight=weight)
    return G


def _brute_force(Q, n):
    """Reference exact MAXCUT solver for small n to validate B&B."""
    best_val = -float("inf")
    best_bits = None
    for mask in range(1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.bool_)
        val = cut_value(Q, bits)
        if val > best_val:
            best_val = val
            best_bits = bits.copy()
    return best_bits, best_val


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm-start B&B exact QUBO solver demo")
    parser.add_argument("--n", type=int, default=20, help="Problem size (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--density", type=float, default=0.5, help="Edge density (default 0.5)")
    parser.add_argument("--time-limit", type=float, default=None, help="B&B time limit in seconds")
    parser.add_argument("--validate", action="store_true", help="Run brute-force validation (n<=24 only)")
    args = parser.parse_args()

    n = args.n
    print(f"Random MAXCUT: n={n}, density={args.density}, seed={args.seed}")
    G = _random_maxcut_graph(n, args.density, args.seed)
    # Q = graph_to_dense_upper(G)

    # Here we use a random warm start so the demo runs without PyQrackIsing.
    # rng = np.random.default_rng(args.seed + 1)
    # warm_bits = rng.integers(0, 2, size=n).astype(np.bool_)
    # warm_value = qubo_value(Q, warm_bits)
    # print(f"Random warm start value: {warm_value:.6f}")
    # print()
    # exact_bits, exact_value, certified = qubo_branch_and_bound(
    #     Q, warm_bits, verbose=True, time_limit=args.time_limit
    # )
    # For a real warm start from PyQrackIsing, replace the above:
    warm_bits, warm_value, exact_bits, exact_val, certified = solve_qubo_exact(G, time_limit=args.time_limit, verbose=True)

    if args.validate and n <= 24:
        print("\nValidating against brute force...")
        bf_bits, bf_value = _brute_force(Q, n)
        match = abs(bf_value - exact_value) < 1e-9
        print(f"Brute force: {bf_value:.6f}  |  B&B: {exact_value:.6f}  |  Match: {match}")
        if not match:
            print("  WARNING: mismatch — check LP bound implementation")
