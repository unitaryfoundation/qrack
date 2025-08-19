# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# After tackling the case where parameters are uniform and independent of time, we generalize the model by averaging per-qubit behavior as if the static case and per-time-step behavior as finite difference. This provides the basis of a novel physics-inspired (adiabatic TFIM) MAXCUT approximate solver that often gives optimal or exact answers on a wide selection of graph types.

from PyQrackIsing import maxcut_tfim
import networkx as nx
import numpy as np


# NP-complete spin glass
def generate_spin_glass_graph(n_nodes=16, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.choice([-1, 1])  # spin glass couplings
    return G


def flip_spin(spins, i):
    new_spins = spins.copy()
    new_spins[i] *= -1
    return new_spins


if __name__ == "__main__":
    # NP-complete spin glass
    G = generate_spin_glass_graph(n_nodes=16, seed=42)

    cut_value, bitstring, cut_edges = maxcut_tfim(G, quality=11)

    print((cut_value, bitstring, cut_edges))

    # Convert bitstring to spins
    spins = {i: 1 if bitstring[i] == '1' else -1 for i in range(len(bitstring))}

    # Reconstruct Ising energy (note: MAXCUT flips sign!)
    E_claim = -sum(G[u][v].get("weight", 1) * spins[u] * spins[v] for u, v in G.edges())

    for i in range(10):  # Try 10 random single spin flips
        idx = np.random.randint(0, len(spins))
        perturbed = flip_spin(spins, idx)
        E_perturbed = -sum(G[u][v].get("weight", 1) * perturbed[u] * perturbed[v] for u, v in G.edges())
        assert E_claim <= E_perturbed  # Should not find a better energy)
