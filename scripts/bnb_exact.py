from pyqrackising import solve_maxcut_exact # solve_maxcut_exact_sparse, solve_maxcut_exact_streaming
import networkx as nx

G = nx.petersen_graph()
solve_maxcut_exact(G)
