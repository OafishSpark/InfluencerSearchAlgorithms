import networkx as nx

import numpy as np

from collections.abc import Callable


def greedy_kkt(
    graph: nx.DiGraph,
    model: Callable[[nx.DiGraph, list, int], int],
    k: int = 1,
    t_max: int = 3,
    rounds: int = 1,
) -> list:
    assert k < len(graph.nodes)
    influencers = []
    others = list(graph.nodes())
    for i in range(k):
        max_estimation = 0
        best_node = ""
        for node in others:
            influencers.append(node)
            estimation = np.mean(
                np.array([len(model(graph, influencers, t_max)) for i in range(rounds)])
            )
            if estimation > max_estimation:
                max_estimation = estimation
                best_node = node
            influencers.remove(node)
        influencers.append(best_node)
        others.remove(best_node)
    return influencers
