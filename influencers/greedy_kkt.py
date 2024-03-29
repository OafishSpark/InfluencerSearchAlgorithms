import networkx as nx

import numpy as np

from collections.abc import Callable

from models.sir_model import sir_inf_model


def greedy_kkt_influencers(
    graph: nx.DiGraph,
    influencers_count: int = 1,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 3,
    rounds: int = 1,
) -> list:
    assert influencers_count < len(graph.nodes)
    influencers = []
    others = list(graph.nodes())
    for i in range(influencers_count):
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
