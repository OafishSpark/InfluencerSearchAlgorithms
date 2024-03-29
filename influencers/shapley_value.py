import networkx as nx

import numpy as np
from numpy import random

from collections.abc import Callable

from models.sir_model import sir_inf_model


def shapley_value_influencers(
    graph: nx.DiGraph,
    influencers_count: int = 1,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 10,
    rounds: int = 1,
    n_permutations: int = 10,
    permutations: dict = None,
) -> list:
    nodes = graph.nodes
    n_nodes = len(nodes)
    if not permutations:
        permutations = [list(random.permutation(nodes)) for _ in range(n_permutations)]
    rates = dict(zip(nodes, [0.0 for i in range(n_nodes)]))
    for p in permutations:
        for node in nodes:
            rates[node] += (
                np.mean(
                    np.array(
                        [
                            len(model(graph, p[: p.index(node) + 1], t_max))
                            - len(model(graph, p[: p.index(node)], t_max))
                            for _ in range(rounds)
                        ]
                    )
                )
                / n_permutations
            )
    influencers = ['' for i in range(influencers_count)]
    n_influencers = 0
    for node in sorted(rates):
        for infl in influencers[:n_influencers]:
            if node in list(graph.adj[infl]):
                continue
        influencers[n_influencers] = node
        n_influencers += 1
        if n_influencers == influencers_count:
            break 
    return influencers
