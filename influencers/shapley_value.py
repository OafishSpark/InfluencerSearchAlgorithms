import networkx as nx

import numpy as np
from numpy import random

from collections.abc import Callable

from models.sir_model import sir_inf_model


def shapley_value_influencers(
    graph: nx.DiGraph,
    k: int = 1,
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
    return sorted(rates)[-k:]
