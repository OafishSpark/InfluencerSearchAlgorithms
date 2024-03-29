import networkx as nx

import numpy as np
from numpy import random

from collections.abc import Callable
from itertools import combinations

from models.sir_model import sir_inf_model
from communities.clustering import topological_distance, single_linkage_dist


def shapley_value_communities(
    graph: nx.DiGraph,
    k: int = 1,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 10,
    rounds: int = 1,
    n_permutations: int = 10,
    permutations: dict = None,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
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
    sorted_nodes = sorted(rates)
    influencers = ["" for i in range(k)]
    n_influencers = 0
    for node in sorted_nodes:
        for infl in influencers[:n_influencers]:
            if node in list(graph.adj[infl]):
                continue
        influencers[n_influencers] = node
        sorted_nodes.remove(node)
        n_influencers += 1
        if n_influencers == k:
            break
    clusters = [[elem] for elem in influencers]
    dist_matrix = dict(
        zip(
            [elem for elem in combinations(nodes, 2)],
            [dist_nodes(graph, elem) for elem in combinations(nodes, 2)],
        )
    )
    for node in sorted_nodes:
        (min(clusters, key=lambda x: single_linkage_dist(dist_matrix, (x, [node])))).append(
            node
        )
    return clusters
