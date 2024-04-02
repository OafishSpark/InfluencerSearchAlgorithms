import networkx as nx

from collections.abc import Callable
from itertools import combinations

from models.sir_model import sir_inf_model
from communities.clustering import topological_distance, single_linkage_dist
from influencers.shapley_value import shapley_value_influencers


def shapley_value_communities(
    graph: nx.DiGraph,
    communities_count: int = 1,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 10,
    rounds: int = 1,
    n_permutations: int = 10,
    permutations: dict = None,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
) -> list:
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    sorted_nodes = shapley_value_influencers(graph, n_nodes, model, t_max, rounds, n_permutations, permutations)
    influencers = ["" for i in range(communities_count)]
    n_influencers = 0
    for node in sorted_nodes:
        for infl in influencers[:n_influencers]:
            if node in list(graph.adj[infl]):
                continue
        influencers[n_influencers] = node
        sorted_nodes.remove(node)
        n_influencers += 1
        if n_influencers == communities_count:
            break
    communities = [[elem] for elem in influencers]
    dist_matrix = dict(
        zip(
            [elem for elem in combinations(nodes, 2)],
            [dist_nodes(graph, elem) for elem in combinations(nodes, 2)],
        )
    )
    for node in sorted_nodes:
        (min(communities, key=lambda x: dist_clusters(dist_matrix, (x, [node])))).append(
            node
        )
    return communities
