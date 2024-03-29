import networkx as nx

from collections.abc import Callable
from itertools import combinations


def topological_distance(graph: nx.Graph, nodes: tuple) -> float:
    first, second = nodes
    k_1 = sum([graph.edges[first, elem]["weight"] for elem in list(graph.adj[first])])
    k_2 = sum([graph.edges[second, elem]["weight"] for elem in list(graph.adj[second])])
    k_12 = sum(
        [
            graph.edges[first, elem]["weight"] + graph.edges[second, elem]["weight"]
            for elem in set(graph.adj[first]).intersection(set(graph.adj[second]))
        ]
    )
    return k_1 + k_2 - 2 * k_12


def single_linkage_dist(
    d_mat: dict,
    clusters: tuple,
) -> float:
    first, second = clusters
    answer = 0
    for a in first:
        for b in second:
            temp = 0
            try:
                temp = d_mat[(a, b)]
            except KeyError:
                temp = d_mat[(b, a)]
            if -answer > -temp:
                answer = temp
    return answer


def clustering_communities(
    graph: nx.DiGraph,
    k: int = 2,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
) -> list:
    nodes = list(graph.nodes)
    clusters = [[node] for node in nodes]
    dist_matrix = dict(
        zip(
            [elem for elem in combinations(nodes, 2)],
            [dist_nodes(graph, elem) for elem in combinations(nodes, 2)],
        )
    )
    for _ in range(0, len(nodes) - k):
        first, second = min(
            [elem for elem in combinations(clusters, 2)],
            key=lambda x: dist_clusters(dist_matrix, x),
        )
        clusters.remove(first)
        clusters.remove(second)
        clusters.append(first + second)
    return clusters
