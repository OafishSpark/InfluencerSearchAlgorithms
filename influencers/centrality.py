import networkx as nx

import numpy as np

from collections.abc import Callable


def centrality_INF(graph: nx.DiGraph, node: str) -> float:
    deg = len(list(graph.adj[node]))
    return (
        sum([list(dict(graph.adj[node]).values())[i]["weight"] for i in range(deg)])
        / deg
    )


def centrality_DC(graph: nx.DiGraph, node: str) -> float:
    deg = len(list(graph.adj[node]))
    # return sum([list(dict(graph.adj[node]).values())[i]["weight"] for i in range(deg)])
    return deg


def centrality_influencers(
    graph: nx.DiGraph,
    k: int = 1,
    centrality_rule: Callable[[nx.Graph, str], float] = centrality_DC,
) -> list:
    nodes = list(graph.nodes)
    node_rate = dict(zip(nodes, [centrality_rule(graph, elem) for elem in nodes]))
    return sorted(node_rate)[-k:]
