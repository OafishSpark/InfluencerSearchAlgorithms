import networkx as nx

from collections.abc import Callable


def centrality_inf(graph: nx.DiGraph, node: str) -> float:
    deg = len(list(graph.adj[node]))
    return (
        sum([list(dict(graph.adj[node]).values())[i]["weight"] for i in range(deg)])
        / deg
    )


def centrality_degree(graph: nx.DiGraph, node: str) -> float:
    deg = len(list(graph.adj[node]))
    # return sum([list(dict(graph.adj[node]).values())[i]["weight"] for i in range(deg)])
    return deg


def centrality_influencers(
    graph: nx.DiGraph,
    influencers_count: int = 1,
    centrality_rule: Callable[[nx.Graph, str], float] = centrality_degree,
) -> list:
    nodes = list(graph.nodes)
    node_rate = dict(zip(nodes, [centrality_rule(graph, elem) for elem in nodes]))
    return sorted(node_rate, key=lambda x: node_rate[x])[-influencers_count:]
