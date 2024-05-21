import networkx as nx

from numpy import random


def independent_cascade_model(
    graph: nx.DiGraph,
    activated_init: list,
    t_max: int = 10,
    probabilities: dict = None,
    thresholds: dict = None,   
) -> list:
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    if probabilities == None:
        probabilities = dict(zip(
            edges,
            random.uniform(0.1, 0.9, graph.number_of_edges()),
        ))
    if thresholds == None:
        thresholds = dict(zip(
            edges,
            random.uniform(0.1, 0.9, graph.number_of_edges()),
        ))
    activating = dict(zip(
        nodes,
        [int(node in activated_init) for node in list(graph.nodes)]
    ))
    # 0 — no activated yet
    # 1 — activated and spreading influence
    # 2 — recently activated (turns to 1 after round ending)
    # 3 — activated and spread all its influence
    number_not_activated = len(nodes) - len(activated_init)
    for _ in range(t_max):
        if number_not_activated == 0:
            break
        for node in nodes:
            if activating[node] == 1:
                # activating stage
                for neighbour in list(graph.adj[node]):
                    if activating[neighbour] == 0:
                        if probabilities[(node, neighbour)] >= thresholds[(node, neighbour)]:
                            activating[neighbour] = 2        
                activating[node] = 3
        for node in nodes:
            if activating[node] == 2:
                activating[node] = 1
    activated = [node for node in filter(lambda x: activating[x] > 0, activating.keys())]
    return activated
