import networkx as nx

from numpy import random

from copy import copy

from collections.abc import Callable


# template for threshold models
def threshold_model(
    graph: nx.DiGraph,
    activated_init: list,
    t_max: int,
    thresholds: dict,
    threshold_function: Callable[[str, nx.DiGraph, list], float],
) -> list:
    nodes = list(graph.nodes)
    active = dict(zip(
        nodes,
        [int(node in activated_init) for node in nodes]
    ))
    # 0 - not active
    # 1 - active
    number_not_activated = len(nodes) - len(activated_init)
    for _ in range(t_max):
        if number_not_activated == 0:
            break
        for node in nodes:
            if active[node] == 0:
                threshold_function_value = threshold_function(node, graph, active)
                if threshold_function_value > thresholds[node]:
                    active[node] = 1
    activated = [node for node in filter(lambda x: active[x] > 0, active.keys())]
    return activated


# linear threshold function (normalized)
def linear_threshold_function_normalized(node: str, graph: nx.DiGraph, active: dict) -> float:
    inputs = graph.in_edges(node)
    sum_of_inps = sum([graph.edges[edge]["weight"] for edge in inputs])
    if sum_of_inps == 0:
        return 0
    answer = 0
    for edge in inputs:
        if active[edge[0]] :
            answer += graph.edges[edge]["weight"]
    return answer / sum_of_inps


# linear threshold function for convinient usage
def linear_threshold_model(
    graph: nx.DiGraph, activated: list, t_max: int = 10, thresholds: dict = None
) -> list:
    if not thresholds:
        thresholds = dict(
            zip(list(graph.nodes), random.uniform(0.01, 0.99, graph.number_of_nodes()))
        )
    return threshold_model(graph, activated, t_max, thresholds, linear_threshold_function_normalized)
