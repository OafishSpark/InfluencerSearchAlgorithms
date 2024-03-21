import networkx as nx

from numpy import random

from copy import copy

from collections.abc import Callable


# template for threshold models
def threshold_model(
    graph: nx.DiGraph,
    activated_init: list,
    t_max: int,
    theta: dict,
    threshold_function: Callable[[str, nx.DiGraph, list], float],
) -> list:
    tr_graph = graph.reverse()
    activated = copy(activated_init)
    not_activated = []
    for elem in list(graph.nodes):
        if elem in activated:
            continue
        else:
            not_activated.append(elem)
    for t in range(0, t_max):
        for elem in not_activated:
            thrshld = threshold_function(elem, tr_graph, activated)
            tht = theta[elem]
            if thrshld > tht:
                not_activated.remove(elem)
                activated.append(elem)
        if not not_activated:
            break
    return activated


# linear threshold function (normalized)
def ltfn(elem: str, tr_graph: nx.DiGraph, activated: list) -> float:
    inputs = list(tr_graph.adj[elem])
    sum_of_inps = sum([tr_graph.edges[elem, inp]["weight"] for inp in inputs])
    if sum_of_inps == 0:
        return 0
    answer = 0
    for inp in inputs:
        if inp in activated:
            answer += tr_graph.edges[elem, inp]["weight"]
    return answer / sum_of_inps


# linear threshold function for convinient usage
def linear_threshold_model(
    graph: nx.DiGraph, activated: list, t_max: int = 10, theta: dict = None
) -> list:
    if not theta:
        theta = dict(
            zip(list(graph.nodes), random.uniform(0.01, 0.99, graph.number_of_nodes()))
        )
    return threshold_model(graph, activated, t_max, theta, ltfn)
