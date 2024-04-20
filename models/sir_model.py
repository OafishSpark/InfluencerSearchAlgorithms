import networkx as nx

from numpy import random

from copy import copy
from collections.abc import Callable


def sir_model(
    graph: nx.DiGraph,
    activated_init: list,
    t_max: int = 10,
    infection_prob: dict = None,
    recover_prob: dict = None,
) -> list:
    # init stage
    susceptible_id, infected_id, recovered_id = 0, 1, 2
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    infected = copy(activated_init)
    if not infection_prob:
        # infection_prob = dict(zip(nodes, random.uniform(0.01, 0.99, n_nodes)))
        max_weight = max([graph.edges[elem]["weight"] for elem in graph.edges])
        infection_prob = dict(zip(
            [edge for edge in list(graph.edges)],
            [0.95 - graph.edges[elem]["weight"] / max_weight / (2 + graph.edges[elem]["weight"] / max_weight) for elem in graph.edges])
        )
    if not recover_prob:
        recover_prob = dict(zip(nodes, random.uniform(0.01, 0.99, n_nodes)))
    # modeling
    status = dict(
        zip(
            list(graph.nodes),
            [infected_id if elem in infected else susceptible_id for elem in nodes],
        )
    )
    for _ in range(t_max):
        for elem in infected:
            for sus in list(graph.adj[elem]):
                if status[sus] == susceptible_id:
                    prob = random.uniform(0, 1)
                    if prob > infection_prob[(elem, sus)]:
                        status[sus] = infected_id
                prob = random.uniform(0, 1)
                if prob > recover_prob[elem]:
                    status[elem] = recovered_id
        for elem in nodes:
            if status[elem] == infected_id:
                infected.append(elem)
        if not infected:
            break    
    answer = []
    for elem in nodes:
        if status[elem] == susceptible_id:
            continue
        else:
            answer.append(elem)
    return answer


def sir_inf_model(graph: nx.DiGraph, activated_init: list, t_max: int = 5):
    nodes = list(graph.nodes)
    susceptible_id, infected_id, recovered_id = 0, 1, 2
    infected = copy(activated_init)
    status = dict(
        zip(
            list(graph.nodes),
            [infected_id if elem in infected else susceptible_id for elem in nodes],
        )
    )
    for _ in range(t_max):
        for elem in infected:
            for sus in list(graph.adj[elem]):
                if status[sus] == susceptible_id:
                    status[sus] = infected_id
            status[elem] = recovered_id
        infected = []
        for node in nodes:
            if status[node] == infected_id:
                infected.append(node)
        if not infected:
            break
    answer = []
    for elem in nodes:
        if status[elem] == susceptible_id:
            continue
        else:
            answer.append(elem)
    return answer