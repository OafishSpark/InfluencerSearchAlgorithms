import networkx as nx

import numpy as np


def new_centrality(graph: nx.DiGraph) -> dict:
    nodes = list(graph.nodes)
    nodes_num = [i for i in range(len(nodes))]
    node_rates = dict(zip(
        nodes_num,
        [0 for _ in nodes]
    ))
    transitive_closure = nx.adjacency_matrix(nx.transitive_closure(graph)).todense()
    set = []
    set_rate = 0
    tr_clsr_row_set = np.array([0 for _ in range(len(nodes))])
    while nodes_num:
        best_node = -1
        best_rate = 0
        for node in nodes_num:
            rate = np.sum(np.logical_or(tr_clsr_row_set, transitive_closure[node]).astype(int))
            rate -= set_rate
            if rate > best_rate:
                best_node = node
                best_rate = rate
        if best_node == -1:
            break
        tr_clsr_row_set = np.logical_or(tr_clsr_row_set, transitive_closure[best_node]).astype(int)
        node_rates[best_node] = best_rate
        set_rate += best_rate
        set.append(best_node)
        nodes_num.remove(best_node)
    return node_rates


def new_centrality_influencers(graph: nx.DiGraph, influencers_count: int) -> list:
    node_rate = new_centrality(graph)
    return sorted(node_rate, key=lambda x: node_rate[x])[-influencers_count:]


def new_centrality_upgraded_influencers(graph: nx.DiGraph, influencers_count: int, t_max: int) -> list:
    node_rate = new_centrality(graph)
    for node in list(graph.nodes):
        node_rate[node] = 1 / t_max * int(graph.out_degree(node)) + (1 - 1/t_max) * node_rate[node]
    return sorted(node_rate, key=lambda x: node_rate[x])[-influencers_count:]
