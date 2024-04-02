import networkx as nx

import numpy as np

from collections.abc import Callable
from itertools import combinations

from influencers.csr import csr_rank_nodes

from communities.clustering import topological_distance, single_linkage_dist

def topological_sort(graph: nx.DiGraph) -> list:
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    sorted_list = []
    # 0 -- not visited
    # 1 -- visited
    vertices = dict(zip(nodes, [0 for i in range(n_nodes)]))

    # recoursive dfs procedure
    def inner_dfs(vertice: str) -> None:
        vertices[vertice] = 1
        for v in graph[vertice]:
            if vertices[v] == 0:
                inner_dfs(v)
        sorted_list.append(vertice)

    for node in nodes:
        if vertices[node] == 0:
            inner_dfs(node)
    return sorted_list


def scc_kosarajus(graph: nx.DiGraph) -> list:
    # create the transposed graph
    nodes = list(graph.nodes)
    n_nodes = len(graph)
    graph_tr = graph.reverse()
    # create the topologically sorted list of vertices
    vertice_queue = topological_sort(graph)
    vertice_queue.reverse()
    # dfs via the transposed graph
    answer = []
    # 0 -- not visited
    # 1 -- visited
    vertices = dict(zip(nodes, [0 for _ in range(n_nodes)]))

    # recoursive dfs procedure
    def inner_dfs(vertice):
        answer[-1].append(vertice)
        vertices[vertice] = 1
        for adj_vertice in list(graph_tr.adj[vertice]):
            if vertices[adj_vertice] == 0:
                inner_dfs(adj_vertice)

    for vertice in vertice_queue:
        if vertices[vertice] == 0:
            answer.append([])
            inner_dfs(vertice)
    return answer


def strongly_components_purdom(graph: nx.DiGraph, n_components: int = None) -> tuple:
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    graph_tr = graph.reverse()
    sc_components = scc_kosarajus(graph)
    if n_components:
        new_graph = graph.copy()
        while n_components > len(sc_components):
            min_edge_weight = min([new_graph.edges[edge]['weight'] for edge in list(new_graph.edges)])
            for edge in list(new_graph.edges):
                new_graph.edges[edge]['weight'] -= min_edge_weight
                if new_graph.edges[edge]['weight'] < 10e-10:
                    new_graph.remove_edge(edge[0], edge[1])
            sc_components = scc_kosarajus(new_graph)
    sc_components.reverse()
    sc_components_d = dict(zip(nodes, [0 for _ in range(n_nodes)]))
    n_scc = len(sc_components)
    for i_scc, scc in enumerate(sc_components):
        for node in scc:
            sc_components_d[node] = i_scc
    answer_scc = [0 for _ in range(n_scc)]
    answer_nodes = dict(zip(nodes, [0 for _ in range(n_nodes)]))
    for i_scc, scc in enumerate(sc_components):
        answer_scc[i_scc] += len(scc)
        visited_scc = [False for _ in range(n_scc)]
        visited_scc[i_scc] = True
        for node in scc:
            answer_nodes[node] = answer_scc[i_scc]
            for neighbour in graph_tr.adj[node]:
                neighbour_ind_scc = sc_components_d[neighbour]
                if not visited_scc[neighbour_ind_scc]:
                    answer_scc[neighbour_ind_scc] += answer_scc[i_scc]
                    visited_scc[neighbour_ind_scc] = True
    return (answer_nodes, sc_components, answer_scc)


def transitive_closure_purdom(graph: nx.DiGraph, n_components: int = None) -> dict:
    return strongly_components_purdom(graph, n_components)[0]


def transitive_closure_floyd(graph: nx.DiGraph) -> dict:
    nodes = list(graph.nodes)
    tr_closure = graph.copy()
    for node_i in nodes:
        for node_e in nodes:
            for node_q in nodes:
                weight_eq = np.inf
                if tr_closure.has_edge(node_e, node_q):
                    weight_eq = tr_closure.edges[node_e, node_q]["weight"]
                weight_eiq = np.inf
                if tr_closure.has_edge(node_e, node_i) and tr_closure.has_edge(
                    node_i, node_q
                ):
                    weight_eiq = (
                        tr_closure.edges[node_e, node_i]["weight"]
                        + tr_closure.edges[node_i, node_q]["weight"]
                    )
                min_w = min(weight_eq, weight_eiq)
                if min_w < np.inf:
                    tr_closure.add_edge(node_e, node_q, weight=min_w)
    answer = dict(zip(nodes, [len(tr_closure.adj[node]) for node in nodes]))
    return answer


def hierarchy_communities_original(
    graph: nx.DiGraph,
    communities_count: int = 1,
    transitive_closure: Callable[[nx.DiGraph], dict] = transitive_closure_purdom,
) -> list:
    tr_closure = transitive_closure(graph, communities_count)
    node_queue = sorted(tr_closure, key=lambda x: tr_closure[x], reverse=True)
    communities = [[node] for node in node_queue[:communities_count]]
    get_node = lambda x: graph.edges[x[0], x[1]]['weight'] if graph.has_edge(x[0], x[1]) else 0
    for ind_node, node in enumerate(node_queue[communities_count:]):
        community_len = ind_node // communities_count + 1
        node_community = 0
        node_estimation = -np.inf
        for ind_community, community in enumerate(communities):
            if len(community) > community_len:
                continue
            estimation = sum([
                max(get_node((node, target)), get_node((target, node))) for target in community
            ])
            if estimation > node_estimation:
                node_estimation = estimation
                node_community = ind_community
        communities[node_community].append(node)
    return communities


def hierarchy_communities_improved(
    graph: nx.DiGraph,
    communities_count: int = 1,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
) -> list:
    nodes = list(graph.nodes)
    tr_closure_rank, sc_components, scc_rank = strongly_components_purdom(graph, communities_count) 
    scc_queue = sorted([i for i in range(len(scc_rank))], reverse=True, key=lambda x: scc_rank[x])
    csr_rank = csr_rank_nodes(graph, sc_components)
    node_queue = sorted(nodes, reverse = True, key = lambda x: tr_closure_rank[x] + csr_rank[x])
    leaders = ['' for _ in range(communities_count)]
    for ind in range(communities_count):
        leader = max(sc_components[scc_queue[ind]], key=lambda x: csr_rank[x])
        leaders[ind] = leader
        node_queue.remove(leader)
    communities = [[leader] for leader in leaders]
    dist_matrix = dict(
        zip(
            [elem for elem in combinations(nodes, 2)],
            [dist_nodes(graph, elem) for elem in combinations(nodes, 2)],
        )
    )
    for node in node_queue:
        (min(communities, key=lambda x: dist_clusters(dist_matrix, (x, [node])))).append(
            node
        )  
    return communities
