import networkx as nx

from numpy import log2

from communities.clustering import clustering_communities 


def community_diversity(graph: nx.graph, node: str, comm_l: list, comm_d: dict) -> float:
    answer = 0
    for comm in comm_l:
        k1 = sum([graph.edges[node, elem]['weight']
                   for elem in list(graph.adj[node])])
        k2 = sum([graph.edges[node, elem]['weight'] 
                  if elem in graph.adj[node]
                    else 0 for elem in comm])
        if k1 < 10e-3 or k2 < 10e-6:
            continue
        answer += k2 / k1 * log2(k2 / k1) 
    return answer


def csr_influencers(
    graph: nx.DiGraph,
    k: int = 1,
    communities_l: list = None,
) -> list:
    if not communities_l:
        communities_l = clustering_communities(graph, k)
    n_communities = len(communities_l)
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    communities_d = dict(zip(nodes, [0 for i in range(len(nodes))]))
    for i, cluster in enumerate(communities_l):
        for elem in cluster:
            communities_d[elem] = i
    node_rate = dict(zip(nodes, [0 for i in range(len(nodes))]))
    for node in nodes:
        deg = len(graph.adj[node])
        modularity = len(communities_l[communities_d[node]]) / n_nodes
        diversity = community_diversity(graph, node, communities_l, communities_d)
        density = nx.density(nx.subgraph(graph, communities_l[communities_d[node]])) / nx.density(graph)
        node_rate[node] = deg * (1 + modularity * diversity * density)
    return sorted(node_rate)[:k]