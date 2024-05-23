from math import ceil

import networkx as nx


from influencers.hierarchy import hierarchy_influencers_with_centrality
from influencers.improved_centrality import (
    count_closure_prob_matrix,
    imporved_centrality_influencers,
)


def community_value(community: list, closure_prob_matrix: dict) -> float:
    value = 0
    for node in community:
        max_prob = 0
        for node_1 in community:
            prob = closure_prob_matrix[(node, node_1)]
            if prob > max_prob:
                max_prob = prob
        value += max_prob
    return value / len(community)


def detect_communities(
    graph: nx.DiGraph, n_communities: int, closure_prob_matrix: dict
):
    nodes = list(graph.nodes)
    leaders = hierarchy_influencers_with_centrality(graph, n_communities)
    communities = []
    scores = []
    for leader in leaders:
        communities += [[leader]]
        nodes.remove(leader)
        scores += [0]
    while nodes:
        ind = min([i for i in range(0, n_communities)], key=lambda x: scores[x])
        community = communities[ind]
        best_score = 0
        best_node = -1
        for node in nodes:
            score = community_value(community + [node], closure_prob_matrix)
            if score >= best_score:
                best_node = node
                best_score = score
        community += [best_node]
        scores[ind] = best_score
        nodes.remove(best_node)
    return communities


def community_detection_value(communities: list, closure_prob_matrix: dict) -> float:
    return sum([community_value(c, closure_prob_matrix) for c in communities]) / len(
        communities
    )


def new_communities(
    graph: nx.DiGraph, probability_dict: dict, n_communities: int = None
) -> list:
    closure_prob_matrix = count_closure_prob_matrix(graph, probability_dict)
    if not n_communities:
        try:
            scores = []
            for n in range(2, int(ceil(graph.number_of_nodes() ** (1 / 1.5)))):
                communities = detect_communities(graph, n, closure_prob_matrix)
                scores += [community_detection_value(communities, closure_prob_matrix)]
            print(scores)
            n_communities = 2 + max(
                [i for i in range(len(scores))], key=lambda x: scores[x]
            )
        except:
            print("smth wrong")
            n_communities = 2
    return detect_communities(graph, n_communities, closure_prob_matrix)
