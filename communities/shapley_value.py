import networkx as nx

from math import ceil

from collections.abc import Callable
from itertools import combinations

from models.sir_model import sir_inf_model
from communities.clustering import topological_distance, single_linkage_dist
from influencers.shapley_value import shapley_value_influencers
from influencers.improved_centrality import count_closure_prob_matrix
from communities.new_communities import community_detection_value


def shapley_value_communities(
    graph: nx.DiGraph,
    communities_count: int = 2,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 10,
    rounds: int = 1,
    n_permutations: int = 10,
    permutations: dict = None,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
) -> list:
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    sorted_nodes = shapley_value_influencers(
        graph, n_nodes, model, t_max, rounds, n_permutations, permutations
    )
    influencers = ["" for i in range(communities_count)]
    n_influencers = 0
    for node in sorted_nodes:
        for infl in influencers[:n_influencers]:
            if node in list(graph.adj[infl]):
                continue
        influencers[n_influencers] = node
        sorted_nodes.remove(node)
        n_influencers += 1
        if n_influencers == communities_count:
            break
    communities = [[elem] for elem in influencers]
    dist_matrix = dict(
        zip(
            [elem for elem in combinations(nodes, 2)],
            [dist_nodes(graph, elem) for elem in combinations(nodes, 2)],
        )
    )
    for node in sorted_nodes:
        (
            min(communities, key=lambda x: dist_clusters(dist_matrix, (x, [node])))
        ).append(node)
    return communities


def new_communities_shapley(
    graph: nx.DiGraph,
    probability_dict: dict,
    n_communities: int = None,
    model: Callable[[nx.DiGraph, list, int], int] = sir_inf_model,
    t_max: int = 10,
    rounds: int = 1,
    n_permutations: int = 10,
    permutations: dict = None,
    dist_nodes: Callable[[nx.Graph, tuple], float] = topological_distance,
    dist_clusters: Callable[[dict, tuple], float] = single_linkage_dist,
) -> list:
    closure_prob_matrix = count_closure_prob_matrix(graph, probability_dict)
    if not n_communities:
        try:
            scores = []
            for n in range(2, int(ceil(graph.number_of_nodes() ** (1 / 2)))):
                communities = shapley_value_communities(
                    graph,
                    n,
                    model,
                    t_max,
                    rounds,
                    n_permutations,
                    permutations,
                    dist_nodes,
                    dist_clusters,
                )
                scores += [community_detection_value(communities, closure_prob_matrix)]
            n_communities = 2 + max(
                [i for i in range(len(scores))], key=lambda x: scores[x]
            )
        except:
            print("smth wrong")
            n_communities = 2
    return shapley_value_communities(
        graph,
        n,
        model,
        t_max,
        rounds,
        n_permutations,
        permutations,
        dist_nodes,
        dist_clusters,
    )
