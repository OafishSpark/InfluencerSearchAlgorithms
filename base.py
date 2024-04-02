import networkx as nx

from models.threshold_model import linear_threshold_model
from models.sir_model import sir_model, sir_inf_model

from influencers.greedy_kkt import greedy_kkt_influencers
from influencers.centrality import centrality_influencers
from influencers.shapley_value import shapley_value_influencers
from influencers.csr import csr_influencers
from influencers.hierarchy import hierarchy_influencers_dull, hierarchy_influencers_with_csr

from communities.clustering import clustering_communities
from communities.shapley_value import shapley_value_communities

from communities.hierarchy import (
    topological_sort,
    scc_kosarajus,
    transitive_closure_purdom,
    transitive_closure_floyd,
    hierarchy_communities_original,
    hierarchy_communities_improved
)


if __name__ == "__main__":
    graph = nx.read_edgelist(
        "./congress_network/congress.edgelist", create_using=nx.DiGraph()
    )
    # influencers = greedy_kkt(graph, 10)
    # influencers = centrality_influencers(graph, 100)
    # influencers = shapley_value_influencers(graph, 100)
    # sorted_nodes = transitive_closure_floyd(graph)
    # print(scc_kosarajus(graph))
    # influencers = csr_influencers(graph, 100, shapley_value_communities(graph, 10))
    # influencers = sorted(sorted_nodes, key=lambda x: sorted_nodes[x])[-7:]
    influencers_csr = csr_influencers(graph, 7, hierarchy_communities_improved(graph, 7))
    influencers = hierarchy_influencers_with_csr(graph, 7)
    result_thrshld, result_sir = 0, 0
    n_experiments = 10
    for i in range(n_experiments):
        result_thrshld += (
            len(linear_threshold_model(graph, influencers, 10)) / n_experiments
        )
        result_sir += len(sir_model(graph, influencers, 10)) / n_experiments
    print(f"Result of the threshold model: {result_thrshld}")
    print(f"Result of the SIR model: {result_sir}")
    print(
        f"Result of the infinite SIR model: {len(sir_inf_model(graph, influencers, 10))}"
    )
