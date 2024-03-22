import networkx as nx

from models.threshold_model import linear_threshold_model
from models.sir_model import sir_model, sir_inf_model

from influencers.greedy_kkt import greedy_kkt
from influencers.centrality import centrality_influencers
from influencers.shapley_value import shapley_value_influencers


if __name__ == "__main__":
    graph = nx.read_edgelist(
        "./congress_network/congress.edgelist", create_using=nx.DiGraph()
    )
    # influencers = greedy_kkt(graph, 10)
    # influencers = centrality_influencers(graph, 10)
    influencers = shapley_value_influencers(graph, 10)
    result_thrshld, result_sir = 0, 0
    n_experiments = 100
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
