import networkx as nx

from models.threshold_model import linear_threshold_model
from models.sir_model import sir_model


if __name__ == "__main__":
    graph = nx.read_edgelist(
        "./congress_network/congress.edgelist", create_using=nx.DiGraph()
    )
    influencers = list(graph.nodes)[:10]
    result_thrshld = 0
    result_sir = 0
    n_experiments = 10
    for i in range(n_experiments):
        result_thrshld += len(linear_threshold_model(graph, influencers, 5)) / n_experiments
        result_sir += len(sir_model(graph, influencers, 5)) / n_experiments
    print(f'Result of the threshold model: {result_thrshld}')
    print(f'Result of the SIR model: {result_sir}')
