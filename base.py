import networkx as nx

from models.threshold_model import linear_threshold_model


if __name__ == "__main__":
    graph = nx.read_edgelist(
        "./congress_network/congress.edgelist", create_using=nx.DiGraph()
    )
    influencers = ['0', '12', '4']
    print(linear_threshold_model(graph, influencers, 10))
