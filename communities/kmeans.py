import networkx as nx

from sklearn import cluster, metrics
from math import ceil


def kmeans(graph: nx.DiGraph, n_communities: int = None):
    nodes = list(graph.nodes)
    graph_matrix = nx.adjacency_matrix(graph).todense()
    communities = []
    if not n_communities:
        try:
            scores = []
            for n in range(2, int(ceil(len(nodes) ** (1 / 2)))):
                model = cluster.KMeans(n_clusters=n, n_init=200)
                model.fit(graph_matrix)
                communities_labels = list(model.labels_)
                scores += [metrics.calinski_harabasz_score(graph_matrix, model.labels_)]
            n_communities = 2 + max(
                [i for i in range(len(scores))], key=lambda x: scores[x]
            )
        except:
            print("smth wrong")
            n_communities = 2
    model = cluster.KMeans(n_clusters=n_communities, n_init=200)
    model.fit(graph_matrix)
    communities_labels = list(model.labels_)
    communities = [[] for _ in range(n_communities)]
    for iv, node in enumerate(nodes):
        communities[communities_labels[iv]] += [node]
    return communities
