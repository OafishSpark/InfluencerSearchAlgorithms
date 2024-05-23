import time

from copy import deepcopy as copy


from tqdm import tqdm

import networkx as nx

from numpy import random

import matplotlib.pyplot as plt


from models.threshold_model import linear_threshold_model
from models.independent_cascade_model import independent_cascade_model

from influencers.greedy_kkt import greedy_kkt_influencers
from influencers.shapley_value import shapley_value_influencers
from influencers.csr import csr_influencers

from influencers.improved_centrality import imporved_centrality_influencers, count_closure_prob_matrix
from influencers.new_centrality import new_centrality_upgraded_influencers

from communities.kmeans import kmeans
from communities.shapley_value import new_communities_shapley

from communities.new_communities import new_communities, community_detection_value


T_MAX = 20
P_UW = 0.5


def model(
    graph: nx.DiGraph,
    active_init: list,
    t: int = 10,
) -> int:
    edges = list(graph.edges)
    cascade_prob = dict(
        zip(edges, [P_UW for _ in edges])
    )
    return independent_cascade_model(
        graph, active_init, t, cascade_prob
    )


def perform_experiment(
    graph: nx.DiGraph,
    num_iter: int,
    n_influencers,
    algorithms={
        "greedy": greedy_kkt_influencers,
        "shapley": shapley_value_influencers,
        "csr": csr_influencers,
        "centrality topology": new_centrality_upgraded_influencers,
        "centrality cascades": imporved_centrality_influencers,
    },
):
    # init stage
    edges = list(graph.edges)
    nodes = list(graph.nodes)
    results_cascades = dict(
        zip(
            algorithms.keys(),
            [[0.0 for _ in range(len(n_influencers))] for _ in range(len(algorithms))],
        )
    )
    results_thresholds = dict(
        zip(
            algorithms.keys(),
            [[0.0 for _ in range(len(n_influencers))] for _ in range(len(algorithms))],
        )
    )
    performance_time = dict(
        zip(
            algorithms.keys(),
            [[0 for _ in range(len(n_influencers))] for _ in range(len(algorithms))],
        )
    )
    # perform stage
    prob = dict(zip(edges, [graph.edges[edge]["weight"] for edge in edges]))
    for ind, n in enumerate(n_influencers):
        influencers = {}
        for algorithm in algorithms.keys():
            time_begin = time.process_time()
            if algorithm == "greedy":
                influencers[algorithm] = algorithms[algorithm](
                    graph, n, model, T_MAX, 10
                )
            elif algorithm == "shapley":
                influencers[algorithm] = algorithms[algorithm](
                    graph, n, model, T_MAX, 10, 10
                )
            elif algorithm == "csr":
                influencers[algorithm] = algorithms[algorithm](
                    graph, n, new_communities(graph, prob, n)
                )
            elif algorithm == "centrality topology":
                influencers[algorithm] = algorithms[algorithm](graph, n, T_MAX)
            elif algorithm == "centrality cascades":
                influencers[algorithm] = algorithms[algorithm](graph, n, prob)
            else:
                assert "Cho to ne tak"
            performance_time[algorithm][ind] += time.process_time() - time_begin
        for _ in tqdm(range(num_iter)):
            thresholds_cascades = dict(
                zip(
                    edges,
                    random.uniform(0.1, 0.9, graph.number_of_edges()),
                )
            )
            thresholds_thresholds = dict(
                zip(
                    nodes,
                    random.uniform(0.1, 0.9, graph.number_of_nodes()),
                )
            )
            for algorithm in algorithms.keys():
                results_cascades[algorithm][ind] += (
                    len(
                        independent_cascade_model(
                            graph,
                            influencers[algorithm],
                            T_MAX,
                            prob,
                            thresholds_cascades,
                        )
                    )
                    / num_iter
                )
                results_thresholds[algorithm][ind] += (
                    len(
                        linear_threshold_model(
                            graph, influencers[algorithm], T_MAX, thresholds_thresholds
                        )
                    )
                    / num_iter
                )
    return (performance_time, results_cascades, results_thresholds)


def visualize(time, cascades, thresholds, n):
    # Создаем фигуру и подграфики
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    markers = ['-|', '-+', '-.', '-s', '-o']
    # Первый график
    for i, t in enumerate(time.keys()):
        axs.plot(n, time[t], markers[i], label=t)
    axs.set_title('Время работы, с')
    axs.set_xlabel('Число инфлюенсеров')
    axs.set_ylabel('Время работы, с')
    plt.tight_layout()
    # Показываем графики
    plt.show()
    # Второй график
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i, c in enumerate(cascades.keys()):
        axs.plot(n, cascades[c], markers[i], label=c)
    axs.set_title('Среднее число активированных вершин в каскадной модели')
    axs.set_xlabel('Число инфлюенсеров')
    axs.set_ylabel('Вершины')
    plt.tight_layout()
    # Показываем графики
    plt.show()
    # Третий график
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i, h in enumerate(thresholds.keys()):
        axs.plot(n, thresholds[h], markers[i], label=h)
    axs.set_title('Среднее число активированных вершин в пороговой модели')
    axs.set_xlabel('Число инфлюенсеров')
    axs.set_ylabel('Вершины')
    # Настраиваем расстояние между подграфиками
    plt.legend(loc="lower right", title="Легенда", frameon=False)
    plt.tight_layout()
    # Показываем графики
    plt.show()


def perform_randoms(n_graphs, n_influencers):
    names = {
        "greedy",
        "shapley",
        "csr",
        "centrality topology",
        "centrality cascades",
    }
    time_g = dict(
        zip(
            names,
            [[0.0 for _ in range(len(n_influencers))] for _ in range(len(names))],
        )
    )
    res_c_g, res_t_g = copy(time_g), copy(time_g)
    for _ in tqdm(range(n_graphs)):
        graph = nx.fast_gnp_random_graph(100, 0.05, directed=True)
        nx.set_edge_attributes(graph, P_UW, "weight")
        time, res_c, res_t = perform_experiment(graph, 1000, n_influencers)
        for name in names:
            for n in range(len(n_influencers)):
                time_g[name][n] += time[name][n] / n_graphs
                res_c_g[name][n] += res_c[name][n] / n_graphs
                res_t_g[name][n] += res_t[name][n] / n_graphs
    return time_g, res_c_g, res_t_g             
    

def perform_communities(graph):
    edges = list(graph.edges)
    prob = dict(zip(edges, [graph.edges[edge]["weight"] for edge in edges]))
    communities = {}
    closure_prob_matrix = count_closure_prob_matrix(graph, prob)
    time_b = time.process_time()
    communities['new'] = new_communities(graph, prob) 
    print(f'new: {time.process_time() - time_b}')
    # time_b = time.process_time()
    # communities['shapley'] = new_communities_shapley(graph, prob, None, model, 10, 10, 10)
    # print(f'shapley: {time.process_time() - time_b}')
    time_b = time.process_time()
    communities['clusters'] = kmeans(graph)
    print(f'clusters: {time.process_time() - time_b}')
    for community in communities.keys():
        print(community)
        print(nx.community.modularity(graph, communities[community]))
        print(nx.community.partition_quality(graph, communities[community]))
        print(community_detection_value(communities[community], closure_prob_matrix))


if __name__ == "__main__":
    n_influencers = [1, 5, 10, 15, 20]
    # time_1, res_c, res_t = perform_randoms(50, n_influencers)
    # graph = nx.read_edgelist("./congress_network/congress.edgelist", create_using=nx.DiGraph())
    graph_email = nx.read_edgelist("./email-Eu-core.txt", create_using=nx.DiGraph)
    graph_email.remove_edges_from(nx.selfloop_edges(graph_email))
    graph_email.remove_nodes_from(list(nx.isolates(graph_email)))
    graph = graph_email.subgraph(list(graph_email.nodes())[:50])
    # graph_wiki_votes = nx.read_edgelist("./wiki-Vote.txt", create_using=nx.DiGraph())
    # nx.set_edge_attributes(graph_wiki_votes, 1, 'weight')
    # graph_wiki_votes.remove_edges_from(nx.selfloop_edges(graph_wiki_votes))
    # graph_wiki_votes.remove_nodes_from(list(nx.isolates(graph_wiki_votes)))
    # graph = nx.subgraph(graph_wiki_votes, list(graph_wiki_votes.nodes)[1530:1730])
    nx.set_edge_attributes(graph, P_UW, "weight")
    # time_1, res_c, res_t = perform_experiment(graph, 1000, n_influencers)
    # print(time_1)
    # print(res_c)
    # print(res_t)
    # visualize(time_1, res_c, res_t, n_influencers)
    perform_communities(graph)
    print(len(graph.nodes))
    print(nx.density(graph))
    print("The End")
    # options = {"with_labels": True, "node_size": 250, "linewidths": 0, "width": 0.5}
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, **options)
