import networkx as nx


def count_closure_prob_matrix(graph: nx.DiGraph, probability_dict: dict) -> dict:
    nodes = list(graph.nodes)
    closure_probability_matrix = dict(zip(
        [(node_a, node_b) for node_a in nodes for node_b in nodes],
        [
            (
                probability_dict[(node_a, node_b)]
                if (node_a, node_b) in probability_dict
                else 0
            )
            for node_a in nodes
            for node_b in nodes
        ],
    ))
    for node_w in nodes:
        for node_u in nodes:
            for node_v in nodes:
                closure_probability_matrix[(node_u, node_v)] = max(
                    [
                        closure_probability_matrix[(node_u, node_v)],
                        closure_probability_matrix[(node_u, node_w)]
                        * closure_probability_matrix[(node_w, node_v)],
                    ]
                )
    return closure_probability_matrix


def improved_centrality(
    graph: nx.DiGraph, probability_dict: dict
) -> dict:
    nodes = list(graph.nodes)
    closure_probability_matrix = count_closure_prob_matrix(graph, probability_dict)
    node_rate = dict(zip(
        nodes,
        [0 for _ in nodes]
    ))
    for node in nodes:
        for neighbour in nodes:
            node_rate[node] += closure_probability_matrix[(node, neighbour)]
        node_rate[node] /= len(nodes)
    return node_rate


def imporved_centrality_influencers(graph: nx.DiGraph, influencers_count: int, probability_dict: dict) -> list:
    node_rate = improved_centrality(graph, probability_dict)
    return sorted(node_rate, key=lambda x: node_rate[x])[-influencers_count:]
