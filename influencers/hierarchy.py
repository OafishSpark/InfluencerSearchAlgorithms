import networkx as nx

from communities.hierarchy import transitive_closure_purdom, strongly_components_purdom

from influencers.csr import csr_rank_nodes


def hierarchy_influencers_dull(
    graph: nx.DiGraph,
    influencers_count: int = 1,
) -> list:
    tr_closure = transitive_closure_purdom(graph, influencers_count)
    return sorted(tr_closure, key=lambda x: tr_closure[x])[-influencers_count:]


def hierarchy_influencers_with_csr(
    graph: nx.DiGraph,
    influencers_count: int = 1,
) -> list:
    _, sc_components, scc_rank = strongly_components_purdom(graph, influencers_count) 
    sc_components_queue = sorted([i for i in range(len(scc_rank))], reverse=True, key=lambda x: scc_rank[x])
    csr_rank_rank = csr_rank_nodes(graph, sc_components)
    leaders = [max(sc_components[sc_components_queue[ind]], key=lambda x: csr_rank_rank[x]) for ind in range(influencers_count)]
    return leaders