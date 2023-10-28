import torch as th
import numpy as np
import unimodal
import igraph as ig


class GraphPruning:

    @classmethod
    def random(cls, adj, prune_ratio):
        adj_edges = th.nonzero(adj)
        pruned_edge_num = round(adj_edges.shape[0] * prune_ratio)
        adj_edge_idx = np.arange(adj_edges.shape[0])
        pruned_edge_idx = np.random.choice(adj_edge_idx, pruned_edge_num, replace=False)
        pruned_edge = adj_edges[pruned_edge_idx]

        for edge in pruned_edge:
            adj[edge[0]][edge[1]] = 0

        return adj

    @classmethod
    def mlf_pruning(cls, adj, prune_ratio):
        v_num = adj.shape[0]
        adj = adj.to_dense().byte()
        edges = th.nonzero(adj).tolist()
        g = ig.Graph(n=v_num, edges=edges)
        g = g.simplify()
        g.es['weight'] = np.random.randint(1, 2, g.ecount())

        # # Create a graph with random integer weights
        # g = ig.Graph.Barabasi(1000,3)
        # g.es['weight'] = np.random.randint(1,40, g.ecount())

        # Get the edgelist dataframe with columns
        # ['source', 'target', 'weight']
        df = g.get_edge_dataframe()

        # Instantiate an MFL object and compute
        # the edge significance
        mlf = unimodal.MLF(directed=False)
        G = mlf.fit_transform(g)
        df_edgelist_1 = G.get_edge_dataframe()

        # # New edge attribute "significance" is created
        # # ['weight', 'significance']
        # print(G.es.attributes())

        # # Apply the transformer to the edgelist
        # # dataframe of the graph
        # mlf = unimodal.MLF(directed=False)
        # df_edgelist_2 = mlf.fit_transform(df)

        ## Pruning
        e_num = G.ecount()
        threshold_index = int(e_num * prune_ratio)
        threshold_value = sorted(G.es['significance'])[threshold_index]

        # pruned_edges = []
        for e in G.es:
            if e['significance'] < threshold_value:
                # pruned_edges.append(e.index)
                # print(e.source, e.target)
                adj[e.source][e.target] = 0
                adj[e.target][e.source] = 0

        # adj_edges = th.nonzero(adj)
        adj = adj.to_sparse()

        return adj

    @classmethod
    def edge_sim_pruning(cls, g, adj, prune_ratio):
        """
        Pruning according to edge similarity
        """
        g_uni = g.remove_self_loop()
        v_adj = GraphPruning.gen_v_adj(g_uni)
        edges = g_uni.edges()
        edge_num = g_uni.num_edges()
        edge_sim = np.zeros(edges[0].shape[0])
        for e_idx in range(edge_num):
            edge_sim[e_idx] = GraphPruning.count_edge_similarity(v_adj, edges[0][e_idx].item(),
                                                                 edges[1][e_idx].item())

        threshold_index = int(edge_num * prune_ratio)
        threshold_value = sorted(edge_sim)[threshold_index]

        # Prune adj
        adj = adj.to_dense().byte()

        # Remove self-loop
        v_num = g.num_nodes()
        for v_idx in range(v_num):
            adj[v_idx][v_idx] = 0

        for e_idx in range(edge_num):
            if edge_sim[e_idx] < threshold_value:
                e_src = edges[0][e_idx].item()
                e_dst = edges[1][e_idx].item()
                adj[e_src][e_dst] = 0
                adj[e_dst][e_src] = 0

        # Add back self-loop for still survived vertices
        for v_idx in range(v_num):
            if adj[v_idx].sum() > 0:
                adj[v_idx][v_idx] = 1

        adj = adj.to_sparse()

        return adj

    @classmethod
    def gen_v_adj(cls, g):
        """
        v_adj (type: list): nghs of each vertex
        """
        v_num = g.num_nodes()
        v_adj = []
        for v_idx in range(v_num):
            v_succ = g.successors(v_idx).tolist()
            v_pred = g.predecessors(v_idx).tolist()
            v_ngh = list(set(v_succ).union(set(v_pred)))
            v_adj.append(v_ngh)

        return v_adj

    @classmethod
    def count_edge_similarity(cls, v_adj, v_i, v_j):
        v_i_ngh = v_adj[v_i]
        v_j_ngh = v_adj[v_j]
        v_ij_inter = list(set(v_i_ngh).intersection(set(v_j_ngh)))
        v_ij_union = list(set(v_i_ngh).union(set(v_j_ngh)))
        e_sim = float(len(v_ij_inter) / len(v_ij_union))

        return e_sim
