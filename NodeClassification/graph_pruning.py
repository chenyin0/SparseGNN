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
        edges = th.nonzero(adj).tolist()
        g = ig.Graph(n=v_num, edges=edges)
        g = g.simplify()
        g.es['weight'] = np.random.randint(1,2, g.ecount())

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

        return adj        

