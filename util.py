import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None,keygate=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.keygate=keygate
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = torch.from_numpy(node_features)

        self.edge_mat = 0

        self.max_neighbor = 0
        self.neighbors = [[] for i in range(len(self.g))]
        for i, j in self.g.edges():
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
        degree_list = []
        for i in range(len(self.g)):
            self.neighbors[i] = self.neighbors[i]
            degree_list.append(len(self.neighbors[i]))
        self.max_neighbor = max(degree_list)
        if self.g.number_of_edges()==0:
            print("Yes we have 0 edges, i will add a self loop")
            self.g.add_edge(0, 0)
        edges = [list(pair) for pair in self.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        deg_list = list(dict(self.g.degree(range(len(self.g)))).values())
        self.edge_mat = torch.LongTensor(edges).transpose(0,1)


