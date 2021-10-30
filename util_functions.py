from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
from util import S2VGraph
import multiprocessing as mp
from itertools import islice

def keygates2subgraphs(A,B, train_pos, train_neg, test_pos, test_neg, val_pos,val_neg, h=2, node_information=None, no_parallel=False, DE_FLAG=True):
    def helper(A,B, links, g_label, DE_FLAG=True):
        g_list = []
        if no_parallel:
            for i in tqdm(links):
                g, n_labels, n_features,ind = subgraph_extraction_labeling(i, A, B,h, node_information, DE_FLAG)
                g_list.append(S2VGraph(g, g_label, n_labels, n_features,i))
            return g_list
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.map_async(
                parallel_worker,
                [(i, A,B, h,  node_information, DE_FLAG) for i in links]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [S2VGraph(g, g_label, n_labels, n_features,ind) for g, n_labels, n_features,ind in results]
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs, val_graphs = None, None, None
    print(type(train_pos))

    if train_pos is not None and train_neg is not None  :
        train_graphs = helper(A,B, train_pos, 0, DE_FLAG) + helper(A, B, train_neg, 1, DE_FLAG)
    if test_pos is not None and test_neg is not None:
        test_graphs = helper(A, B, test_pos, 0, DE_FLAG) + helper(A, B, test_neg, 1, DE_FLAG)
    if val_pos is not None and  val_neg is not None:
        val_graphs = helper(A, B, val_pos, 0, DE_FLAG) + helper(A, B, val_neg, 1, DE_FLAG)
    return train_graphs, test_graphs,val_graphs

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(ind, A, B, h=2,node_information=None,DE_FLAG=True):
    print("Inside Subgraph Extraction Labeling")
    # extract the h-hop enclosing subgraph around node (key-gate) 'ind'
    dist = 0
    nodes =[]
    visited = []
    fringe = []
    nodes.append(ind)
    visited.append(ind)
    fringe.append(ind)
    labels=np.zeros(1) # the key-gate gets a label 0
    for dist in range(1, h+1):
        if dist==1:
            fan_out, fan_in=direction(ind,A,B)
            visited = union_list(visited,fan_out)
            for e in fan_out:
                labels=np.append(labels,[(dist*-1)],0)
                nodes = union_list(nodes,fan_out)
            visited = union_list(visited,fan_in)

            for e in fan_in:
                labels=np.append(labels,[dist],0)
                nodes = union_list(nodes,fan_in)
        else:
            if len(fan_out)>0:
                fringe=fan_out
                fringe = neighbors(fringe, A)
                fringe = subtract_list(fringe,visited)
                visited = union_list(visited,fringe)

                for e in fringe:
                    labels=np.append(labels,[(dist*-1)],0)
                nodes = union_list(nodes,fringe)
                fan_out=fringe

            if len(fan_in)>0:
                fringe=fan_in
                fringe = neighbors(fringe, A)
                fringe = subtract_list(fringe,visited)

                visited = union_list(visited,fringe)
                for e in fringe:
                    labels=np.append(labels,[dist],0)

                nodes = union_list(nodes,fringe)
                fan_in=fringe
    subgraph = A[nodes, :][:, nodes]
    print(nodes)
    print(labels)
    labels=labels.astype(int)
    # get node features
    features = []
    if node_information is not None:
        i=0
        for node in nodes:
            vector=[]
            vector=list(node_information[node])
            one_hot=None
            if labels[i]==-2:
                one_hot=[1,0,0,0,0]
            elif labels[i]==1:
                one_hot=[0,0,0,1,0]
            elif labels[i]==2:
                one_hot=[0,0,0,0,1]
            elif labels[i]==-1:
                one_hot=[0,1,0,0,0]
            elif labels[i]==0:
                one_hot=[0,0,1,0,0]
            if DE_FLAG:
                print("We are performing distance encoding")
                vector.extend(one_hot)
            else:
                print("We are not performing distance encoding")
            print("For node "+str(node)+" the feature is now ")
            print(vector)
            features.append(vector)
            i=i+1
    features_=np.array(features)
    features_ = np.float32(features_)
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    g.nodes.data('foo', default=ind)    
    print(labels.shape)
    if len(nodes)==1:
        print("Yes I have a single node\n")
        print(labels.shape)
        labels=labels.reshape(1,1)
    return g, labels.tolist(), features_, ind
def direction(node,A,B):
    print("Inside Direction for node "+str(node))
    fri=set()
    fri.add(node)
    res=neighbors(fri,A)
    print("Those are neighbors!")
    print(res)
    fan_in=neighbors_fanin(fri,B)
    print("Those are fanin neighbors!")
    print(fan_in)
    fan_out=neighbors_fanout(fri,B)
    print("Those are fan out neighbors!")
    print(fan_out)
    return fan_out, fan_in
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = []
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = list(nei)
        res = union_list(res,nei)
    return res

def neighbors_fanin(fringe, A):
    res = []
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = list(nei)
        res = union_list(res,nei)
    return res

def neighbors_fanout(fringe, A):
    res = []
    for node in fringe:
        _, nei, _ = ssp.find(A[node, :])
        nei = list(nei)
        res = union_list(res,nei)
    return res
def union_list(first_list,second_list):
    resulting_list = list(first_list)
    resulting_list.extend(x for x in second_list if x not in resulting_list)
    return resulting_list

def subtract_list(first_list,second_list):
    resulting_list = []#list(first_list)
    resulting_list.extend(x for x in first_list if x not in second_list)
    return resulting_list



