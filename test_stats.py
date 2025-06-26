import os
import sys
print(os.getcwd())
print(sys.path[0])

# from stats import degree_stats
from evaluation.stats import degree_stats,spectral_stats,orbit_stats_all
import networkx as nx
import random


len = 2
graph_ref_list = []
graph_pred_list = []

for i in range(len):
    graph_ref_list.append(nx.random_graphs.random_regular_graph(3, 30))  # 20 nodes & each node has 3 neghbours
    graph_pred_list.append(nx.random_graphs.erdos_renyi_graph(30, 0.5))  # n=20 nodes, probablity p = 0.2

degree_stats(graph_ref_list, graph_pred_list,is_parallel=False)
spectral_stats(graph_ref_list, graph_pred_list)
orbit_stats_all(graph_ref_list, graph_pred_list)