import numpy as np
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from collections import deque
import torch
import pandas as pd
from scipy.stats import norm
import pickle  # 用于保存结果
class GrangerCausalityNetwork:
    def __init__(self, data, target_feature, max_lag=1, threshold_p_value=0.05):
        self.data = data
        self.target_feature = target_feature
        self.max_lag = max_lag
        self.threshold_p_value = threshold_p_value

    def construct_macro_graph(self):
        num_features = len(self.data.columns)
        A_Ma = np.zeros((num_features, num_features))

        for i in self.data.columns:
            for j in self.data.columns:
                if i != j:
                    try:
                        test_result = grangercausalitytests(self.data[[i, j]], self.max_lag, verbose=False)
                        min_p_value = min(test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, self.max_lag + 1))
                        A_Ma[self.data.columns.get_loc(i), self.data.columns.get_loc(j)] = min_p_value
                    except Exception as e:
                        print(f"Error testing {i} and {j}: {e}")
                        continue
        return A_Ma

    def construct_micro_graph(self, window_data):
        num_features = len(window_data.columns)
        A_Mi = np.zeros((num_features, num_features))

        for i in window_data.columns:
            for j in window_data.columns:
                if i != j:
                    try:
                        test_result = grangercausalitytests(window_data[[i, j]], self.max_lag, verbose=False)
                        min_p_value = min(test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, self.max_lag + 1))
                        A_Mi[window_data.columns.get_loc(i), window_data.columns.get_loc(j)] = min_p_value
                    except Exception as e:
                        print(f"Error testing {i} and {j}: {e}")
                        continue
        return A_Mi

    def fuse_graphs(self, A_Ma, A_Mi):
        num_features = A_Ma.shape[0]
        A_fused = np.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    p_ma = A_Ma[i, j]
                    p_mi = A_Mi[i, j]
                    if p_ma > 0 and p_mi > 0:
                        z_ma = norm.ppf(1 - p_ma)
                        z_mi = norm.ppf(1 - p_mi)
                        rho = 0.5
                        Z = (z_ma + z_mi) / np.sqrt(2 + 2 * rho)
                        p_fused = 1 - norm.cdf(Z)
                        A_fused[i, j] = p_fused
                    else:
                        A_fused[i, j] = 1.0
        return A_fused

    def add_edges_based_on_fused_graph(self, A_fused):
        G = nx.DiGraph()
        for column in self.data.columns:
            G.add_node(column)
        p_values = []
        indices = []
        for i in range(A_fused.shape[0]):
            for j in range(A_fused.shape[1]):
                if i != j:
                    p_values.append(A_fused[i, j])
                    indices.append((i, j))
        p_values = np.array(p_values)
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=self.threshold_p_value, method='fdr_bh')
        for idx, (i, j) in enumerate(indices):
            p_corrected = corrected_p_values[idx]
            if p_corrected < self.threshold_p_value:
                if p_corrected <= 0:
                    p_corrected = 1e-10
                weight = -np.log(p_corrected)
                G.add_edge(self.data.columns[i], self.data.columns[j], weight=weight)
        return G
def construct_edge_index(G):
    edges = list(G.edges())

    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).T

    return edge_index

