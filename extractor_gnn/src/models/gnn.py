# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


def generate_batch_data(memory, src_index, tgt_index):
    edges = torch.stack([src_index, tgt_index], dim=1)
    data_list = [Data(x=x, edge_index=edge_index) for x, edge_index in zip(memory, edges)]
    batch_data = Batch.from_data_list(data_list)

    return batch_data.x.to(memory.device), batch_data.edge_index.to(memory.device)


class GNN(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.5):
        super(GNN, self).__init__()

        self.gnn1 = GATConv(hidden_size, hidden_size, num_heads, concat=False, dropout=dropout)
        self.gnn2 = GATConv(hidden_size, hidden_size, num_heads, concat=False, dropout=dropout)
        self.gnn3 = GATConv(hidden_size, hidden_size, num_heads, concat=False, dropout=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)
        self.activation_1 = torch.nn.ReLU()
        self.activation_2 = torch.nn.ReLU()

    def forward(self, node_embeddings, src_index, tgt_index):
        """

        Args:
            node_embeddings: (batch_size, num_nodes, hidden_size)
            src_index: (batch_size, num_edges)
            tgt_index: (batch_size, num_edges)

        Returns:
            output_embeddings: (batch_size, num_nodes, hidden_size)
        """

        batch_size, num_nodes, hidden_size = node_embeddings.shape

        memory, edge_index = generate_batch_data(node_embeddings, src_index, tgt_index)
        memory = self.gnn1(memory, edge_index)
        memory = self.activation_1(self.layer_norm_1(memory))
        memory = self.gnn2(memory, edge_index)
        memory = self.activation_2(self.layer_norm_2(memory))
        memory = self.gnn3(memory, edge_index)
        node_embeddings = memory.view([batch_size, num_nodes, hidden_size])

        return node_embeddings
