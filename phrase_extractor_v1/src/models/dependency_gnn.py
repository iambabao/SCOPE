# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/29
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


def generate_batch_data(memory, src_index, tgt_index):
    edges = torch.stack([src_index, tgt_index], dim=1)  # (batch_size, 2, num_edges)
    data_list = [Data(x=x, edge_index=edge_index) for x, edge_index in zip(memory, edges)]
    batch_data = Batch.from_data_list(data_list)

    return batch_data.x.to(memory.device), batch_data.edge_index.to(memory.device)


class DependencyGNN(nn.Module):
    def __init__(self, node_hidden_size, gnn_hidden_size, num_heads, dropout=0.1):
        super(DependencyGNN, self).__init__()

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gnn1 = GATConv(node_hidden_size, gnn_hidden_size, num_heads, dropout=dropout)
        self.gnn2 = GATConv(gnn_hidden_size * num_heads, gnn_hidden_size, num_heads, dropout=dropout)

    def forward(self, node_embeddings, src_index, tgt_index):
        """

        Args:
            node_embeddings: (batch_size, num_nodes, node_hidden_size)
            src_index: (batch_size, num_edges)
            tgt_index: (batch_size, num_edges)

        Returns:
            output_embeddings: (batch_size, num_nodes, gnn_hidden_size * num_heads)
        """

        batch_size = node_embeddings.shape[0]
        num_nodes = node_embeddings.shape[1]

        memory, edge_index = generate_batch_data(node_embeddings, src_index, tgt_index)
        memory = self.gnn1(memory, edge_index)
        memory = self.dropout(self.activation(memory))
        memory = self.gnn2(memory, edge_index)
        node_embeddings = memory.view([batch_size, num_nodes, -1])

        return node_embeddings
