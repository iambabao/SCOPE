# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/12
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


def generate_batch_data(memory, seq_length):
    src_index, tgt_index = [], []
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            src = i * seq_length + j

            # edges to siblings
            ii, jj = i - 1, j - 1
            if 0 < ii < seq_length and 0 < jj < seq_length:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)
            ii, jj = i + 1, j + 1
            if 0 < ii < seq_length and 0 < jj < seq_length:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)

            # edges to first parent
            ii, jj = i, j + 1
            if 0 < ii < seq_length and 0 < jj < seq_length:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)

            # edges to second parent
            ii, jj = i - 1, j
            if 0 < ii < seq_length and 0 < jj < seq_length:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)
    edge_index = torch.tensor([src_index, tgt_index], dtype=torch.long)  # (2, num_edges)

    data_list = [Data(x=x, edge_index=edge_index) for x in memory]
    batch_data = Batch.from_data_list(data_list)

    return batch_data.x.to(memory.device), batch_data.edge_index.to(memory.device)


class PyramidGNN(nn.Module):
    def __init__(self, node_hidden_size, gnn_hidden_size, num_heads, dropout=0.1):
        super(PyramidGNN, self).__init__()

        self.gnn1 = GATConv(node_hidden_size, gnn_hidden_size, num_heads, concat=False, dropout=dropout)
        self.gnn2 = GATConv(gnn_hidden_size, gnn_hidden_size, num_heads, concat=False, dropout=dropout)

    def forward(self, node_embeddings):
        """

        Args:
            node_embeddings: (batch_size, seq_length, seq_length, node_hidden_size)

        Returns:
            output_embeddings: (batch_size, seq_length, seq_length, gnn_hidden_size)
        """

        batch_size = node_embeddings.shape[0]
        seq_length = node_embeddings.shape[1]

        memory = node_embeddings.view([batch_size, seq_length * seq_length, -1])
        memory, edge_index = generate_batch_data(memory, seq_length)
        memory = self.gnn1(memory, edge_index)
        memory = self.gnn2(memory, edge_index)
        node_embeddings = memory.view([batch_size, seq_length, seq_length, -1])

        return node_embeddings
