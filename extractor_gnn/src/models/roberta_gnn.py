# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/5/1
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig

from .gnn import GNN
from .utils import ce_loss, pu_loss_with_ce as pu_loss


class RobertaGNNExtractor(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.loss_type = kwargs.get('loss_type', 'ce')
        self.prior = 0.0002 * kwargs.get('prior', 0.00)  # SQuAD: 0.00021
        # self.prior = 0.0001 * kwargs.get('prior', 0.00)  # DROP: 0.00010

        self.roberta = RobertaModel(config)
        self.pos_embedding = nn.Embedding(17, 128)
        self.dense = nn.Linear(config.hidden_size + 128, config.hidden_size)
        self.gnn = GNN(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.phrase_layer = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            token_mapping=None,
            pos_tag=None,
            num_tokens=None,
            src_index=None,
            tgt_index=None,
            phrase_labels=None,
    ):
        batch_size = input_ids.shape[0]
        max_num_tokens = input_ids.shape[1]  # equal to max_seq_length
        # mask for real tokens with shape (batch_size, max_num_tokens)
        token_mask = torch.arange(max_num_tokens).expand(batch_size, max_num_tokens).to(self.device) < num_tokens.unsqueeze(1)
        # mask for valid phrases with shape (batch_size, max_num_tokens, max_num_tokens)
        phrase_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_num_tokens),
            token_mask.unsqueeze(-2).expand(-1, max_num_tokens, -1),
        ).triu()

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]

        # reconstruct sequence_output to token_embeddings
        expanded_token_mapping = token_mapping.unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        zeros = torch.zeros([batch_size, max_num_tokens, self.config.hidden_size], dtype=torch.float).to(self.device)
        token_embeddings = torch.scatter_add(zeros, 1, expanded_token_mapping, sequence_output)
        zeros = torch.zeros([batch_size, max_num_tokens], dtype=torch.long).to(self.device)
        token_counter = torch.scatter_add(zeros, 1, token_mapping, torch.ones_like(token_mapping))
        token_counter = torch.max(token_counter, torch.ones_like(token_counter))
        token_embeddings = token_embeddings / token_counter.unsqueeze(-1)  # (batch_size, max_num_tokens, hidden_size)

        # add feature embedding
        pos_em = self.pos_embedding(pos_tag)
        token_embeddings = self.dense(torch.cat([token_embeddings, pos_em], dim=-1))

        # gnn
        node_embeddings = self.gnn(token_embeddings, src_index, tgt_index)
        token_embeddings = self.layer_norm(token_embeddings + node_embeddings)

        start_expanded = token_embeddings.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
        end_expanded = token_embeddings.unsqueeze(1).expand(-1, max_num_tokens, -1, -1)
        phrase_matrix = torch.cat([start_expanded, end_expanded], dim=-1)
        phrase_logits = self.phrase_layer(phrase_matrix)  # (batch_size, max_num_tokens, max_num_tokens, 2)
        outputs = (phrase_mask.unsqueeze(-1) * phrase_logits,) + outputs

        if phrase_labels is not None:
            phrase_labels = phrase_labels.reshape(-1, max_num_tokens, max_num_tokens)
            if self.loss_type == 'ce':
                loss = ce_loss(phrase_logits, phrase_labels, phrase_mask)
            elif self.loss_type == 'pu':
                loss = pu_loss(phrase_logits, phrase_labels, self.prior, phrase_mask)
            else:
                raise ValueError('{} is not supported for loss.'.format(self.loss_type))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
