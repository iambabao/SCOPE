# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/12/29
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from .dependency_gnn import DependencyGNN
from .utils import ce_loss, pu_loss


class BertGNNExtractor(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.gnn_hidden_size = kwargs.get('gnn_hidden_size', 256)
        self.num_gnn_heads = kwargs.get('num_gnn_heads', 2)
        self.num_labels = config.num_labels
        self.loss_type = kwargs.get('loss_type')
        self.prior_token = kwargs.get('prior_token')
        self.prior_phrase = kwargs.get('prior_phrase')

        self.bert = BertModel(config)
        self.dependency_gnn = DependencyGNN(
            config.hidden_size,
            self.gnn_hidden_size,
            self.num_gnn_heads,
            dropout=config.hidden_dropout_prob,
        )
        self.dense = nn.Linear(self.gnn_hidden_size * self.num_gnn_heads, self.gnn_hidden_size)
        self.start_layer = nn.Sequential(
            self.dense,
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.gnn_hidden_size, config.num_labels),
        )
        self.end_layer = nn.Sequential(
            self.dense,
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.gnn_hidden_size, config.num_labels),
        )
        self.phrase_layer = nn.Sequential(
            nn.Linear(2 * self.gnn_hidden_size * self.num_gnn_heads, self.gnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.gnn_hidden_size, config.num_labels),
        )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            token_spans=None,
            src_index=None,
            tgt_index=None,
            num_tokens=None,
            start_labels=None,
            end_labels=None,
            phrase_labels=None,
    ):
        batch_size = input_ids.shape[0]
        max_num_tokens = token_spans.shape[1]
        # mask for real tokens with shape (batch_size, max_num_tokens)
        token_mask = torch.arange(max_num_tokens).expand(len(num_tokens), max_num_tokens).to(self.device) < num_tokens.unsqueeze(1)
        # mask for valid phrases with shape (batch_size, max_num_tokens, max_num_tokens)
        phrase_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_num_tokens),
            token_mask.unsqueeze(-2).expand(-1, max_num_tokens, -1),
        ).triu()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        token_embeddings = []
        for i in range(batch_size):
            token_embeddings.append(
                torch.stack([torch.mean(sequence_output[i][start:end], dim=0) for start, end in token_spans[i]], dim=0)
            )
        token_embeddings = torch.stack(token_embeddings, dim=0)  # (batch_size, max_num_tokens, hidden_size)

        # (batch_size, max_num_tokens, gnn_hidden_size * num_gnn_heads)
        token_embeddings = self.dependency_gnn(token_embeddings, src_index, tgt_index)

        start_logits = self.start_layer(token_embeddings)  # (batch_size, max_num_tokens, 2)
        end_logits = self.end_layer(token_embeddings)  # (batch_size, max_num_tokens, 2)
        outputs = (token_mask.unsqueeze(-1) * start_logits, token_mask.unsqueeze(-1) * end_logits) + outputs

        start_extended = token_embeddings.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
        end_extended = token_embeddings.unsqueeze(1).expand(-1, max_num_tokens, -1, -1)
        phrase_matrix = torch.cat([start_extended, end_extended], dim=-1)
        phrase_logits = self.phrase_layer(phrase_matrix)  # (batch_size, max_num_tokens, max_num_tokens, 2)
        outputs = (phrase_mask.unsqueeze(-1) * phrase_logits,) + outputs

        if None not in [start_labels, end_labels, phrase_labels]:
            phrase_labels = phrase_labels.reshape(-1, max_num_tokens, max_num_tokens)
            if self.loss_type == 'ce':
                start_loss = ce_loss(start_logits, start_labels, self.num_labels, attention_mask == 1)
                end_loss = ce_loss(end_logits, end_labels, self.num_labels, attention_mask == 1)
                phrase_loss = ce_loss(phrase_logits, phrase_labels, self.num_labels, phrase_mask)
            elif self.loss_type == 'pu':
                p_risk = pu_loss(start_logits, 1, attention_mask & (start_labels == 1))
                u_risk = pu_loss(start_logits, 0, attention_mask & (start_labels == 0))
                n_risk = u_risk - self.prior_token * pu_loss(start_logits, 0, attention_mask & (start_labels == 1))
                if n_risk >= 0: start_loss = self.prior_token * p_risk + n_risk
                else: start_loss = -n_risk

                p_risk = pu_loss(end_logits, 1, attention_mask & (end_labels == 1))
                u_risk = pu_loss(end_logits, 0, attention_mask & (end_labels == 0))
                n_risk = u_risk - self.prior_token * pu_loss(end_logits, 0, attention_mask & (end_labels == 1))
                if n_risk >= 0: end_loss = self.prior_token * p_risk + n_risk
                else: end_loss = -n_risk

                golden_mask = torch.logical_and(
                    start_labels.unsqueeze(-1).expand(-1, -1, max_num_tokens),
                    end_labels.unsqueeze(-2).expand(-1, max_num_tokens, -1)
                )  # mask for golden start and end tokens
                predicted_mask = torch.logical_and(
                    (torch.argmax(start_logits, dim=-1) == 1).unsqueeze(-1).expand(-1, -1, max_num_tokens),
                    (torch.argmax(end_logits, dim=-1) == 1).unsqueeze(-2).expand(-1, max_num_tokens, -1)
                )  # mask for predicted start and end tokens
                phrase_mask &= (golden_mask | predicted_mask)  # maks for valid phrases
                p_risk = pu_loss(phrase_logits, 1, phrase_mask & (phrase_labels == 1))
                u_risk = pu_loss(phrase_logits, 0, phrase_mask & (phrase_labels == 0))
                n_risk = u_risk - self.prior_phrase * pu_loss(phrase_logits, 0, phrase_mask & (phrase_labels == 1))
                if n_risk >= 0: phrase_loss = self.prior_phrase * p_risk + n_risk
                else: phrase_loss = -n_risk
            else:
                raise ValueError('{} is not supported for loss.'.format(self.loss_type))

            loss = start_loss + end_loss + phrase_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
