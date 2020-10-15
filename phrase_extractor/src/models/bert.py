# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/10/15
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from .utils import ce_loss, bce_loss, pu_loss


class BertExtractor(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.loss_type = kwargs.get('loss_type')
        self.prior = kwargs.get('prior')
        self.gamma = kwargs.get('gamma')
        self.frozen_layers = kwargs.get('frozen_layers')

        self.bert = BertModel(config)
        self.index_output_layer = nn.Linear(config.hidden_size, 2)
        self.phrase_output_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self.init_weights()

        if self.frozen_layers >= 0:
            no_gradient = ['bert.embeddings.']
            for i in range(self.frozen_layers):
                no_gradient.append('bert.encoder.layer.{}.'.format(i))
            for n, p in self.named_parameters():
                if any(nd in n for nd in no_gradient):
                    p.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_labels=None,
        end_labels=None,
        phrase_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        max_seq_length = sequence_output.shape[1]
        # mask for available tokens
        phrase_mask = torch.logical_and(
            attention_mask.unsqueeze(-1).expand(-1, -1, max_seq_length),
            attention_mask.unsqueeze(-2).expand(-1, max_seq_length, -1)
        ).triu()

        index_logits = self.index_output_layer(sequence_output)
        start_logits = index_logits[:, :, 0]  # (batch_size, max_seq_length)
        end_logits = index_logits[:, :, 1]  # (batch_size, max_seq_length)
        outputs = (attention_mask * start_logits, attention_mask * end_logits) + outputs

        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, max_seq_length, -1)
        end_extend = sequence_output.unsqueeze(1).expand(-1, max_seq_length, -1, -1)
        phrase_matrix = torch.cat([start_extend, end_extend], dim=-1)
        phrase_logits = self.phrase_output_layer(phrase_matrix)  # (batch_size, max_seq_length, max_seq_length, 2)
        outputs = (phrase_mask.unsqueeze(-1) * phrase_logits,) + outputs

        if None not in [start_labels, end_labels, phrase_labels]:
            start_loss = bce_loss(start_logits, start_labels, attention_mask == 1)
            end_loss = bce_loss(end_logits, end_labels, attention_mask == 1)

            if self.loss_type == 'ce-loss':
                phrase_loss = ce_loss(phrase_logits, phrase_labels, self.num_labels, phrase_mask)
            elif self.loss_type == 'masked-ce-loss':
                golden_mask = torch.logical_and(
                    start_labels.unsqueeze(-1).expand(-1, -1, max_seq_length),
                    end_labels.unsqueeze(-2).expand(-1, max_seq_length, -1)
                )  # mask for golden start and end tokens
                predicted_mask = torch.logical_and(
                    (start_logits > 0).unsqueeze(-1).expand(-1, -1, max_seq_length),
                    (end_logits > 0).unsqueeze(-2).expand(-1, max_seq_length, -1)
                )  # mask for predicted start and end tokens
                positive_mask = phrase_mask & (golden_mask | predicted_mask)  # maks for all available positive tokens
                phrase_loss = ce_loss(phrase_logits, phrase_labels, self.num_labels, positive_mask)
            elif self.loss_type == 'pu-loss':
                u_risk = pu_loss(phrase_logits, 0, phrase_mask & (phrase_labels == 0))
                p_risk = pu_loss(phrase_logits, 1, phrase_mask & (phrase_labels == 1))
                n_risk = u_risk - self.prior * pu_loss(phrase_logits, 0, phrase_mask & (phrase_labels == 1))
                if n_risk >= 0:
                    phrase_loss = self.gamma * self.prior * p_risk + n_risk
                else:
                    phrase_loss = -n_risk
            else:
                raise ValueError('{} is not supported'.format(self.loss_type))

            loss = start_loss + end_loss + phrase_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
