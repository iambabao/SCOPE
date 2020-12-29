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

from .utils import ce_loss, pu_loss


class BertExtractor(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.loss_type = kwargs.get('loss_type')
        self.prior_token = kwargs.get('prior_token')
        self.prior_phrase = kwargs.get('prior_phrase')

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.start_layer = nn.Sequential(
            self.dense,
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.end_layer = nn.Sequential(
            self.dense,
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.phrase_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
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
            start_labels=None,
            end_labels=None,
            phrase_labels=None,
    ):
        max_seq_length = input_ids.shape[1]
        # mask for real tokens
        phrase_mask = torch.logical_and(
            attention_mask.unsqueeze(-1).expand(-1, -1, max_seq_length),
            attention_mask.unsqueeze(-2).expand(-1, max_seq_length, -1)
        ).triu()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        start_logits = self.start_layer(sequence_output)  # (batch_size, max_seq_length, 2)
        end_logits = self.end_layer(sequence_output)  # (batch_size, max_seq_length, 2)
        outputs = (attention_mask.unsqueeze(-1) * start_logits, attention_mask.unsqueeze(-1) * end_logits) + outputs

        start_expanded = sequence_output.unsqueeze(2).expand(-1, -1, max_seq_length, -1)
        end_expanded = sequence_output.unsqueeze(1).expand(-1, max_seq_length, -1, -1)
        phrase_matrix = torch.cat([start_expanded, end_expanded], dim=-1)
        phrase_logits = self.phrase_layer(phrase_matrix)  # (batch_size, max_seq_length, max_seq_length, 2)
        outputs = (phrase_mask.unsqueeze(-1) * phrase_logits,) + outputs

        if None not in [start_labels, end_labels, phrase_labels]:
            phrase_labels = phrase_labels.reshape(-1, max_seq_length, max_seq_length)
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
                    start_labels.unsqueeze(-1).expand(-1, -1, max_seq_length),
                    end_labels.unsqueeze(-2).expand(-1, max_seq_length, -1)
                )  # mask for golden start and end tokens
                predicted_mask = torch.logical_and(
                    (torch.argmax(start_logits, dim=-1) == 1).unsqueeze(-1).expand(-1, -1, max_seq_length),
                    (torch.argmax(end_logits, dim=-1) == 1).unsqueeze(-2).expand(-1, max_seq_length, -1)
                )  # mask for predicted start and end tokens
                phrase_mask &= (golden_mask | predicted_mask)  # maks for all positive tokens
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
