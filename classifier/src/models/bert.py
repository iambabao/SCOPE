# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/1/18
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output, dim=1)  # (batch_size, hidden_size)

        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
