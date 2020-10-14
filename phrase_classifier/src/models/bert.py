# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/8/10
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.frozen_layers = kwargs.get('frozen_layers')

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        phrase_spans=None,
        num_phrases=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)

        phrase_embeddings = []
        for i in range(phrase_spans.shape[0]):
            phrase_embeddings.append(
                torch.stack([sequence_output[i][start:end].mean(dim=0) for start, end in phrase_spans[i]], dim=0)
            )
        phrase_embeddings = torch.stack(phrase_embeddings, dim=0)  # (batch_size, max_num_phrases, hidden_size)

        # (batch_size, max_num_phrases)
        logits = self.classifier(phrase_embeddings)  # (batch_size, max_num_phrases, 2)
        outputs = (logits,) + outputs

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            mask = torch.arange(labels.shape[1])[None, :].to(self.device) < num_phrases[:, None]
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) * mask.view(-1))
            loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
