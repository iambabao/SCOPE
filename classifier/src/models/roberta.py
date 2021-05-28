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

from .utils import ce_loss


class RobertaClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mappings=None,
            num_phrases=None,
            labels=None,
    ):
        batch_size, max_num_phrases, _ = mappings.shape
        phrase_mask = torch.arange(max_num_phrases).expand(batch_size, max_num_phrases).to(self.device) < num_phrases.unsqueeze(1)

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        phrase_embeddings = torch.gather(
            sequence_output.unsqueeze(2).repeat(1, 1, max_num_phrases, 1), 1,
            mappings.unsqueeze(-1).repeat(1, 1, 1, self.config.hidden_size),
        )  # (batch_size, max_num_phrases, 2, hidden_size)
        phrase_embeddings = phrase_embeddings.reshape(batch_size, max_num_phrases, -1)

        logits = self.classifier(phrase_embeddings)
        outputs = (logits,) + outputs

        if labels is not None:
            loss = ce_loss(logits, labels, phrase_mask)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
