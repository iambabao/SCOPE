# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/9/28
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/9/28
"""

import torch
from torch import nn
from transformers import RobertaConfig, BertPreTrainedModel, RobertaModel

class RobertaClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        phrase_spans=None,
        num_phrases=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]

        phrase_embeddings = []
        for i in range(phrase_spans.shape[0]):
            phrase_embeddings.append(
                torch.stack([sequence_output[i][start:end].mean(dim=0) for start, end in phrase_spans[i]], dim=0)
            )
        phrase_embeddings = torch.stack(phrase_embeddings, dim=0)

        logits = self.classifier(phrase_embeddings)
        outputs = (logits,) + outputs

        if labels is not None:
            mask = torch.arange(labels.shape[1])[None, :].to(self.device) < num_phrases[:, None]
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) * mask.view(-1))
            loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
