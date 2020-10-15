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


class RobertaPUClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.prior = kwargs.get('prior')
        self.gamma = kwargs.get('gamma')

        self.roberta = RobertaModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_labels),
            nn.Softmax(dim=-1),
        )

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
            u_risk = self.loss_fct(0, logits, labels == 0)
            p_risk = self.loss_fct(1, logits, labels == 1)
            n_risk = u_risk - self.prior * self.loss_fct(0, logits, labels == 1)
            loss = self.gamma * self.prior * p_risk + n_risk
            if n_risk < 0:
                loss = -n_risk

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...

    def loss_fct(self, label, logits, phrase_mask):
        label_mask = torch.eye(2, dtype=torch.float32)[label][None, None, :].to(self.device)  # (1, 1, 2)
        loss = torch.sum(torch.sum(label_mask * (1 - logits), dim=-1) * phrase_mask) / torch.sum(phrase_mask)
        return loss
