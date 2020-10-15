# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/9
"""

import torch
import torch.nn as nn


def ce_loss(logits, labels, num_labels, mask=None):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    loss = loss_function(logits.view(-1, num_labels), labels.view(-1))
    if mask is not None:
        loss = torch.sum(loss * mask.view(-1), dim=-1) / torch.sum(mask)
    else:
        loss = torch.mean(loss)
    return loss


def bce_loss(logits, labels, mask=None):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(logits.view(-1), labels.view(-1).type(torch.float))
    if mask is not None:
        loss = torch.sum(loss * mask.view(-1), dim=-1) / torch.sum(mask)
    else:
        loss = torch.mean(loss)
    return loss


def pu_loss(logits, golden, mask=None):
    label_mask = torch.eye(2, dtype=torch.float)[golden].to(logits.device)
    for _ in range(logits.dim() - 1):
        label_mask = label_mask.unsqueeze(0)
    logits = torch.softmax(logits, dim=-1)
    if mask is not None:
        loss = torch.sum(torch.sum(label_mask * (1 - logits), dim=-1) * mask) / torch.sum(mask)
    else:
        loss = torch.mean(torch.sum(label_mask * (1 - logits), dim=-1))
    return loss
