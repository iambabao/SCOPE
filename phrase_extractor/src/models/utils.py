# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/11/10
"""

import torch
import torch.nn as nn


def bce_loss(logits, labels, mask=None):
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_function(logits.view(-1), labels.view(-1).type(torch.float))
    if mask is not None:
        loss = torch.sum(mask.view(-1) * loss) / torch.sum(mask)
    else:
        loss = torch.mean(loss)
    return loss


def ce_loss(logits, labels, num_labels, mask=None):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    loss = loss_function(logits.view(-1, num_labels), labels.view(-1))
    if mask is not None:
        loss = torch.sum(mask.view(-1) * loss) / torch.sum(mask)
    else:
        loss = torch.mean(loss)
    return loss


def pu_loss(logits, golden, mask=None):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    logits = logits.view(-1, 2)
    labels = golden * torch.ones([logits.shape[0]], dtype=torch.long).to(logits.device)
    loss = loss_function(logits, labels)
    if mask is not None:
        if torch.sum(mask) != 0:
            loss = torch.sum(mask.view(-1) * loss) / torch.sum(mask)
        else:
            loss = 0
    else:
        loss = torch.mean(loss)
    return loss
