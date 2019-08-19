import torch
import torch.nn.functional as F
from torch import nn


def weighted_focal_loss_with_logits(logits, labels, weights, gamma=2.):
    """
    weighted binary focal loss with raw score (before sigmoid)

    logits: raw score, vector
    labels: corresponding labels, vector
    weights: corresponding weights, vector
    gamma: hyper parameters for focal loss, by default 2
    """
    log_probs = F.logsigmoid(logits)
    probs = F.sigmoid(logits)

    pos_logprobs = log_probs[labels == 1]
    neg_logprobs = torch.log(torch.clamp(1 - probs[labels == 0], 1e-6, 1))
    pos_probs = probs[labels == 1]
    neg_probs = 1 - probs[labels == 0]
    pos_weights = weights[labels == 1]
    neg_weights = weights[labels == 0]

    pos_probs = pos_probs.detach()
    neg_probs = neg_probs.detach()

    pos_loss = - pos_logprobs * (1 - pos_probs) ** gamma
    neg_loss = - neg_logprobs * (1 - neg_probs) ** gamma
    loss = ((pos_loss * pos_weights).sum() + (neg_loss * neg_weights).sum()) / (weights.sum() + 1e-12)

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] < 0.5).sum()
    neg_total = (labels == 0).sum()

    return loss, pos_correct, pos_total, neg_correct, neg_total

    log_probs[labels == 0] = torch.log(1 - probs[labels == 0])
    probs[labels == 0] = 1 - probs[labels == 0]

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] > 0.5).sum()
    neg_total = (labels == 0).sum()

    return loss, pos_correct, pos_total, neg_correct, neg_total


def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train'):
    """
    Calculate the loss for rpn branch
    The loss consists of a binary classification loss and
    six bounding box regression losses

    logits: raw score for each anchor box
    deltas: regression terms for each anchor box
    labels: class label for each anchor box
    label_weights: weights for each anchor box
    targets: regression gt for each anchor box
    target_weights: regression gt weights for each anchor box, should not be used here
    cfg: config dict
    mode

    return binary classification loss, sum of six regression terms,
           [# true positives, # total positives, # true negatives, # total negatives,
            loss of dz, loss of dy, loss of dx, loss of dd, loss of dh, loss of dw]
    """
    batch_size, num_windows, num_classes = logits.size()
    batch_size_k = batch_size
    labels = labels.long()

    # classification loss
    pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
    batch_size = batch_size*num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
        weighted_focal_loss_with_logits(logits, labels, label_weights)

    # bounding box regression losses
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)

    index = (labels != 0).nonzero()[:, 0]
    deltas = deltas[index]
    targets = targets[index]

    rpn_reg_loss = 0
    reg_losses = []
    for i in range(6):
        l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
        rpn_reg_loss += l
        reg_losses.append(l.data.item())

    return rpn_cls_loss, rpn_reg_loss, [pos_correct, pos_total, neg_correct, neg_total,
                                        reg_losses[0], reg_losses[1], reg_losses[2],
                                        reg_losses[3], reg_losses[4], reg_losses[5]]
