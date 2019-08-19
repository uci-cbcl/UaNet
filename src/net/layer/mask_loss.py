from . import *
import torch
import numpy as np
import torch.nn.functional as F
from net.layer.rpn_loss import weighted_focal_loss_with_logits


def mask_loss(probs, targets):
    """
    Compute the loss for mask branch

    probs: list of mask predictions
    targets: list of mask targets
    """
    loss_func = torch.nn.BCEWithLogitsLoss()
    cnt = 0
    losses = torch.zeros((len(probs))).cuda()
    weight = torch.ones((len(probs))).cuda()

    for i in range(len(probs)):
        target = targets[i]
        prob = probs[i]

        prob = prob.view(-1)
        target = target.view(-1)

        # If this is a false positive, then do not count for loss        
        # Soft dice loss
        if (target == 1).sum():
            prob = F.sigmoid(prob)
            alpha = 0.5
            beta  = 0.5

            p0 = prob
            p1 = 1 - prob
            g0 = target
            g1 = 1 - target

            num = torch.sum(p0 * g0)
            den = num + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
            
            loss = 1 - num / (den + 1e-5)
            losses[i] = loss

    return ((losses) * weight).sum(), losses.detach().cpu().numpy()
