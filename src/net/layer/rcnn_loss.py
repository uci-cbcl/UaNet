import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def rcnn_loss(logits, deltas, labels, targets):
    """
    Calculate the loss for rcnn branch
    The loss consists of a multi-class classification loss and
    six bounding box regression losses
    logits: raw score for each of the rpn proposals
    deltas: regression terms for each of the rpn proposals
    labels: class label of each of the rpn proposals
    targets: regression gt for each of the rpn proposals

    return binary classification loss, sum of six regression terms,
           [loss of dz, loss of dy, loss of dx, loss of dd, loss of dh, loss of dw, confusion matric]
    """
    batch_size, num_class   = logits.size(0),logits.size(1)

    # weighted cross entropy
    # calculate the weights for each class
    weight = torch.ones(num_class).cuda()
    total = len(labels)
    for i in range(num_class):
        num_pos = float((labels == i).sum())
        num_pos = max(num_pos, 1)
        weight[i] = total / num_pos

    # cross entropy loss
    # TODO: In case of missing annotations for an OAR, we should mask out the contribution of the loss
    # Otherwise, treating a "true positive" as negative may confuse the model and result false negative.
    weight = weight / weight.sum()
    rcnn_cls_loss = F.cross_entropy(logits, labels, weight=weight, size_average=True)

    # compute the confusion matric
    confusion_matrix = np.zeros((num_class, num_class))
    probs = F.softmax(logits, dim=1)
    v, cat = torch.max(probs, dim=1)
    for i in labels.nonzero():
        i = i.item()
        confusion_matrix[labels.long().detach()[i].item()][cat[i].detach().item()] += 1

    # compute the regression losses
    num_pos = len(labels.nonzero())

    if num_pos > 0:
        # one hot encode
        select = Variable(torch.zeros((batch_size, num_class))).cuda()
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0
        select = select.view(batch_size, num_class, 1).expand((batch_size, num_class, 6)).contiguous().byte()

        deltas = deltas.view(batch_size, num_class, 6)
        deltas = deltas[select].view(-1, 6)

        rcnn_reg_loss = 0
        reg_losses = []

        # sum of the six regression terms
        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
            rcnn_reg_loss += l
            reg_losses.append(l.data.item())
    else:
        rcnn_reg_loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()

    return rcnn_cls_loss, rcnn_reg_loss, [reg_losses[0], reg_losses[1], reg_losses[2],
                                        reg_losses[3], reg_losses[4], reg_losses[5], confusion_matrix]


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


 
