from net.layer import *
try:
    from utils.pybox import torch_nms, torch_overlap
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap

from net.layer.util import box_transform, box_transform_inv, clip_boxes
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable


def rcnn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)


def rcnn_decode(window, delta, weight):
    return  box_transform_inv(window, delta, weight)


def rcnn_nms(cfg, mode, inputs, proposals, logits, deltas):
    """
    Given the output of for each rpn proposals, return detection results by 
    1) performing non maximum suppression, and 
    2) applying the regression terms to rpn proposals.
    cfg: cfg dict
    mode: which phase is current, one of ['train', 'eval', 'valid', 'test']
    inputs: original input to the network (input data x)
    proposals: a list of rpn proposals from frist stage
    logits_flat: classification score for each anchor box, flattened
    deltas_flat: regression terms for each anchor box, flattened

    return
    windows: list of rpn proposals, [b, prob, z, y, x, d, h, w]
    """

    if mode in ['train']:
        nms_pre_score_threshold = cfg['rcnn_train_nms_pre_score_threshold']
        nms_overlap_threshold = cfg['rcnn_train_nms_overlap_threshold']

    elif mode in ['valid', 'test', 'eval']:
        nms_pre_score_threshold = cfg['rcnn_test_nms_pre_score_threshold']
        nms_overlap_threshold = cfg['rcnn_test_nms_overlap_threshold']
    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)

    batch_size, _, depth, height, width = inputs.size()
    num_class = cfg['num_class']

    # Only one label for each rpn proposal
    probs = F.softmax(logits).cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
    proposals = proposals.cpu().data.numpy()

    detections = []
    keeps = []
    for b in range(batch_size):
        detection = [np.empty((0, 9), np.float32),]

        index = np.where(proposals[:, 0] == b)[0]
        if len(index) > 0:
            prob = probs[index]
            delta = deltas[index]
            proposal = proposals[index]
            cats = np.argmax(prob, 1)

            # skip background
            for j in range(1, num_class): 
                idx = np.where(cats == j)[0]
                if len(idx) > 0:
                    p = prob[idx, j].reshape(-1, 1)
                    d = delta[idx, j]
                    box = rcnn_decode(proposal[idx, 2:8], d, cfg['box_reg_weight'])
                    box = clip_boxes(box, inputs.shape[2:])

                    js = np.expand_dims(np.array([j] * len(p)), axis=-1)
                    output = np.concatenate((p, box, js), 1)

                    if len(output) > 0:
                        output = torch.from_numpy(output).float()
                        output, keep = torch_nms(output, nms_overlap_threshold)

                    num = len(output)

                    if num > 0:
                        det = np.zeros((num, 9),np.float32)
                        det[:, 0] = b
                        det[:, 1:] = output
                        detection.append(det)
                        keeps.extend(index[idx[keep.numpy()]].tolist())

        detection = np.vstack(detection)
        detections.append(detection)

    detections = Variable(torch.from_numpy(np.vstack(detections))).cuda()

    return detections, keeps

    
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
