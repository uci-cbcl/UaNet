import sys

from .layer import *

from config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm


bn_momentum = 0.1
affine = True

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32))

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64))

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            ResBlock3d(192, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(128, out_channels),
            ResBlock3d(out_channels, out_channels),
            ResBlock3d(out_channels, out_channels))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.preBlock(x)
        out_pool = out
        out1 = self.forw1(out_pool)
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))

        return [x, out1, out2, comb3]


class RpnHead(nn.Module):
    """
    Region proposal network (RPN) head

    Applying two 1x1x1 3D convs to the feature map, to generate
    1. binary classification score for each anchor box at each sliding window position
    2. six regression terms for each anchor box at each sliding window positions
    """
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                  nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        
        return logits, deltas


class RcnnHead(nn.Module):
    """
    Regional convolution neural network (RCNN) head

    1. 3D ROI pooling (use Adaptive pooling from PyTorch).
       I am not sure whether they are exactly the same, but serve our purpose
    2. multi-class classification for each rpn proposals
    3. six regression terms for each rpn proposals
    """
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas

class MaskHead(nn.Module):
    """
    Mask head for the proposed network

    Only upsample the region that contains ROI, up to the original image scale
    """
    def __init__(self, cfg, in_channels=128):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, detections, features):
        img, f_2, f_4, f_8 = features  

        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)
        f_8 = f_8.squeeze(0)

        _, _, D, H, W = img.shape
        out = []

        for detection in detections:
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            up1 = self.up1(f_8[b, :, z_start / 8:z_end / 8, y_start / 8:y_end / 8, x_start / 8:x_end / 8].unsqueeze(0))
            up1 = self.back1(torch.cat((up1, f_4[b, :, z_start / 4:z_end / 4, y_start / 4:y_end / 4, x_start / 4:x_end / 4].unsqueeze(0)), 1))
            up2 = self.up2(up1)
            up2 = self.back2(torch.cat((up2, f_2[b, :, z_start / 2:z_end / 2, y_start / 2:y_end / 2, x_start / 2:x_end / 2].unsqueeze(0)), 1))
            up3 = self.up3(up2)
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)
            up3 = self.back3(torch.cat((up3, im), 1))

            # Get one of the head out of the 28, acoording to the predicted class \hat{c} (cat variable here)
            logits = getattr(self, 'logits' + str(int(cat)))(up3)
            logits = logits.squeeze()
#             logits = F.sigmoid(logits).squeeze()

            mask = Variable(torch.zeros((D, H, W))).cuda()
            mask[z_start:z_end, y_start:y_end, x_start:x_end] = logits
            mask = mask.unsqueeze(0)
            out.append(mask)

        out = torch.cat(out, 0)

        return out


def crop_mask_regions(masks, crop_boxes):
    """
    Crop the region bounded by crop_boxes, to save memory
    """
    out = []
    for i in range(len(crop_boxes)):
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
        m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
        out.append(m)
    
    return out


def top1pred(boxes):
    """
    Select the detection with highest probability for each class (OAR)
    """
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res


def random1pred(boxes):
    """
    Random select one detection for each class (OAR)
    """
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res


class CropRoi(nn.Module):
    """
    3D ROI pooling (use Adaptive pooling from PyTorch)
    TODO: A faster and more efficient implementation should be use C++

    The input is a lists of rpn proposal [b, z, y, x, d, h, w]
    The return is a list of pooled features of size rcnn_crop_size
    """
    def __init__(self, cfg, rcnn_crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']

    def forward(self, f, inputs, proposals):
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = int(p[0])
            center = p[2:5]
            side_length = p[5:8]

            # left bottom corner
            c0 = center - side_length / 2 
            # right upper corner
            c1 = c0 + side_length 

            # corresponding point on the downsampled feature map
            c0 = (c0 / self.scale).floor().long()
            c1 = (c1 / self.scale).ceil().long()
            minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            maximum = torch.LongTensor(
                np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

            # clip the boxes, to make sure (0, 0, 0) <= (z0, y0, x0) and (z1, y1, x1) < (D, H, W)
            c0 = torch.cat((c0.unsqueeze(0), minimum), 0)
            c1 = torch.cat((c1.unsqueeze(0), maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            # This should never happen
            if np.any((c1 - c0).cpu().data.numpy() < 1):
                print(p)
                print('c0:', c0, ', c1:', c1)
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class UaNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(UaNet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config, 1, 128)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=128)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.mask_head = MaskHead(config, in_channels=128)
        self.use_rcnn = False
        self.use_mask = False
                

    def forward(self, inputs, truth_boxes, truth_labels, truth_masks, masks):
        """
        Forward function for the network.
        I admit this is a bit strange: the forward takes in multiple arguments.
        As people might wonder, how to set the variables in test mode since no ground truth labels are available.
        So, in test mode, simply set truth_boxes, truth_labels, truth_masks, masks to None
        """

        # Feature extraction backbone
        features = data_parallel(self.feature_net, (inputs))

        # Get feature_map_8
        fs = features[-1]

        # RPN branch
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

        b, D, H, W, _, num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)

        # Generating anchor boxes
        self.rpn_window = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []

        # Only in evalutation mode, or in training mode and we need use rcnn branch,
        # we will perform nms to rpn results
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)

        # Generate the labels for each anchor box, and regression terms for positive anchor boxes
        # Generate the labels for each RPN proposal, and corresponding regression terms
        if self.mode in ['train', 'valid']:
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels )

            if self.use_rcnn:
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels, truth_masks)

        # RCNN branch
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.mask_probs = []
        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                rcnn_crops = self.rcnn_crop(fs, inputs, self.rpn_proposals)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals, 
                                                                        self.rcnn_logits, self.rcnn_deltas)

            # Mask branch
            if self.use_mask:
                # keep batch index, z, y, x, d, h, w, class
                self.crop_boxes = []
                if len(self.detections):
                    # [batch_id, z, y, x, d, h, w, class]
                    self.crop_boxes = self.detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()

                    # Use left bottom and right upper points to represent a bounding box
                    # [batch_id, z0, y0, x0, z1, y1, x1]
                    self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
                    self.crop_boxes = self.crop_boxes.astype(np.int32)

                    # Round the coordinates to the nearest multiple of 8
                    self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 8)

                    # Clip the coordinates, so the points fall within the size of the input data
                    # More specifically, make sure (0, 0, 0) <= (z0, y0, x0) and (z1, y1, x1) < (D, H, W) 
                    self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
                
                # In evaluation mode, we keep the detection with the highest probability for each OAR
                if self.mode in ['eval', 'test']:
                    self.crop_boxes = top1pred(self.crop_boxes)
                else:
                    # In training mode, we random select one detection for each OAR
                    self.crop_boxes = random1pred(self.crop_boxes)

                # Generate mask labels for each detection
                if self.mode in ['train', 'valid']:
                    self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
                        truth_boxes, truth_labels, masks)

                # Make sure to keep feature maps not splitted by data parallel
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))
                self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)

    def forward2(self, inputs, bboxes):
        """
        Test the segmentation accuracy with ground truth box as input
        """
        features = data_parallel(self.feature_net, (inputs)); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        self.crop_boxes = []
        for b in range(len(bboxes)):
            self.crop_boxes.append(np.column_stack((np.zeros((len(bboxes[b]) + b, 1)), bboxes[b])))

        self.crop_boxes = np.concatenate(self.crop_boxes, 0)
        self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
        self.crop_boxes = self.crop_boxes.astype(np.int32)
        self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 8)
        self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
#         self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
#             truth_boxes, truth_labels, masks)

        # Make sure to keep feature maps not splitted by data parallel
        features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
        self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))
        self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)

    def loss(self, targets=None):
        """
        Loss for the network
        """
        cfg  = self.cfg
    
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rcnn_stats = None
        mask_stats = None

        self.mask_loss = torch.zeros(1).cuda()
    
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)

        if self.use_mask:
            self.mask_loss, mask_losses = mask_loss(self.mask_probs, self.mask_targets)

            # Compute the loss for each OAR
            mask_stats = np.zeros(cfg['num_class'] - 1) - 1
            for i in range(len(self.crop_boxes)):
                cat = int(self.crop_boxes[i][-1]) - 1
                mask_stats[cat] = mask_losses[i]
            mask_stats[mask_stats == -1] = np.nan
    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss \
                          + self.mask_loss

    
        return self.total_loss, rpn_stats, rcnn_stats, mask_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()


if __name__ == '__main__':
    net = UaNet(config)

    input = torch.rand([4, 1, 128, 128, 128])
    input = Variable(input)

