from __future__ import print_function

from functools import partial
from multiprocessing import Pool
import numpy as np
import pydicom
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.model import UaNet as Net
from dataset.brain_reader import BrainReader, train_collate, eval_collate, test_collate
from config import config
from utils.visualize import draw_gt, draw_pred, generate_image_anim, generate_image_pngs
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, crop_boxes2mask
from utils.preprocess import preprocess_image
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import zoom

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str,
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=None,
                    help="path to save the results")


# miccai 15 test spacings, used for computing 95% HD
miccai_spacings = {
    '0522c0661': [3.0, 0.9800000190734862, 0.9800000190734862], 
    '0522c0878': [3.0, 1.1699999570846558, 1.1699999570846558], 
    '0522c0708': [2.500629186630249, 1.2200000286102295, 1.2200000286102295], 
    '0522c0667': [2.4999997615814205, 1.1200000047683716, 1.1200000047683716], 
    '0522c0727': [3.0, 1.1699999570846558, 1.1699999570846558], 
    '0522c0669': [2.5, 1.1100000143051147, 1.1100000143051147], 
    '0522c0806': [3.0, 0.9800000190734862, 0.9800000190734862], 
    '0522c0746': [3.0, 0.9800000190734862, 0.9800000190734862], 
    '0522c0659': [2.4999997615814205, 1.1799999475479126, 1.1799999475479126], 
    '0522c0845': [2.5, 1.2699999809265137, 1.2699999809265137], 
    '0522c0857': [3.0, 1.1699999570846558, 1.1699999570846558], 
    '0522c0555': [2.0, 0.9800000190734862, 0.9800000190734862], 
    '0522c0598': [3.0, 1.1699999570846558, 1.1699999570846558], 
    '0522c0788': [3.0, 1.1699999570846558, 1.1699999570846558], 
    '0522c0576': [2.4999992847442627, 0.9800000190734862, 0.9800000190734862]
}


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'test':
        if args.weight == None:
            logging.error('weight must be specified if using test mode')
            return -1
        if args.dicom_path == None:
            logging.error('dicom file path must be specified if using test mode')
            return -1
        if args.out_dir == None:
            logging.error('output directory must be specified if using test mode')
            return -1

        test_single(args.weight, args.dicom_path, args.out_dir,
            config['do_postprocess'], None)
    elif args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        test_set_name = config['test_set_name']
        num_workers = 0
        initial_checkpoint = args.weight

        net = Net(config).cuda()


        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            out_dir = 'output'
            epoch = 'random_epoch'
            
            # Load pre-trained model weights
            state_dict = net.state_dict()
            state_dict.update({k: v for k, v in checkpoint['state_dict'].items() if k in state_dict})
            net.load_state_dict(state_dict)
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'videos')):
            os.makedirs(os.path.join(save_dir, 'videos'))

        dataset = BrainReader(data_dir, test_set_name, mode='eval')
        eval(net, dataset, save_dir, config['do_postprocess'], [])
    else:
        logging.error('Mode %s is not supported' % (args.mode))
        

def predict(net, input):
    '''
    Given trained net and torch input, return numpy mask prediction
    '''
    net.set_mode('eval')
    net.use_rcnn = True
    net.use_mask = True

    with torch.no_grad():
        net.forward(input, None, None, None, None)

    crop_boxes = net.crop_boxes
    segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]

    pred_mask = crop_boxes2mask(crop_boxes[:, 1:], segments, input.shape[2:])
    pred_mask = pred_mask.astype(np.uint8)

    return pred_mask


def test_single(weight_path, file_path, save_dir, do_postprocess=False, anchor_params=None):
    pid = file_path.split('/')[-1]

    logging.info('Starting processing, dicom file path %s, output dir %s' % (file_path, save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load image data
    standard_spacing = np.array([2.5, 0.976562, 0.976562])
    raw_img, origin, spacing = load_dicom_image(file_path)

    logging.info('Spacing %s', str(spacing))
    logging.info('Shape %s', str(raw_img.shape))

    dicom_file_single_path = None
    for dirpath, dirnames, filenames in os.walk(file_path):
        dicom_file_single_path = os.path.join(dirpath, filenames[0])
    ds = pydicom.dcmread(dicom_file_single_path, force=True)


    # Preprocess raw image
    logging.info('Preprocessing...')
    processed_image = raw_img.copy()
    npy_mask = preprocess_image(sitk.GetImageFromArray(raw_img))
    processed_image[npy_mask == 0] = -1024

    # Crop only brain region to reduce image size
    # Only choose certain number of slices
    D_raw, H_raw, W_raw = raw_img.shape
    z_start = max(0, D_raw - config['test_max_size'][0])
    z_end = D_raw
    _, yy, xx = np.where(processed_image > -1024)
    y_start = yy.min()
    y_end = yy.max()
    x_start = xx.min()
    x_end = xx.max()

    # Ensure only certain XY resolution
    y_center, x_center = H_raw // 2, W_raw // 2
    if y_end - y_start > config['test_max_size'][1]:
        y_start = max(0, y_center - config['test_max_size'][1] // 2)
        y_end = min(H_raw, y_center + config['test_max_size'][1] // 2)

    if x_end - x_start > config['test_max_size'][2]:
        x_start = max(0, x_center - config['test_max_size'][2] // 2)
        x_end = min(W_raw, x_center + config['test_max_size'][2] // 2)

    processed_image = processed_image[z_start:z_end, y_start:y_end, x_start:x_end]
    logging.info('Image input size: %s' % (str(processed_image.shape)))

    # Data to torch tensor
    D, H, W = processed_image.shape
    input = processed_image.copy()
    input = pad2factor(input)
    input = input[np.newaxis, ...].astype(np.float32)
    input = normalize(input)
    input = torch.from_numpy(input).float()
    input = input.unsqueeze(0).cuda()

    # Load net
    logging.info('Predicting...')
    net = Net(config).cuda()
    if not weight_path:
        logging.error('No model weight file specified')
        return

    if os.path.isfile(weight_path):
        # Single model prediction
        logging.info('Single model prediction mode')
        logging.info('Loading model from %s' % weight_path)
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['state_dict'])

        pred_mask = predict(net, input)
    else:
        # Multiple model predictions, using majority ensemble
        logging.info('Multi models prediction mode')
        input_d, input_h, input_w = input.shape[2:]
        pred_mask = np.zeros((len(config['roi_names']), 
                               input_d, input_h, input_w))
        ckpts = os.listdir(weight_path)
        n_consensus = (len(ckpts) + 1) // 2
        logging.info('Number of consensus %d' % (n_consensus))

        for ckpt in ckpts:
            one_weight_path = os.path.join(weight_path, ckpt)
            logging.info('Loading model from %s' % one_weight_path)
            checkpoint = torch.load(one_weight_path)
            net.load_state_dict(checkpoint['state_dict'])

            pred_mask += predict(net, input)

        pred_mask = (pred_mask >= n_consensus).astype(np.uint8)

    pred_mask = pred_mask[:, :D, :H, :W]

    # Post process
    if do_postprocess:
        raise NotImplementedError
    
    # Convert back into raw image size
    raw_pred_mask = np.zeros((len(config['roi_names']), D_raw, H_raw, W_raw), dtype=np.uint8)
    raw_pred_mask[:, z_start:z_end, y_start:y_end, x_start:x_end] = pred_mask

    normalized_img = (normalize(raw_img) + 1) / 2
    raw_pred_contours = merge_contours(get_contours_from_masks(raw_pred_mask))
    pred_img = draw_pred(normalized_img, raw_pred_contours)

    np.save(os.path.join(save_dir, '%s_raw_mask.npy' % (pid)), raw_pred_mask)
    np.save(os.path.join(save_dir, '%s_raw_contours.npy' % (pid)), raw_pred_contours)

    png_save_dir = os.path.join(save_dir, '%s' % (pid))
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    logging.info('Saving predicted pngs to %s' % (png_save_dir))
    generate_image_pngs(raw_img, raw_pred_mask, png_save_dir)
    # generate_image_anim(pred_img, save_path=os.path.join(save_dir, '%s.mp4' % (pid)))

    logging.info('Finished')


def eval(net, dataset, save_dir=None, do_postprocess=False, anchor_params=None):
    net.set_mode('eval')
    net.use_mask = True
    net.use_rcnn = True
    avg_dice = []
    avg_HD95 = []
    raw_dir = config['raw_dir']
    data_dir = config['data_dir']
    preprocessed_dir = config['preprocessed_data_dir']

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels, truth_masks, mask, image) in enumerate(dataset):
        try:
            D, H, W = image.shape
            pid = dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid), image.shape)
            gt_mask = mask.astype(np.uint8)

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                net.forward(input, truth_bboxes, truth_labels, truth_masks, mask)

                crop_boxes = net.crop_boxes
                segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]

            pred_mask = crop_boxes2mask(crop_boxes[:, 1:], segments, input.shape[2:])
            pred_mask = pred_mask[:, :D, :H, :W]
            pred_mask = pred_mask.astype(np.uint8)
            np.save(os.path.join(save_dir, '%s.npy' % (pid)), pred_mask)

            if do_postprocess:
                raise NotImplementedError

            # Read raw image, the one before being preprocessed and crop start and end
            # coordiantes
            # spacing = miccai_spacings[dataset.filenames[i]]
            dicom_img, header = nrrd.read(os.path.join(data_dir, pid, 'img.nrrd'))
            # from [x, y, z] to [z, y, x]
            spacing = header['spacings'][::-1] 
            print('spacing ', spacing)
            # raw_img, _ = nrrd.read(os.path.join(data_dir, pid, 'img.nrrd'))
            # raw_img = np.swapaxes(raw_img, 0, -1)
            # normalized_img = (normalize(raw_img) + 1) / 2
            # D_raw, H_raw, W_raw = raw_img.shape

            # start, end = np.load(os.path.join(preprocessed_dir, '%s_bbox.npy' % (pid)))
            # raw_gt_mask = np.zeros((len(config['roi_names']), D_raw, H_raw, W_raw), dtype=np.uint8)
            # raw_gt_mask[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = gt_mask
            # raw_pred_mask = np.zeros((len(config['roi_names']), D_raw, H_raw, W_raw), dtype=np.uint8)
            # raw_pred_mask[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = pred_mask

            # # Generate comparison image
            # gt_contours = merge_contours(get_contours_from_masks(raw_gt_mask))
            # gt_img = draw_gt(normalized_img, gt_contours)
            # pred_contours = merge_contours(get_contours_from_masks(raw_pred_mask))
            # pred_img = draw_pred(normalized_img, pred_contours)
            # full = np.concatenate((gt_img, pred_img), 2)

            # np.save(os.path.join(save_dir, '%s_raw_mask.npy' % (pid)), raw_pred_mask)
            # np.save(os.path.join(save_dir, '%s_raw_contour.npy' % (pid)), get_contours_from_masks(raw_pred_mask))
            # generate_image_anim(full, save_path=os.path.join(save_dir, 'videos', '%s.mp4' % (pid)))

            # Generate and print dice score for each class
            HD95 = hausdorff_distance(get_contours_from_masks(pred_mask), get_contours_from_masks(gt_mask), spacing=spacing, percent=0.95)
            # HD95 = [None] * len(config['roi_names'])
            score = dice_score_seperate(pred_mask, gt_mask)
            avg_HD95.append(np.array(HD95))
            avg_dice.append(np.array(score))

            for i in range(0, len(score)):
                print(config['roi_names'][i], ' DC', score[i], ', 95% HD', HD95[i])
            print()

            # Clear gpu memory
            del input, truth_bboxes, truth_labels, truth_masks, mask, image, pred_mask#, gt_mask, gt_img, pred_img, full, score
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels, truth_masks, mask, image,
            torch.cuda.empty_cache()
            traceback.print_exc()

            print()
            return

    print('====================')
    print('Average dice:')

    avg_dice = np.array(avg_dice)
    avg_HD95 = np.array(avg_HD95)
    for i in range(0, len(avg_dice[0])):
        s_region = avg_dice[:, i]
        s_region = s_region[s_region != None]

        HD_region = avg_HD95[:, i]
        HD_region = HD_region[HD_region != None]

        if len(s_region) and len(HD_region):
            print(config['paper_roi_names'][i], ',', round(np.mean(s_region), 4) * 100, ',', round(np.std(s_region), 4) * 100, ',', \
                    round(np.mean(HD_region), 4), ',', round(np.std(HD_region), 4))
        else:
            print(config['paper_roi_names'][i], ', None, None, None, None')
        
    # print('avg, DC ', round(np.mean(avg_dice[avg_dice != None]), 4) * 100)
    print()


def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]

    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks


if __name__ == '__main__':
    main()
