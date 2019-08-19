import os

# 1 :  Brachial Plexus
# 2 :  Brain Stem
# 3 :  ConstrictorNaris
# 4 :  Ear-L
# 5 :  Ear-R
# 6 :  Eye-L
# 7 :  Eye-R
# 8 :  Hypophysis
# 9 :  Larynx
# 10 :  Lens L
# 11 :  Lens R
# 12 :  Mandible
# 13 :  Optical Chiasm
# 14 :  Optical Nerve L
# 15 :  Optical Nerve R
# 16 :  Oral Cavity
# 17 :  Parotid L
# 18 :  Parotid R
# 19 :  SmgL
# 20 :  SmgR
# 21 :  Spinal Cord
# 22 :  Sublingual Gland
# 23 :  Temporal Lobe L
# 24 :  Temporal Lobe R
# 25 :  Thyroid
# 26 :  TMJL
# 27 :  TMJR
# 28 :  Trachea


# Preprocessing using preserved HU in dilated part of mask
data_config = {
    'raw_dir':  None,
    'data_dir': '../data/raw',
    'preprocessed_data_dir': '../data/preprocessed',
    # 28 OAR names, names from the original dicom RT
    'roi_names': ['Brachial Plexus', 'Brain Stem', 'ConstrictorNaris', 'Ear-L', 'Ear-R', 'Eye-L', 'Eye-R',
        'Hypophysis', 'Larynx', 'Lens L', 'Lens R', 'Mandible', 'Optical Chiasm', 'Optical Nerve L',
        'Optical Nerve R', 'Oral Cavity', 'Parotid L', 'Parotid R', 'SmgL', 'SmgR', 'Spinal Cord',
        'Sublingual Gland', 'Temporal Lobe L', 'Temporal Lobe R', 'Thyroid', 'TMJL', 'TMJR', 'Trachea'],
    
    # name used for legend for the figures in the paper, for better consistency
    'paper_roi_names': ['Brachial Plexus', 'Brain Stem', 'Constrictor Naris', 'Ear L', 'Ear R', 'Eye L', 'Eye R',
        'Hypophysis', 'Larynx', 'Lens L', 'Lens R', 'Mandible', 'Optical Chiasm', 'Optical Nerve L',
        'Optical Nerve R', 'Oral Cavity', 'Parotid L', 'Parotid R', 'SMG L', 'SMG R', 'Spinal Cord',
        'Sublingual Gland', 'Temporal Lobe L', 'Temporal Lobe R', 'Thyroid', 'TMJ L', 'TMJ R', 'Trachea'],

    # data configuration
    # maximum z, x, y slices to load, in order to reduce data preparation time
    # These numbers are chosen according to the max_crop_size and jitter
    # since the max input would be centered at the image with size train_max_crop_size,
    # there is no need to load more than that.
    'num_slice': 180,
    'num_x': 272,
    'num_y': 272,

    # maximum input size to the network
    'train_max_crop_size': [160, 240, 240], 
    'bbox_border': 8,
    'pad_value': -1024,
    'jitter_range': [4, 16, 16],
    'stride': [16, 32, 32],
    'test_max_size': [256, 320, 320], 

    # whether to do affine and elastic transformation
    'do_elastic': True,
    'do_postprocess': False,
}


def get_anchors(bases, aspect_ratios):
    """
    Generate anchor according to the scale and aspect ratios

    bases: the scale for each anchor box
    aspect ratios: d:h:w for each anchor box
    """
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors


bases = [7, 15, 30, 50]
aspect_ratios = [[1, 2.5, 2.5], [1, 2.5, 5.], [1, 5., 2.5]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),

    # # of input channel, since it is CT image, we only have one channel
    'chanel': 1,

    # The feature map used for detection is a downsampled by stride
    'stride': 8,

    # The smallest feature map in the network is downsampled by max_stride
    'max_stride': 16,

    # Random sample num_neg negative samples for rpn proposals
    'num_neg': 80000,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.1,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.5,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.5,

    # detection network configuration
    # extra 1 for background
    'num_class': len(data_config['roi_names']) + 1,

    # ROI pooling size
    'rcnn_crop_size': (6, 6, 6),
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.2,
    'rcnn_train_batch_size': 128,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.5,
    'rcnn_test_nms_pre_score_threshold': 0.5,
    'rcnn_test_nms_overlap_threshold': 0.5,

    # controlling the strength of bounding box regression losses
    'box_reg_weight': [10., 10., 10., 5., 5., 5.]
}


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'UaNet',
    'batch_size': 1,

    'lr_schedule': lr_shedule,
    'optimizer': 'Adam',
    'momentum' : 0.9,
    'weight_decay': 1e-4,

    # total # of epochs
    'epochs': 200,

    # save check point (model weights) every epoch_save epochs
    'epoch_save': 1,

    # starting epoch_rcnn, add the rcnn branch for training
    'epoch_rcnn': 40,

    # starting epoch_mask, add the mask branch for training
    'epoch_mask': 45,

    # num_workers for data loader
    'num_workers': 4,

    # training data is the combination of dataset1, dataset2 training data (A total of 215)
    # Because of the patient privacy, the access to the training data in dataset 1 
    # will be granted on a case by case basis by submitting a request to the corresponding authors, 
    # subjecting to the review and approval by IRB.
    # 
    # validation data was extracted from the training data, around 10%
    # 
    # TODO:
    # You will have to generate your own training data somehow, in order to train the model
    # put all the training filenames into csv, and set the csv path here
    'train_set_name': 'split/dataset1_2_train.csv',
    'val_set_name': 'split/dataset1_2_val.csv',
    'test_set_name': 'split/release_dataset1_test.csv',
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3


train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'experiment_1')
train_config['initial_checkpoint'] = None # train_config['out_dir'] + '/model/xxx.ckpt'

config = dict(data_config, **net_config)
config = dict(config, **train_config)
