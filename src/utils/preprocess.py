import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import os
import SimpleITK as sitk
import nrrd
from config import data_config
from multiprocessing import Pool
from utils.util import annotation2multi_mask, annotation2masks


roi_names = data_config['roi_names']
data_dir = data_config['data_dir']
save_dir = data_config['preprocessed_data_dir']
z_starts = None

def morphology_process(itk_img, radius=7):
    """
    First use threshold to get rough brain region, then 
    use morphology closing and opening to remove region outside the brain
    """
    connected_img = 1 - sitk.OtsuThreshold(itk_img)
    closed_img = sitk.BinaryMorphologicalClosing(connected_img, radius)
    opened_img = sitk.BinaryMorphologicalOpening(closed_img, radius)
    
    H, W = sitk.GetArrayFromImage(itk_img).shape
    seed = [(H / 2, W / 2)]
    mask_img = sitk.ConnectedThreshold(opened_img, seedList=seed, lower=1)
    
    return mask_img

def preprocess_image(itk_img):
    """
    Preprocess itk image slice by slice
    """
    width, height, depth = itk_img.GetWidth(), itk_img.GetHeight(), itk_img.GetDepth()
    npy_mask = np.zeros((depth, height, width))
    for i in range(depth):
        npy_mask[i, :, :] = sitk.GetArrayFromImage(morphology_process(itk_img[:, :, i]))
        
    return npy_mask


def main():
    pids = os.listdir(data_dir)
    
    pool = Pool(processes=10)
    pool.map(preprocess, pids)
    
    pool.close()
    pool.join()
    
def preprocess(params):
    pid = params
    image, meta = nrrd.read(os.path.join(data_dir, pid, 'img.nrrd'))
    image = np.swapaxes(image, 0, -1)
    processed_image = image.copy()

    if z_starts is not None:
        z_start = z_starts[pid]
    else:
        z_start = 0
    processed_image = processed_image[z_start:, :, :]

    # Get binary mask for brain region, remove human hair and other tissues
    npy_mask = preprocess_image(sitk.GetImageFromArray(processed_image))
    processed_image[npy_mask == 0] = -1024

    # Crop only brain region to reduce image size
    _, yy, xx = np.where(processed_image > -1024)
    y_start = yy.min()
    y_end = yy.max()
    x_start = xx.min()
    x_end = xx.max()
    processed_image = processed_image[:, y_start:y_end, x_start:x_end]

    bbox = np.array([[z_start, y_start, x_start], [z_start + image.shape[0], y_end, x_end]])
    np.save(os.path.join(save_dir, '%s_raw.npy' % (pid)), image)
    np.save(os.path.join(save_dir, '%s_bbox.npy' % (pid)), bbox)
    nrrd.write(os.path.join(save_dir, '%s_clean.nrrd' % (pid)), processed_image)
    print(pid, ' ', processed_image.shape)

    masks = {}
    for roi in roi_names:
        if os.path.isfile(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi))):
            mask, meta = nrrd.read(os.path.join(data_dir, pid, 'structures', '%s.nrrd' % (roi)))
            mask = np.swapaxes(mask, 0, -1)
            mask = mask[z_start:, y_start:y_end, x_start:x_end]
            mask = mask.astype(np.uint8)
            masks[roi] = mask
            nrrd.write(os.path.join(save_dir, '%s_%s.nrrd' % (pid, roi)), mask)

    masks = annotation2masks(masks).astype(np.uint8)
    nrrd.write(os.path.join(save_dir, '%s_masks.npy' % (pid)), masks)
    # np.save(os.path.join(save_dir, '%s_masks.npy' % (pid)), masks)
                

if __name__ == '__main__':
    main()
