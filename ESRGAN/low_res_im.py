from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

result_path = './data/DIV2K/train_LR_c/'
path = './data/DIV2K/train_HR/'
for img_name in os.listdir(path):
    raw_img = cv2.imread(os.path.join(path, img_name))
    lr_img, hr_img = create_lr_hr_pair(raw_img, 4)
    #bic_img = imresize_np(lr_img, 4).astype(np.uint8)
    bic_img = lr_img
    results_img = bic_img
    result_img_path = result_path + img_name
    cv2.imwrite(result_img_path, results_img)
    print ('saved ' + img_name)

print ('Done!')