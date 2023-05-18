import glob
import time
import cv2
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf

data_dir = ".\\check\\"

all_images = glob.glob(data_dir)
images_batch = all_images

low_resolution_shape = (32, 32)
high_resolution_shape = (128, 128)

i = 1

for i in range(1, 3):
    img = "D:\\Documents\\UiT Internship\\Task 2\\my-code-ESRGAN-TensorFlow_tf2\\esrgan-tf2-master\\data\\check\\" + str(i) + ".png"
    img1 = cv2.imread(img, 1)
    print(img)
    #img1 = img1.astype(np.float32)
    img1_high_resolution = cv2.resize(img1, (128, 128))
    img1_low_resolution = cv2.resize(img1, (32, 32))
    filename_low = str(i) + "low.png"
    filename_high = str(i) + "high.png"
    cv2.imwrite(filename_low, img1_low_resolution)
    cv2.imwrite(filename_high, img1_high_resolution)
    i = i+1