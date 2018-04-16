import os
import numpy as np
import pandas as pd
import glob
import glob2
from matplotlib import pylab as plt
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage.filters.rank import median
from skimage.morphology import disk
import logging
from tqdm import tqdm

plt.rcParams['image.cmap'] = 'gray'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def resize(image_array, target_max=224):
    logging.info('image size before {}'.format(image_array.shape))
    shape = image_array.shape
    max_side = max(shape)
    ratio = target_max / max_side
    logging.info('resize ratio {}'.format(ratio))
    if ratio > 1:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA        
    resize_array = cv2.resize(image_array, (0, 0), fx=ratio, fy=ratio, interpolation=interpolation)
    logging.info('image size after {}'.format(resize_array.shape))
    return resize_array


def pad_to_square(image_array, bg_white=True):
    shape = image_array.shape
    assert len(shape) == 2
    max_side = max(shape)
    median_margin = np.median(list(image_array[:5, :].flatten()) + list(image_array[-5:, :].flatten()) + 
                            list(image_array[:, :5].flatten()) + list(image_array[:, -5:].flatten()))
    logging.info('median_margin {}'.format(median_margin))
    if median_margin < 1:
        canvas = np.zeros((max_side, max_side))
    else:
        canvas = np.ones((max_side, max_side))
    pad_size = (max_side - np.array(shape)) // 2
    canvas[pad_size[0]:(pad_size[0] + shape[0]), pad_size[1]:(pad_size[1] + shape[1])] = image_array
    if bg_white and median_margin < 1:
        canvas = 1 - canvas
    return canvas


def normalize(image_array):
    return image_array.astype(float) / 255


def preprocess_image(filepath):
    """Return preprocessed image from filepath
    """
    image_array = plt.imread(filepath, -1)
    normal_array = normalize(image_array) # to [0, 1]
    resize_array = resize(normal_array, target_max=224)
    blurred_array = gaussian(resize_array, 1)
    th = threshold_otsu(blurred_array)
    binary_array = (blurred_array > th).astype(np.uint8)
    square_array = pad_to_square(binary_array, bg_white=True)
    preproc_array = (square_array * 255).astype(np.uint8)
    return image_array, binary_array, preproc_array


def show_preprocess(image_array, binary_array, square_array):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image_array)
    plt.title('raw image')
    plt.subplot(132)
    plt.imshow(binary_array)
    plt.title('binary image')
    plt.subplot(133)
    plt.imshow(square_array)
    plt.title('preprocessed image')


if __name__ == '__main__':
    print('test preprocessing')
    filepath = '/Users/megatron/DL/train/和/4cb35c1fd4c9b2df39c31da1ffe1544d67c67d23.jpg'

    filepath = '/Users/megatron/DL/train/从/8a37d0a2ecd3ed6372f6cf45eea4c20e0405d118.jpg'
    image_array, binary_array, square_array = preprocess_image(filepath)
    show_preprocess(image_array, binary_array, square_array)
    plt.show()