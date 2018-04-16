import os
import random
import numpy as np
import struct
from PIL import Image
from PIL import ImageFilter
import pickle
import glob2
from matplotlib import pylab as plt



class DataSet:
    def __init__(self, search_path, is_train=True, char_dict=None):
        self.file_counter = 0
        self.is_train = is_train
        self.iter_index = 0
        self.files = glob2.glob(search_path)
        self.char_dict = char_dict
        self.use_filter = False
        self.use_rotation = True
        self.get_char_dict()
    
    def get_char_dict(self):
        self.num_dict = dict(enumerate(sorted(set([self.gen_tagcode(file) for file in self.files]))))
        self.char_dict = {val:key for key, val in self.num_dict.items()}

    @staticmethod
    def gen_tagcode(file_name):
        return file_name.split(os.sep)[-2]
    
    @staticmethod
    def read_image(filepath):
        image_array = plt.imread(filepath, -1).astype(np.uint8)
        image_array = 255 - image_array
        return image_array

    def load_next_file(self):
        filepath = np.random.choice(self.files, replace=True)
        image_array = self.read_image(filepath)
        tag = self.gen_tagcode(filepath)
        num = self.char_dict[tag]
        if self.use_filter:
            image_array = self.apply_filter(image_array)
        if self.use_rotation:
            image_array = self.rotate(image_array)
        return image_array, num
    
    def load_all(self, files):
        x = []
        y = []
        for filepath in files:
            image_array = self.read_image(filepath)
            tag = self.gen_tagcode(filepath)
            num = self.char_dict[tag]
            if self.use_filter:
                image_array = self.apply_filter(image_array)
            if self.use_rotation:
                image_array = self.rotate(image_array)
            image_array = np.expand_dims(image_array/255, axis=2)
            x.append(image_array)
            y.append(num)
        return np.array(x), np.array(y)
    
    def rotate(self, image):
        im = Image.fromarray(image)
        im = im.rotate(random.randint(-10, 10)) # rotate slightly and randomly
        im = im.resize([64, 64])
        new_image = np.asarray(im)
        return new_image

    def apply_filter(self,image):
        im = Image.fromarray(image)
        filters = [ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.EMBOSS]
        im = im.filter(random.choice(filters))
        im = im.resize([64, 64])
        new_image = np.asarray(im)
        print('after filtering shape', new_image.shape)
#         new_image = new_image.reshape(new_image.shape[0], new_image.shape[1], 1)
        return new_image

    def train_valid_split(self, train_ratio=0.7, seed=42):
        n_files = len(self.files)
        np.random.seed(seed=seed)
        train_set = np.random.choice(sorted(self.files), int(train_ratio * n_files))
        valid_set = list(set(self.files) - set(train_set))
        return train_set, valid_set


