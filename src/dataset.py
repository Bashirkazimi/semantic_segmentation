"""
Author: Bashir Kazimi
"""

from glob import glob
import os
from src import utils
import numpy as np
from keras.utils import Sequence


class Data(Sequence):
    """
    Data class
    """

    def __init__(self, data_dir, num_classes=5, wildcard=r"*.tif", image_size=128, num_channels=1,
                 batch_size=32, preprocess=False, label_dir=r'/media/kazimi/Data/data/bmbp_data'):
        """
        Initialize dataset. data_dir contains input images ending the wildcard extensions. label_dir contains labels
        for each image, where each file is named/contains the full name of the corresponding input image.
        :param data_dir: path to data directory, which contains x: input image folder, y:label folder
        :param num_classes: number of classes
        :param wildcard: wildcard to look for specific file patterns, e.g., r'*.tif'
        :param image_size: size of the images, e.g., 128 pixels
        :param num_channels: number of channels
        :param batch_size: batch size
        :param preprocess: preprocess input images if true
        :param valid: true if validation data
        :param label_dir: path to corresponding label directory
        """
        self.preprocess = preprocess
        self.num_channels = num_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_shape = (image_size, image_size, num_channels)

        # x,y pairs for training
        self.datafiles = glob(os.path.join(data_dir, wildcard))
        self.datalabels = []
        # look for patter in input image file name and retrieve the corresponding label file name!
        for f in self.datafiles:
            pth, fn = os.path.split(f)
            new_fn = fn.split('_')[0]+'.tif' if '_' in fn else fn
            self.datalabels.append(os.path.join(label_dir, new_fn))

        # self.datafiles = self.datafiles[:20]
        # self.datalabels = self.datalabels[:20]

        self.num_examples = len(self.datafiles)

        # this is used to shuffle the dataset at each epoch!
        self.ids = list(range(len(self.datafiles)))
        self.num_classes = num_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_examples / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        reads and returns a batch of image,label pairs.
        :param idx: index to start from
        :return: returns batch_size image label pairs starting from index idx.
        """
        batch = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        ret_x = np.empty((len(batch), self.image_size, self.image_size, self.num_channels))
        ret_y = np.empty((len(batch), self.image_size, self.image_size, self.num_channels))
        for i in range(len(batch)):
            x = utils.read_tif(self.datafiles[batch[i]])
            if self.preprocess:
                x = utils.zero_one(x)
            y = utils.read_tif(self.datalabels[batch[i]])
            actual_size = x.shape[0]
            begin_index = (actual_size - self.image_size) // 2
            x = x[begin_index:begin_index + self.image_size, begin_index:begin_index + self.image_size]
            y = y[begin_index:begin_index + self.image_size, begin_index:begin_index + self.image_size]
            ret_x[i] = np.expand_dims(x, -1)
            ret_y[i] = np.expand_dims(y, -1)
        return ret_x, ret_y

    def on_epoch_end(self):
        """
        shuffles the dataset indices at the end of each epoch.
        :return:
        """
        np.random.shuffle(self.ids)
