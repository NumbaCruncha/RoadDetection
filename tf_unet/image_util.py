# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import glob
import numpy as np
from PIL import Image
import cv2

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):

        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        if np.amax(data) != 0:
            data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """

        row, col = data.shape[:2]
        bottom = data[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        bordersize = 25
        data_with_border = cv2.copyMakeBorder(data, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        #
        #
        # row, col = labels.shape[:2]
        # bottom = labels[row - 2:row, 0:col]
        # mean = cv2.mean(bottom)[0]
        #
        # labels_with_border = cv2.copyMakeBorder(data, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[mean])



        # shape = data.shape
        # w = shape[1]
        # h = shape[0]
        #
        # base_size = h + 30, w + 30, 3
        # # make a 3 channel image for base which is slightly larger than target img
        # base = np.zeros(base_size, dtype=np.uint8)
        # cv2.rectangle(base, (0, 0), (w + 30, h + 30), (255, 255, 255), 30)  # really thick white rectangle
        # base[10:h + 10, 10:w + 10] = data


        # shape = labels.shape
        # w = shape[1]
        # h = shape[0]
        #
        # base_size = h + 30, w + 30, 2
        # # make a 3 channel image for base which is slightly larger than target img
        # base = np.zeros(base_size, dtype=np.uint8)
        # cv2.rectangle(base, (0, 0), (w + 30, h + 30), (255, 255, 255), 30)  # really thick white rectangle
        # base[10:h + 10, 10:w + 10] = labels

        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class=2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".jpg", mask_suffix='_mask.tif', shuffle_data=True, n_class=2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        try:
            assert len(self.data_files) > 0, "No training files"
        except Exception:
            print('Dang!')
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path + r'\*', recursive=True)
        all_files = [i for i in all_files if r'.xml' not in i]
        all_files = [i for i in all_files if r'.ovr' not in i]
        all_files = [i for i in all_files if r'.jgw' not in i]
        # return [name for name in all_files if self.data_suffix in name and self.mask_suffix not in name]

        exclusions = ['BX24_500_013026', 'BX24_500_013027', 'BX24_500_013028', 'BX24_500_013034', 'BX24_500_013035', 'BX24_500_014028', 'BX24_500_014029', 'BX24_500_014030',
                      'BX24_500_013026_mask', 'BX24_500_013027_mask', 'BX24_500_013028_mask', 'BX24_500_013034_mask', 'BX24_500_013035_mask', 'BX24_500_014028_mask', 'BX24_500_014029_mask',
                      'BX24_500_014030_mask']

        all_files_with_exclusions = [i for i in all_files if i.split('.')[0].split('\\')[-1] not in exclusions]
        return all_files_with_exclusions ## COMPLETE EXCLUSIONS
    
    def _load_file(self, path, dtype=np.float32, format='image_file'):
        if format == 'image_file':
            images = cv2.imread(path)
            images_scaled = cv2.resize(images, (400, 600))
            return np.array(images_scaled, dtype)
        elif format == 'GeoTiff':
            mask = np.squeeze(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            mask_scaled = cv2.resize(mask, (400, 600))
            return np.array(mask_scaled, np.bool)

    def _cycle_file(self):
        self.file_idx += 1
        if self.file_idx >= len(sorted([i for i in self.data_files if i.split('.')[1] == self.data_suffix.split('.')[1]])):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cycle_file()

        image_name = sorted([i for i in self.data_files if i.split('.')[1] == self.data_suffix.split('.')[1]])[self.file_idx]

        try:
            label_name = sorted([i for i in self.data_files if i.split('.')[1] == self.mask_suffix.split('.')[1]])[self.file_idx]
        except IndexError:
            print('Index Out of Range')
            label_name = None

        img = self._load_file(image_name, np.float32, 'image_file')
        if label_name not in self.data_files:
            # label = np.array(np.random.rand(400, 600), np.bool)
            label = np.empty(shape=(400, 600), dtype=np.bool)


        else:
            label = self._load_file(label_name, np.bool, 'GeoTiff')
        return img, label
