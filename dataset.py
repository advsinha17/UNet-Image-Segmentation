import os
from PIL import Image
import numpy as np
import random
from labeldata import id_to_color
import re

import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    
    """

    Class that generates batches of (input, target) pairs after preprocessing the images.

    Args:
        images_dir: Directory from which to obtain the images.
        id_to_color: Dictionary used to convert encoded images into appropriate labels.
        batch_size: Size of batch generated.
        shuffle: True if images are to be shuffled at end of each epoch, false otherwise.
        predicting: True if generated data is being used to make predictions. (This is required as os.listdir() generates a list
        in arbitrary order. While predicting, the list is sorted so that prediction are made in the same order the images are 
        present in the input directory.)

    """

    def __init__(self, images_dir, batch_size = 16, shuffle = True, predicting = False):
        
        super(DataGenerator, self).__init__()

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_images = os.listdir(images_dir)
        if predicting:
            self._sort_dir()
        self.image_size = (128, 128)
        self.image_shape = self.image_size + (3,)
        self.id_to_color = id_to_color
        self.on_epoch_end()

    def _sort_dir(self):   

        """
        Sort the list of image paths.
        """
        self.list_images.sort(key = self._alnum_key)

    
    def _tryint(self, dir):

        """
        Return an int if possible, or `dir` unchanged.

        Args:
            dir: Path of an image.
            
        """
        try:
            return int(dir)
        
        except ValueError:
            return dir
        
    def _alnum_key(self, dir):

        """
        Turn a string into a list of string and number chunks.
        """
        return [self._tryint(c) for c in re.split('([0-9]+)', dir)]
    

    def __len__(self):

        """
        Returns number of batches per epoch
        """
        return int(np.ceil(len(self.list_images) / self.batch_size))
    
    def _image_mask_split(self, filename):

        """
        Splits the input image and the target image.

        Args:
            filename: Path of the image.

        Returns:
            Tuple consisting of encoding of image and mask.

        """
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path)
        image, mask = image.crop([0, 0, 256, 256]), image.crop([256, 0, 512, 256])
        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)
        image = np.array(image)/255.0
        mask = np.array(mask)
        return image, mask
    
    def _vectorized_labels(self, mask, mapping):

        """
        Returns label for each pixel of the target mask.

        Args:
            mask: Encoding of the mask.
            mapping: Dictionary mapping each category to a pixel encoding.
        
        Returns:
            closest_category: Label for each pixel of the mask image.

        """
        height, width, _ = mask.shape

        closest_distance = np.full((height, width), np.inf)
        closest_category = np.full((height, width), -1, dtype=int)
        
        for id, color in mapping.items():
            dist = np.linalg.norm(mask - np.array(color).reshape(1, 1, -1), axis=-1)
            
            is_closer = closest_distance > dist
            
            closest_distance = np.where(is_closer, dist, closest_distance)
            closest_category = np.where(is_closer, id, closest_category)

        return closest_category



    def __getitem__(self, index):

        """
        Generates a batch of data.
        """

        batches = self.list_images[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        masks = []
        for filename in batches:
            image, mask = self._image_mask_split(filename)
            images.append(image)
            masks.append(self._vectorized_labels(mask, id_to_color))

        return np.stack(images).astype('float32'), np.stack(masks).astype('float32')
    
    def on_epoch_end(self):

        """
        Shuffles list of images on end of each epoch.
        """
        if self.shuffle:
            random.shuffle(self.list_images)
