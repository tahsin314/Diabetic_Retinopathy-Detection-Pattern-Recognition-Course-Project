import os
import random
import numpy as np
import cv2 
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
'''
Adapted from here: https://www.kaggle.com/c/siim-isic-DR-classification/discussion/154876#867412
'''

class ColorConstancy(ImageOnlyTransform):
    def __init__(
            self, power:float = 4, gamma:float = None,
            always_apply=False,
            p=0.5,
    ):
        super(ColorConstancy, self).__init__(always_apply, p)
        self.power = power
        self.gamma = gamma

    def apply(self, image, **params):
        """
        Args:
            img (PIL Image): Image to apply color constancy on.

        Returns:
            Image: Image with color constancy.
        """
        img = self.color_constancy(image, self.power, self.gamma)
        return img

    def color_constancy(self, img, power=6, gamma=None):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_dtype = img.dtype

        if gamma is not None:
            img = img.astype('uint8')
            look_up_table = np.ones((256,1), dtype='uint8') * 0
            for i in range(256):
                look_up_table[i][0] = 255*pow(i/255, 1/gamma)
            img = cv2.LUT(img, look_up_table)

        img = img.astype('float32')
        img_power = np.power(img, power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return img.astype(img_dtype)