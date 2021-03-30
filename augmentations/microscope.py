# Courtesy: https://www.kaggle.com/c/siim-isic-DR-classification/discussion/159476
# Albumentations version is also added

import os
import random
import numpy as np
import cv2 
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

class Microscope:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

class MicroscopeAlbumentations(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(MicroscopeAlbumentations, self).__init__(always_apply, p)

    def apply(self, img, **params):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img