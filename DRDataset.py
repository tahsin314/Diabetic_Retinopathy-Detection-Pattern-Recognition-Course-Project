import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from random import choices
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target.astype('int')] = 1.
    return vec

class DRDataset(Dataset):
    def __init__(self, image_ids, labels=None, dim=256, target_type='regression', crop = False, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.crop = crop
        self.transforms = transforms
        self.dim = dim
        self.target_type=target_type
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.crop:
            image = crop_image_from_gray(image)
        image = cv2.resize(image, (self.dim, self.dim))
        
        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        if self.labels is not None:
            target = self.labels[idx]
            return image_id, image, target
            # return image_id, image, onehot(5, target)
        else:
            return image_id, image

    def __len__(self):
        return len(self.image_ids)
    
    def target_processor(self, target):
        if self.target_type == 'regression': return target
        elif self.target_type == 'classification': return one_hot(5, target)
        elif self.target_type == 'ordinal_regression':
            tmp = np.zeros(target.shape[0], 5)
            for i in range(target+1):
                tmp[:, i] = 1
            return tmp

    def get_labels(self):
        return list(self.labels)