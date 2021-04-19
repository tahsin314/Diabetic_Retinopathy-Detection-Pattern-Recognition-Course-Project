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
import pytorch_lightning as pl
from tqdm import tqdm_notebook as tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target.astype('int')] = 1.
    return vec

class DRDataset(Dataset):
    def __init__(self, image_ids, labels=None, dim=256, target_type='regression', crop = False, ben_color=False, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.crop = crop
        self.ben_color = ben_color
        self.transforms = transforms
        self.dim = dim
        self.target_type=target_type
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.crop:
            image = crop_image_ilovescience(image)
        if self.ben_color:
            image = load_ben_color(image, self.dim)
        image = cv2.resize(image, (self.dim, self.dim))
        
        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        if self.labels is not None:
            target = self.labels[idx]
            return image_id, image, self.target_processor(target)
            # return image_id, image, onehot(5, target)
        else:
            return image_id, image

    def __len__(self):
        return len(self.image_ids)
    
    def target_processor(self, target):
        if self.target_type == 'regression': return target
        elif self.target_type == 'classification': return one_hot(5, target)
        elif self.target_type == 'ordinal_regression':
            tmp = np.zeros((1, 5))
            for i in range(target+1):
                tmp[:, i] = 1
            return tmp

    def get_labels(self):
        return list(self.labels)

class DRDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, valid_ds, test_ds, 
    batch_size=32, sampler=None, shuffle=True, num_workers=4):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        # self.train_transform = train_transform
        # self.valid_transform = valid_transform
        # self.test_transform = test_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sampler = sampler

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds,batch_size=self.batch_size, 
        sampler=self.sampler, shuffle=self.shuffle, drop_last=True,
        num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_ds,batch_size=self.batch_size, drop_last=True,
        shuffle=self.shuffle,
         num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds,batch_size=self.batch_size, 
        shuffle=self.shuffle, drop_last=True, num_workers=self.num_workers,
         pin_memory=True)
        return test_loader
        
