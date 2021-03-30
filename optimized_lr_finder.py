import sys
sys.path.insert(0, 'pytorch-lr-finder/torch_lr_finder')
from config import *
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import threading
import curses 
import gc
import time
from random import choices
from itertools import chain
import numpy as np
import pandas as pd
import sklearn
import cv2
from tqdm import tqdm as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_kappa_score
from apex import amp
import torch, torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from DRDataset import DRDataset
from catalyst.data.sampler import BalanceClassSampler
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from config import *
from lr_finder import LRFinder

if mixed_precision:
  scaler = torch.cuda.amp.GradScaler()
  
prev_epoch_num = 0
best_valid_loss = np.inf
best_valid_kappa = 0.0
balanced_sampler = False
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
pseduo_df = rank_based_pseudo_label_df(pseduo_df, 0.2, 0.99)
pseudo_labels = list(pseduo_df['target'])
print("Pseudo data length: {}".format(len(pseduo_df)))
print("Negative label: {}, Positive label: {}".format(pseudo_labels.count(0), pseudo_labels.count(1))) 
df = pd.read_csv('data/train_768.csv')

pseduo_df['fold'] = np.nan
pseduo_df['fold'] = pseduo_df['fold'].map(lambda x: 16)
# pseduo_df = meta_df(pseduo_df, test_image_path)
    
df['fold'] = df['fold'].astype('int')
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
train_folds = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16]
valid_folds = [4, 9, 14]
train_df = df[df['fold'] == train_folds[0]]
valid_df = df[df['fold'] == valid_folds[0]]
for i in train_folds[1:]:
  train_df = pd.concat([train_df, df[df['fold'] == i]])
for i in valid_folds[1:]:
  valid_df = pd.concat([valid_df, df[df['fold'] == i]])

train_df = pd.concat([train_df, pseduo_df], ignore_index=True)
test_df = pseduo_df
train_meta = np.array(train_df[meta_features].values, dtype=np.float32)
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)
# model = seresnext(pretrained_model).to(device)
model = EffNet(pretrained_model=pretrained_model, freeze_upto=freeze_upto).to(device)

train_ds = DRDataset(train_df.image_name.values, train_meta, train_df.target.values, dim=sz, transforms=train_aug)
if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=BalanceClassSampler(labels=train_ds.get_labels(), mode="downsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=4)

plist = [
        {'params': model.backbone.parameters(),  'lr': learning_rate/50},
        {'params': model.meta_fc.parameters(),  'lr': learning_rate},
        # {'params': model.metric_classify.parameters(),  'lr': learning_rate},
    ]

optimizer = optim.Adam(plist, lr=learning_rate)
# lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=10*learning_rate, step_size_up=2000, step_size_down=2000, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

criterion = criterion_margin_focal_binary_cross_entropy
if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  # optimizer.load_state_dict(tmp['optim'])
  # lr_reduce_scheduler.load_state_dict(tmp['scheduler'])
  # cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
  # amp.load_state_dict(tmp['amp'])
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['best_loss']
  del tmp
  print('Model Loaded!')
# model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=100, num_iter=500,  accumulation_steps=accum_step)
lr_finder.plot() # to inspect the loss-learning rate graph
