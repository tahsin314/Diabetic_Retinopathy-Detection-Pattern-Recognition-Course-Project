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
import torch, torchvision
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
from model.resnest import Resnest, Mixnet, Attn_Resnest
from config import *

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
train_folds = [0, 1, 2, 5, 6, 8, 9, 10, 12, 13, 15, 16]
valid_folds = [3,7,11,14]
train_df = df[df['fold'] == train_folds[0]]
valid_df = df[df['fold'] == valid_folds[0]]

for i in valid_folds[1:]:
  valid_df = pd.concat([valid_df, df[df['fold'] == i]])
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
# model = Mixnet(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = Resnest(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = Attn_Resnest(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = EffNet_ArcFace(pretrained_model=pretrained_model, use_meta=use_meta, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)
model = EffNet(pretrained_model=pretrained_model, use_meta=use_meta, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)
# model = seresnext(pretrained_model, use_meta=True).to(device)
# model = EffNet(pretrained_model=pretrained_model, use_meta=True, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)

valid_ds = DRDataset(valid_df.image_name.values, valid_meta, valid_df.target.values, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_aug = Compose([Normalize()])
tta_aug1 = Compose([
  ShiftScaleRotate(p=1,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    Normalize(always_apply=True)])
tta_aug2 = Compose([
  Cutout(p=1.0, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    Normalize(always_apply=True)])
tta_aug3 = Compose([
  RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=1.0),
    Normalize(always_apply=True)])
tta_aug4 = Compose([
  AdvancedHairAugmentationAlbumentations(p=1.0),
    Normalize(always_apply=True)])
tta_aug5 = Compose([
  GaussianBlur(blur_limit=3, p=1),
    Normalize(always_apply=True)])
tta_aug6 = Compose([
  HueSaturationValue(p=1.0),
    Normalize(always_apply=True)])
tta_aug7 = Compose([
  HorizontalFlip(1.0),
    Normalize(always_apply=True)])

tta_aug8 = Compose([
  VerticalFlip(1.0),
    Normalize(always_apply=True)])

tta_aug9 = Compose([
  ColorConstancy(p=1.0),
    Normalize(always_apply=True)])

tta_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    ], p=0.20),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
    HueSaturationValue(p=0.4),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    Normalize(always_apply=True)
    ]
      )
augs = [test_aug, tta_aug1, tta_aug1, tta_aug1, tta_aug3, tta_aug3, tta_aug3, tta_aug6, tta_aug6, tta_aug6, tta_aug7, tta_aug7, tta_aug7, tta_aug8, tta_aug8, tta_aug8]
# augs = [test_aug, tta_aug1]

def evaluate():
   model.eval()
   PREDS = np.zeros((len(valid_df), 1))
   IMG_IDS = []
   LAB = []
   with torch.no_grad():
    for t in range(len(augs)):
      print('TTA {}'.format(t+1))
      test_ds = DRDataset(image_ids=valid_df.image_name.values, meta_features=valid_meta, dim=sz, transforms=augs[t])
      test_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)

      img_ids = []
      preds = []
      lab = []
      
      for idx, (img_id, inputs, meta, labels) in T(enumerate(test_loader),total=len(test_loader)):
        inputs = inputs.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float(), meta)
        img_ids.extend(img_id)        
        preds.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
        lab.extend(torch.argmax(labels, 1).cpu().numpy())
      # zippedList =  list(zip(img_ids, preds))
      print(np.array(preds).shape, np.array(lab).shape)
      score_diff_tta = np.abs(np.array(preds)-np.array(lab)).reshape(len(valid_loader.dataset), 1)
      # print(score_diff_tta.shape)
    # print(np.array(PREDS).shape, np.array(LAB).shape)
      zippedList_tta =  list(zip(img_ids, lab, np.squeeze(preds), np.squeeze(score_diff_tta)))
      temp_df = pd.DataFrame(zippedList_tta, columns = ['image_name','label', 'predictions', 'difference'])
      temp_df.to_csv(f'submission_TTA{t}.csv', index=False)
      IMG_IDS = img_ids
      LAB = lab
      PREDS += np.array(preds).reshape(len(valid_loader.dataset), 1)
    PREDS /= len(augs)
    score_diff = np.abs(np.array(PREDS)-np.array(LAB).reshape(len(valid_loader.dataset), 1))
    # print(np.array(PREDS).shape, np.array(LAB).shape)
    zippedList =  list(zip(IMG_IDS, LAB, np.squeeze(PREDS), np.squeeze(score_diff)))
    submission = pd.DataFrame(zippedList, columns = ['image_name','label', 'prediction', 'difference'])
    submission = submission.sort_values(by=['difference'], ascending=False)
    submission.to_csv('val_report.csv', index=False)      
  

def train_val(epoch, dataloader, optimizer, choice_weights= [0.8, 0.1, 0.1], rate=1):
  t1 = time.time()
  running_loss = 0
  epoch_samples = 0
  img_ids = []
  pred = []
  lab = []
  probs = []
  model.eval()
  print("Initiating val phase ...")
  for idx, (img_id, inputs,meta,labels) in enumerate(dataloader):
    with torch.set_grad_enabled(False):
        inputs = inputs.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        epoch_samples += len(inputs)
        choice_weights = [1.0, 0, 0]
        choice = choices(opts, weights=choice_weights)
        optimizer.zero_grad()
        outputs = model(inputs.float(), meta)
        loss = ohem_loss(1.00, criterion, outputs, labels)
        running_loss += loss.item() 
        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
        img_ids.extend(img_id)
        probs.extend(torch.softmax(outputs,1).detach().cpu().numpy())
        pred.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
        lab.extend(torch.argmax(labels, 1).cpu().numpy())
        msg = f'Epoch {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
        print(msg, end= '\r')
  # exps = np.linspace(-5, -2, 40)
  # for ex in exps:
  # pred = [((p+1e-8)**-1) for p in pred]
  score_diff = np.abs(np.array(pred)-np.array(lab))      
  kappa = roc_kappa_score(lab, pred)
  zippedList =  list(zip(img_ids, lab, pred, score_diff))
  submission = pd.DataFrame(zippedList, columns = ['image_name','label', 'target', 'difference'])
  submission = submission.sort_values(by=['difference'], ascending=False)
  submission.to_csv('val_report.csv', index=False)
  msg = f'Validation Loss: {running_loss/epoch_samples:.4f} Validation kappa: {kappa:.4f}'
  print(msg)
  return running_loss/epoch_samples, kappa

def evaluatev2(test_df, test_meta):
   model.eval()
   PREDS = np.zeros((len(test_df), 1))
   with torch.no_grad():
     for t in range(len(augs)):
      print('TTA {}'.format(t+1))
      test_ds = DRDataset(image_ids=test_df.image_name.values, meta_features=test_meta, dim=sz, transforms=augs[t])
      test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

      img_ids = []
      preds = []
      for idx, (img_id, inputs, meta) in T(enumerate(test_loader),total=len(test_loader)):
        inputs = inputs.to(device)
        meta = meta.to(device)
        outputs = model(inputs.float(), meta)
        img_ids.extend(img_id)        
        preds.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
      zippedList =  list(zip(img_ids, preds))
      temp_df = pd.DataFrame(zippedList, columns = ['image_name',f'target{t}'])
      temp_df.to_csv(f'submission_TTA{t}.csv', index=False)

# Effnet model
plist = [ 
        {'params': model.backbone.parameters(),  'lr': learning_rate/50},
        # {'params': model.meta_fc.parameters(),  'lr': learning_rate},
        # {'params': model.output.parameters(),  'lr': learning_rate},
    ]

optimizer = optim.Adam(plist, lr=learning_rate)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

# nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
criterion = criterion_margin_focal_binary_cross_entropy

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  if mixed_precision:
    scaler.load_state_dict(tmp['scaler'])
  # amp.load_state_dict(tmp['amp'])
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['best_loss']
  best_valid_kappa = tmp['best_kappa']
  print(best_valid_kappa)
  print('Model Loaded!')

# valid_loss, valid_kappa = train_val(-1, valid_loader, optimizer=optimizer, rate=1.00)
# evaluate()
evaluatev2(valid_df, valid_meta)