import os
import shutil
import sys
import glob
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
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from DRDataset import DRDataset
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from model.resnest import Resnest, Mixnet, Attn_Resnest
# from model.densenet import *
from config import *

if mixed_precision:
  scaler = torch.cuda.amp.GradScaler()

history = pd.DataFrame()
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
test_df = pd.read_csv('data/test_768.csv')
test_image_path = 'data/test_768'
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)

model = EffNet(pretrained_model=pretrained_model, use_meta=use_meta, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)
# model = Attn_Resnest(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = Resnest(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = Mixnet(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = EffNet_ArcFace(pretrained_model=pretrained_model, use_meta=use_meta, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)
# model = EffNet(pretrained_model=pretrained_model, use_meta=True, freeze_upto=freeze_upto, out_neurons=500, meta_neurons=250).to(device)
# model = seresnext(pretrained_model, use_meta=True).to(device)
pred_cols = ['image_name'].extend([f'TTA{i}' for i in range(TTA)])

# augs = [test_aug, tta_aug1, tta_aug2, tta_aug3, tta_aug4, tta_aug5, tta_aug6, tta_aug7, tta_aug8, tta_aug9]
# augs = [test_aug, tta_aug1, tta_aug3, tta_aug6, tta_aug7, tta_aug8]
augs = [test_aug, tta_aug1, tta_aug1, tta_aug1, tta_aug3, tta_aug3, tta_aug3, tta_aug6, tta_aug6, tta_aug6, tta_aug7, tta_aug7, tta_aug7, tta_aug8, tta_aug8, tta_aug8]

def rank_data(sub, t):
    sub[f'target{t}'] = sub[f'target{t}'].rank() / sub[f'target{t}'].rank().max()
    return sub

def evaluate():
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
      # PREDS += np.array(preds).reshape(len(test_loader.dataset), 1)
    #  PREDS /= TTA     
  #  return img_ids, list(PREDS[:, 0])
  #  return img_ids, temp_df

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  if mixed_precision:
    scaler.load_state_dict(tmp['scaler'])
  print("Best kappa: {:4f}".format(tmp['best_kappa']))
  del tmp
  print('Model Loaded!')

evaluate()

# submission = pd.read_csv('submission_TTA0.csv')
# submission = rank_data(submission, 0)
# submission.columns = ['image_name', 'target0']

sub0 = pd.read_csv('submission_TTA0.csv')
sub0 = rank_data(sub0, 0)
sub1 = pd.read_csv('submission_TTA1.csv')
sub1 = rank_data(sub1, 1)
sub2 = pd.read_csv('submission_TTA2.csv')
sub2 = rank_data(sub2, 2)
sub3 = pd.read_csv('submission_TTA3.csv')
sub3 = rank_data(sub3, 3)
sub4 = pd.read_csv('submission_TTA4.csv')
sub4 = rank_data(sub4, 4)
sub5 = pd.read_csv('submission_TTA5.csv')
sub5 = rank_data(sub5, 5)
# sub6 = pd.read_csv('submission_TTA6.csv')
# sub6 = rank_data(sub6, 6)
sub0.columns = ['image_name', 'target0']
sub1.columns = ['image_name', 'target1']
sub2.columns = ['image_name', 'target2']
sub3.columns = ['image_name', 'target3']
sub4.columns = ['image_name', 'target4']
sub5.columns = ['image_name', 'target5']
# sub6.columns = ['image_name', 'target6']

f_sub = sub0.merge(sub1, on = 'image_name').merge(sub2, on = 'image_name').merge(sub3, on = 'image_name').merge(sub4, on = 'image_name').merge(sub5, on = 'image_name')
f_sub['target'] = (f_sub['target0'] + f_sub['target1'] + f_sub['target2'] + f_sub['target3'] + f_sub['target4'] + f_sub['target5'])/TTA
f_sub = f_sub[['image_name', 'target']]
f_sub['image_name'] = f_sub['image_name'].map(lambda x: x.replace(test_image_path, '').replace('.jpg', '').replace('/', ''))
f_sub.to_csv('blend_sub.csv', index = False)

# for i in range(1, TTA):
#   sub = pd.read_csv(f'submission_TTA{i}.csv')
#   sub = rank_data(sub, i)
#   sub.columns = ['image_name', f'target{i}']
#   submission.merge(sub, on = 'image_name')
#   print(submission.keys())

# submission['target'] = submission['target0']
# for i in range(1, TTA):
#   submission['target'] += submission[f'target{i}']
# submission['target'] = submission['target']/TTA
# submission = submission[['image_name', 'target']]

# # zippedList =  list(zip(IMG_IDS, TARGET_PRED))
# # submission = pd.DataFrame(zippedList, columns = ['image_name','target'])
# submission['image_name'] = submission['image_name'].map(lambda x: x.replace(test_image_path, '').replace('.jpg', '').replace('/', ''))
# submission.to_csv('submission.csv', index=False)