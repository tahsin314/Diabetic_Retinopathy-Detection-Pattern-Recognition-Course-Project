import os

from matplotlib.pyplot import axis
from config import *
import shutil
import sys
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
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
from sklearn.metrics import cohen_kappa_score

import torch, torchvision
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# import labml
# from labml import experiment
# from labml.utils.lightening import LabMLLighteningLogger
from DRDataset import DRDataset, DRDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from losses.dice import HybridLoss
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet
from model.resnest import Resnest, Mixnet, Attn_Resnest
from model.hybrid import Hybrid
sys.path.insert(0, 'pytorch_lr_finder')
from torch_lr_finder import LRFinder
from over9000.over9000 import Over9000, Ralamb
import wandb

seed_everything(SEED)

wandb_logger = WandbLogger(project="Diabetic_Retinopathy", config=params, settings=wandb.Settings(start_method='fork'))
wandb.init(project="Diabetic_Retinopathy", config=params, settings=wandb.Settings(start_method='fork'))
wandb.run.name= model_name
m_p = mixed_precision
if m_p:
  scaler = torch.cuda.amp.GradScaler() 
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
df = pd.read_csv(f'{data_dir}/trainLabels_cropped.csv')
test_df = pd.read_csv(f'{data_dir}/train.csv')

df['image_id'] = df['image'].map(lambda x: f"{image_path}/{x}.jpeg")
df['diagnosis'] = df['level'].map(lambda x: x)
df = df[['image_id', 'diagnosis']]
df_messidor = Messidor_Process(f'{data_dir}/Messidor Dataset')
df_idrid = IDRID_Process(data_dir)
test_df['image_id'] = test_df['id_code'].map(lambda x: f"{test_image_path}/{x}.png")
test_df['diagnosis'] = test_df['diagnosis'].map(lambda x: x)
df = pd.concat([df_messidor, df, df_idrid], ignore_index=True)
# Delete later
# df = test_df
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df = df[['image_id', 'diagnosis']]
df['fold'] = np.nan 
X = df['image_id']
y = df['diagnosis']
train_idx = []
val_idx = []
for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    train_idx = train_index
    val_idx = val_index
    df.loc[val_idx, 'fold'] = i

df['fold'] = df['fold'].astype('int')
train_df = df[(df['fold']!=fold)]
valid_df = df[df['fold']==fold]
# test_df = df[df['fold']==n_fold-1]
# print(len(train_df), len(valid_df), len(test_df))
if 'eff' in model_name:
  model = EffNet(pretrained_model=pretrained_model, num_class=num_class, freeze_upto=freeze_upto).to(device)
else:
  model = Resnest(pretrained_model, num_class=num_class).to(device)
# model = Mixnet(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
# model = Hybrid().to(device)
model = torch.nn.DataParallel(model)
# print(model.module.backbone.conv_head)
wandb.watch(model)

train_ds = DRDataset(train_df.image_id.values, train_df.diagnosis.values, target_type=target_type, crop=crop, ben_color=ben_color, dim=sz, transforms=train_aug)

if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=BalanceClassSampler(labels=train_ds.get_labels(), mode="upsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

valid_ds = DRDataset(valid_df.image_id.values, valid_df.diagnosis.values, target_type=target_type, crop=crop, ben_color=ben_color, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_ds = DRDataset(test_df.image_id.values, test_df.diagnosis.values, dim=sz, target_type=target_type, crop=crop, ben_color=ben_color, transforms=val_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

# Hybrid model
plist = [ 
        {'params': model.module.backbone.parameters(),  'lr': learning_rate/20},
        # {'params': model.module.Attn_Resnest.resnest.parameters(),  'lr': learning_rate/100},
        # {'params': model.module.effnet.parameters(),  'lr': learning_rate/100},
        # {'params': model.module.attn1.parameters(), 'lr': learning_rate},
        # {'params': model.module.attn2.parameters(), 'lr': learning_rate},
        # {'params': model.module.head_res.parameters(), 'lr': learning_rate}, 
        # {'params': model.module.eff_conv.parameters(),  'lr': learning_rate}, 
        # {'params': model.module.eff_attn.parameters(),  'lr': learning_rate}, 
        {'params': model.module.head.parameters(),  'lr': learning_rate},
        # {'params': model.module.output.parameters(), 'lr': learning_rate},
        # {'params': model.module.output1.parameters(),  'lr': learning_rate},
    ]
# optimizer = optim.AdamW(plist, lr=learning_rate)
optimizer = Ralamb(plist, lr=learning_rate)
# nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
# criterion = nn.BCEWithLogitsLoss(reduction='sum')
if target_type == 'regression':
  criterion = nn.MSELoss(reduction='none')
else:criterion = criterion_margin_focal_binary_cross_entropy
# criterion = ArcFaceLoss().to(device)
# criterion = HybridLoss(alpha=2, beta=1).to(device)
# lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader=train_loader, accumulation_steps=accum_step, end_lr=50, num_iter=300, step_mode="exp")
# _, suggested_lr = lr_finder.plot(suggest_lr=True)
# lr_finder.reset()
# print(f"Suggested LR: {suggested_lr}")
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
cyclic_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[learning_rate/20, learning_rate], epochs=n_epochs, steps_per_epoch=len(train_loader), pct_start=0.7, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=5.0, final_div_factor=100.0, last_epoch=-1)
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[learning_rate/20,  learning_rate], max_lr=[learning_rate/10, 2*learning_rate], step_size_up=5*len(train_loader), step_size_down=5*len(train_loader), mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[learning_rate/250, learning_rate/10, learning_rate/10, learning_rate/10], max_lr=[learning_rate/25, learning_rate, learning_rate, learning_rate], step_size_up=5*len(train_loader), step_size_down=5*len(train_loader), mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)



class LightningDR(pl.LightningModule):
  def __init__(self, model, loss_fn, optim, plist, 
  batch_size, lr_scheduler, num_class=1, patience=3, factor=0.5,
  target_type='regression', learning_rate=1e-3):
      super().__init__()
      self.model = model
      self.num_class = num_class
      self.loss_fn = loss_fn
      self.optim = optim
      self.plist = plist 
      self.target_type= target_type
      self.lr_scheduler = lr_scheduler
      self.patience = patience
      self.factor = factor
      self.learning_rate = learning_rate
      self.batch_size = batch_size
  
  def forward(self, x):
      return self.model(x)

  def configure_optimizers(self):
        optimizer = self.optim(self.plist, self.learning_rate)
        lr_sc = self.lr_scheduler(optimizer, mode='min', factor=0.5, 
        patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
        return {
       'optimizer': optimizer,
       'lr_scheduler': lr_sc,
       'monitor': 'val_loss'
   }
  #  { 
  #    'cyclic_scheduler': self.cyclic_scheduler,
  #  })
    
  def loss_func(self, logits, labels):
      return self.loss_fn(logits, labels)
  
  def training_step(self, train_batch, batch_idx):
      _, x, y = train_batch
      x, y = x.float(), y.float()
      logits = self.forward(x)
      loss = self.loss_func(logits, y)
      self.log('train_loss', loss)
      return loss

  def validation_step(self, val_batch, batch_idx):
      _, x, y = val_batch
      x, y = x.float(), y.float()
      logits = self.forward(x)
      loss = self.loss_func(logits, y)
      self.log('val_loss', loss)
      return {'val_loss':loss, 'probs':logits, 'gt':y}

  
  def test_step(self, test_batch, batch_idx):
      _, x, y = test_batch
      x, y = x.float(), y.float()
      logits = self.forward(x)
      loss = self.loss_func(logits, y)
      self.log('test_loss', loss)
      return {'test_loss':loss, 'probs':logits, 'gt':y}

  def label_processor(self, probs, gt):
    if self.target_type == 'regression':
          pr = np.round(probs.view(-1).detach().cpu().numpy())
          pr = np.clip(pr, 0, 4)
          la = gt.view(-1).cpu().numpy()

    if self.target_type == 'ordinal_regression':
      pr = torch.round(torch.sum((torch.sigmoid(probs)>0.5).float(), axis=1)-1).view(-1).detach().cpu().numpy()
      pr = np.clip(pr, 0, 4)
      la = torch.round(torch.sum(gt, axis=1)-1).view(-1).detach().cpu().numpy()

    return pr, la


  def validation_step_end(self, outputs):
      avg_loss = outputs['val_loss']
      probs = outputs['probs']
      gt = outputs['gt']
      pr, la = self.label_processor(probs, gt)
      kappa = torch.tensor(cohen_kappa_score(pr, la, weights='quadratic'))
      val_logs = {'val_loss': avg_loss, 'val_kappa': kappa}
      return {'val_loss': avg_loss, 'probs': pr, 'gt':la}

  def validation_epoch_end(self, outputs):
    # This is what happens at the end of validation epoch. Usually gathering all predictions
    # outputs is a list of dictionary from each step.
    # avg_loss = torch.cat([out['val_loss'] for out in outputs]).mean()
    avg_loss = torch.Tensor([out['val_loss'] for out in outputs]).mean().numpy()
    # print(avg_loss)
    # probs = torch.Tensor([out['probs'] for out in outputs])
    probs = torch.cat([torch.tensor(out['probs']) for out in outputs], dim=0)
    # gt = torch.Tensor([out['gt'] for out in outputs])
    gt = torch.cat([torch.tensor(out['gt']) for out in outputs], dim=0)
    # print(gt.size(), probs.size()) 
    pr, la = self.label_processor(probs, gt)

    kappa = torch.tensor(cohen_kappa_score(pr, la, weights='quadratic'))
    print(f'Epoch: {self.current_epoch} Loss : {avg_loss:.2f}, kappa: {kappa:.4f}')
    val_logs = {'val_loss': avg_loss, 'val_kappa': kappa}
    return {'avg_val_loss': avg_loss, 'log': val_logs}
    
  def test_epoch_end(self, outputs):
      # This is what happens at the end of validation epoch. Usually gathering all predictions
      # outputs is a list of dictionary from each step.
      avg_loss = torch.cat([out['test_loss'] for out in outputs], dim=0).mean()
      probs = torch.cat([out['probs'] for out in outputs], dim=0)
      gt = torch.cat([out['gt'] for out in outputs], dim=0)
      probs = probs.detach().cpu().numpy()
      gt = gt.detach().cpu().numpy()
      
      pr, la = self.label_processor(probs, gt)

      kappa = torch.tensor(cohen_kappa_score(pr, la, weights='quadratic'))
      test_logs = {'test_loss': avg_loss, 'test_kappa': kappa}
      print(f'Loss: {avg_loss:.2f}, kappa: {kappa:.4f}')

      return {'avg_test_loss': avg_loss, 'log': test_logs}


data_module = DRDataModule(train_ds, valid_ds, test_ds, batch_size=batch_size)

# train
model = LightningDR(model, nn.MSELoss(reduction='mean'), Ralamb, plist, batch_size, 
lr_reduce_scheduler, num_class, target_type, learning_rate = learning_rate)
checkpoint_callback1 = ModelCheckpoint(
    monitor='val_loss',
    dirpath='model_dir',
    filename=f"{model_name}_loss",
    save_top_k=1,
    mode='min',
)

checkpoint_callback2 = ModelCheckpoint(
    monitor='val_kappa',
    dirpath='model_dir',
    filename=f"{model_name}_kappa",
    save_top_k=1,
    mode='max',
)

trainer = pl.Trainer(max_epochs=n_epochs, precision=16, auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                  # gradient_clip_val=1,
                  num_sanity_val_steps=10,
                  accumulate_grad_batches = accum_step,
                  logger=wandb_logger,  # Comment that out to reactivate sanity but the ROC will fail if the sample has only class 0
                #  checkpoint_callback=checkpoint_callback,
                  gpus=1,
                  auto_scale_batch_size=True,
                  # early_stop_callback=False,
                  progress_bar_refresh_rate=1, 
                  callbacks=[checkpoint_callback1, checkpoint_callback1])
trainer.train_dataloader = data_module.train_dataloader
# trainer.tune(model)
# Run learning rate finder
print(model.learning_rate)
# lr_finder = trainer.tuner.lr_find(model, train_loader, max_lr=50, num_training=200)

# Results can be found in
# print(lr_finder.results)

# Plot with
# fig = lr_finder.plot(suggest=True, show=True)
# fig.savefig('lr_finder.png')
# fig.show()

# Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# print(f"Suggested LR: {new_lr}")
# update hparams of the model
# model.hparams.lr = new_lr
# model.learning_rate = new_lr
# wandb.log({'Suggested LR': new_lr})
# with experiment.record(name='sample', exp_conf=params, disable_screen=True):
        # trainer.fit(model, data_loader)
trainer.fit(model, data_module)
