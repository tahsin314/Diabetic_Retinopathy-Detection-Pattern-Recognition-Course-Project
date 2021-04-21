import os
import glob
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
from sklearn.metrics import cohen_kappa_score, r2_score

import torch, torchvision
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, 
LearningRateMonitor, StochasticWeightAveraging,) 
from pytorch_lightning.loggers import WandbLogger
# import labml
# from labml import experiment
# from labml.utils.lightening import LabMLLighteningLogger
from DRDataset import DRDataset, DRDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.regression_loss import *
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from losses.dice import HybridLoss
from utils import *
from data_processor import *
from model.seresnext import seresnext
from model.effnet import EffNet
from model.resnest import Resnest, Mixnet, Attn_Resnest
from model.hybrid import Hybrid
from model.vit import ViT
from model.botnet import BotNet
from optimizers.over9000 import AdamW, Over9000, Ralamb, LookaheadAdam
import wandb

seed_everything(SEED)
os.system("rm -rf *.png")
wandb_logger = WandbLogger(project="Diabetic_Retinopathy", config=params, settings=wandb.Settings(start_method='fork'))
wandb.init(project="Diabetic_Retinopathy", config=params, settings=wandb.Settings(start_method='fork'))
wandb.run.name= model_name

np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

if 'eff' in model_name:
  base = EffNet(pretrained_model=pretrained_model, num_class=num_class, freeze_upto=freeze_upto).to(device)
elif 'vit' in model_name:
  base = ViT(pretrained_model, num_class=num_class) # Not Working 
else:
  if model_type == 'Normal':
    base = Resnest(pretrained_model, num_class=num_class).to(device)
  elif model_type == 'Attention':
    base = Attn_Resnest(pretrained_model, num_class=num_class).to(device)
  elif model_type == 'Bottleneck':
    base = BotNet(pretrained_model, dim=sz, num_class=num_class).to(device)
# base = torch.nn.DataParallel(base)
wandb.watch(base)

train_ds = DRDataset(train_df.image_id.values, train_df.diagnosis.values, target_type=target_type, 
crop=crop, ben_color=ben_color, dim=sz, transforms=train_aug)

if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=
  BalanceClassSampler(labels=train_ds.get_labels(), mode="upsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, 
  shuffle=True, drop_last=True, num_workers=4)

valid_ds = DRDataset(valid_df.image_id.values, valid_df.diagnosis.values, 
target_type=target_type, crop=crop, ben_color=ben_color, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_ds = DRDataset(test_df.image_id.values, test_df.diagnosis.values, dim=sz, 
target_type=target_type, crop=crop, ben_color=ben_color, transforms=val_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

# Hybrid model
plist = [ 
        {'params': base.backbone.parameters(),  'lr': learning_rate/20},
        {'params': base.head.parameters(),  'lr': learning_rate}
    ]

optimizer = AdamW
# optimizer = AdamW(plist, lr=learning_rate)
# nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
# criterion = nn.BCEWithLogitsLoss(reduction='sum')
if target_type == 'regression':
  criterion = nn.MSELoss(reduction='mean')
  # criterion = hybrid_regression_loss
else:
  # criterion = nn.MSELoss(reduction='mean')
  # criterion = nn.BCEWithLogitsLoss(reduction='sum')
  criterion = criterion_margin_focal_binary_cross_entropy

lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
# cyclic_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer(plist, lr=learning_rate), max_lr=[learning_rate/20, learning_rate], 
# epochs=n_epochs, steps_per_epoch=len(train_loader), pct_start=0.7, anneal_strategy='cos', cycle_momentum=True, 
# base_momentum=0.85, max_momentum=0.95, div_factor=10.0, final_div_factor=20.0, last_epoch=-1)
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer(plist, lr=learning_rate), base_lr=[learning_rate/80,  learning_rate/4], 
# max_lr=[3*learning_rate/80, 3*learning_rate/4], step_size_up=5*len(train_loader), 
# step_size_down=5*len(train_loader), mode='exp_range', gamma=1.0, scale_fn=None, scale_mode='cycle',
#  cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
# cyclic_scheduler = None
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[learning_rate/250, learning_rate/10, 
# learning_rate/10, learning_rate/10], max_lr=[learning_rate/25, learning_rate, learning_rate, learning_rate], 
# step_size_up=5*len(train_loader), step_size_down=5*len(train_loader), mode='triangular', gamma=1.0, 
# scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, lr=learning_rate), 
5*len(train_loader), 2, learning_rate/5, -1)


class LightningDR(pl.LightningModule):
  def __init__(self, model, loss_fn, optim, plist, 
  batch_size, lr_scheduler, cyclic_scheduler, random_id, num_class=1, patience=3, factor=0.5,
  target_type='regression', learning_rate=1e-3):
      super().__init__()
      self.model = model
      self.num_class = num_class
      self.loss_fn = loss_fn
      self.optim = optim
      self.plist = plist 
      self.target_type= target_type
      self.lr_scheduler = lr_scheduler
      self.cyclic_scheduler = cyclic_scheduler
      self.random_id = random_id
      self.patience = patience
      self.factor = factor
      self.learning_rate = learning_rate
      self.batch_size = batch_size
      self.epoch_end_output = [] # Ugly hack for gathering results from multiple GPUs
  
  def forward(self, x):
      out = self.model(x)
      out = out.type_as(x)
      return out

  def configure_optimizers(self):
        optimizer = self.optim(self.plist, self.learning_rate)
        lr_sc = self.lr_scheduler(optimizer, mode='min', factor=0.5, 
        patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
        return ({
       'optimizer': optimizer,
       'lr_scheduler': lr_sc,
       'monitor': 'val_loss',
       'cyclic_scheduler': self.cyclic_scheduler}
        )
 
  def loss_func(self, logits, labels):
      return self.loss_fn(logits, labels)
  
  def step(self, batch):
    _, x, y = batch
    x, y = x.float(), y.float()
    logits = self.forward(x)
    loss = self.loss_func(torch.squeeze(logits), torch.squeeze(y))
    return loss, logits, y  
  
  def training_step(self, train_batch, batch_idx):
    loss, _, _ = self.step(train_batch)
    self.log('train_loss', loss)
    if self.cyclic_scheduler:
      self.cyclic_scheduler.step()
    return loss

  def validation_step(self, val_batch, batch_idx):
      loss, logits, y = self.step(val_batch)
      self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True) 
      val_log = {'val_loss':loss, 'probs':logits, 'gt':y}
      self.epoch_end_output.append({k:v.cpu() for k,v in val_log.items()})
      return val_log

  def test_step(self, test_batch, batch_idx):
      loss, logits, y = self.step(test_batch)
      self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
      test_log = {'test_loss':loss, 'probs':logits, 'gt':y}
      self.epoch_end_output.append({k:v.cpu() for k,v in test_log.items()})
      return test_log

  def label_processor(self, probs, gt):
    if self.target_type == 'regression':
      raw_pr = probs.view(-1).detach().cpu().numpy()
      raw_pr = np.nan_to_num(raw_pr, nan=4.0, posinf=4.0)
      pr = np.round(raw_pr)
      pr = np.clip(pr, 0, 4).astype('int')
      la = gt.view(-1).cpu().numpy().astype('int')

    if self.target_type == 'ordinal_regression':
      raw_pr = (torch.sum((torch.sigmoid(probs)).float(), axis=1)-1).view(-1).detach().cpu().numpy()
      pr = torch.round(torch.sum((torch.sigmoid(probs)>0.5).float(), axis=1)-1).view(-1).detach().cpu().numpy()
      pr = np.clip(pr, 0, 4)
      la = torch.round(torch.sum(gt, axis=1)-1).view(-1).detach().cpu().numpy()

    return raw_pr, pr, la

  def distributed_output(self, outputs):
    if torch.distributed.is_initialized():
      print('TORCH DP')
      torch.distributed.barrier()
      gather = [None] * torch.distributed.get_world_size()
      torch.distributed.all_gather_object(gather, outputs)
      outputs = [x for xs in gather for x in xs]
    return outputs

  def epoch_end(self, mode, outputs):
    outputs = self.epoch_end_output
    avg_loss = torch.Tensor([out[f'{mode}_loss'].mean() for out in outputs]).mean()
    probs = torch.cat([torch.tensor(out['probs']).view(-1, 1) for out in outputs], dim=0)
    gt = torch.cat([torch.tensor(out['gt']).view(-1, 1) for out in outputs], dim=0)
    raw_pr, pr, la = self.label_processor(torch.squeeze(probs), torch.squeeze(gt))
    kappa = torch.tensor(cohen_kappa_score(pr, la, weights='quadratic'))
    r2 = r2_score(la, raw_pr)
    print(f'Epoch: {self.current_epoch} Loss : {avg_loss.numpy():.2f}, kappa: {kappa:.4f}, R2: {r2:.4f}')
    logs = {f'{mode}_loss': avg_loss, f'{mode}_kappa': kappa, f'{mode}_R2':r2}
    self.log(f'{mode}_loss', avg_loss)
    self.log( f'{mode}_kappa', kappa)
    self.log(f'{mode}_R2', r2)
    self.epoch_end_output = []
    return pr, la, {f'avg_{mode}_loss': avg_loss, 'log': logs}

  def validation_epoch_end(self, outputs):
    _, _, log_dict = self.epoch_end('val', outputs)
    self.epoch_end_output = []
    return log_dict

  def test_epoch_end(self, outputs):
    predictions, actual_labels, log_dict = self.epoch_end('test', outputs)
    plot_confusion_matrix(predictions, actual_labels, 
    [i for i in range(5)], self.random_id)
    conf = cv2.imread(f'./conf_{self.random_id}.png', cv2.IMREAD_COLOR)
    conf = cv2.cvtColor(conf, cv2.COLOR_BGR2RGB)
    wandb.log({"Confusion Matrix": 
    [wandb.Image(conf, caption="Confusion Matrix")]})
    return log_dict

data_module = DRDataModule(train_ds, valid_ds, test_ds, batch_size=batch_size)

model = LightningDR(base, criterion, optimizer, plist, batch_size, 
lr_reduce_scheduler, cyclic_scheduler, num_class, target_type=target_type, learning_rate = learning_rate)
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
lr_monitor = LearningRateMonitor(logging_interval='step')
swa_callback =StochasticWeightAveraging()

trainer = pl.Trainer(max_epochs=n_epochs, precision=16, auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                  gradient_clip_val=100,
                  num_sanity_val_steps=10,
                  profiler="simple",
                  weights_summary='top',
                  accumulate_grad_batches = accum_step,
                  logger=wandb_logger, 
                  checkpoint_callback=True,
                  gpus=gpu_ids, num_processes=4*len(gpu_ids),
                  stochastic_weight_avg=True,
                  auto_scale_batch_size='power',
                  benchmark=True,
                  distributed_backend='dp',
                  # plugins='deepspeed', # Not working 
                  # early_stop_callback=False,
                  progress_bar_refresh_rate=1, 
                  callbacks=[checkpoint_callback1, checkpoint_callback2,
                  lr_monitor])
# trainer.train_dataloader = data_module.train_dataloader
# Run learning rate finder
# lr_finder = trainer.tuner.lr_find(model, train_loader, min_lr=1e-6, max_lr=100, num_training=500)

# # Results can be found in
# # print(lr_finder.results)

# # Plot with
# fig = lr_finder.plot(suggest=True, show=True)
# fig.savefig('lr_finder.png')
# fig.show()
# lr_img = cv2.imread('lr_finder.png', cv2.IMREAD_COLOR)
# lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
# wandb.log({"LR Finder": [wandb.Image(lr_img, caption="Learning Rate Finder")]})

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# print(f"Suggested LR: {new_lr}")
# # # update hparams of the model
# # model.hparams.lr = new_lr
# # model.learning_rate = new_lr
# # wandb.log({'Suggested LR': new_lr})
# # with experiment.record(name='sample', exp_conf=params, disable_screen=True):
#         # trainer.fit(model, data_loader)
# wandb.log(params)
# model.cyclic_scheduler = cyclic_scheduler
trainer.fit(model, datamodule=data_module)
try:
  print(f"Best Model path: {checkpoint_callback1.best_model_path} Best Score: {checkpoint_callback1.best_model_score:.4f}")
except:
  pass
chk_path = checkpoint_callback1.best_model_path
model2 = LightningDR.load_from_checkpoint(chk_path, model=base, loss_fn=criterion, optim=optimizer,
 plist=plist, batch_size=batch_size, 
lr_scheduler=lr_reduce_scheduler, cyclic_scheduler=cyclic_scheduler, 
num_class=num_class, target_type=target_type, learning_rate = learning_rate, random_id=random_id)

trainer.test(model=model2, test_dataloaders=test_loader)

# CAM Generation
model2.eval()
plot_heatmap(model2, image_path, test_df, val_aug, random_id=random_id, crop=crop, ben_color=ben_color, cam_layer_name=cam_layer_name, sz=sz)
cam = cv2.imread(f'./heatmap_{random_id}.png', cv2.IMREAD_COLOR)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
wandb.log({"CAM": [wandb.Image(cam, caption="Class Activation Mapping")]})
