import os
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
from DRDataset import DRDataset
from catalyst.data.sampler import BalanceClassSampler
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from losses.dice import HybridLoss
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from model.resnest import Resnest, Mixnet, Attn_Resnest
from model.hybrid import Hybrid
sys.path.insert(0, 'pytorch_lr_finder')
from torch_lr_finder import LRFinder
import wandb

seed_everything(SEED)

wandb.init(project="Diabetic_Retinopathy", config=params)
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
df = pd.concat([df_messidor, df, df_idrid, test_df], ignore_index=True)
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
train_df = df[(df['fold']!=fold) & (df['fold']!=n_fold-1)]
valid_df = df[df['fold']==fold]
test_df = df[df['fold']==n_fold-1]
# print(len(train_df), len(valid_df), len(test_df))
# model = Resnest(pretrained_model, use_meta=use_meta).to(device)
# model = Mixnet(pretrained_model, use_meta=use_meta, out_neurons=500, meta_neurons=250).to(device)
model = EffNet(pretrained_model=pretrained_model, freeze_upto=freeze_upto).to(device)
# model = Hybrid().to(device)
model = torch.nn.DataParallel(model)
wandb.watch(model)
# print(model.module.backbone.conv_head)
# model.to(f'cuda:{model.device_ids[0]}')
train_ds = DRDataset(train_df.image_id.values, train_df.diagnosis.values, crop=True, dim=sz, transforms=train_aug)


if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=BalanceClassSampler(labels=train_ds.get_labels(), mode="upsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

valid_ds = DRDataset(valid_df.image_id.values, valid_df.diagnosis.values, crop=True, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_ds = DRDataset(test_df.image_id.values, test_df.diagnosis.values, dim=sz, crop=True, transforms=val_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

def train_val(epoch, dataloader, model, optimizer, choice_weights= [0.8, 0.1, 0.1], rate=1, train=True, mode='train'):
  global m_p
  global result
  global accum_step
  global batch_size
  t1 = time.time()
  running_loss = 0
  epoch_samples = 0
  batch_kappa = []
  pred = []
  lab = []
  if train:
    model.train()
    print("Initiating train phase ...")
  else:
    model.eval()
    print("Initiating val phase ...")
  for idx, (_, inputs,labels) in enumerate(dataloader):
    with torch.set_grad_enabled(train):
      inputs = inputs.to(device)
      labels = labels.view(-1, 1).float().to(device)
      epoch_samples += len(inputs)
      if not train:
        choice_weights = [1.0, 0, 0]
      choice = choices(opts, weights=choice_weights)
      optimizer.zero_grad()
      with torch.cuda.amp.autocast(m_p):
        if choice[0] == 'normal':
          outputs = model(inputs.float())
          loss = ohem_loss(rate, criterion, outputs, labels)
          running_loss += loss.item()
        
        elif choice[0] == 'mixup':
          inputs, targets = mixup(inputs, labels, np.random.uniform(0.8, 1.0))
          outputs = model(inputs.float())
          loss = mixup_criterion(outputs, targets, criterion=criterion, rate=rate)
          running_loss += loss.item()
        
        elif choice[0] == 'cutmix':
          inputs, targets = cutmix(inputs, labels, np.random.uniform(0.8, 1.0))
          outputs = model(inputs.float())
          loss = cutmix_criterion(outputs, targets, criterion=criterion, rate=rate)
          running_loss += loss.item()
      
        loss = loss/accum_step
      
        if train:
          if m_p:
            scaler.scale(loss).backward()
            if (idx+1) % accum_step == 0:
              scaler.step(optimizer) 
              scaler.update() 
              optimizer.zero_grad()
              cyclic_scheduler.step()
          else:
            loss.backward()
            if (idx+1) % accum_step == 0:
              optimizer.step()
              optimizer.zero_grad()
              # cyclic_scheduler.step()    
      elapsed = int(time.time() - t1)
      eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
      pred.extend(torch.round(outputs).view(-1).detach().cpu().numpy())
      lab.extend(labels.view(-1).cpu().numpy())
      batch_kappa.append(cohen_kappa_score(torch.round(outputs).view(-1).detach().cpu().numpy(), labels.view(-1).cpu().numpy(), weights='quadratic'))
      # pred.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
      # lab.extend(torch.argmax(labels, dim=1).cpu().numpy())
      if train:
        wandb.log({"Train Loss": running_loss/epoch_samples})
        progress = int(30*(idx/len(dataloader)))
        # bar = "\033[92m"+"⚪"*progress+"◯"*(30-progress)+"  "+str(int(100*progress/30)+"%/100%\033[92m")
        bar = f"\033[92m {'⚪'*progress}{'◯ '*(30-progress)}  {int(100*progress/30)}%/100%\033[92m"
        msg = f"Epoch: {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
      else:
        wandb.log({f"{mode} Loss": running_loss/epoch_samples})
        progress = int(30*(idx/len(dataloader)))
        # bar = "\033[92m"+"⚪"*progress+"◯"*(30-progress)+"  "+str(int(100*progress/30)+"%/100%\033[92m")
        bar = f"\033[92m {'⚪'*progress}{'◯ '*(30-progress)}  {int(100*progress/30)}%/100%\033[92m"
        msg = f'Epoch {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
      # print(bar, end= '\r')
      print(msg, end='\r')
  history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
  history.loc[epoch, f'{mode}_time'] = elapsed
  history.loc[epoch, 'rate'] = rate  
  if mode !='train':
    pred = np.clip(np.array(pred), 0, 4)
    lab = np.array(lab)
    try:
      kappa = cohen_kappa_score(pred, lab, weights='quadratic')
      kappa_mean = np.mean(batch_kappa)
      wandb.log({"Epoch":epoch,  f"{mode} Loss": running_loss/epoch_samples, f"{mode} Kappa Score":kappa, f"{mode} Mean Kappa Score":kappa_mean})
      plot_confusion_matrix(lab, pred, [i for i in range(5)])
      try:
        plot_heatmap(model, image_path, valid_df, val_aug, sz=sz)
        cam = cv2.imread('./heatmap.png', cv2.IMREAD_COLOR)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        wandb.log({"CAM": [wandb.Image(cam, caption="Class Activation Mapping")]})
      except:
        pass

      conf = cv2.imread('./conf.png', cv2.IMREAD_COLOR)
      conf = cv2.cvtColor(conf, cv2.COLOR_BGR2RGB)

      wandb.log({"Confusion Matrix": [wandb.Image(conf, caption="Confusion Matrix")]})
    except Exception as e:
      print(f"\033[91mException: {e}\033[91m")
      print('\033[91mMixed Precision\033[0m rendering nan value. Forcing \033[91mMixed Precision\033[0m to be False ...')
      batch_size = batch_size//2
      m_p = False
      accum_step = 2*accum_step

    lr_reduce_scheduler.step(running_loss)
    msg = f'{mode} Loss: {running_loss/epoch_samples:.4f} \n {mode} kappa: {kappa:.4f}'
    print(msg)
    
    history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
    history.loc[epoch, f'{mode}_kappa'] = kappa
    history.to_csv(f'{history_dir}/history_{model_name}_{sz}.csv', index=False)
    return running_loss/epoch_samples, kappa

# Hybrid model
plist = [ 
        {'params': model.module.backbone.parameters(),  'lr': learning_rate/10},
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
optimizer = optim.Adam(plist, lr=learning_rate)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
# cyclic_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[learning_rate/20, 3*learning_rate], epochs=n_epochs, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=20.0, final_div_factor=100.0, last_epoch=-1)
cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[learning_rate/10,  learning_rate], max_lr=[learning_rate/5, 2*learning_rate], step_size_up=5*len(train_loader), step_size_down=5*len(train_loader), mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[learning_rate/250, learning_rate/10, learning_rate/10, learning_rate/10], max_lr=[learning_rate/25, learning_rate, learning_rate, learning_rate], step_size_up=5*len(train_loader), step_size_down=5*len(train_loader), mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

# nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
# criterion = criterion_margin_focal_binary_cross_entropy
criterion = nn.MSELoss(reduction='sum')
# criterion = ArcFaceLoss().to(device)
# criterion = HybridLoss(alpha=2, beta=1).to(device)

# lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
# lr_finder.plot()


def main():
  prev_epoch_num = 0
  best_valid_loss = np.inf
  best_valid_kappa = 0.0

  if load_model:
    tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
    model.load_state_dict(tmp['model'])
    optimizer.load_state_dict(tmp['optim'])
    lr_reduce_scheduler.load_state_dict(tmp['scheduler'])
    cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
    scaler.load_state_dict(tmp['scaler'])
    prev_epoch_num = tmp['epoch']
    best_valid_loss = tmp['best_loss']
    best_valid_loss, best_valid_kappa = train_val(prev_epoch_num+1, valid_loader, optimizer=optimizer, rate=1, train=False, mode='val')
    del tmp
    print('Model Loaded!')
  
  for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())
    rate = 1
    if epoch < 20:
      rate = 1
    elif epoch>=20 and rate>0.65:
      rate = np.exp(-(epoch-20)/40)
    else:
      rate = 0.65

    train_val(epoch, train_loader, model, optimizer=optimizer, choice_weights=choice_weights, rate=rate, train=True, mode='train')
    wandb.log(params)
    try:
      valid_loss, valid_kappa = train_val(epoch, valid_loader, model, optimizer=optimizer, rate=1.00, train=False, mode='val')
      print("#"*20)
      print(f"Epoch {epoch} Report:")
      print(f"Validation Loss: {valid_loss :.4f} Validation kappa: {valid_kappa :.4f}")
      best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler':lr_reduce_scheduler.state_dict(), 
      
      'cyclic_scheduler':cyclic_scheduler.state_dict(), 
            'scaler': scaler.state_dict(),
      'best_loss':valid_loss, 'best_kappa':valid_kappa, 'epoch':epoch}
      best_valid_loss, best_valid_kappa = save_model(valid_loss, valid_kappa, best_valid_loss, best_valid_kappa, best_state, os.path.join(model_dir, model_name))
      print("#"*20)
    except Exception as e:
      print(f'\033[91mException: {e}\033[91m')
      print("Can not calculate Kappa Score. Moving on to the next epoch. ")
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  test_loss, test_kappa = train_val(-1, test_loader, model, optimizer=optimizer, rate=1.00, train=False, mode='test')
  print(test_loss, test_kappa)
   
if __name__== '__main__':
  main()


