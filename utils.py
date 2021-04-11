import os
from config import *
import random
import numpy as np
import cv2
import pandas as pd 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F_alb
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from gradcam import GradCAM, GradCAMpp
# from gradcam.utils import visualize_cam


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)

def visualize_cam(mask, img, alpha=0.8, beta=0.15):
    
    """
    Courtesy: https://github.com/vickyliin/gradcam_plus_plus-pytorch/blob/master/gradcam/utils.py
    Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()*beta
    result = result.div(result.max()).squeeze()

    return heatmap, result


def grad_cam_gen(model, img, mixed_precision = False, device = 'cuda'):     
    configs = [dict(model_type='resnet', arch=model, layer_name='conv_head')]
    # configs = [dict(model_type='resnet', arch=model, layer_name='layer4')]
    for config in configs:
        config['arch'].to(device).eval()
    # print(config['arch'])
    cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs]

    for _, gradcam_pp in cams:
        mask_pp, _ = gradcam_pp(img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, img)
        result_pp = result_pp.cpu().numpy()
        #convert image back to Height,Width,Channels
        result_pp = np.transpose(result_pp, (1,2,0))
        return result_pp/np.max(result_pp)

def plot_heatmap(model, path, valid_df, val_aug, crop=True, 
ben_color=False, device='cuda', layer_name='conv_head', sz=384):
    
    fig = plt.figure(figsize=(70, 56))
    valid_df['path'] = valid_df['image_id'].map(lambda x: x)
    for class_id in sorted(valid_df['diagnosis'].unique()):
        for i, (idx, row) in enumerate(valid_df.loc[valid_df['diagnosis'] == class_id].sample(5, random_state=42).iterrows()):
            ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
            path=f"{row['path']}"
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if crop:
                image = crop_image_ilovescience(image)
            if ben_color:
                image = load_ben_color(image, sz)
            image = cv2.resize(image, (sz, sz))
            aug = val_aug(image=image)
            image = aug['image'].reshape(sz, sz, 3).transpose(2, 0, 1)
            image = torch.FloatTensor(image)
            prediction = model(torch.unsqueeze(image.to(device), dim=0))
            prediction = prediction.data.cpu().numpy()
            image = grad_cam_gen(model.backbone, torch.unsqueeze(image, dim=0).cuda(), layer_name=layer_name)
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            plt.imshow(image)
            ax.set_title('Label: %s Prediction: %s' % (row['diagnosis'], int(np.clip(np.round(np.ravel(prediction)[0]), 0, 4))))
            plt.savefig('heatmap_0.png')

def plot_confusion_matrix(predictions, actual_labels, labels):
    cm = confusion_matrix(predictions, actual_labels, labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('conf_0.png')

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


def crop_image_ilovescience(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        flag = 0
        return image, flag
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x); y = int(y); r = int(r)
    flag = 1
    #print(x,y,r)
    if r > 100:
        return output[0 + (y-r)*int(r<y):-1 + (y+r+1)*int(r<y),0 + (x-r)*int(r<x):-1 + (x+r+1)*int(r<x)]
    else:
        print('none!')
        flag = 0
        return image,flag

def load_ben_color(image, IMG_SIZE):
    sigmaX=10
    image = crop_image_ilovescience(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


def Messidor_Process(dirname):
    bases = [11, 12, 14, 21, 22, 23, 24, 31, 32, 33]
    df_messidor = pd.DataFrame()
    for i in bases:
        train_dir_messidor = os.path.join(dirname,'Base '+str(i))
        csvfiles = os.listdir(dirname)
        df_tmp = pd.DataFrame()
        for f in csvfiles:
            if str(i) in f and 'csv' in f:
                df_tmp = pd.read_csv(os.path.join(dirname, f), encoding = "ISO-8859-1")
                break
        col_tmp = list(df_tmp.columns)
        df_tmp.drop(columns = [col_tmp[1]])
        df_tmp['image_id'] = df_tmp['Image name'].map(lambda x: os.path.join(train_dir_messidor, x))
        df_tmp['diagnosis'] = df_tmp['Retinopathy grade']
        df_tmp.drop(columns = col_tmp, axis=1, inplace=True)
        df_tmp = df_tmp.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle dataframe
        df_messidor = pd.concat([df_messidor, df_tmp], ignore_index=True)
    return df_messidor

def IDRID_Process(dirname):
    image_dirname = os.path.join(dirname, 'B.%20Disease%20Grading/B. Disease Grading/1. Original Images')
    gt_dirname = os.path.join(dirname, 'B.%20Disease%20Grading/B. Disease Grading/2. Groundtruths')
    df_train = pd.read_csv(f'{gt_dirname}/a. IDRiD_Disease Grading_Training Labels.csv')
    df_test = pd.read_csv(f'{gt_dirname}/b. IDRiD_Disease Grading_Testing Labels.csv')
    df_train['image_id'] = df_train['Image name'].map(lambda x: f"{image_dirname}/a. Training Set/{x}.jpg")
    df_test['image_id'] = df_test['Image name'].map(lambda x: f"{image_dirname}/b. Testing Set/{x}.jpg")
    df_idrid = pd.concat([df_train, df_test], ignore_index=True)
    df_idrid['diagnosis'] = df_idrid['Retinopathy grade']
    df_idrid = df_idrid[['image_id', 'diagnosis']]
    df_idrid = df_idrid.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle dataframe
    return df_idrid

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets, shuffled_targets, lam]
    return data, targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets, shuffled_targets, lam]
    return data, targets

def cutmix_criterion(preds, targets, criterion, rate=0.7):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    return lam * ohem_loss(rate, criterion, preds, targets1) + (1 - lam) * ohem_loss(rate, criterion, preds, targets2)
    # return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def mixup_criterion(preds, targets, criterion, rate=0.7):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    return lam * ohem_loss(rate, criterion, preds, targets1) + (1 - lam) * ohem_loss(rate, criterion, preds, targets2)
    # return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view).cuda()
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)

def ohem_loss(rate, base_crit, cls_pred, cls_target):

    batch_size = cls_pred.size(0) 
    ohem_cls_loss = base_crit(cls_pred, cls_target)
    if rate==1:
        return ohem_cls_loss.sum()
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min((sorted_ohem_loss.size())[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss * batch_size

def save_model(valid_loss, valid_kappa, best_valid_loss, best_valid_kappa, best_state, savepath):
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}. Saving checkpoint')
        torch.save(best_state, savepath+'_loss.pth')
        best_valid_loss = valid_loss
    if valid_kappa>best_valid_kappa:
        print(f'Validation kappa has increased from:  {best_valid_kappa:.4f} to: {valid_kappa:.4f}. Saving checkpoint')
        torch.save(best_state, savepath + '_kappa.pth')
        best_valid_kappa = valid_kappa
    else:
        torch.save(best_state, savepath + '_last.pth')
    return best_valid_loss, best_valid_kappa 
