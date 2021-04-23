import os
from config import *
import random
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from gradcam.gradcam import GradCAM, GradCAMpp


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


def grad_cam_gen(model, img, mixed_precision = False, cam_layer_name='conv_head', device = 'cuda'):     
    configs = [dict(model_type='resnet', arch=model, layer_name=cam_layer_name)]
    # configs = [dict(model_type='resnet', arch=model, cam_layer_name='layer4')]
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

def plot_heatmap(model, path, valid_df, val_aug, random_id, crop=True, 
ben_color=False, device='cuda', cam_layer_name='conv_head', sz=384):
    
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
            image = grad_cam_gen(model.model.backbone, torch.unsqueeze(image, dim=0).cuda(), cam_layer_name=cam_layer_name)
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            plt.imshow(image)
            ax.set_title('Label: %s Prediction: %s' % (row['diagnosis'], 
            int(np.clip(np.round(np.ravel(prediction)[0]), 0, 4))), fontsize=40)
            plt.savefig(f'heatmap_{random_id}.png')

def plot_confusion_matrix(predictions, actual_labels, labels, random_id):
    cm = confusion_matrix(predictions, actual_labels, labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'conf_{random_id}.png')

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

