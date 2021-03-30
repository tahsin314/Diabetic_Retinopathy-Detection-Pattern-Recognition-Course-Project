import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import pandas as pd
from tqdm import tqdm as T
import torch
from tfrecord.torch.dataset import TFRecordDataset
import cv2
import gc
from p_tqdm import p_map

def decode_image(features):
    # get BGR image from bytes
    features["image"] = cv2.imdecode(features["image"], -1)
    return features

data_dir = 'data'
tfrec_dir = f'{data_dir}/tfrecords'
train_dir = f"{data_dir}/malignant"
test_dir = f"{data_dir}/test_768"
filelist = os.listdir(tfrec_dir)
filelist = [f for f in filelist if '.tfrec' in f]
attributes = ['patient_id', 'target', 'sex', 'width', 'height', 'age_approx', 'anatom_site_general_challenge', 'diagnosis', 'image_name', 'image']
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
train = pd.DataFrame()
test = pd.DataFrame()

train_row = 0
test_row = 0

# Creating index files
for i in T(filelist):
    tfrec_file = os.path.join(tfrec_dir, i)
    index_file = os.path.join(tfrec_dir, i).replace('.tfrec', '.index')
    if not os.path.exists(index_file):
        os.system(f"python3 -m tfrecord.tools.tfrecord2idx {tfrec_file} {index_file}")

def tfrec_extract(filename):
    global train_row
    global test_row
    global train
    global test
    tfrecord_path = os.path.join(tfrec_dir, filename)
    index_path = tfrecord_path.replace('.tfrec', '.index')
    
    if 'train' in filename:
        savedir = train_dir
    else:savedir = test_dir
    dataset = TFRecordDataset(tfrecord_path, index_path, transform=decode_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for data in T(loader):
        # print(len(loader))
        if 'train' in filename:
            train_row += 1
        else: test_row += 1
        img_name = data['image_name'].squeeze().data.cpu().numpy().copy()
        img_name = os.path.join(savedir, ''.join(map(chr, img_name)))
        img_name += '.jpg'
        image_file = data['image'].squeeze().data.cpu().numpy()
        cv2.imwrite(img_name, image_file)
        del data['image']
        del data['image_name']
        for k, v in data.items():
            if 'train' in filename:
                train.loc[train_row, 'image_name'] = img_name
                train.loc[train_row, k] = v.squeeze().data.cpu().numpy()
                train.loc[train_row, 'tfrec'] = filename.replace('.tfrec', '')
            else:
                test.loc[test_row, 'image_name'] = img_name
                test.loc[test_row, k] = v.squeeze().data.cpu().numpy()
                test.loc[test_row, 'tfrec'] = filename.replace('.tfrec', '')


# p_map(tfrec_extract, filelist)
for f in T(filelist):
    tfrec_extract(f)

train.to_csv(f"{data_dir}/malignant.csv", index=False)
print(f"total {train_row} train images")
# test.to_csv(f"{data_dir}/test_768.csv", index=False)
# print(f"total {train_row} train images and {test_row} test images")