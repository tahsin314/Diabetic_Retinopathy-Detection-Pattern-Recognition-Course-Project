import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from torchvision.utils import save_image
from config import *
from DRDataset import *
from utils import *

denorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def visualize(original_image):
    fontsize = 18
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(8,8,1)
    plt.axis('off')
    ax.imshow(original_image, cmap='gray')
    for i in range(63):
        augment = train_aug(image = image)
        aug_img = augment['image']
        ax = fig.add_subplot(8,8,i+2)
        plt.axis('off')
        ax.imshow(aug_img, cmap='gray')
    fig.savefig('aug.png')

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


train_ds = DRDataset(df.image_id.values, df.diagnosis.values, target_type=target_type, crop=False, ben_color=True, dim=sz, transforms=train_aug)
train_loader = DataLoader(train_ds,batch_size=64, shuffle=False, num_workers=4)
_, im, _= iter(train_loader).next()
# print(im.size(), torch.max(denorm(im)))
save_image(im.float(), 'Aug.png', nrow=8, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)