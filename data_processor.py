from config import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

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

