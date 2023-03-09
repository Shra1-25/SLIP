import os
import pandas as pd
import shutil
from tqdm import tqdm
from datasets import pil_loader
import numpy as np

root = "/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/"
img_names = os.listdir(root + 'full_data')

train_meta = pd.read_csv(root + 'train_split_metadata.csv')
val_meta = pd.read_csv(root + 'val_split_metadata.csv')
test_meta = pd.read_csv(root + 'test_data.csv')

# for folder, meta in {'train':train_meta, 'val':val_meta, 'test':test_meta}.items():
#     print('Splitting '+folder+' data:')
#     for name in tqdm(meta['image_name']):
#         if name or (name+'.jpg') in img_names:
#             if folder=='test':
#                 shutil.copy(root+'full_data/'+name+'.jpg', root+folder+'/')
#             else:
#                 shutil.copy(root+'full_data/'+name, root+folder+'/')
        
#         else:
#             print('File ',name,'not found in original dataset.')
#             exit()
all_means = np.zeros((len(train_meta),3))
all_stds = np.zeros((len(train_meta),3))
for idx, name in enumerate(tqdm(train_meta['image_name'])):
    img = pil_loader(root+'full_data/'+name)
    all_means[idx] = np.mean(np.mean(img,0),0)
    all_stds[idx] = np.std(np.std(img, 0), 0)
print(np.mean(all_means,0))
print(np.std(all_stds,0))


