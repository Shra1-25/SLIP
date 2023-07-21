# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets

import utils
import pandas as pd


ImageFile.LOAD_TRUNCATED_IMAGES = True
# diagnosis_map = {"NV":0, "SCC":1, "BKL":2, "AK":3, "BCC":4, "MEL":5, "DF":6, "VASC":7}
diagnosis_map = {"MALIGNANT":0, "BENIGN":1, "BENIGN_WITHOUT_CALLBACK":2}

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')

class ISICValDataset(torch.utils.data.Dataset):
    def __init__(self, val_transform, root, val_path, context_length=26):
        annotations = pd.read_csv(val_path)
        self.samples = [(annotations.loc[i,'image_name'], annotations.loc[i, 'description'], annotations.loc[i,'diagnosis']) for i in range(len(annotations))]
        self.root = root
        self.transform = val_transform
        # self.tokenizer = tokenizer 
        self.context_length=context_length
    def __getitem__(self, i):
        image_id, caption, target = self.samples[i]
        path = os.path.join(self.root, 'full_data/', image_id)
        img = pil_loader(path)
        image = self.transform(img)
        target = diagnosis_map[target]
        # caption = self.tokenizer.encode_plus(caption, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')
        
        return image, caption, target
    def __len__(self):
        return len(self.samples)

class ISICE2ETrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform, root, val_path, tokenizer, context_length=26):
        annotations = pd.read_csv(val_path)
        self.samples = [(annotations.loc[i,'image_name'], annotations.loc[i, 'description'], annotations.loc[i,'diagnosis']) for i in range(len(annotations))]
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer 

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # normalize = transforms.Normalize(mean=[170.611, 134.134, 132.450], std=[10.039, 8.356, 8.342])
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.context_length = context_length

    def __getitem__(self, i):
        image_id, caption, target = self.samples[i]
        path = os.path.join(self.root, 'full_data/', image_id)
        img = pil_loader(path)
        image = self.transform(img)
        target = diagnosis_map[target]
        caption = self.tokenizer.encode_plus(caption, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')
        
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return image, caption, target, aug1, aug2
    def __len__(self):
        return len(self.samples)

class CBISValDataset(torch.utils.data.Dataset):
    def __init__(self, val_transform, root, val_path, context_length=26):
        annotations = pd.read_csv(val_path)
        self.samples = [(annotations.loc[i,'image_path'], annotations.loc[i, 'description'], annotations.loc[i,'pathology']) for i in range(len(annotations))]
        self.root = root
        self.transform = val_transform
        # self.tokenizer = tokenizer 
        self.context_length=context_length
    def __getitem__(self, i):
        image_id, caption, target = self.samples[i]
        path = os.path.join(self.root, image_id)
        img = pil_loader(path)
        image = self.transform(img)
        target = diagnosis_map[target]
        # caption = self.tokenizer.encode_plus(caption, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')
        
        return image, caption, target
    def __len__(self):
        return len(self.samples)

class CBISE2ETrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform, root, val_path, tokenizer, context_length=26):
        annotations = pd.read_csv(val_path)
        self.samples = [(annotations.loc[i,'image_path'], annotations.loc[i, 'description'], annotations.loc[i,'pathology']) for i in range(len(annotations))]
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer 

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # normalize = transforms.Normalize(mean=[170.611, 134.134, 132.450], std=[10.039, 8.356, 8.342])
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.context_length = context_length

    def __getitem__(self, i):
        image_id, caption, target = self.samples[i]
        path = os.path.join(self.root, image_id)
        img = pil_loader(path)
        image = self.transform(img)
        target = diagnosis_map[target]
        caption = self.tokenizer.encode_plus(caption, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')
        
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return image, caption, target, aug1, aug2
    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]
        elif self.dataset == 'isic':
            annotations = pd.read_csv(root+'train_split_metadata.csv')
            self.samples = [(annotations.loc[i,'image_name'], annotations.loc[i,'description']) for i in range(len(annotations))]
        elif self.dataset == 'cbis':
            annotations = pd.read_csv(root+'train_split_metadata.csv')
            self.samples = [(annotations.loc[i,'image_path'], annotations.loc[i,'description']) for i in range(len(annotations))]

    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, title, desc = self.samples[i]
            caption = np.random.choice([title, desc])
            img = yfcc_loader(self.root, index)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc3m':
            ann = self.samples[i]
            filename, captions = ann['image_id'], ann['captions']
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc12m':
            ann = self.samples[i]
            filename, captions = ann['image_name'], ann['captions']
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)
        elif self.dataset == 'isic':
            image_id, caption = self.samples[i]
            path = os.path.join(self.root, 'full_data/', image_id)
            img = pil_loader(path)
            # print(img.shape, caption.shape)
        elif self.dataset == 'cbis':
            image_id, caption = self.samples[i]
            path = os.path.join(self.root, image_id)
            img = pil_loader(path)
        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer.encode_plus(caption, max_length=26, padding='max_length', truncation=True, return_tensors='pt')

        return image, caption['input_ids'][0]


class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, tokenizer=None, context_length=26):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)
        
        if self.tokenizer is not None:
            # caption = self.tokenizer(caption)
            caption = self.tokenizer.encode_plus(caption, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')
        
        # return image, caption['input_ids'][0], aug1, aug2
        return image, caption['input_ids'][0], aug1, aug2

class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment):
        super().__init__(dataset, root, metadata)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset


def get_dataset(train_transform, tokenizer, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # normalize = transforms.Normalize(mean=[170.611, 134.134, 132.450], std=[10.039, 8.356, 8.342])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.model.startswith('SIMCLR'):
        return ImageCaptionDatasetSSL(args.dataset, args.root, args.metadata, augment)
    elif args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer)
    elif args.model.startswith('SLIP'):
        return ImageCaptionDatasetSLIP(args.dataset, args.root, args.metadata, train_transform, augment, tokenizer, context_length=args.context_length)