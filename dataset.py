import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    CLASSES = ['cancer'] 

    def __init__(
            self, 
            images_dir, 
            masks_dir,
            csv_path,
            split, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.df = pd.read_csv(csv_path)
        if split == 'train':
            self.df = self.df[self.df['split'] == 'train']
        elif split == 'test':
            self.df = self.df[self.df['split'] == 'test']

        self.ids = list(self.df['image_name'])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
