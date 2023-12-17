import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from configs.config import Config  # Import a generic Config class

class Dataset(BaseDataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.CLASSES = self.config.CLASSES

        self.df = pd.read_csv(self.config.CSV_PATH)
        if split == 'train':
            self.df = self.df[self.df['split'] == 'train']
        elif split == 'test':
            self.df = self.df[self.df['split'] == 'test']

        self.ids = list(self.df['image_name'])
        self.images_fps = [os.path.join(self.config.IMAGES_DIR, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(self.config.MASKS_DIR, image_id) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.config.CLASSES]
        self.augmentation = self.config.get_training_augmentation() if split == 'train' else self.config.get_validation_augmentation()
        self.preprocessing = self.config.get_preprocessing()

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
