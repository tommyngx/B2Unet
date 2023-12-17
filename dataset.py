import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from configs.config import Config  # Import a generic Config class

class Dataset(BaseDataset):
    def __init__(self, config, split, classes=None, augmentation=None, preprocessing=None):
        self.config = config  # Use the provided Config instance

        if split == 'train':
            df = pd.read_csv(self.config.TRAIN_CSV_PATH)
        elif split == 'valid':
            df = pd.read_csv(self.config.VALID_CSV_PATH)
        else:
            raise ValueError("Invalid split. Use 'train' or 'valid'.")

        self.ids = list(df['image_name'])
        self.images_fps = [os.path.join(self.config.IMAGES_DIR, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(self.config.MASKS_DIR, image_id) for image_id in self.ids]

        self.class_values = [classes.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

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
