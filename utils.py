
# Update 30 Nov 2023
# Tommy bugs
# Miscellaneous utilities.
#
import os, pytz, time, shutil, requests, glob, re,sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from IPython.display import Image, display

def get_training_augmentation():
    train_transform = [
        albu.Resize(512, 512),
        albu.HorizontalFlip(p=0.5),
        # Add more augmentations as needed
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        # Add more augmentations as needed
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.Resize(512, 512),
        albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def preprocessing_fn(ENCODER, ENCODER_WEIGHTS):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return preprocessing_fn

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def visualize2(**images):
    """Display images."""
    for name, image in images.items():
        #print(name, image.shape)
        display(Image(image, width=300, height=300))


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def plot_training_history(train_history, valid_history, metric_name='IoU', plot_title='Training and Validation Metrics'):
    """Plot the training and validation history."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label=f'Training {metric_name}')
    plt.plot(valid_history, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(plot_title)
    plt.legend()
    plt.show()

def calculate_iou(pred, target, smooth=1e-5):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate evaluation metrics (IoU, F1 Score, Accuracy, etc.)."""
    predictions = (predictions > threshold).astype(np.uint8)
    targets = targets.astype(np.uint8)

    iou = calculate_iou(predictions, targets)
    f1_score = 2 * np.sum(predictions * targets) / (np.sum(predictions) + np.sum(targets))
    accuracy = np.sum(predictions == targets) / np.prod(targets.shape)

    return iou, f1_score, accuracy

def inference(model, dataloader, device):
    """Run model inference on a dataloader."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model.predict(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions


def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
