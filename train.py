import argparse
import importlib
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from utils import get_training_augmentation,get_validation_augmentation, get_preprocessing, visualize
from models.unetplusplus import UnetPlusPlus
from models.unet import Unet
import segmentation_models_pytorch as smp
from loss import DiceLoss
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model.")
    parser.add_argument("--config", type=str, default="config_unetplusplus", help="Name of the configuration file without extension")
    parser.add_argument("--model", type=str, default="unetplusplus", help="Name of the model to use")
    return parser.parse_args()

def main():
    # Load configurations
    args = parse_args()
    # Dynamically import the specified configuration module
    config_module = importlib.import_module(f'configs.{args.config}')
    model_config  = config_module.Config()

    ENCODER = model_config.ENCODER
    ENCODER_WEIGHTS=model_config.ENCODER_WEIGHTS
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load train and validation datasets
    train_dataset = Dataset(
        images_dir=model_config.IMAGES_DIR,
        masks_dir=model_config.MASKS_DIR,
        csv_path=model_config.TRAIN_CSV_PATH,
        split="train",
        classes=model_config.CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        images_dir=model_config.IMAGES_DIR,
        masks_dir=model_config.MASKS_DIR,
        csv_path=model_config.VALID_CSV_PATH,
        split="test",
        classes=model_config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Load model
    model = model_config.get_model()
    print(f"Model Configuration: {model_config.MODEL_NAME}")
    model.to(model_config.DEVICE)


    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)
    loss_fn = DiceLoss()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=model_config.BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=model_config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Create epoch runners
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss_fn,
        metrics=model_config.METRICS_TRAIN,
        optimizer=optimizer,
        device=model_config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss_fn,
        metrics=model_config.METRICS_VALID,
        device=model_config.DEVICE,
        verbose=True,
    )

    # Training loop
    now = datetime.datetime.now()

    max_score = 0
    for epoch in range(1, model_config.EPOCHS + 1):
        print(f"\nEpoch: {epoch}/{model_config.EPOCHS}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f"{model_config.MODEL_SAVE_PATH}_{now.strftime('%Y%m%d')}.pth")
            print('Model saved!')

        if epoch == 20:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

if __name__ == "__main__":
    main()
