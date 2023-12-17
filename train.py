import argparse
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from utils import get_training_augmentation, get_preprocessing, visualize
from models import Unet, UnetPlusPlus  # Update with your actual module and class names
from configs.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", type=str, default="config_unetplusplus.py", help="Path to the configuration file")
    parser.add_argument("--model", type=str, default="unetplusplus", help="Name of the model to use")
    return parser.parse_args()

def main():
    # Load configurations
    args = parse_args()
    config_module = __import__(args.config.replace(".py", ""))
    model_config = config_module.Config()

    # Load train and validation datasets
    train_dataset = Dataset(
        images_dir=model_config.IMAGES_DIR,
        masks_dir=model_config.MASKS_DIR,
        csv_path=model_config.TRAIN_CSV_PATH,
        split="train",
        classes=model_config.CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(model_config.get_preprocessing())
    )

    valid_dataset = Dataset(
        images_dir=model_config.IMAGES_DIR,
        masks_dir=model_config.MASKS_DIR,
        csv_path=model_config.VALID_CSV_PATH,
        split="valid",
        classes=model_config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(model_config.get_preprocessing())
    )

    # Load model
    model_config.MODEL_NAME = args.model
    model = model_config.get_model()
    model.to(model_config.DEVICE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)
    loss_fn = model_config.get_loss()

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
    for epoch in range(1, model_config.EPOCHS + 1):
        print(f"\nEpoch: {epoch}/{model_config.EPOCHS}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Do something (save model, change lr, etc.)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{model_config.MODEL_SAVE_PATH}_{epoch}.pth")

if __name__ == "__main__":
    main()
