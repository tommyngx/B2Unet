import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from models import UnetPlusPlus  # Update with your actual module and class names
from utils import get_validation_augmentation, get_preprocessing  # Update with your actual module names
from loss import DiceLoss  # Update with your actual loss class from loss.py
from configs.config import Config

def main():
    # Load configurations
    config_module = __import__("config_unetplusplus")  # Update with the correct configuration file
    model_config = config_module.Config()

    # Load test dataset
    test_dataset = Dataset(config=model_config, split='test', classes=model_config.CLASSES, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=model_config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Define the model, loss, and metrics
    model = UnetPlusPlus()  # Update with your actual model class
    model.load_state_dict(torch.load(model_config.MODEL_SAVE_PATH))  # Load the trained model weights
    model.eval()

    loss = DiceLoss()  # Update with your actual loss class from loss.py
    metrics = model_config.METRICS_TEST  # Update with your actual test metrics

    # Define test epoch
    test_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics,
        device=model_config.DEVICE,
        verbose=True,
    )

    # Run the test epoch
    test_logs = test_epoch.run(test_loader)

    # Print or save the test results
    print("Test Results:")
    for key, value in test_logs.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
