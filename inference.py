import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import Dataset  # Import your Dataset class
from loss import DiceLoss  # Import your custom loss or use smp.utils.losses.DiceLoss
#from configs import Config  # Import your Config class
from utils import visualize  # Import your visualization function
import importlib
from utils import get_training_augmentation,get_validation_augmentation, get_preprocessing, visualize
import numpy as np

def main(config_file, model_save_link):
    # Load config
    config_module = importlib.import_module(f'configs.{args.config}')
    model_config  = config_module.Config()

    ENCODER = model_config.ENCODER
    ENCODER_WEIGHTS=model_config.ENCODER_WEIGHTS
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load model
    #checkpoint = torch.load(f'{args.model_path}', map_location=model_config.DEVICE)
    #best_model = model_config.get_model()
    #best_model.load_state_dict(checkpoint)
    #best_model = checkpoint['model']
    best_model = torch.load(f'{args.model_path}')
    best_model.to(model_config.DEVICE)

    # Load test dataset
    test_dataset = Dataset(
        images_dir=model_config.IMAGES_DIR,
        masks_dir=model_config.MASKS_DIR,
        csv_path=model_config.TEST_CSV_PATH,
        split="test",
        classes=model_config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)

    # Define metrics for evaluation
    metrics_all = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
    ]

    # Create test epoch
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=DiceLoss(),  # Use your custom loss or smp.utils.losses.DiceLoss
        metrics=metrics_all,
        device=model_config.DEVICE,
    )

    test_dataset_vis = Dataset(
            images_dir=model_config.IMAGES_DIR,
            masks_dir=model_config.MASKS_DIR,
            csv_path=model_config.TEST_CSV_PATH,
            split="test",
            classes=model_config.CLASSES,
            augmentation=get_validation_augmentation(),
)

    # Run evaluation on the test set
    #logs = test_epoch.run(test_dataloader)

    # Visualize results
    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(model_config.DEVICE).unsqueeze(0)
        best_model.eval()
        with torch.no_grad():
            pr_mask = best_model(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        #print(image_vis.shape, gt_mask.shape, pr_mask.shape)
        visualize2(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the segmentation model on the test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder where models are saved")
    args = parser.parse_args()

    main(args.config, args.model_path)
