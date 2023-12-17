#from models import Unet, UnetPlusPlus  # Update with your actual module and class names
from models.unet import Unet
from models.unetplusplus import UnetPlusPlus
import segmentation_models_pytorch as smp

class Config:
    def __init__(self, model_name="unetplusplus"):
        self.MODEL_NAME = "unetplusplus"

        # General
        self.PROJECT_NAME = "BreastCancerSegmentation"
        self.DEVICE = "cuda"  # or "cpu" if you want to run on CPU

        # Model
        self.ENCODER = "efficientnet-b7"
        self.ENCODER_WEIGHTS = "imagenet"
        self.CLASSES = ["cancer"]
        self.ACTIVATION = "sigmoid"

        # Training
        self.EPOCHS = 50
        self.BATCH_SIZE = 4
        self.LEARNING_RATE = 0.0001

        # Data
        self.DATA_DIR = "/content/CSAW_1class"  # Update with the path to your dataset directory
        self.IMAGES_DIR = f"{self.DATA_DIR}/images"
        self.MASKS_DIR = f"{self.DATA_DIR}/masks"
        self.TRAIN_CSV_PATH = f"{self.DATA_DIR}/dataset.csv"
        self.VALID_CSV_PATH = f"{self.DATA_DIR}/dataset.csv"  # Assuming you have a separate validation split
        self.TEST_CSV_PATH = f"{self.DATA_DIR}/dataset.csv"

        # Model Saving
        self.MODEL_SAVE_PATH = f"/content{self.PROJECT_NAME}_model.pth"

        # Metrics
        self.METRICS_TRAIN = self.get_metrics()
        self.METRICS_VALID = self.get_metrics()
        self.METRICS_TEST = self.get_metrics()

    def get_model(self):
        if self.MODEL_NAME == "unetplusplus":
            return UnetPlusPlus(
                encoder_name=self.ENCODER,
                encoder_weights=self.ENCODER_WEIGHTS,
                in_channels=3,
                classes=len(self.CLASSES),
                activation=self.ACTIVATION
            )
        elif self.MODEL_NAME == "unet":
            return Unet(
                encoder_name=self.ENCODER,
                encoder_weights=self.ENCODER_WEIGHTS,
                in_channels=3,
                classes=len(self.CLASSES),
                activation=self.ACTIVATION
            )
        # Add more conditions for other models as needed

    def get_metrics(self):
        if self.MODEL_NAME == "unetplusplus":
            return [
                smp.utils.metrics.IoU(threshold=0.5),
                smp.utils.metrics.Fscore(threshold=0.5),
                # Add more metrics as needed
            ]
        elif self.MODEL_NAME == "unet":
            return [
                smp.utils.metrics.IoU(threshold=0.5),
                smp.utils.metrics.Fscore(threshold=0.5),
                # Add more metrics as needed
            ]
            # Define metrics for the Unet model
            # ...
        # Add more conditions for other models as needed
