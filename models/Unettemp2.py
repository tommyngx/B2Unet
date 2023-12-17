import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetPlusPlus(nn.Module):
    def __init__(self, encoder_name='efficientnet-b7', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        # Add any additional post-processing if needed
        return torch.sigmoid(self.model(x))

class Unet(nn.Module):
    def __init__(self, encoder_name='efficientnet-b7', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'):
        super(Unet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        # Add any additional post-processing if needed
        return torch.sigmoid(self.model(x))
