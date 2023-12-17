import torch.nn as nn
import segmentation_models_pytorch as smp

class Unet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes, activation):
        super(Unet, self).__init__()

        # Use the provided encoder and other parameters
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)
