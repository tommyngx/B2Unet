
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

class DiceLoss(smp.utils.losses.DiceLoss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    # You can add any additional customization or logic here if needed
