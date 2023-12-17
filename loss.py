import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        true_1_hot = F.one_hot(targets, num_classes).float()

        probs = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous()
        probs = probs.permute(0, 2, 3, 1).contiguous()

        intersection = torch.sum(probs * true_1_hot, dim=(0, 1, 2))
        cardinality = torch.sum(probs + true_1_hot, dim=(0, 1, 2))

        dice_loss = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - dice_loss.mean()
