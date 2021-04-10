from torch.nn.modules.loss import _Loss
import torch.nn as nn
from pytorch_toolbelt.losses import DiceLoss


class MyLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary", log_loss=True)

    def forward(self, *input):
        return self.bce_loss(*input) + self.dice_loss(*input)
