import numpy as np


def dice_torch_batch(y_pred, true, threshold=0.5, reduction="mean"):
    batch_size = y_pred.shape[0]
    p = y_pred.view(batch_size, -1) > threshold
    t = true.view(batch_size, -1)
    true_zero_mask = p.sum(dim=1) == 0
    pred_zero_mask = t.sum(dim=1) == 0
    ones_mask = true_zero_mask * pred_zero_mask
    dice = (2 * (p * t).sum(dim=1)) / (p.sum(dim=1) + t.sum(dim=1) + 1e-4)
    dice[ones_mask] = 1
    if reduction == "mean":
        return dice.mean().item()
    elif reduction == "numpy":
        return dice.cpu().data.numpy()
    else:
        return dice


def dice_numpy(y_pred, true, threshold=0.5):
    p = (y_pred.reshape(-1,) > threshold).astype(np.uint8)
    t = true.reshape(-1,)
    if p.sum() == 0 and t.sum() == 0:
        return 1
    dice = (2 * (p * t).sum()) / (p.sum() + t.sum() + 1e-4)
    return dice
