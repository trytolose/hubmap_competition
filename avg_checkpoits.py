from typing import List
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from src.datasets.zarr_dataset import ZarrValidDataset
from src.loops.loops import validation_full_image
from src.transforms.transform import valid_transform
from src.datasets.dataset import ImageDatasetV2
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd

FOLD_IMGS = {
    0: ["4ef6695ce", "0486052bb", "2f6ecfcdf"],
    1: ["c68fe75ea", "095bf7a1f", "aaa6a05cc"],
    2: ["afa5e8098", "1e2425f28", "b2dc8411c"],
    3: ["cb2d976f4", "8242609fa", "54f2eec69"],
    4: ["26dc41664", "b9a3865fc", "e79de561c"],
}
BATCH_SIZE = 24
TRAIN_IMG_SIZE = 1024
CROP_SIZE = 1024 * 4
# input_path = Path("../input/train_v3_4096_1024/images")
input_path = "../input/zarr_train_orig"

df = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train.csv").set_index("id", drop=True)


def get_score(model, fold):
    dice_mean = []
    for img_id in FOLD_IMGS[fold]:

        val_ds = ZarrValidDataset(
            img_id,
            img_path=input_path,
            transform=valid_transform(TRAIN_IMG_SIZE),
            crop_size=CROP_SIZE,
            step=CROP_SIZE,
        )

        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )
        metrics_val = validation_full_image(
            val_loader,
            model,
            torch.nn.BCEWithLogitsLoss(),
            rle=df.loc[img_id, "encoding"],
        )
        dice_mean.append(metrics_val["dice_full"])
        del val_loader

    image_dice_mean = np.mean(dice_mean)
    return image_dice_mean


def average_weights(state_dicts: List[dict]):
    everage_dict = OrderedDict()
    for k in state_dicts[0].keys():
        everage_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(
            state_dicts
        )
    return everage_dict


def avg(model, fold, checkpoints_weights_paths):
    all_weights = [
        torch.load(path, map_location="cuda") for path in checkpoints_weights_paths
    ]
    best_score = 0
    best_weights = []
    for w in all_weights:
        current_weights = best_weights + [w]
        average_dict = average_weights(current_weights)
        model.load_state_dict(average_dict)
        score = get_score(model, fold)
        print(score)
        if score > best_score:
            best_score = score
            best_weights.append(w)
    return best_score, best_weights


def get_avg_checkpoint(weights_path, fold=0):
    val_img_paths = [
        str(img)
        for img in Path(input_path).glob("*.png")
        if img.stem.split("_")[0] in FOLD_IMGS[fold]
    ]
    model = smp.Unet("resnet34").cuda()
    weights_paths = sorted(
        list((Path(weights_path) / f"fold_{fold}").glob("*.pth")),
        key=lambda x: x.stem.split("_")[-1],
        reverse=True,
    )
    [print(w) for w in weights_paths]
    best_score, best_weights = avg(model, fold, weights_paths)
    w_avg = average_weights(best_weights)
    torch.save(
        w_avg, Path(weights_path) / f"fold_{fold}/fold_{fold}_avg_{best_score:.4f}.pth"
    )


if __name__ == "__main__":
    weights_path = "./weights/zarr_full_image_val"
    for i in range(1):
        print("FOLD:", i)
        get_avg_checkpoint(weights_path, i)
