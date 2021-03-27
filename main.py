from functools import partial
from albumentations.augmentations import transforms

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

from src.datasets.dataset import ImageDataset, ImageDatasetV2
from src.datasets.zarr_dataset import ZarrTrainDataset, ZarrValidDataset
from src.loops.loops import train, validation, validation_full_image
from src.transforms.transform import (
    base_transform,
    valid_transform,
    baseline_aug,
    baseline_aug_v2,
)
from src.utils.checkpoint import CheckpointHandler
from src.utils.utils import IMAGE_SIZES
import argparse

FOLD_IMGS = {
    0: ["4ef6695ce", "0486052bb", "2f6ecfcdf"],
    1: ["c68fe75ea", "095bf7a1f", "aaa6a05cc"],
    2: ["afa5e8098", "1e2425f28", "b2dc8411c"],
    3: ["cb2d976f4", "8242609fa", "54f2eec69"],
    4: ["26dc41664", "b9a3865fc", "e79de561c"],
}


def _get_loader(dataset, batch_size, num_workers, sampler=None, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        sampler=sampler,
    )


def main(args):
    FOLD = int(args.fold)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 10
    START_LR = 0.001
    EPOCH = args.epoch
    CROP_SIZE = args.crop_size
    TRAIN_IMG_SIZE = args.train_img_size
    WEIGHT_PATH = f"./crop_4096_1024/fold_{FOLD}"
    ITERS = 100
    
    df = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train.csv").set_index("id", drop=True)
    input_path = "../input/zarr_train"

    train_img_ids = [
        x for fold, fold_imgs in FOLD_IMGS.items() for x in fold_imgs if fold != FOLD
    ]
    val_img_ids = FOLD_IMGS[FOLD]

    print(f"FOLD: {FOLD}")

    get_loader = partial(
        _get_loader, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
    )
    train_loader = get_loader(
        ZarrTrainDataset(
            img_ids=train_img_ids,
            img_path=input_path,
            transform=baseline_aug(TRAIN_IMG_SIZE),
            iterations=ITERS*BATCH_SIZE,
        )
    )
    model = smp.Unet("resnet34").cuda()
    # model.load_state_dict(torch.load("./first_launch/epoch_7_score_0.9234.pth"))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=START_LR)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=5, verbose=True
    )
    cp_handler = CheckpointHandler(model, WEIGHT_PATH, 5)

    for e in range(1, EPOCH + 1):
        metrics_train = train(train_loader, model, optimizer, loss_fn, scaler)
        images_dice = {}
        dice_mean = []
        val_loss = []
        for img_id in val_img_ids:

            val_loader = get_loader(
                ZarrValidDataset(
                    img_id,
                    img_path=input_path,
                    transform=valid_transform(CROP_SIZE),
                    crop_size=CROP_SIZE,
                    step=CROP_SIZE,
                )
            )
            metrics_val = validation_full_image(val_loader, model, loss_fn, rle=df.loc[img_id, "encoding"])
            images_dice[img_id] = metrics_val["dice_full"]
            dice_mean.append(metrics_val["dice_mean"])
            val_loss.append(metrics_val["loss_val"])
            del val_loader

        image_dice_mean = np.mean(list(images_dice.values()))
        dice_mean = np.mean(dice_mean)
        val_loss = np.mean(val_loss)
        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {val_loss:.4f}; "
        log += f"avg_dice: {dice_mean:.4f}; full_mask_dice: {image_dice_mean:.4f} "
        print(log, end="")
        cp_handler.update(e, image_dice_mean)
        scheduler.step(image_dice_mean)
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_save_path", default="./", type=str, help="valid path")

    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--epoch", type=int, default=60, help="total epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--crop_size", type=int, default=1024, help="batch size")
    parser.add_argument("--train_img_size", type=int, default=1024, help="batch size")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    main(args)

