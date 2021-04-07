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
    FOLD = args.fold
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 10
    START_LR = 0.001
    EPOCH = args.epoch
    CROP_SIZE = args.crop_size
    TRAIN_IMG_SIZE = args.train_img_size
    WEIGHT_PATH = f"./weights/crop_4096_1024_old_loader/fold_{FOLD}"

    # df = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train_v1_1024/split_v2.csv")
    # input_path = Path("/hdd/kaggle/hubmap/input_v2/train_v2_2048/images")
    input_path = Path("../input/train_v3_4096_1024/images")

    train_img_paths = [
        str(img)
        for img in Path(input_path).glob("*.png")
        if img.stem.split("_")[0] not in FOLD_IMGS[FOLD]
    ]
    val_img_paths = [
        str(img)
        for img in Path(input_path).glob("*.png")
        if img.stem.split("_")[0] in FOLD_IMGS[FOLD]
    ]

    print(f"FOLD: {FOLD}")

    get_loader = partial(
        _get_loader, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
    )
    train_loader = get_loader(
        ImageDatasetV2(train_img_paths, baseline_aug(TRAIN_IMG_SIZE)), sampler=None,
    )
    val_loader = get_loader(
        ImageDatasetV2(val_img_paths, valid_transform(TRAIN_IMG_SIZE)), shuffle=False,
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
        metrics_val = validation(val_loader, model, loss_fn)
        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {metrics_val['loss_val']:.4f}; "
        log += f"dice_mean: {metrics_val['dice_mean']:.4f}; dice_pos: {metrics_val['dice_pos']:.4f}; "
        log += f"dice_neg: {metrics_val['dice_neg']:.4f} "
        print(log, end="")
        cp_handler.update(e, metrics_val["dice_mean"])
        scheduler.step(metrics_val["dice_mean"])
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_save_path", default="./", type=str, help="valid path")

    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--epoch", type=int, default=60, help="total epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--crop_size", type=int, default=4096, help="batch size")
    parser.add_argument("--train_img_size", type=int, default=1024, help="batch size")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    main(args)
