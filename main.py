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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _get_loader(dataset, batch_size, num_workers, sampler=None, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )


def main(args):
    FOLD = int(args.fold)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 10
    START_LR = 0.001
    EPOCH = args.epoch
    CROP_SIZE = args.crop_size
    TRAIN_IMG_SIZE = args.train_img_size
    WEIGHT_PATH = f"./weights/zarr_full_image_val_no_pdf/fold_{FOLD}"
    ITERS = 100
    IS_OLD_VALIDATION = False

    df = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train.csv").set_index("id", drop=True)
    input_path = "../input/zarr_train_orig"
    pdf_path = "../input/zarr_pdf"
    crop_img_path = Path("../input/train_v3_4096_1024/images")

    train_img_ids = [
        x for fold, fold_imgs in FOLD_IMGS.items() for x in fold_imgs if fold != FOLD
    ]
    val_img_ids = FOLD_IMGS[FOLD]

    val_img_paths = [
        str(img)
        for img in Path(crop_img_path).glob("*.png")
        if img.stem.split("_")[0] in FOLD_IMGS[FOLD]
    ]

    print(f"FOLD: {FOLD}")

    get_loader = partial(
        _get_loader, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
    )
    train_loader = get_loader(
        ZarrTrainDataset(
            img_ids=train_img_ids,
            img_path=input_path,
            transform=baseline_aug(TRAIN_IMG_SIZE),
            iterations=ITERS * BATCH_SIZE,
            pdf_path=None,
            crop_size=CROP_SIZE,
        )
    )
    old_val_loader = get_loader(
        ImageDatasetV2(val_img_paths, valid_transform(TRAIN_IMG_SIZE)), shuffle=False,
    )

    model = smp.Unet("resnet34").cuda()
    # model.load_state_dict(
    #     # torch.load("../submission/fold_0_zarr_pdf_epoch_34_score_0.9123.pth")
    #     # torch.load("../submission/fold_0_4096to1024_epoch_49_score_0.9339.pth")
    #     torch.load("weights/crop_4096_1024_old_loader/fold_0/epoch_38_score_0.9254.pth")
    # )
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
        dice_pos, dice_neg = [], []
        if IS_OLD_VALIDATION is True:
            metrics_val = validation(old_val_loader, model, loss_fn)
            image_dice_mean = 0
            dice_mean = metrics_val["dice_mean"]
            val_loss = metrics_val["loss_val"]
            dice_pos = metrics_val["dice_pos"]
            dice_neg = metrics_val["dice_neg"]
        else:
            for img_id in val_img_ids:
                val_loader = get_loader(
                    ZarrValidDataset(
                        img_id,
                        img_path=input_path,
                        transform=valid_transform(TRAIN_IMG_SIZE),
                        crop_size=CROP_SIZE,
                        step=CROP_SIZE,
                    ),
                    shuffle=False,
                )
                metrics_val = validation_full_image(
                    val_loader, model, loss_fn, rle=df.loc[img_id, "encoding"]
                )
                images_dice[img_id] = metrics_val["dice_full"]
                print(
                    f'{np.mean(metrics_val["dice_mean"]):.4f} {metrics_val["dice_full"]:.4f} '
                    + f'{np.mean(metrics_val["loss_mask"]):.4f}'
                )
                dice_mean.extend(metrics_val["dice_mean"])
                val_loss.append(metrics_val["loss_val"])
                dice_pos.append(metrics_val["dice_pos"])
                dice_neg.append(metrics_val["dice_neg"])

                del val_loader

            image_dice_mean = np.mean(list(images_dice.values()))
            dice_mean = np.mean(dice_mean)
            val_loss = np.mean(val_loss)
            dice_pos = np.mean(dice_pos)
            dice_neg = np.mean(dice_neg)

        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {val_loss:.4f}; "
        log += f"avg_dice: {dice_mean:.4f}; full_mask_dice: {image_dice_mean:.4f} "
        log += f"dice_neg: {dice_neg:.4f}; dice_pos: {dice_pos:.4f}"
        print(log, end="")
        cp_handler.update(e, dice_mean)
        scheduler.step(dice_mean)
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

