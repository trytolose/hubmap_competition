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
    public_augs,
    vitaly_augs,
)
from src.utils.checkpoint import CheckpointHandler
from src.utils.utils import IMAGE_SIZES
import argparse
from madgrad import MADGRAD
from pytorch_toolbelt.losses import DiceLoss

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


from torch.nn.modules.loss import _Loss


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


def main(args):
    FOLD = args.fold
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 10
    START_LR = 0.001
    EPOCH = args.epoch
    CROP_SIZE = args.crop_size
    TRAIN_IMG_SIZE = args.train_img_size
    BACKGROUND_WEIGHTS = [0.5, 0.25, 0.01]
    WEIGHT_PATH = f"./weights/1024_weighted/fold_{FOLD}"

    # df = pd.read_csv("../input/train_1024_256/split_v1.csv")
    df = pd.read_csv("../input/train_v1_1024/split_v2.csv")

    df_pseudo = pd.read_csv(
        "/home/trytolose/rinat/kaggle/hubmap/input/train_1024_256_pseudo_v1/meta.csv"
    )

    # df = df[df["img_id"] != "aaa6a05cc"].reset_index(drop=True)
    # input_path = Path("/hdd/kaggle/hubmap/input_v2/train_v2_2048/images")
    # input_path = Path("../input/train_v3_4096_1024/images")
    input_path = Path("../input/train_v1_1024")

    df["fold"] = -1
    for fold_idx, img_ids in FOLD_IMGS.items():
        df.loc[df["img_id"].isin(img_ids), "fold"] = fold_idx

    df["file"] = df["file"].apply(lambda x: str(input_path / "images" / Path(x).name))
    df_train = df[df["fold"] != FOLD].reset_index(drop=True)
    df_valid = df[df["fold"] == FOLD].reset_index(drop=True)
    print(f"FOLD: {FOLD}")

    # df_train = pd.concat([df_train, df_pseudo], axis=0, ignore_index=True)
    df_train["back_prob"] = -1
    counts = df_train["glomerulus_pix"].value_counts()
    zero_gl = counts[0]
    non_zero_gl = len(df_train) - zero_gl
    df_train.loc[df_train["glomerulus_pix"] == 0, "back_prob"] = (1 / zero_gl) * 40
    df_train.loc[df_train["glomerulus_pix"] != 0, "back_prob"] = (1 / non_zero_gl) * 60
    # ) * BACKGROUND_WEIGHTS[2]
    # df_train.loc[df_train["back_class"] == 1, "back_prob"] = (
    #     1 / counts[1]
    # ) * BACKGROUND_WEIGHTS[0]
    # df_train.loc[df_train["back_class"] == 2, "back_prob"] = (
    #     1 / counts[2]
    # ) * BACKGROUND_WEIGHTS[1]

    sampler = WeightedRandomSampler(df_train["back_prob"].values, len(df_train))

    get_loader = partial(
        _get_loader, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
    )
    train_loader = get_loader(
        ImageDataset(df_train, public_augs(TRAIN_IMG_SIZE)),
        sampler=sampler,
        shuffle=False,
    )
    val_loader = get_loader(
        ImageDataset(df_valid, valid_transform(TRAIN_IMG_SIZE)), shuffle=False,
    )

    model = smp.Unet("resnet34").cuda()
    # model.load_state_dict(torch.load("./first_launch/epoch_7_score_0.9234.pth"))

    loss_fn = MyLoss()
    optimizer = Adam(model.parameters(), lr=START_LR)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    cp_handler = CheckpointHandler(model, WEIGHT_PATH, 1)
    for e in range(1, EPOCH + 1):
        metrics_train = train(train_loader, model, optimizer, loss_fn, scaler)
        metrics_val = validation_full_image(val_loader, model, loss_fn)
        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {metrics_val['loss_val']:.4f}; "
        log += f"dice_mean: {metrics_val['dice_mean']:.4f}; "  # dice_pos: {metrics_val['dice_pos']:.4f}; "
        # log += f"dice_neg: {metrics_val['dice_neg']:.4f} "
        print(log, end="")
        cp_handler.update(e, metrics_val["dice_mean"])
        scheduler.step(metrics_val["dice_mean"])
        print("")

    # for e in range(1, EPOCH + 1):
    #     metrics_train = train(train_loader, model, optimizer, loss_fn, scaler)
    #     images_dice = {}
    #     dice_mean = []
    #     val_loss = []
    #     for img_id in df_valid["img_id"].unique():
    #         df_img = df_valid[df_valid["img_id"] == img_id].reset_index(drop=True)
    #         val_loader = get_loader(
    #             ImageDataset(df_img, valid_transform(CROP_SIZE)), shuffle=False,
    #         )
    #         metrics_val = validation_full_image(val_loader, model, loss_fn)
    #         images_dice[img_id] = metrics_val["dice_full"]
    #         dice_mean.append(metrics_val["dice_mean"])
    #         val_loss.append(metrics_val["loss_val"])
    #         del val_loader

    #     image_dice_mean = np.mean(list(images_dice.values()))
    #     dice_mean = np.mean(dice_mean)
    #     val_loss = np.mean(val_loss)
    #     log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {val_loss:.4f}; "
    #     log += f"avg_dice: {dice_mean:.4f}; full_mask_dice: {image_dice_mean:.4f} "
    #     print(log, end="")
    #     cp_handler.update(e, image_dice_mean)
    #     scheduler.step(image_dice_mean)
    #     print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_save_path", default="./", type=str, help="valid path")

    parser.add_argument("--fold", type=int, default=-1, help="fold")
    parser.add_argument("--epoch", type=int, default=60, help="total epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--crop_size", type=int, default=2048, help="batch size")
    parser.add_argument("--train_img_size", type=int, default=512, help="batch size")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    main(args)

