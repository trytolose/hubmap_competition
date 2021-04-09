from functools import partial
from pathlib import Path
from pydoc import locate

import hydra
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_toolbelt.losses import DiceLoss
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.dataset import ImageDataset, ImageDatasetV2
from src.datasets.zarr_dataset import ZarrTrainDataset, ZarrValidDataset
from src.loops.loops import train, validation, validation_full_image
from src.transforms.transform import (
    base_transform,
    baseline_aug,
    baseline_aug_v2,
    public_hard_aug,
    valid_transform,
)
from src.utils.checkpoint import CheckpointHandler
from src.utils.utils import IMAGE_SIZES, get_lr
from torch.utils.tensorboard import SummaryWriter

FOLD_IMGS = {
    0: ["4ef6695ce", "0486052bb", "2f6ecfcdf"],
    1: ["c68fe75ea", "095bf7a1f", "aaa6a05cc",],
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


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg))

    exp_dir_name = f"{cfg.EXP_NAME}_{cfg.DATASET.MODE}_{cfg.DATASET.CROP_SIZE}_{cfg.DATASET.IMG_SIZE}"
    cp_path = Path(cfg.CP.CP_DIR) / exp_dir_name / str(cfg.FOLD)

    df = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train.csv").set_index("id", drop=True)
    # zarr_input_path = "../input/zarr_train_orig"
    # crop_img_path = Path("../input/train_v3_4096_1024")

    if cfg.DATASET.MODE == "prepaired":
        df_crops_meta = pd.read_csv(Path(cfg.PREPAIRED.CROP_PATH) / "meta.csv")
        df_crops_meta["fold"] = -1
        for fold_idx, img_ids in FOLD_IMGS.items():
            df_crops_meta.loc[df_crops_meta["img_id"].isin(img_ids), "fold"] = fold_idx

        # df["file"] = df["file"].apply(lambda x: str(input_path / "images" / Path(x).name))
        df_train = df_crops_meta[df_crops_meta["fold"] != cfg.FOLD].reset_index(
            drop=True
        )
        # df_valid = df_crops_meta[df_crops_meta["fold"] == FOLD].reset_index(drop=True)
        df_train["back_prob"] = -1
        counts = df_train["glomerulus_pix"].value_counts()
        zero_gl = counts[0]
        non_zero_gl = len(df_train) - zero_gl
        sampler_weigths = cfg.PREPAIRED.BATCH_TARGET_WEIGHTS
        df_train.loc[df_train["glomerulus_pix"] == 0, "back_prob"] = (
            1 / zero_gl
        ) * sampler_weigths[0]
        df_train.loc[df_train["glomerulus_pix"] != 0, "back_prob"] = (
            1 / non_zero_gl
        ) * sampler_weigths[1]

        sampler = WeightedRandomSampler(df_train["back_prob"].values, len(df_train))

        if cfg.DEBUG_MODE is True:
            df_train = df_train[:40]
            sampler = None
        get_loader = partial(
            _get_loader,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            shuffle=True,
        )
        train_loader = get_loader(
            ImageDataset(df_train, baseline_aug(cfg.DATASET.IMG_SIZE)),
            sampler=sampler,
            shuffle=False,
        )
    if cfg.DATASET.MODE == "zarr":
        train_img_ids = [
            x
            for fold, fold_imgs in FOLD_IMGS.items()
            for x in fold_imgs
            if fold != cfg.FOLD
        ]
        get_loader = partial(
            _get_loader,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            shuffle=True,
        )
        iter_counts = cfg.TRAIN.ITERATIONS_PER_EPOCH * cfg.TRAIN.BATCH_SIZE
        if cfg.DEBUG_MODE is True:
            iter_counts = 40
        train_loader = get_loader(
            ZarrTrainDataset(
                img_ids=train_img_ids,
                img_path=cfg.ZARR.ZARR_PATH,
                transform=baseline_aug(cfg.DATASET.IMG_SIZE),
                iterations=iter_counts,
                pdf_path=cfg.ZARR.PDF,
                crop_size=cfg.DATASET.CROP_SIZE,
            )
        )

    val_img_paths = [
        str(img)
        for img in (Path(cfg.PREPAIRED.CROP_PATH) / "images").glob("*.png")
        if img.stem.split("_")[0] in FOLD_IMGS[cfg.FOLD]
    ]
    val_loader = get_loader(
        ImageDatasetV2(val_img_paths, valid_transform(cfg.DATASET.IMG_SIZE)),
        shuffle=False,
    )

    model = smp.Unet(cfg.MODEL.ENCODER, encoder_weights=cfg.MODEL.WEIGHTS).cuda()

    loss_fn = locate(cfg.LOSS_FN.NAME)()

    optimizer = locate(cfg.OPTIMIZER.NAME)(
        params=model.parameters(), **cfg.OPTIMIZER.CFG
    )

    scaler = GradScaler()
    scheduler = locate(cfg.OPTIMIZER.SCHEDULER.NAME)(
        optimizer=optimizer, **cfg.OPTIMIZER.SCHEDULER.CFG
    )
    if cfg.DEBUG_MODE is False:
        cp_handler = CheckpointHandler(model, cp_path, cfg.CP.BEST_CP_COUNT)
        writer = SummaryWriter(
            log_dir=Path(cfg.LOGGING.TENSORBOARD_LOG_DIR) / exp_dir_name
        )

    for e in range(1, cfg.TRAIN.EPOCH + 1):
        metrics_train = train(train_loader, model, optimizer, loss_fn, scaler)
        metrics_val = validation(val_loader, model, loss_fn)
        dice_mean = metrics_val["dice_mean"]
        val_loss = metrics_val["loss_val"]

        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {val_loss:.4f}; "
        log += f"avg_dice: {dice_mean:.4f}; "
        print(log, end="")
        if cfg.DEBUG_MODE is False:
            writer.add_scalar("Loss/train", metrics_train["loss_train"], e)
            writer.add_scalar("Loss/valid", metrics_val["loss_val"], e)
            writer.add_scalar("Dice_mean/valid", metrics_val["dice_mean"], e)
            writer.add_scalar("Learning rate", get_lr(optimizer), e)

            cp_handler.update(e, dice_mean)
        scheduler.step(e - 1)
        # scheduler.step(dice_mean)
        print("")


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()

