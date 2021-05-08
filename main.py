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
from src.datasets.zarr_dataset import ZarrTrainDataset, ZarrValidDataset, ZarrDatasetV2
from src.loops.loops import (
    train,
    validation,
    validation_full_image,
    validation_full_zar,
)
from src.transforms.transform import (
    base_transform,
    baseline_aug,
    baseline_aug_v2,
    public_hard_aug,
    valid_transform,
    public_hard_aug_v2,
)
from src.utils.checkpoint import CheckpointHandler
from src.utils.utils import IMAGE_SIZES, get_lr
from torch.utils.tensorboard import SummaryWriter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import sys

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

    print(OmegaConf.to_yaml(cfg))

    exp_dir_name = f"FOLD_{cfg.FOLD}_{cfg.EXP_NAME}_{cfg.DATASET.MODE}_{cfg.DATASET.CROP_SIZE}_{cfg.DATASET.IMG_SIZE}"
    cp_path = Path(cfg.CP.CP_DIR) / exp_dir_name / str(cfg.FOLD)

    df_train_reference = pd.read_csv("/hdd/kaggle/hubmap/input_v2/train.csv").set_index(
        "id", drop=True
    )
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
    if cfg.DATASET.MODE == "prepaired_new_split":

        df_crops_meta = pd.read_csv(Path(cfg.PREPAIRED.CROP_PATH) / "meta.csv")
        strf_cols = [
            "glomerulus_pix",
            "medulla",
            "cortex",
            "outer_stripe",
            "Inner medulla",
            "Outer Medulla",
        ]
        for col in strf_cols:
            df_crops_meta[col] = pd.cut(df_crops_meta[col], 10, labels=np.arange(10))

        df_crops_meta["fold"] = 0
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for fold, (_, test_index) in enumerate(
            mskf.split(df_crops_meta, df_crops_meta[["img_id"] + strf_cols])
        ):
            df_crops_meta.loc[test_index, "fold"] = fold

        df_train = df_crops_meta[df_crops_meta["fold"] != cfg.FOLD].reset_index(
            drop=True
        )
        df_valid = df_crops_meta[df_crops_meta["fold"] == cfg.FOLD].reset_index(
            drop=True
        )

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
            # sampler=sampler,
            shuffle=True,
        )
        val_loader = get_loader(
            ImageDataset(df_valid, valid_transform(cfg.DATASET.IMG_SIZE)),
            # sampler=sampler,
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

    if cfg.DATASET.MODE == "zarr_prepaired":
        train_img_ids = [
            x
            for fold, fold_imgs in FOLD_IMGS.items()
            for x in fold_imgs
            if fold != cfg.FOLD
        ]
        pseudo_ids = cfg.DATASET.PSEUDO_IDS
        print(pseudo_ids)
        if len(pseudo_ids) > 0:
            train_img_ids.extend(pseudo_ids)

        df_coord_name = f"train_fold{cfg.FOLD}_crop_{cfg.DATASET.CROP_SIZE}_img_{cfg.DATASET.IMG_SIZE}_step_{cfg.DATASET.STEP}.csv"
        df_path = Path(cfg.ZARR.CALC_COORD_PATH) / df_coord_name
        zarr_ds = ZarrDatasetV2(
            img_ids=train_img_ids,
            img_path=cfg.ZARR.ZARR_PATH,
            transform=public_hard_aug_v2(cfg.DATASET.IMG_SIZE),
            crop_size=cfg.DATASET.CROP_SIZE,
            step=cfg.DATASET.CROP_SIZE,
            df_path=df_path,
        )

        df_train = zarr_ds.df.copy()
        df_train["back_prob"] = -1

        df_train["density_cls"] = pd.cut(
            df_train["glomerulus_pix"], 5, labels=np.arange(5)
        )
        prob_vc = df_train["density_cls"].value_counts()
        for idx in prob_vc.index:
            df_train.loc[df_train["density_cls"] == idx, "back_prob"] = 1 / prob_vc[idx]
        # counts = df_train["glomerulus_pix"].value_counts()
        # zero_gl = counts[0]
        # non_zero_gl = len(df_train) - zero_gl
        # sampler_weigths = cfg.PREPAIRED.BATCH_TARGET_WEIGHTS
        # df_train.loc[df_train["glomerulus_pix"] == 0, "back_prob"] = (
        #     1 / zero_gl
        # ) * sampler_weigths[0]
        # df_train.loc[df_train["glomerulus_pix"] != 0, "back_prob"] = (
        #     1 / non_zero_gl
        # ) * sampler_weigths[1]

        # print(df_train["back_prob"].value_counts())
        # sys.exit()

        sampler = WeightedRandomSampler(df_train["back_prob"].values, len(df_train))

        train_loader = DataLoader(
            dataset=zarr_ds,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            sampler=sampler,
            shuffle=False,
        )
        df_coord_name = f"valid_fold{cfg.FOLD}_crop_{cfg.DATASET.CROP_SIZE}_img_{cfg.DATASET.IMG_SIZE}_step_{cfg.DATASET.CROP_SIZE}.csv"
        df_path = Path(cfg.ZARR.CALC_COORD_PATH) / df_coord_name

        val_loader = DataLoader(
            dataset=ZarrDatasetV2(
                img_ids=FOLD_IMGS[cfg.FOLD],
                img_path=cfg.ZARR.ZARR_PATH,
                transform=valid_transform(cfg.DATASET.IMG_SIZE),
                crop_size=cfg.DATASET.CROP_SIZE,
                step=cfg.DATASET.CROP_SIZE,
                df_path=df_path,
                mode="valid",
            ),
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            shuffle=False,
        )
    # model = smp.Unet(cfg.MODEL.ENCODER, encoder_weights=cfg.MODEL.WEIGHTS).cuda()
    model = smp.Unet(**cfg.MODEL.CFG)

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
        # metrics_val = validation(val_loader, model, loss_fn)
        if cfg.DATASET.MODE == "prepaired_new_split":
            metrics_val = validation(val_loader, model, loss_fn)
        else:
            metrics_val = validation_full_zar(
                val_loader, model, loss_fn, cfg.DATASET.CROP_SIZE, thr=0.5,
            )

        dice_mean = metrics_val["dice_mean"]
        val_loss = metrics_val["loss_val"]

        log = f"epoch: {e:03d}; loss_train: {metrics_train['loss_train']:.4f}; loss_val: {val_loss:.4f}; "
        log += f"avg_dice: {dice_mean:.4f}; "
        if metrics_val.get("dice_full_mean", None) is not None:
            log += f"dice_full_mean: {metrics_val['dice_full_mean']:.4f}; "
        if metrics_val.get("dice_pos", None) is not None:
            log += f"dice_pos: {metrics_val['dice_pos']:.4f} "
        if metrics_val.get("cls_rocauc", None) is not None:
            log += (
                f"roc_auc: {metrics_val['cls_rocauc']:.4f} f1: {metrics_val['f1']:.4f}"
            )
        # log += f"avg_dice_x4: {metrics_val_x4['dice_mean']:.4f}; "

        print(log, end="")
        if cfg.DEBUG_MODE is False:
            writer.add_scalar("Loss/train", metrics_train["loss_train"], e)
            writer.add_scalar("Loss/valid", metrics_val["loss_val"], e)
            writer.add_scalar("Dice_mean/valid", metrics_val["dice_mean"], e)
            if metrics_val.get("dice_full_mean", None) is not None:
                writer.add_scalar("Dice_full/valid", metrics_val["dice_full_mean"], e)
            writer.add_scalar("Learning rate", get_lr(optimizer), e)

            cp_handler.update(e, metrics_val[cfg.KEY_METRIC])
        # scheduler.step(e - 1)
        scheduler.step(metrics_val[cfg.KEY_METRIC])
        print("")


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()

