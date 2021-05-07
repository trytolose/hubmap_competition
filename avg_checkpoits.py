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
from src.datasets.zarr_dataset import ZarrTrainDataset, ZarrValidDataset, ZarrDatasetV2
from src.loops.loops import validation_full_zar
import numpy as np
import pandas as pd
from pydoc import locate
import hydra
from omegaconf import DictConfig, OmegaConf


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


def get_score(model, cfg):
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
    loss_fn = locate(cfg.LOSS_FN.NAME)()
    metrics_val = validation_full_zar(
        val_loader, model, loss_fn, cfg.DATASET.CROP_SIZE, thr=0.5,
    )

    return metrics_val[cfg.KEY_METRIC]


def average_weights(state_dicts: List[dict]):
    everage_dict = OrderedDict()
    for k in state_dicts[0].keys():
        everage_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(
            state_dicts
        )
    return everage_dict


def avg(model, checkpoints_weights_paths, cfg):
    all_weights = [
        torch.load(path, map_location="cuda") for path in checkpoints_weights_paths
    ]
    best_score = 0
    best_weights = []
    for w in all_weights:
        current_weights = best_weights + [w]
        average_dict = average_weights(current_weights)
        model.load_state_dict(average_dict)
        score = get_score(model, cfg)
        print(score)
        if score > best_score:
            best_score = score
            best_weights.append(w)
    return best_score, best_weights


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig):

    model = smp.Unet(**cfg.MODEL.CFG).cuda()
    exp_dir_name = f"FOLD_{cfg.FOLD}_{cfg.EXP_NAME}_{cfg.DATASET.MODE}_{cfg.DATASET.CROP_SIZE}_{cfg.DATASET.IMG_SIZE}"
    cp_path = Path(cfg.CP.CP_DIR) / exp_dir_name / str(cfg.FOLD)
    weights_paths = sorted(
        list(cp_path.glob("*.pth")), key=lambda x: x.stem.split("_")[-1], reverse=True,
    )
    [print(w) for w in weights_paths]
    best_score, best_weights = avg(model, weights_paths, cfg)
    w_avg = average_weights(best_weights)
    torch.save(w_avg, cp_path / f"fold{cfg.FOLD}_avg_{best_score:.4f}.pth")


if __name__ == "__main__":
    main()
