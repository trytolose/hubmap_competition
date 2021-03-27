from typing import List
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from src.loops.loops import validation_full_image
from src.transforms.transform import valid_transform
from src.datasets.dataset import ImageDatasetV2
import segmentation_models_pytorch as smp

FOLD_IMGS = {
    0: ["4ef6695ce", "0486052bb", "2f6ecfcdf"],
    1: ["c68fe75ea", "095bf7a1f", "aaa6a05cc"],
    2: ["afa5e8098", "1e2425f28", "b2dc8411c"],
    3: ["cb2d976f4", "8242609fa", "54f2eec69"],
    4: ["26dc41664", "b9a3865fc", "e79de561c"],
}
BATCH_SIZE = 24
TRAIN_IMG_SIZE = 1024
input_path = Path("../input/train_v3_4096_1024/images")


def average_weights(state_dicts: List[dict]):
    everage_dict = OrderedDict()
    for k in state_dicts[0].keys():
        everage_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(
            state_dicts
        )
    return everage_dict


def avg(model, data_loader, checkpoints_weights_paths):
    all_weights = [
        torch.load(path, map_location="cuda") for path in checkpoints_weights_paths
    ]
    best_score = 0
    best_weights = []
    for w in all_weights:
        current_weights = best_weights + [w]
        average_dict = average_weights(current_weights)
        model.load_state_dict(average_dict)
        score = validation_full_image(data_loader, model, torch.nn.BCEWithLogitsLoss())[
            "dice_mean"
        ]
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

    val_loader = DataLoader(
        dataset=ImageDatasetV2(val_img_paths, valid_transform(TRAIN_IMG_SIZE)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        pin_memory=False,
    )
    weights_paths = list((Path(weights_path) / f"fold_{fold}").glob("*.pth"))
    weights_paths = sorted(weights_paths, reverse=True)
    print(weights_paths)
    best_score, best_weights = avg(model, val_loader, weights_paths)
    w_avg = average_weights(best_weights)
    torch.save(
        w_avg, Path(weights_path) / f"fold_{fold}/fold_{fold}_avg_{best_score}.pth"
    )


if __name__ == "__main__":
    weights_path = "./crop_4096_1024"
    for i in range(5):
        print("FOLD:", i)
        get_avg_checkpoint(weights_path, i)

# checkpoints_weights_paths: List[str] = [
#     os.path.join(w_path, str(x) + ".pt") for x in [4, 3, 2, 1, 0]
# ]
# best_score, best_weights = avg(checkpoints_weights_paths)
# w_avg = average_weights(best_weights)
