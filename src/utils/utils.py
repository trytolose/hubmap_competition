import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import zarr
from rasterio.windows import Window
from tqdm import tqdm

CLS_DICT = {
    "Medulla": 1,
    "Cortex": 2,
    "Outer Stripe": 3,
    "Inner medulla": 4,
    "Outer Medulla": 5,
}
IMAGE_SIZES = {
    "095bf7a1f": (38160, 39000),
    "4ef6695ce": (39960, 50680),
    "c68fe75ea": (26840, 49780),
    "b9a3865fc": (31295, 40429),
    "afa5e8098": (36800, 43780),
    "cb2d976f4": (34940, 49548),
    "8242609fa": (31299, 44066),
    "2f6ecfcdf": (31278, 25794),
    "0486052bb": (25784, 34937),
    "b2dc8411c": (14844, 31262),
    "54f2eec69": (30440, 22240),
    "e79de561c": (16180, 27020),
    "26dc41664": (38160, 42360),
    "1e2425f28": (26780, 32220),
    "aaa6a05cc": (18484, 13013),
}


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def create_masks(img_id, rle, path="/hdd/kaggle/hubmap/input_v2/train/"):
    img_path = Path(path) / f"{img_id}.tiff"
    img = rasterio.open(img_path, num_threads="all_cpus")
    h, w = img.height, img.width

    json_path = Path(path) / f"{img_id}-anatomical-structure.json"
    mask = rle2mask(rle, (w, h))
    mask_main = zarr.array(mask)
    del mask
    gc.collect()
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(json_path, "r") as read_file:
        data = json.load(read_file)
        for d in data:
            class_name = d["properties"]["classification"]["name"]
            points = d["geometry"]["coordinates"][0]
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            mask = cv2.fillPoly(mask, [points], CLS_DICT[class_name])

    mask_second = zarr.array(mask)
    del mask
    gc.collect()
    return mask_main, mask_second


def read_from_layers(layers, window):
    if len(layers) == 1:
        return np.stack([layers[0].read(x, window=window) for x in [1, 2, 3]], 2)
    else:
        return np.stack([layers[x].read(1, window=window) for x in range(3)], 2)


def create_dataset(
    img_id: str,
    rle: str,
    crop_size: int,
    folder_to_save,
    step=None,
    resize=None,
    path="/hdd/kaggle/hubmap/input_v2/train/",
) -> np.ndarray:
    if step is None:
        step = crop_size
    s_th = 40  # saturation blancking threshold
    p_th = 1000 * (crop_size // 256) ** 2  # threshold for the minimum number of pixels

    img_path = Path(path) / f"{img_id}.tiff"
    dataset = rasterio.open(img_path, num_threads="all_cpus")
    mask_main, mask_second = create_masks(img_id, rle, path)

    h, w = dataset.height, dataset.width
    layers = []
    if dataset.count != 3:
        subdatasets = dataset.subdatasets
        if len(subdatasets) > 0:
            for i, subdataset in enumerate(subdatasets, 0):
                layers.append(rasterio.open(subdataset))
    else:
        layers.append(rasterio.open(img_path))

    #     mask = np.zeros((h, w), dtype=np.bool)
    images_path = Path(folder_to_save) / "images"
    masks_path = Path(folder_to_save) / "masks"
    images_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)

    meta_info = []
    for x in range(0, w, step):
        if x + crop_size > w:
            x = w - crop_size
        for y in range(0, h, step):
            if y + crop_size > h:
                y = h - crop_size
            window = Window(x, y, crop_size, crop_size)
            crop = read_from_layers(layers, window=window)[:, :, ::-1]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            hh, ss, vv = cv2.split(hsv)

            background = (
                False if (ss > s_th).sum() <= p_th or crop.sum() <= p_th else True
            )
            if background:
                meta_dict = {}
                meta_dict["img_id"] = img_id
                mask = mask_main[y : y + crop_size, x : x + crop_size]
                mask_back = mask_second[y : y + crop_size, x : x + crop_size]
                meta_dict["glomerulus_pix"] = mask.sum()
                meta_dict["medulla"] = (mask_back == 1).sum()
                meta_dict["cortex"] = (mask_back == 2).sum()
                meta_dict["outer_stripe"] = (mask_back == 3).sum()
                meta_dict["Inner medulla"] = (mask_back == 4).sum()
                meta_dict["Outer Medulla"] = (mask_back == 5).sum()

                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    print("!!!!", crop.shape)

                #             mask[y:y+crop_size, x:x+crop_size] = background
                crop_path = images_path / f"{img_id}_{x}_{y}.png"
                mask_path = masks_path / f"{img_id}_{x}_{y}.png"
                meta_dict["file"] = str(crop_path)
                if resize is not None:
                    crop = cv2.resize(crop, (resize, resize))
                    mask = cv2.resize(mask, (resize, resize))
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(mask_path), mask)
                meta_info.append(meta_dict)
    del dataset, mask_main, mask_second, layers
    gc.collect()

    return meta_info
