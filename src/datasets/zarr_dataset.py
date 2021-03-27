from torch.utils.data import Dataset
import cv2
from pathlib import Path
import zarr
import numpy as np
import math

IMG_SIZES_X4 = {
    "2f6ecfcdf": (7819, 6448),
    "8242609fa": (7824, 11016),
    "aaa6a05cc": (4621, 3253),
    "cb2d976f4": (8735, 12387),
    "b9a3865fc": (7823, 10107),
    "b2dc8411c": (3711, 7815),
    "0486052bb": (6446, 8734),
    "e79de561c": (4045, 6755),
    "095bf7a1f": (9540, 9750),
    "54f2eec69": (7610, 5560),
    "4ef6695ce": (9990, 12670),
    "26dc41664": (9540, 10590),
    "c68fe75ea": (6710, 12445),
    "afa5e8098": (9200, 10945),
    "1e2425f28": (6695, 8055),
}


class ZarrTrainDataset(Dataset):
    def __init__(self, img_ids, img_path, transform, iterations=1000, crop_size=1024):
        self.crop_size = crop_size
        self.transform = transform
        self.iterations = iterations
        self.img_ids = img_ids
        self.img_path = img_path

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        img_id = np.random.choice(self.img_ids)[0]
        h, w = self.img_dict[img_id]

        x = np.random.randint(0, w - self.crop_size)
        y = np.random.randint(0, h - self.crop_size)

        with zarr.open(self.img_path, mode="r") as img_zarr:
            img = img_zarr[img_id][y : y + self.crop_size, x : x + self.crop_size]

        with zarr.open(self.img_path, mode="r") as mask_zarr:
            mask = mask_zarr[img_id + "_mask"][
                y : y + self.crop_size, x : x + self.crop_size
            ]

        transormed = self.transform(image=img, mask=mask)
        return transormed["image"], transormed["mask"].unsqueeze(0).float()


class ZarrValidDataset(Dataset):
    def __init__(self, tiff_id, transform, crop_size=1024, step=512, rle=None):
        self.crop_size = crop_size
        self.transform = transform
        self.step = step
        self.tiff_id = tiff_id
        self.h, self.w = IMG_SIZES_X4[tiff_id]
        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)
        self.mask = None

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        with zarr.open(self.img_path, mode="r") as img_zarr:
            img = img_zarr[self.tiff_id][y : y + self.crop_size, x : x + self.crop_size]

        if self.mask is not None:
            with zarr.open(self.img_path, mode="r") as mask_zarr:
                crop_mask = mask_zarr[self.tiff_id + "_mask"][
                    y : y + self.crop_size, x : x + self.crop_size
                ]

            transormed = self.transform(image=img, mask=crop_mask)
            return (
                transormed["image"],
                transormed["mask"].unsqueeze(0).float(),
                f"{x}_{y}",
            )
        else:
            transormed = self.transform(image=img)
            return transormed["image"], f"{x}_{y}"

