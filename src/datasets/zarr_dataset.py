from torch.utils.data import Dataset
import cv2
from pathlib import Path
import zarr
import numpy as np
import math
from scipy.special import softmax

IMG_SIZES = {
    "2f6ecfcdf": (31278, 25794),
    "8242609fa": (31299, 44066),
    "aaa6a05cc": (18484, 13013),
    "cb2d976f4": (34940, 49548),
    "b9a3865fc": (31295, 40429),
    "b2dc8411c": (14844, 31262),
    "0486052bb": (25784, 34937),
    "e79de561c": (16180, 27020),
    "095bf7a1f": (38160, 39000),
    "54f2eec69": (30440, 22240),
    "4ef6695ce": (39960, 50680),
    "26dc41664": (38160, 42360),
    "c68fe75ea": (26840, 49780),
    "afa5e8098": (36800, 43780),
    "1e2425f28": (26780, 32220),
}

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
    def __init__(
        self,
        img_ids,
        img_path,
        transform,
        iterations=1000,
        crop_size=1024,
        pdf_path=False,
    ):
        self.crop_size = crop_size
        self.transform = transform
        self.iterations = iterations
        self.img_ids = img_ids
        self.img_path = img_path
        self.zarr = zarr.open(self.img_path, mode="r")
        self.pdf = None
        self.coord = np.arange(512 ** 2)
        if pdf_path is True:
            self.pdf = zarr.open("../input/zarr_pdf", mode="r")

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        img_id = np.random.choice(self.img_ids)
        h, w = IMG_SIZES[img_id]

        if self.pdf is True:
            pdf_mask = self.pdf[img_id]
            x, y = self._get_corner(pdf_mask)
            scale_x, scale_y = w / 512, h / 512
            x = int(scale_x * x)
            y = int(scale_y * y)
            if x + self.crop_size > w:
                x = w - self.crop_size
            if y + self.crop_size > h:
                y = h - self.crop_size
        else:
            x = np.random.randint(0, w - self.crop_size)
            y = np.random.randint(0, h - self.crop_size)
            img = self.zarr[img_id][y : y + self.crop_size, x : x + self.crop_size]
            # while not _check_background(img, self.crop_size):
            #     x = np.random.randint(0, w - self.crop_size)
            #     y = np.random.randint(0, h - self.crop_size)
            #     img = self.zarr[img_id][y : y + self.crop_size, x : x + self.crop_size]
            # print(f"x: {x} y: {y} count: {count}")

        mask = self.zarr[img_id + "_mask"][
            y : y + self.crop_size, x : x + self.crop_size
        ]
        transormed = self.transform(image=img, mask=mask)
        return {
            "image": transormed["image"],
            "mask": transormed["mask"].unsqueeze(0).float(),
        }

    def _get_corner(self, pdf):
        probs = softmax(pdf).reshape(-1,)
        left_top_point = np.random.choice(self.coord, size=1, p=probs)[0]
        x, y = np.unravel_index([left_top_point], (512, 512))
        x, y = x[0], y[0]
        return x, y


class ZarrValidDataset(Dataset):
    def __init__(self, tiff_id, img_path, transform, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.transform = transform
        self.step = step
        self.tiff_id = tiff_id
        self.h, self.w = IMG_SIZES[tiff_id]
        self.h_orig, self.w_orig = IMG_SIZES[tiff_id]
        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)
        self.img_path = img_path
        self.zarr = zarr.open(self.img_path, mode="r")

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img = self.zarr[self.tiff_id][y : y + self.crop_size, x : x + self.crop_size]
        crop_mask = self.zarr[self.tiff_id + "_mask"][
            y : y + self.crop_size, x : x + self.crop_size
        ]
        background = _check_background(img, self.crop_size)
        transormed = self.transform(image=img, mask=crop_mask)
        return {
            "image": transormed["image"],
            "mask": transormed["mask"].unsqueeze(0).float(),
            "file_name": f"{x}_{y}",
            "is_background": background,
        }


def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blancking threshold
    p_th = 1000 * (crop_size // 256) ** 2  # threshold for the minimum number of pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)
    background = False if (ss > s_th).sum() <= p_th or img.sum() <= p_th else True
    return background
