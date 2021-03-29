from torch.utils.data import Dataset
import cv2
from pathlib import Path
import rasterio
from rasterio.windows import Window
from src.utils.utils import read_from_layers, rle2mask
import math


class ImageDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "file"]
        mask_path = img_path.replace("images", "masks")
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        transormed = self.transforms(image=img, mask=mask)
        return (
            transormed["image"],
            transormed["mask"].unsqueeze(0).float(),
            Path(img_path).stem,
        )


class ImageDatasetV2(Dataset):
    def __init__(self, img_paths, transforms):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = img_path.replace("images", "masks")
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        transormed = self.transforms(image=img, mask=mask)
        return {
            "image": transormed["image"],
            "mask": transormed["mask"].unsqueeze(0).float(),
            "file_name": Path(img_path).stem,
        }


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, transform, crop_size=1024, step=512, rle=None):
        self.crop_size = crop_size
        self.transform = transform
        self.step = step
        dataset = rasterio.open(tiff_path, num_threads="all_cpus")
        self.h = dataset.height
        self.w = dataset.width
        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)
        self.mask = None
        if rle is not None:
            self.mask = rle2mask(rle, (self.w, self.h))

        self.layers = []
        if dataset.count != 3:
            subdatasets = dataset.subdatasets
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(
                        rasterio.open(subdataset, num_threads="all_cpus")
                    )
        else:
            self.layers.append(rasterio.open(tiff_path, num_threads="all_cpus"))

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size
        window = Window(x, y, self.crop_size, self.crop_size)
        img = read_from_layers(self.layers, window=window)  # [:, :, ::-1]
        if self.mask is not None:
            crop_mask = self.mask[y : y + self.crop_size, x : x + self.crop_size]
            transormed = self.transform(image=img, mask=crop_mask)
            return {
                "image": transormed["image"],
                "mask": transormed["mask"].unsqueeze(0).float(),
                "file_name": f"{x}_{y}",
            }
        else:
            transormed = self.transform(image=img)
            return transormed["image"], f"{x}_{y}"


# dice_mean: 0.9438; image_dice_mean: 0.9068

