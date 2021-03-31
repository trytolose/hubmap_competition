import albumentations as A
from albumentations.augmentations.transforms import GridDistortion, ShiftScaleRotate
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
import cv2


def base_transform(img_size=512):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )


def baseline_aug(img_size=512):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # A.ShiftScaleRotate(shift_limit_x=300, shift_limit_y=300, rotate_limit=90),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RGBShift(),
            A.IAAAdditiveGaussianNoise(),
            A.Rotate(limit=360),
            A.RandomContrast(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )


def baseline_aug_v2(img_size=512):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit_x=300, shift_limit_y=300, rotate_limit=90
                    ),
                    A.GridDistortion(num_steps=8, distort_limit=(-0.59, 0.62)),
                ],
                p=0.2,
            ),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RGBShift(),
            A.IAAAdditiveGaussianNoise(),
            A.RandomContrast(limit=(-0.24, 1.0)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )


def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )


def vitaly_augs(img_size):
    train_transform = [
        A.Resize(img_size, img_size),
        # A.CropNonEmptyMaskIfExists(DIM, DIM, p=1)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.IAAPiecewiseAffine(p=0.4),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
                ),
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(p=0.4),
            ],
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def public_augs(img_size):

    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightness(limit=0.2, p=1),
                    A.RandomContrast(limit=0.2, p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [A.Blur(blur_limit=3, p=1), A.MedianBlur(blur_limit=3, p=1)], p=0.25
            ),
            A.OneOf([A.GaussNoise(0.002, p=0.5), A.IAAAffine(p=0.5),], p=0.25),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(
                num_holes=10,
                max_h_size=int(0.1 * img_size),
                max_w_size=int(0.1 * img_size),
                p=0.25,
            ),
            A.ShiftScaleRotate(p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )
