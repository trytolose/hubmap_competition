import albumentations as A
from albumentations.augmentations.transforms import GridDistortion, ShiftScaleRotate
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2


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

