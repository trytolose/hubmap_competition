import albumentations as A
from albumentations.augmentations.transforms import (
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
)
from albumentations.pytorch import ToTensorV2


def base_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )


def baseline_aug(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
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


def baseline_aug_v2(img_size):
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


def public_hard_aug(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
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


def public_hard_aug_v2(img_size):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightness(limit=0.45, p=1),
                    A.RandomContrast(limit=0.45, p=1),
                    A.RandomGamma((20, 80), p=1),
                ],
                p=0.5,
            ),
            A.Blur(blur_limit=20, p=0.4),
            A.OneOf([A.GaussNoise(mean=-20, p=0.5), A.IAAAffine(p=0.5),], p=0.4),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(
                num_holes=10,
                max_h_size=int(0.1 * img_size),
                max_w_size=int(0.1 * img_size),
                p=0.3,
            ),
            A.OneOf(
                [
                    A.ShiftScaleRotate(scale_limit=0.3, p=0.25),
                    A.OpticalDistortion(distort_limit=(0.3, 1.0), p=0.25),
                    A.ElasticTransform(alpha=3, sigma=100, alpha_affine=80, p=0.25),
                    A.GridDistortion(p=0.25),
                ],
                p=0.25,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ]
    )
