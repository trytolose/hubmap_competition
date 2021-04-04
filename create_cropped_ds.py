from pathlib import Path

import pandas as pd
from src.utils.utils import create_dataset
from tqdm import tqdm


def manual_bin(square):
    if square == 0:
        return 0
    elif 0 < square <= 0.1:
        return 1
    elif 0.1 < square <= 0.2:
        return 2
    elif 0.2 < square <= 0.3:
        return 3
    elif 0.3 < square <= 0.4:
        return 4
    elif 0.4 < square <= 0.6:
        return 5
    elif 0.6 < square <= 0.8:
        return 6
    elif 0.8 < square:
        return 7


def bin_squares(df, img_size):
    bining_cols = [
        "glomerulus_pix",
        "medulla",
        "cortex",
        "outer_stripe",
        "Inner medulla",
        "Outer Medulla",
    ]
    for col in bining_cols:
        df[col] = df[col] / img_size ** 2
        df[col] = df[col].apply(manual_bin)
    return df


if __name__ == "__main__":
    CROP_SIZE = 1024
    STEP = 1024
    IMG_SIZE = 256
    INPUT_PATH = Path("/hdd/kaggle/hubmap/input_v2")
    PATH_FOR_CROPS = "/home/trytolose/rinat/kaggle/hubmap/input/train_1024_256"
    df_train = pd.read_csv(INPUT_PATH / "train.csv")
    train_tiffs = list((INPUT_PATH / "train").glob("*.tiff"))

    # meta_data = []
    # for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    #     meta = create_dataset(
    #         row["id"],
    #         row["encoding"],
    #         CROP_SIZE,
    #         PATH_FOR_CROPS,
    #         resize=IMG_SIZE,
    #         step=STEP,
    #     )
    #     meta_data.extend(meta)

    # df = bin_squares(pd.DataFrame(meta_data), IMG_SIZE)
    # df.to_csv(PATH_FOR_CROPS + "/meta.csv", index=False)

    PATH_FOR_CROPS = (
        "/home/trytolose/rinat/kaggle/hubmap/input/train_1024_256_pseudo_v1"
    )
    pseudo_tiff_path = INPUT_PATH / "test"
    df_pseudo = pd.read_csv("../input/d48_hand_labelled.csv")
    rle_pseudo = df_pseudo.iloc[0, 1]

    meta_data = []
    meta = create_dataset(
        "d488c759a",
        rle_pseudo,
        CROP_SIZE,
        PATH_FOR_CROPS,
        resize=IMG_SIZE,
        step=STEP,
        path=pseudo_tiff_path,
    )
    meta_data.extend(meta)

    df = bin_squares(pd.DataFrame(meta_data), IMG_SIZE)
    df.to_csv(PATH_FOR_CROPS + "/meta.csv", index=False)
