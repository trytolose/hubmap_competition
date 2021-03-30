from pathlib import Path

import pandas as pd
from src.utils.utils import create_dataset
from tqdm import tqdm

if __name__ == "__main__":

    INPUT_PATH = Path("/hdd/kaggle/hubmap/input_v2")
    PATH_FOR_CROPS = "../input/train_v4_4096_1024"
    df_train = pd.read_csv(INPUT_PATH / "train.csv")
    train_info = pd.read_csv(INPUT_PATH / "HuBMAP-20-dataset_information.csv")
    train_tiffs = list((INPUT_PATH / "train").glob("*.tiff"))

    meta_data = []
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        meta = create_dataset(
            row["id"], row["encoding"], 4096, PATH_FOR_CROPS, resize=1024, step=2048,
        )
        meta_data.extend(meta)

    df = pd.DataFrame(meta_data)
    df.to_csv(PATH_FOR_CROPS + "/meta.csv", index=False)
