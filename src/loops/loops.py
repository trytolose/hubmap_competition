import numpy as np
import torch
from src.utils.metrics import dice_numpy, dice_torch_batch
from torch.cuda.amp import autocast
from tqdm import tqdm
from src.utils.utils import rle2mask
from src.datasets.zarr_dataset import IMG_SIZES
import cv2


def train(data_loader, model, optimizer, loss_fn, scaler):
    model.cuda()
    model.train()
    train_loss = []
    for batch in tqdm(data_loader, ncols=70, leave=False):
        optimizer.zero_grad()
        image = batch["image"].cuda()
        mask = batch["mask"].cuda()
        with autocast():
            pred = model(image)
            loss = loss_fn(pred, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(loss.item())
    metrics = {}
    metrics["loss_train"] = np.mean(train_loss)
    return metrics


def validation(data_loader, model, loss_fn):
    model.eval()
    val_loss = []
    dice_per_crop = []
    non_empty_indexes = []
    for batch in tqdm(data_loader, ncols=70, leave=False):
        with torch.no_grad():
            image = batch["image"].cuda()
            mask = batch["mask"].cuda()
            pred = model(image)
            val_loss.append(loss_fn(pred, mask).item())
            dice_per_crop.append(dice_torch_batch(pred, mask, reduction="numpy"))
            non_empty_indexes.append((mask.sum(dim=(1, 2, 3)) > 0).cpu().data.numpy())
    non_empty_indexes = np.concatenate(non_empty_indexes)
    dice_per_crop = np.concatenate(dice_per_crop)
    metrics = {}
    metrics["dice_mean"] = np.array(dice_per_crop).mean()
    metrics["dice_pos"] = dice_per_crop[non_empty_indexes].mean()
    metrics["dice_neg"] = dice_per_crop[~non_empty_indexes].mean()
    metrics["loss_val"] = np.mean(val_loss)
    return metrics


def validation_full_image(data_loader, model, loss_fn, rle, return_mask=False):
    crop_size = data_loader.dataset.crop_size
    h, w = data_loader.dataset.h_orig, data_loader.dataset.w_orig
    model.eval()
    val_loss = []
    dice_metric = []
    dice_per_crop = []
    mask_true = rle2mask(rle, (w, h))
    # mask_true = cv2.resize(mask_true, (data_loader.dataset.w, data_loader.dataset.h))
    mask_pred = np.zeros(
        (data_loader.dataset.h, data_loader.dataset.w), dtype=np.float32
    )
    non_empty_indexes = []
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    for batch in tqdm(data_loader, ncols=70, leave=False):
        with torch.no_grad():
            image = batch["image"].cuda()
            mask = batch["mask"].cuda()
            is_non_empty = batch["is_background"]
            crop_names = batch["file_name"]
            pred = model(image)
            loss = loss_fn(pred, mask).mean(dim=(1, 2, 3)).cpu().data.numpy()
            val_loss.append(loss)

            pred = pred.sigmoid().squeeze()
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)
            # dice_metric.append(dice_torch_batch(pred, mask, reduction="mean"))
            dice_per_crop.append(dice_torch_batch(pred, mask, reduction="numpy"))
            pred = pred.cpu().data.numpy().astype(np.float32)
            # non_empty_indexes.append((mask.sum(dim=(1, 2, 3)) > 0).cpu().data.numpy())
            is_non_empty = is_non_empty.data.numpy()
            non_empty_indexes.append(is_non_empty)
            for predict_single, crop_name in zip(pred, crop_names):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])
                mask_pred[y : y + crop_size, x : x + crop_size] = cv2.resize(
                    predict_single, (crop_size, crop_size)
                )
    metrics = {}
    dice_per_crop = np.concatenate(dice_per_crop)
    non_empty_indexes = np.concatenate(non_empty_indexes)
    # print(f"{non_empty_indexes.sum()}/{len(dice_per_crop)}")
    # df_val = data_loader.dataset.df
    # non_empty_mask = df_val["glomerulus_pix"] > 0
    # empty_mask = df_val["glomerulus_pix"] == 0
    metrics["dice_pos"] = 0  # dice_per_crop[non_empty_indexes].mean()
    metrics["dice_neg"] = 0  # dice_per_crop[~non_empty_indexes].mean()
    metrics["dice_full"] = dice_numpy(mask_pred, mask_true)
    metrics["dice_mean"] = dice_per_crop[non_empty_indexes].tolist()
    # [print(f"{x:.3f}", end=", ") for x in metrics["dice_mean"]]
    # print()
    val_loss = np.concatenate(val_loss)
    metrics["loss_mask"] = val_loss[non_empty_indexes].mean()
    metrics["loss_val"] = np.mean(val_loss)
    if return_mask is True:
        del mask_true
        mask_pred = mask_pred.astype(np.float32)
        mask_pred = cv2.resize(mask_pred, (w, h))
        return metrics, mask_pred
    return metrics


def inference(data_loader, model, crop_size, train_img_size):
    model.eval()
    mask_pred = np.zeros(
        (data_loader.dataset.h, data_loader.dataset.w), dtype=np.float16
    )
    for image, crop_names in tqdm(data_loader, ncols=70, leave=True):
        with torch.no_grad():
            image = image.cuda()
            pred = model(image)
            pred = pred.sigmoid().squeeze()
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)
            pred = pred.cpu().data.numpy().astype(np.float16)
            for predict_single, crop_name in zip(pred, crop_names):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])
                if crop_size != train_img_size:
                    predict_single = cv2.resize(
                        predict_single.astype(np.float32), (crop_size, crop_size)
                    ).astype(np.float16)

                # predict_single[:10, :] = 1
                # predict_single[-10:, :] = 1
                # predict_single[:, :10] = 1
                # predict_single[:, -10:] = 1
                mask_pred[y : y + crop_size, x : x + crop_size] = predict_single

    return mask_pred


def inference_overlap(data_loader, model, crop_size):
    model.eval()
    mask_pred_overlap = np.zeros(
        (data_loader.dataset.h, data_loader.dataset.w), dtype=np.uint8
    )
    mask_pred = np.zeros(
        (data_loader.dataset.h, data_loader.dataset.w), dtype=np.float16
    )
    for image, crop_names in tqdm(data_loader, ncols=70, leave=True):
        with torch.no_grad():
            image = image.cuda()
            pred = model(image)
            pred = pred.sigmoid().squeeze()
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)
            pred = pred.cpu().data.numpy().astype(np.float16)
            for predict_single, crop_name in zip(pred, crop_names):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])
                mask_pred_overlap[y : y + crop_size, x : x + crop_size] += 1
                # dumb = np.zeros((1024, 1024))
                # dumb[:20, :] = 1
                # dumb[-20:, :] = 1
                # dumb[:, :20] = 1
                # dumb[:, -20:] = 1
                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    return mask_pred / mask_pred_overlap
