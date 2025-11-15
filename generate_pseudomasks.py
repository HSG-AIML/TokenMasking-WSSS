"""
Generate pseudo-masks from ViTWithTokenDropout for different datasets (DFC, ADE20K).

This script:
- Loads a trained ViTWithTokenDropout checkpoint.
- Runs it on a dataset.
- Uses attention maps to build pseudo-segmentation masks.
- Saves pseudo-masks, input images, and ground-truth masks as .npy files.

Supported datasets:
- "dfc":    GRSS DFC 2020 (8 land cover classes).
- "ade20k": ADE20K semantic segmentation (151 classes).
"""

import os
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import torch
from model import ViTWithTokenDropout
from torch.utils.data import DataLoader

from recorder_tokendropout import Recorder

DATASET = "dfc"  # or "ade20k"


def fill_closest_values(matrix: np.ndarray, to_fill: int = -1) -> np.ndarray:
    """
    Fill all entries equal to `to_fill` with the value of the closest
    non-`to_fill` neighbor using BFS.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of labels with some cells equal to `to_fill`.
    to_fill : int, optional
        Label value to be replaced by nearest valid labels.

    Returns
    -------
    np.ndarray
        Matrix with all `to_fill` entries replaced.
    """
    rows, cols = matrix.shape
    queue = deque()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    # Initialize the queue with all known values adjacent to `to_fill`
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != to_fill:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and matrix[ni, nj] == to_fill:
                        queue.append((ni, nj, matrix[i, j]))

    # Perform BFS
    while queue:
        i, j, value = queue.popleft()
        # If this spot has been updated in the meantime with a closer valid value, skip it
        if matrix[i, j] != to_fill:
            continue
        matrix[i, j] = value
        # Add all adjacent `to_fill`s to the queue
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and matrix[ni, nj] == to_fill:
                queue.append((ni, nj, value))

    return matrix


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Given a directory of Lightning checkpoints named like:
        epoch=E-val_loss=L.ckpt (or similar pattern),
    select the checkpoint with the lowest validation loss, then highest epoch.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoint files.

    Returns
    -------
    str
        Full path to the selected checkpoint.
    """
    checkpoint_names = os.listdir(checkpoint_dir)
    parsed_checkpoints = []

    for name in checkpoint_names:
        parts = name.split("-")
        if len(parts) < 3:
            continue  # skip files that don't match expected pattern
        try:
            epoch = int(parts[1].split("=")[1])
            val_loss = float(parts[2].split("=")[1].replace(".ckpt", ""))
        except (IndexError, ValueError):
            continue
        parsed_checkpoints.append((epoch, val_loss, name))

    if not parsed_checkpoints:
        raise RuntimeError(f"No valid checkpoints found in {checkpoint_dir}")

    # Sort by validation loss (ascending) then epoch (descending)
    parsed_checkpoints.sort(key=lambda x: (x[1], -x[0]))
    return os.path.join(checkpoint_dir, parsed_checkpoints[0][2])


if DATASET == "dfc":
    # -------------------- DFC CONFIG --------------------
    from datasets.dfc import DFCDataset

    CLASSES_MAP = {
        "Forest": 0,
        "Shrubland": 1,
        "Grassland": 2,
        "Wetland": 3,
        "Cropland": 4,
        "Urban": 5,
        "Barren": 6,
        "Water": 7,
    }
    CLASS_NAMES = list(CLASSES_MAP.keys())
    NUM_CLASSES = len(CLASSES_MAP)

    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    PATCH_SIZE = 14
    IN_CHANS = 13

    DATA_ROOT = "/netscratch/lscheibenreif/grss-dfc-20/"
    SET = "validation"

    PROJECT = "1z9zz5ea"
    CHECKPOINT_DIR = f"ViT_pruning/{PROJECT}/checkpoints"
    CHECKPOINT_PATH = None  # if you want a specific checkpoint, put its path here

    def build_dataset():
        return DFCDataset(
            DATA_ROOT,
            mode=SET,
            clip_sample_values=True,
            used_data_fraction=1.0,
            image_px_size=IMAGE_SIZE[0],
            cover_all_parts=False,
            seed=42,
            frac=0.1,
        )

    def dfc_display_mct(attn: np.ndarray) -> List[np.ndarray]:
        """
        Convert attention maps to upsampled class-specific attention masks.

        Parameters
        ----------
        attn : np.ndarray
            Attention maps for all classes, shape (num_classes, num_patches).

        Returns
        -------
        List[np.ndarray]
            List of upsampled attention masks, shape (H, W, 1) each.
        """
        mask_rs = []
        grid_h = IMAGE_SIZE[0] // PATCH_SIZE
        grid_w = IMAGE_SIZE[1] // PATCH_SIZE

        for class_idx in range(NUM_CLASSES):
            if class_idx >= attn.shape[0]:
                break
            mask = attn[class_idx]
            mask = mask.reshape(grid_h, grid_w)
            mask_r = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]))[..., np.newaxis]
            mask_rs.append(mask_r)
        return mask_rs

    def dfc_generate_pseudomask(
        y_pred: np.ndarray,
        mask: np.ndarray,
        cts: List[np.ndarray],
        logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate a pseudo-mask for DFC using class-wise attention thresholds.

        Parameters
        ----------
        y_pred : np.ndarray
            Binary predictions per class (0/1), shape (num_classes,).
        mask : np.ndarray
            Ground-truth mask (unused, only for shape).
        cts : list of np.ndarray
            Class-wise attention maps, each of shape (H, W, 1).
        logits : np.ndarray, optional
            Raw logits per class, used to rank classes if provided.

        Returns
        -------
        np.ndarray
            Pseudo-mask of shape like `mask`, with entries in [0, num_classes-1].
        """
        pseudomask_true = np.ones(mask.shape) * -1

        if logits is None:
            for idx, el in enumerate(y_pred):
                if el == 1:
                    cts_temp = cts[idx].squeeze()
                    mean_pix = np.mean(cts_temp)
                    pseudomask_true[cts_temp >= mean_pix] = idx
        else:
            sorted_indices = sorted(range(len(logits)), key=lambda k: logits[k])
            for idx in sorted_indices:
                el = y_pred[idx]
                if el == 1:
                    cts_temp = cts[idx].squeeze()
                    mean_pix = np.mean(cts_temp)
                    pseudomask_true[cts_temp >= mean_pix] = idx

        pseudomask_true = fill_closest_values(pseudomask_true, -1)
        return pseudomask_true

elif DATASET == "ade20k":
    # -------------------- ADE20K CONFIG --------------------
    from datasets.ade import ADE20KSegmentation

    NUM_CLASSES = 151
    IMAGE_SIZE: Tuple[int, int] = (208, 320)
    PATCH_SIZE = 8
    IN_CHANS = 3

    DATA_ROOT = "/ds2/computer_vision/ADEChallengeData2016"
    SET = "train"

    # If you have a Lightning checkpoint dir instead, you can use find_best_checkpoint.
    CHECKPOINT_DIR = None
    CHECKPOINT_PATH = "/netscratch2/jhanna/vit_wsss_mct/args.dataset=0-epoch=10-val_accuracy=0.95.ckpt"

    def build_dataset():
        return ADE20KSegmentation(
            DATA_ROOT,
            split=SET,
            transform=True,
        )

    def ade_display_mct(attn: np.ndarray) -> List[np.ndarray]:
        """
        Convert attention maps (for active classes) to upsampled masks.

        Parameters
        ----------
        attn : np.ndarray
            Attention maps for selected classes, shape (K, num_patches).

        Returns
        -------
        List[np.ndarray]
            List of upsampled attention masks, shape (H, W, 1) each.
        """
        mask_rs = []
        grid_h = IMAGE_SIZE[0] // PATCH_SIZE
        grid_w = IMAGE_SIZE[1] // PATCH_SIZE

        for i in range(attn.shape[0]):
            mask = attn[i]
            mask = mask.reshape(grid_h, grid_w)
            mask_r = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]))[..., np.newaxis]
            mask_rs.append(mask_r)
        return mask_rs

    def ade_generate_pseudomask(
        y_pred_indices: np.ndarray,
        label_indices: np.ndarray,
        mask: np.ndarray,
        cts: List[np.ndarray],
        logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate a pseudo-mask for ADE20K using class indices and attention.

        Parameters
        ----------
        y_pred_indices : np.ndarray
            Indices of predicted positive classes (where model prediction is 1).
        label_indices : np.ndarray
            Indices of positive ground-truth classes (where label == 1).
        mask : np.ndarray
            Ground-truth mask array, used for shape (H, W) or (H, W, 1).
        cts : list of np.ndarray
            Attention maps corresponding to `label_indices`.
        logits : np.ndarray, optional
            Logits for positive classes, aligned with `label_indices`.

        Returns
        -------
        np.ndarray
            Pseudo-mask with class indices (0..150).
        """
        if mask.ndim == 3:
            base_shape = mask.shape[:2]
        else:
            base_shape = mask.shape

        pseudomask_true = np.ones(base_shape) * -1

        if logits is None:
            for idx, class_id in enumerate(label_indices):
                cts_temp = cts[idx].squeeze()
                mean_pix = np.mean(cts_temp)
                pseudomask_true[cts_temp >= mean_pix] = class_id
        else:
            sorted_indices = sorted(range(len(logits)), key=lambda k: logits[k])
            for idx in sorted_indices:
                class_id = label_indices[idx]
                cts_temp = cts[idx].squeeze()
                mean_pix = np.mean(cts_temp)
                if mean_pix != 0:
                    pseudomask_true[cts_temp >= mean_pix] = class_id

        pseudomask_true = fill_closest_values(pseudomask_true, -1)
        # Optionally also close small holes labeled as 0
        pseudomask_true = fill_closest_values(pseudomask_true, 0)
        return pseudomask_true

else:
    raise ValueError(f"Unknown DATASET '{DATASET}', expected 'dfc' or 'ade20k'")


def load_model() -> Recorder:
    """
    Instantiate ViTWithTokenDropout, load checkpoint, wrap in Recorder, and move to device.

    Returns:
    Recorder
        Model wrapped with Recorder that returns intermediate attention maps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTWithTokenDropout(
        patch_size=PATCH_SIZE,
        prune=False,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        return_att=True,
    )

    if CHECKPOINT_PATH is not None:
        ckpt_path = CHECKPOINT_PATH
    elif CHECKPOINT_DIR is not None:
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
    else:
        raise RuntimeError("Either CHECKPOINT_PATH or CHECKPOINT_DIR must be set.")

    state = torch.load(ckpt_path, map_location=device)["state_dict"]
    # Remove "model." prefix used by Lightning
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    model = Recorder(model, prune=False).to(device)
    return model


def generate_all_pseudomasks(dataset, model, batch_size: int = 1):
    """
    Run the model on a dataset and build pseudo-masks for all samples.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset returning dicts with keys "img", "y", "mask".
    model : nn.Module
        Model wrapped in Recorder.
    batch_size : int, optional
        Batch size for the DataLoader.

    Returns
    -------
    pms : np.ndarray
        Pseudo-masks, stacked as (N, H, W) or (N, H, W, 1).
    imgs : np.ndarray
        Input images, stacked.
    masks : np.ndarray
        Ground-truth masks, stacked.
    """
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pms = []
    imgs = []
    masks = []

    for batch in loader:
        img = batch["img"].float().to(device)
        gt_mask = batch["mask"]
        labels = batch["y"]

        with torch.no_grad():
            out, x, *_ = model(img)
            logits = out.cpu().numpy().squeeze()  # shape (num_classes,)
            preds = np.round(logits).astype(int)  # binary multi-label

        # Recorder is assumed to return attention for class tokens in x[0]
        attn_cls = x[0].cpu().numpy()  # shape (num_classes, num_patches)

        if DATASET == "dfc":
            # DFC: labels is binary vector per class
            # attn_cls: (8, num_patches)
            cts = dfc_display_mct(attn_cls)
            pm = dfc_generate_pseudomask(
                y_pred=preds,
                mask=gt_mask.squeeze().numpy(),
                cts=cts,
                logits=logits,
            )

        elif DATASET == "ade20k":
            # ADE: labels is binary vector of length 151
            label_bin = labels.squeeze().numpy()  # (151,)
            pos_mask = label_bin == 1
            label_indices = np.where(pos_mask)[0]

            # Only keep attention for positive GT classes
            attn_selected = attn_cls[label_indices]
            cts = ade_display_mct(attn_selected)

            # Also select logits for positive GT classes
            logits_pos = logits[label_indices]

            pm = ade_generate_pseudomask(
                y_pred_indices=np.where(preds == 1)[0],
                label_indices=label_indices,
                mask=gt_mask.squeeze().numpy(),
                cts=cts,
                logits=logits_pos,
            )

        else:
            raise ValueError("Unknown DATASET in generation loop.")

        pms.append(pm)
        imgs.append(img.cpu().numpy().squeeze())
        masks.append(gt_mask.squeeze().numpy())

    return np.array(pms), np.array(imgs), np.array(masks)


if __name__ == "__main__":
    dataset = build_dataset()
    model = load_model()

    pms, imgs, masks = generate_all_pseudomasks(dataset, model)

    np.save(f"pms_{DATASET}_{SET}.npy", pms)
    np.save(f"imgs_{DATASET}_{SET}.npy", imgs)
    np.save(f"masks_{DATASET}_{SET}.npy", masks)
    print(f"Saved pseudo-masks for {DATASET} ({SET})")
