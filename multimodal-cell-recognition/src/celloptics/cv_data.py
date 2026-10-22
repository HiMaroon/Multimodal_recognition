import csv
import random
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv_config as cfg
from protocol_guard import MODALITY_COLUMNS, validate_cells, freeze


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def read_manifest(path: Path = cfg.MANIFEST_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = validate_cells(resolve_paths(list(csv.DictReader(f))))
    freeze("cell_manifest", [path])
    return rows


def resolve_paths(rows: list[dict]) -> list[dict]:
    path_columns = ("rd_path", "rb_path", "t_path", "mask_path")
    for row in rows:
        for column in path_columns:
            value = Path(row[column])
            if not value.is_absolute():
                row[column] = str(cfg.RELEASE_ROOT / value)
    return rows


def read_fold_assignments(path: Path = cfg.FOLD_ASSIGNMENT_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = validate_cells(resolve_paths(list(csv.DictReader(f))), development_only=True)
    freeze("cv_manifest", [path])
    return rows


def rows_for_fold(fold: int, train: bool) -> list[dict]:
    rows = read_fold_assignments()
    if train:
        return [row for row in rows if int(row["fold"]) != fold]
    return [row for row in rows if int(row["fold"]) == fold]


@dataclass
class BatchSpec:
    modalities: tuple[str, ...]
    train: bool = False


class ManifestDataset(Dataset):

    def __init__(self, rows: list[dict], spec: BatchSpec):
        self.rows = rows
        self.spec = spec

    def __len__(self) -> int:
        repeat = cfg.TRAIN_AUG_REPEAT if self.spec.train else 1
        return len(self.rows) * repeat

    def _image_preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, 0)
        image = np.transpose(image, [0, 3, 1, 2]).astype("float32")
        return image / 255.0

    def _resize_chw(self, array: np.ndarray, is_mask: bool = False) -> np.ndarray:
        c = array.shape[1]
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        out = [
            cv2.resize(array[0, ci], cfg.PATCH_SIZE[::-1], interpolation=interp)
            for ci in range(c)
        ]
        return np.stack(out, axis=0)

    def _rotate_chw(
        self, img: np.ndarray, angle: float, is_mask: bool = False
    ) -> np.ndarray:
        (h, w) = (img.shape[1], img.shape[2])
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        out = [
            cv2.warpAffine(
                img[ci],
                matrix,
                (w, h),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for ci in range(img.shape[0])
        ]
        return np.stack(out, axis=0)

    def _augment(
        self, images: list[np.ndarray], mask: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        images = [self._resize_chw(image, is_mask=False) for image in images]
        mask = np.round(self._resize_chw(mask, is_mask=True))
        if self.spec.train:
            if random.random() < 0.5:
                images = [image[:, :, ::-1].copy() for image in images]
                mask = mask[:, :, ::-1].copy()
            if random.random() < 0.5:
                images = [image[:, ::-1, :].copy() for image in images]
                mask = mask[:, ::-1, :].copy()
            if random.random() < 0.5:
                angle = random.uniform(-20.0, 20.0)
                images = [
                    self._rotate_chw(image, angle, is_mask=False) for image in images
                ]
                mask = self._rotate_chw(mask, angle, is_mask=True)
        return (images, mask)

    def __getitem__(self, index: int):
        row = self.rows[index % len(self.rows)]
        path_by_modality = {m: Path(row[c]) for m, c in MODALITY_COLUMNS.items()}
        images = [
            self._image_preprocess(read_image(path_by_modality[m]))
            for m in self.spec.modalities
        ]
        mask = np.round(self._image_preprocess(read_image(Path(row["mask_path"]))))
        (images, mask) = self._augment(images, mask)
        label = int(row["label"])
        return {
            "images": torch.from_numpy(np.concatenate(images, axis=0)),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "image_name": row["image_name"],
            "sample_id": row["sample_id"],
            "cell_type": row["cell_type"],
        }
