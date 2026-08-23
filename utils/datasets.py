"""FG-SBIR dataset loaders for Sketchy, ShoeV2, ChairV2, and TU-Berlin.

The manuscript uses four datasets with NOT identical directory layouts and split
protocols. We do not assume a single layout: each dataset has its own configurable
root and split-file paths (see configs/default.yaml -> data.datasets).

Split files are plain text CSV/TSV with columns:
  - train.csv / val.csv / test.csv :  image_path,sketch_path,label      (paired)
  - query.csv                      :  sketch_path,                     label
  - gallery.csv                    :  photo_path,                      label
The loader treats the header row as optional (detected by the column names).
Paths are resolved relative to the dataset root unless absolute.

Three dataset classes are provided:
  - FGSBIRTrainDataset: paired (sketch, photo, label) for training/val triplet.
  - SketchQueryDataset:  queries only -> (sketch, label)  for retrieval.
  - PhotoGalleryDataset:  gallery photos -> (photo, label) for retrieval.
"""

from __future__ import annotations

import csv
import os

import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# split file parsing
# ---------------------------------------------------------------------------

def _read_split(csv_file: str):
    """Return list of dicts. Detects columns. Tolerates "," or \\t."""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(
            f"Split file not found: {csv_file}. Provide it under the dataset root; "
            "paths are configurable in configs/default.yaml -> data.datasets.")
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)
        all_rows = [r for r in reader if any(c.strip() for c in r)]
    if not all_rows:
        return rows
    header = [h.strip().lower() for h in all_rows[0]]
    body = all_rows[1:]
    # detect column roles
    def col(name_candidates):
        for h in header:
            if h in name_candidates:
                return header.index(h)
        return None
    c_img = col(["image_path", "photo_path", "image", "photo"])
    c_sk = col(["sketch_path", "sketch"])
    c_ph = col(["photo_path", "photo"])
    c_lab = col(["label", "class_id", "class", "label_id"])
    # if there is no header (unknown columns), fall back to positional fields
    has_header = any(x is not None for x in [c_img, c_sk, c_lab])
    for r in body:
        r = [c.strip() for c in r]
        if not has_header:
            # positional: assume image_path,sketch_path,label OR sketch,label OR photo,label
            rec = {}
            if len(r) >= 3:
                rec = dict(image_path=r[0], sketch_path=r[1], label=r[2])
            elif len(r) == 2:
                rec = dict(sketch_path=r[0], label=r[1])
        else:
            rec = {}
            if c_lab is not None and c_lab < len(r):
                rec["label"] = r[c_lab]
            # paired: image_path + sketch_path
            if c_img is not None and c_img < len(r):
                rec["image_path"] = r[c_img]
            if c_sk is not None and c_sk < len(r):
                rec["sketch_path"] = r[c_sk]
            if c_ph is not None and c_ph < len(r):
                rec["photo_path"] = r[c_ph]
        if rec:
            rows.append(rec)
    return rows


def _resolve(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def _load_image(path: str, mode: str) -> Image.Image:
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {path}")
    return img.convert(mode)


# ---------------------------------------------------------------------------
# dataset classes
# ---------------------------------------------------------------------------

class FGSBIRTrainDataset(Dataset):
    """Paired (sketch, photo, label) training/validation dataset."""

    def __init__(self, root: str, split_file: str, transform, size: int | None = None):
        self.root = root
        self.transform = transform  # tuple (sketch_tf, photo_tf)
        self.samples = _read_split(os.path.join(root, split_file))
        if not self.samples:
            raise RuntimeError(f"No samples parsed from {os.path.join(root, split_file)}")
        if size is not None:
            self.samples = self.samples[:size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        photo_path = s.get("image_path") or s.get("photo_path")
        sketch_path = s["sketch_path"]
        label = int(s["label"])
        photo = _load_image(_resolve(self.root, photo_path), "RGB")
        sketch = _load_image(_resolve(self.root, sketch_path), "L")
        if self.transform is not None:
            sketch_tf, photo_tf = self.transform
            sketch = sketch_tf(sketch)
            photo = photo_tf(photo)
        return sketch, photo, label


class SketchQueryDataset(Dataset):
    """Query sketches only -> (sketch, label) for retrieval."""

    def __init__(self, root: str, split_file: str, transform, size: int | None = None):
        self.root = root
        self.sketch_tf = transform[0] if isinstance(transform, tuple) else transform
        self.samples = _read_split(os.path.join(root, split_file))
        if not self.samples:
            raise RuntimeError(f"No queries parsed from {os.path.join(root, split_file)}")
        if size is not None:
            self.samples = self.samples[:size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sketch_path = s.get("sketch_path") or s.get("image_path") or s.get("photo_path")
        label = int(s["label"])
        sketch = _load_image(_resolve(self.root, sketch_path), "L")
        if self.sketch_tf is not None:
            sketch = self.sketch_tf(sketch)
        return sketch, label


class PhotoGalleryDataset(Dataset):
    """Gallery photos only -> (photo, label) for retrieval."""

    def __init__(self, root: str, split_file: str, transform, size: int | None = None,
                 restricted_gallery_size: int | None = None):
        self.root = root
        self.photo_tf = transform[1] if isinstance(transform, tuple) else transform
        self.samples = _read_split(os.path.join(root, split_file))
        if not self.samples:
            raise RuntimeError(f"No gallery photos parsed from {os.path.join(root, split_file)}")
        if restricted_gallery_size is not None:
            # truncate to the manuscript gallery size if the split is larger
            self.samples = self.samples[:restricted_gallery_size]
        if size is not None:
            self.samples = self.samples[:size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        photo_path = s.get("photo_path") or s.get("image_path")
        label = int(s["label"])
        photo = _load_image(_resolve(self.root, photo_path), "RGB")
        if self.photo_tf is not None:
            photo = self.photo_tf(photo)
        return photo, label


# ---------------------------------------------------------------------------
# dataset config registry + builders
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "sketchy":   dict(num_classes=25, gallery_size=500),
    "shoev2":    dict(num_classes=20, gallery_size=400),
    "chairv2":   dict(num_classes=15, gallery_size=250),
    "tu_berlin": dict(num_classes=50, gallery_size=400),
}


def get_dataset_cfg(cfg: dict, name: str) -> dict:
    """Resolve a dataset config, merging the registry defaults with the YAML."""
    d = DATASET_REGISTRY.get(name, {})
    user = cfg.get("data", {}).get("datasets", {}).get(name, {})
    return {**d, **user}  # user overrides registry defaults


def build_train_dataset(cfg: dict, name: str, transform, split: str = "train",
                        size: int | None = None):
    """Build a paired (sketch, photo, label) dataset for train or val."""
    dc = get_dataset_cfg(cfg, name)
    key = {"train": "train_split", "val": "val_split", "test": "test_split"}[split]
    return FGSBIRTrainDataset(dc["root"], dc[key], transform, size=size)


def build_split_dataset(cfg: dict, name: str, split: str, transform, restricted_gallery_size=None):
    """Build a query or gallery dataset for retrieval / validation.

    split: 'train' | 'val' | 'test' | 'query' | 'gallery'
    For 'train/val/test' we return the paired train dataset (used for val loss).
    For 'query'/'gallery' we return query-only / gallery-only datasets.
    """
    dc = get_dataset_cfg(cfg, name)
    if split in ("train", "val", "test"):
        key = {"train": "train_split", "val": "val_split", "test": "test_split"}[split]
        return FGSBIRTrainDataset(dc["root"], dc[key], transform)
    if split == "query":
        return SketchQueryDataset(dc["root"], dc["query_split"], transform)
    if split == "gallery":
        return PhotoGalleryDataset(dc["root"], dc["gallery_split"], transform,
                                   restricted_gallery_size=restricted_gallery_size)
    raise ValueError(f"unknown split '{split}'")
