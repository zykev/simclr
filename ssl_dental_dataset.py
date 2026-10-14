"""Unlabelled two-view dental dataset for SimCLR continued pretraining.

Records come from the case-level split, but their FDI/sextant/view labels are
deliberately never returned to the training loop.  The default uses only the
original full-mouth images so annotation bboxes are not a hidden source of
spatial supervision.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# The workspace itself also has a top-level ``utils`` package.  Import this
# baseline's vendored records helper by its explicit directory, so launching
# from the workspace root and launching from ``baselines/simclr`` behave alike.
_LOCAL_UTILS = str(Path(__file__).resolve().parent / "utils")
if _LOCAL_UTILS not in sys.path:
    sys.path.insert(0, _LOCAL_UTILS)
import dental_records  # noqa: E402

resolve_compatible_path = dental_records.resolve_compatible_path

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _crop_from_full(full_img: Image.Image, box_original):
    if box_original is None:
        return full_img
    x1, y1, x2, y2 = box_original
    return full_img.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))


class TwoViewDentalDataset(Dataset):
    """Returns two independently augmented views of one unlabelled record."""

    def __init__(self, records, transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["source_image_path"]) as opened:
            image = _crop_from_full(opened.convert("RGB"), record["box_original"])
        return self.transform(image), self.transform(image)


def build_ssl_transform(input_size: int):
    """Canonical SimCLR-style augmentation, adapted only for RGB dental images."""
    color = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.05)
    kernel = max(3, int(round(input_size * 0.1)) | 1)
    return transforms.Compose([
        transforms.RandomResizedCrop(
            input_size, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def build_ssl_dataset(data_path, split_json, split="train", categories="full", input_size=448):
    if isinstance(categories, str):
        categories = [item.strip() for item in categories.split(",") if item.strip()]
    records = dental_records.build_records(data_path, split_json, split, categories)
    if not records:
        raise ValueError("no dental records selected for SimCLR SSL pretraining")
    return TwoViewDentalDataset(records, build_ssl_transform(input_size)), records
