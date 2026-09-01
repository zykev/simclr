"""Supervised fine-tuning dataset for SimCLRv2 on the dental dataset.

Mirrors ``baselines/Swin-MAE/tooth_dataset.py`` conventions (case-level split,
re-cutting crops from the full-resolution source image, ImageNet normalization)
but returns supervised labels aligned with the ``compare/`` probe tasks:

  task          crop_type   label source              example
  -----------   ---------   ------------------------  -------
  fdi           tooth       str(FDI)                  "16"
  tooth_class   tooth       str(int(FDI) % 10)        "6"   (1..8, compare §5)
  sextant       sextant     str(sextant id)           "S1"
  view          full        str(view)                 "F"

Task label maps are built from the *train* records (sorted label strings) and
stored in the checkpoint so ``ft_eval.py`` reproduces the exact mapping.
Test samples whose gold label is absent from the map are excluded from that
task's metrics (compare's "unsupported" convention).
"""
import random

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import dental_records

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_TASKS = "fdi,tooth_class,sextant,view"
DEFAULT_CATEGORIES = "full,tooth,sextant"  # group 标签空间过大，不用于监督微调

# task -> (crop_type it consumes, label extraction)
TASK_DEFS = {
    "fdi": {"crop_type": "tooth", "label_fn": lambda rec: str(rec["label"])},
    "tooth_class": {"crop_type": "tooth",
                    "label_fn": lambda rec: str(int(rec["label"]) % 10)},
    "sextant": {"crop_type": "sextant", "label_fn": lambda rec: str(rec["label"])},
    "view": {"crop_type": "full", "label_fn": lambda rec: str(rec["label"])},
}

CROP_TYPES_FOR_TASKS = sorted({d["crop_type"] for d in TASK_DEFS.values()})


def parse_tasks(spec):
    if spec is None:
        return list(TASK_DEFS)
    if isinstance(spec, str):
        tasks = [t.strip() for t in spec.split(",") if t.strip()]
    else:
        tasks = list(spec)
    unknown = set(tasks) - set(TASK_DEFS)
    if unknown:
        raise ValueError(f"unsupported tasks: {sorted(unknown)} (expected subset of {list(TASK_DEFS)})")
    return tasks


def parse_task_weights(spec, tasks):
    """'fdi:1.0,tooth_class:0.5' -> {task: float}; unspecified tasks default to 1.0."""
    weights = {task: 1.0 for task in tasks}
    if spec:
        for token in str(spec).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                task, w = token.split(":", 1)
                weights[task.strip()] = float(w)
            else:
                weights[token] = 1.0
    return weights


def build_label_maps(records, tasks):
    """{task: {label_str: class_id}} built from train records, sorted by label."""
    maps = {}
    for task in tasks:
        crop_type = TASK_DEFS[task]["crop_type"]
        labels = sorted({TASK_DEFS[task]["label_fn"](r) for r in records
                         if r["crop_type"] == crop_type})
        maps[task] = {label: i for i, label in enumerate(labels)}
    return maps


def _crop_from_full(full_img, box_original):
    """Cut a crop out of the full-resolution image. Boxes are floats; floor/ceil
    the bounds so every valid box covers at least one source pixel."""
    x1, y1, x2, y2 = box_original
    return full_img.crop((int(x1), int(y1), int(x2) + 1, int(y2) + 1))


class SupervisedDentalDataset(Dataset):
    """Crop records (full/tooth/sextant) with supervised labels for the tasks.

    A record contributes a label to every task whose crop_type it matches
    (e.g. a tooth record contributes to both ``fdi`` and ``tooth_class``).
    Labels outside the train label map are encoded as -1 (ignored in the loss
    and in eval metrics).
    """

    def __init__(self, records, label_maps, tasks, transform=None, is_train=True,
                 seed=0):
        self.records = [r for r in records if r["crop_type"] in CROP_TYPES_FOR_TASKS]
        self.label_maps = label_maps
        self.tasks = list(tasks)
        self.transform = transform
        self.is_train = is_train
        self.seed = seed
        self.exposures = {}  # sample_key -> count (coverage reporting)

    def set_epoch(self, epoch):
        if not self.is_train:
            return
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)
        self.records = [self.records[i] for i in indices]

    def __len__(self):
        return len(self.records)

    def _labels_for(self, record):
        labels = {}
        for task in self.tasks:
            if TASK_DEFS[task]["crop_type"] != record["crop_type"]:
                continue
            label_str = TASK_DEFS[task]["label_fn"](record)
            labels[task] = self.label_maps[task].get(label_str, -1)
        return labels

    def __getitem__(self, index):
        record = self.records[index]
        if self.is_train:
            self.exposures[record["sample_key"]] = self.exposures.get(
                record["sample_key"], 0) + 1
        full_img = PIL.Image.open(record["source_image_path"]).convert("RGB")
        if record["crop_type"] == "full":
            img = full_img
        else:
            img = _crop_from_full(full_img, record["box_original"])
        if self.transform is not None:
            img = self.transform(img)
        return img, self._labels_for(record)


def collate_supervised(batch, tasks):
    """(images, {task: LongTensor with -1 for non-applicable samples})."""
    images = [b[0] for b in batch]
    images = torch.stack(images, dim=0)
    labels = {}
    for task in tasks:
        labels[task] = torch.tensor([b[1].get(task, -1) for b in batch],
                                    dtype=torch.long)
    return images, labels


def build_transform(is_train, args):
    """Mirror Swin-MAE's transform; input_size driven; --train_aug selects strength."""
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    input_size = getattr(args, "input_size", 224)

    if is_train:
        aug = getattr(args, "train_aug", "default")
        if aug == "none":
            return build_transform(False, args)
        if aug == "strong":
            scale = (0.5, 1.0)
        else:
            scale = (0.7, 1.0)
        ops = [
            transforms.RandomResizedCrop(
                input_size, scale=scale, ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if aug == "strong":
            ops += [
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                           saturation=0.15, hue=0.03)], p=0.5),
                transforms.RandomRotation(degrees=10,
                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                          fill=0),
            ]
        ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(ops)

    # eval transform
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    size = int(input_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_datasets(args):
    """Build (train_dataset, val_dataset, label_maps, split_metadata) from the
    case-level split; label maps always come from the *train* partition."""
    from utils import dental_records
    tasks = parse_tasks(getattr(args, "tasks", None) or DEFAULT_TASKS)
    categories = getattr(args, "categories", None) or DEFAULT_CATEGORIES
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]

    train_records = dental_records.build_records(
        args.data_path, args.split_json, args.split, categories)
    val_split = "test" if args.split == "train" else "train"
    val_records = dental_records.build_records(
        args.data_path, args.split_json, val_split, categories)

    label_maps = build_label_maps(train_records, tasks)

    train_dataset = SupervisedDentalDataset(
        train_records, label_maps, tasks,
        transform=build_transform(True, args), is_train=True, seed=args.seed)
    val_dataset = SupervisedDentalDataset(
        val_records, label_maps, tasks,
        transform=build_transform(False, args), is_train=False, seed=args.seed)

    split_metadata = {
        "split_json": str(args.split_json),
        "split": args.split,
        "categories": ",".join(categories),
        "tasks": tasks,
        "data_root": str(args.data_path),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "label_maps": {t: list(m) for t, m in label_maps.items()},
    }
    return train_dataset, val_dataset, label_maps, split_metadata
