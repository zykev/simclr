"""Evaluate a fine-tuned SimCLRv2 checkpoint on a dental split.

Loads a ``checkpoint-<epoch>.pth`` produced by ``ft_train.py`` (which stores
``args`` / ``label_maps`` / ``split_metadata``) and reports per-task
accuracy / macro-F1 / per-class recall + confusion matrices on the requested
partition (default ``test`` — the case-disjoint held-out split).

Metrics mirror the ``compare/`` probe conventions:
  - labels come from the train-built ``label_maps`` (stored in the ckpt);
  - test samples whose gold label is absent from the map are ``unsupported``
    and excluded from accuracy/F1 (reported as counts).

Usage:
  python ft_eval.py --checkpoint exp/simclrv2_r50_full_224/checkpoint-100.pth \\
      --data_path .datasets/intraoral --split_json .../train_test_seed0_test0p2.json \\
      --split test --output_dir exp/simclrv2_r50_full_224/eval
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import simclrv2_resnet as simclr_resnet
from dental_dataset import (
    TASK_DEFS,
    build_datasets,
    build_transform,
    collate_supervised,
    parse_tasks,
)
from ft_train import FineTuneModel, evaluate
from utils.dental_records import resolve_compatible_path


def parse_args():
    parser = argparse.ArgumentParser("SimCLRv2 fine-tuned eval", add_help=False)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="trained checkpoint-<epoch>.pth")
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--split_json", default=None, type=str)
    parser.add_argument("--split", default="test", type=str,
                        choices=("train", "test"))
    parser.add_argument("--categories", default=None, type=str)
    parser.add_argument("--tasks", default=None, type=str)
    parser.add_argument("--input_size", default=None, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default=None, type=str,
                        help="default: <checkpoint parent>/eval")
    return parser.parse_args()


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args")
    ckpt_args_d = ckpt_args.__dict__ if hasattr(ckpt_args, "__dict__") else {}

    # --- resolve config: CLI > ckpt args > defaults --------------------------
    data_path = str(resolve_compatible_path(
        args.data_path or ckpt_args_d.get("data_path", ".datasets/intraoral")))
    split_json = str(resolve_compatible_path(
        args.split_json or ckpt_args_d.get("split_json"))) \
        if (args.split_json or ckpt_args_d.get("split_json")) else None
    if split_json is None:
        raise ValueError("--split_json required (not stored in ckpt args)")
    categories = args.categories or ckpt_args_d.get("categories", "full,tooth,sextant")
    tasks = parse_tasks(args.tasks or ckpt_args_d.get("tasks") or "fdi,tooth_class,sextant,view")
    input_size = args.input_size or int(ckpt_args_d.get("input_size", 224))
    args.input_size = input_size  # build_transform reads args.input_size

    label_maps = ckpt.get("label_maps")
    if label_maps is None:  # 兜底：从 train records 重建
        from dental_dataset import build_label_maps
        from utils import dental_records
        train_records = dental_records.build_records(
            data_path, split_json, "train",
            [c.strip() for c in str(categories).split(",") if c.strip()])
        label_maps = build_label_maps(train_records, tasks)

    # --- rebuild model --------------------------------------------------------
    arch = ckpt_args_d.get("model_arch") or {}
    backbone = simclr_resnet.SimCLRv2Backbone(
        depth=int(arch.get("depth", ckpt_args_d.get("depth", 50))),
        width_multiplier=int(arch.get("width_multiplier", ckpt_args_d.get("width", 1))),
        sk_ratio=float(arch.get("sk_ratio", ckpt_args_d.get("sk_ratio", 0.0625))))
    model = FineTuneModel(backbone, {t: len(m) for t, m in label_maps.items()})
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- dataset ----------------------------------------------------------------
    from utils import dental_records
    records = dental_records.build_records(
        data_path, split_json, args.split,
        [c.strip() for c in str(categories).split(",") if c.strip()])

    from dental_dataset import SupervisedDentalDataset
    dataset = SupervisedDentalDataset(
        records, label_maps, tasks,
        transform=build_transform(False, args), is_train=False)

    if len(dataset) == 0:
        raise ValueError(f"no records for split={args.split} under {split_json}")

    # --- evaluate ----------------------------------------------------------------
    metrics = evaluate(model, dataset, tasks, label_maps, device,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers)
    print("\n=== eval summary ===")
    for task in tasks:
        m = metrics[task]
        print(f"task[{task}]: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"n={m['n_samples']} unsupported={m['n_unsupported']}")

    # --- confusion matrices / per-class recall ------------------------------------
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(output_dir, exist_ok=True)
    summary = {"checkpoint": args.checkpoint, "split": args.split, "metrics": metrics}
    with torch.no_grad():
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False,
                            collate_fn=lambda b: collate_supervised(b, tasks))
        per_class = {}
        confusions = {}
        for task in tasks:
            num_cls = len(label_maps[task])
            conf = np.zeros((num_cls, num_cls), dtype=int)
            recall = defaultdict(int)
            support = defaultdict(int)
            for images, labels in loader:
                images = images.to(device)
                lab = labels[task]
                mask = lab >= 0
                if mask.sum() == 0:
                    continue
                logits = model(images)[task][mask]
                preds = logits.argmax(dim=1).cpu().numpy()
                gold = lab[mask].numpy()
                for g, p in zip(gold, preds):
                    conf[g, p] += 1
                    support[g] += 1
            inv = {i: label for label, i in label_maps[task].items()}
            per_class[task] = {}
            for i in range(num_cls):
                tp = conf[i, i]
                rec = tp / support[i] if support[i] else 0.0
                per_class[task][inv[i]] = {
                    "support": int(support[i]),
                    "recall": round(float(rec), 4),
                }
            confusions[task] = {"classes": [inv[i] for i in range(num_cls)],
                                "matrix": conf.tolist()}
            np.save(os.path.join(output_dir, f"confusion_{task}.npy"), conf)
            with open(os.path.join(output_dir, f"confusion_{task}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"classes": [inv[i] for i in range(num_cls)],
                           "matrix": conf.tolist()}, f, ensure_ascii=False, indent=2)
    summary["per_class_recall"] = per_class

    summary_path = os.path.join(output_dir, f"eval_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main(parse_args())
