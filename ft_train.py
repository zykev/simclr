"""SimCLRv2 dental fine-tuning entry point (PyTorch, DDP + single-GPU).

Loads a lightly-ai SimCLRv2 pretrained checkpoint (``{'resnet': ..., 'head': ...}``,
Separius architecture — see ``simclrv2_resnet.py``), attaches multi-task linear
heads aligned with the ``compare/`` probe tasks, and fine-tunes on the same
case-level dental split used by the other baselines.

Tasks (multi-task CE on the same backbone features):

  fdi / tooth_class  <- tooth crops (str(FDI) / str(FDI % 10))
  sextant            <- sextant crops
  view               <- full-image crops

Modes:
  linear   freeze the whole backbone (BN eval; classic linear probe, SGD 0.01)
  partial  freeze stem + first ``--freeze_blocks`` stage(s) (default 2)
  full     fine-tune everything (AdamW 3e-4)

Checkpoints follow the Swin-MAE convention (``checkpoint-<epoch>.pth`` with
``{'model', 'optimizer', 'epoch', 'args', 'label_maps', 'split_metadata'}``) so
the ``compare/`` adapter can consume them.

Usage:
  torchrun --nproc_per_node=8 ft_train.py --checkpoint <pretrained.pth> \\
      --data_path <root> --split_json <split.json> --split train --output_dir <exp>
  python ft_train.py --checkpoint <pretrained.pth> --batch_size 32 --device cuda:0 ...
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import simclrv2_resnet as simclr_resnet
from dental_dataset import (
    DEFAULT_TASKS,
    TASK_DEFS,
    build_datasets,
    collate_supervised,
    parse_task_weights,
)


# --------------------------------------------------------------------------
# distributed helpers (mirror baselines/Swin-MAE/utils/misc.py)
# --------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        print(f"| distributed init (rank {args.rank}, world {args.world_size}, gpu {args.gpu})",
              flush=True)
    else:
        args.distributed = False
        print("Not using distributed mode")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class FineTuneModel(nn.Module):
    """SimCLRv2 backbone + one linear head per task."""

    def __init__(self, backbone, task_num_classes):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            task: nn.Linear(backbone.channels_out, n)
            for task, n in task_num_classes.items()})

    def forward(self, x):
        feats = self.backbone.global_features(x)
        return {task: head(feats) for task, head in self.heads.items()}


def _backbone_block_index(name):
    """'backbone.net.2.blocks.0...' -> 2 (net.0 = stem, net.1..4 = stages)."""
    parts = name.split(".")
    if "net" in parts:
        i = parts.index("net") + 1
        if i < len(parts) and parts[i].isdigit():
            return int(parts[i])
    return None


def apply_finetune_mode(model, mode, freeze_blocks):
    for p in model.parameters():
        p.requires_grad = True
    if mode == "linear":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif mode == "partial":
        for name, p in model.backbone.named_parameters():
            idx = _backbone_block_index(name)
            if idx is not None and idx <= freeze_blocks:
                p.requires_grad = False
    # heads always train
    return model


def set_bn_eval(module):
    """Freeze BN running statistics while training (SimCLRv2 small-batch caveat)."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# --------------------------------------------------------------------------
# loss / metrics
# --------------------------------------------------------------------------

def build_class_weights_from_records(label_maps, records, tasks):
    """Inverse-frequency class weights per task (computed on train records)."""
    from collections import Counter
    weights = {}
    for task in tasks:
        crop_type = TASK_DEFS[task]["crop_type"]
        counter = Counter()
        for r in records:
            if r["crop_type"] == crop_type:
                counter[TASK_DEFS[task]["label_fn"](r)] += 1
        total = sum(counter.values())
        n = len(label_maps[task])
        w = torch.zeros(n)
        for label, c in counter.items():
            if label in label_maps[task]:
                w[label_maps[task][label]] = total / max(1, c * n)
        weights[task] = w
    return weights


def compute_loss(model, images, labels, tasks, task_weights, class_weights):
    logits = model(images)
    total = torch.zeros((), device=images.device)
    per_task = {}
    for task in tasks:
        lab = labels[task].to(images.device)
        mask = lab >= 0
        if mask.sum() == 0:
            continue
        lg = logits[task][mask]
        w = class_weights.get(task) if class_weights else None
        if w is not None:
            w = w.to(images.device)
        loss = F.cross_entropy(lg, lab[mask], weight=w)
        total = total + loss * task_weights[task]
        per_task[task] = loss.item()
    return total, per_task


@torch.no_grad()
def evaluate(model, dataset, tasks, label_maps, device, batch_size=64,
             num_workers=4, pin_mem=True):
    """Per-task accuracy / macro-F1 on a dataset (single process).

    Samples whose gold label is absent from the train label map are counted as
    ``unsupported`` and excluded from accuracy/F1 (compare/ probe convention).
    """
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        collate_fn=lambda b: collate_supervised(b, tasks), pin_memory=pin_mem)
    stats = {task: {"correct": 0, "total": 0, "unsupported": 0,
                    "tp": defaultdict(int), "fp": defaultdict(int),
                    "fn": defaultdict(int)}
             for task in tasks}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        for task in tasks:
            st = stats[task]
            lab = labels[task]
            mask = lab >= 0
            if mask.sum() == 0:
                continue
            gold = lab[mask].tolist()
            preds = logits[task][mask].argmax(dim=1).cpu().tolist()
            for g, p in zip(gold, preds):
                if g == p:
                    st["correct"] += 1
                    st["tp"][g] += 1
                else:
                    st["fp"][p] += 1
                    st["fn"][g] += 1
            st["total"] += len(gold)
        for task, lab in labels.items():
            stats[task]["unsupported"] += int((lab < 0).sum())

    metrics = {}
    for task in tasks:
        st = stats[task]
        acc = st["correct"] / st["total"] if st["total"] else 0.0
        classes = sorted(set(st["tp"]) | set(st["fp"]) | set(st["fn"]))
        f1s = []
        for c in classes:
            tp, fp, fn = st["tp"][c], st["fp"][c], st["fn"][c]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        metrics[task] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "n_samples": st["total"],
            "n_unsupported": st["unsupported"],
        }
    return metrics


# --------------------------------------------------------------------------
# scheduler
# --------------------------------------------------------------------------

def lr_lambda_fn(warmup_steps, total_steps, min_lr, base_lr):
    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
    return f


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

def save_checkpoint(args, epoch, model_without_ddp, optimizer, label_maps,
                    task_weights, split_metadata, val_metrics):
    if not is_main_process():
        return None
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"checkpoint-{epoch:03d}.pth")
    torch.save({
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "label_maps": label_maps,
        "task_weights": task_weights,
        "split_metadata": split_metadata,
        "val_metrics": val_metrics,
    }, path)
    print(f"[save] {path}", flush=True)
    return path


def load_resume(args, model_without_ddp, optimizer):
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("epoch", 0))
    print(f"[resume] {args.resume} -> start_epoch={start}")
    return start


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, args, tasks,
                    task_weights, class_weights, scaler):
    model.train()
    if getattr(args, "freeze_bn", False):
        set_bn_eval(model)
    header = f"epoch [{epoch}]"
    running = defaultdict(float)
    n_steps = 0
    accum = max(1, args.accum_iter)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, per_task = compute_loss(
                    model, images, labels, tasks, task_weights, class_weights)
            scaler.scale(loss / accum).backward()
        else:
            loss, per_task = compute_loss(
                model, images, labels, tasks, task_weights, class_weights)
            (loss / accum).backward()

        for task, v in per_task.items():
            running[f"loss_{task}"] += v
        running["loss"] += loss.item()

        if (step + 1) % accum == 0:
            if scaler is not None:
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.clip_grad)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_steps += 1

    if (step + 1) % accum != 0:  # flush tail
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        n_steps += 1

    elapsed = time.time() - t0
    n_seen = len(loader) * images.shape[0]
    stats = {k: round(v / max(1, len(loader)), 4) for k, v in running.items()}
    stats["lr"] = optimizer.param_groups[0]["lr"]
    stats["samples_per_sec"] = round(n_seen / max(1e-6, elapsed), 1)
    if is_main_process():
        print(f"{header} " + " ".join(f"{k}={v}" for k, v in stats.items()),
              flush=True)
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("SimCLRv2 dental fine-tuning", add_help=False)
    # data
    parser.add_argument("--data_path", default=".datasets/intraoral", type=str)
    parser.add_argument("--split_json", default=None, type=str,
                        help="case-level split JSON (datasets/split.py)")
    parser.add_argument("--split", default="train", type=str,
                        choices=("train", "test"))
    parser.add_argument("--categories", default="full,tooth,sextant", type=str)
    parser.add_argument("--tasks", default=DEFAULT_TASKS, type=str,
                        help="comma list of fdi,tooth_class,sextant,view")
    parser.add_argument("--task_weights", default=None, type=str,
                        help="e.g. 'fdi:1.0,tooth_class:0.5'")
    parser.add_argument("--class_weighted", action="store_true",
                        help="inverse-frequency class weights per task")
    # model / pretrained
    parser.add_argument("--checkpoint", default="", type=str,
                        help="lightly-ai SimCLRv2 .pth (e.g. simclrv2_r50_1x_sk1.pth)")
    parser.add_argument("--depth", default=None, type=int, choices=(50, 101))
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--sk_ratio", default=None, type=float)
    parser.add_argument("--input_size", default=448, type=int,
                        help="train/eval resolution; default 448 (compare/ models.yaml 对齐)")
    parser.add_argument("--finetune_mode", default="full",
                        choices=("linear", "full", "partial"))
    parser.add_argument("--freeze_blocks", default=2, type=int,
                        help="partial mode: freeze stem + first N stages")
    parser.add_argument("--freeze_bn", default=None, type=lambda s: s.lower() in ("1", "true", "yes"),
                        help="freeze BN stats during training; default: True for linear")
    # optimization
    parser.add_argument("--opt", default=None, choices=("adamw", "sgd"))
    parser.add_argument("--lr", default=None, type=float,
                        help="default: 0.01 (linear/sgd) or 3e-4 (full/partial/adamw)")
    parser.add_argument("--min_lr", default=None, type=float,
                        help="cosine 最终学习率；默认 0.1×lr（与预训练 lr_end 惯例一致，微调不衰减到 0）")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="per-GPU batch; default 32 (448 分辨率合理值; 224 可调大)")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--train_aug", default="default",
                        choices=("default", "strong", "none"))
    # misc
    parser.add_argument("--output_dir", default="./exp/simclrv2_dental_448", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int, dest="local_rank")
    return parser.parse_args()


def main(args):
    init_distributed_mode(args)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.distributed:
        device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # -- data ---------------------------------------------------------------
    train_dataset, val_dataset, label_maps, split_metadata = build_datasets(args)
    tasks = train_dataset.tasks
    task_weights = parse_task_weights(args.task_weights, tasks)
    if len(train_dataset) == 0:
        raise ValueError(f"No training records under {args.data_path}")

    # -- model ---------------------------------------------------------------
    if not args.checkpoint:
        raise ValueError("--checkpoint (lightly-ai SimCLRv2 .pth) is required")
    backbone = simclr_resnet.build_backbone_from_ckpt_name(
        args.checkpoint, depth=args.depth, width=args.width, sk_ratio=args.sk_ratio)
    model = FineTuneModel(
        backbone, {task: len(m) for task, m in label_maps.items()})
    simclr_resnet.load_simclrv2_backbone(
        model.backbone, args.checkpoint, strict=True)
    apply_finetune_mode(model, args.finetune_mode, args.freeze_blocks)
    model.to(device)

    # 记录真实架构到 args（adapter / eval 读取）
    args.depth = backbone.depth
    args.width = backbone.width_multiplier
    args.sk_ratio = backbone.sk_ratio
    args.model_arch = {
        "depth": backbone.depth,
        "width_multiplier": backbone.width_multiplier,
        "sk_ratio": backbone.sk_ratio,
    }

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model_without_ddp.parameters()
                      if p.requires_grad)
    n_total = sum(p.numel() for p in model_without_ddp.parameters())
    if is_main_process():
        print(f"[model] depth={backbone.depth} width={backbone.width_multiplier} "
              f"sk_ratio={backbone.sk_ratio} channels={backbone.channels_out}")
        print(f"[model] mode={args.finetune_mode} trainable={n_trainable/1e6:.2f}M / "
              f"{n_total/1e6:.2f}M total")
        print(f"[data] train_records={len(train_dataset)} val_records={len(val_dataset)} "
              f"tasks={tasks}")
        for task in tasks:
            print(f"  task[{task}] classes={len(label_maps[task])} "
                  f"labels={list(label_maps[task])[:12]}{'...' if len(label_maps[task]) > 12 else ''}")

    # -- optimizer / scheduler ------------------------------------------------
    if args.opt is None:
        args.opt = "sgd" if args.finetune_mode == "linear" else "adamw"
    if args.lr is None:
        args.lr = 0.01 if args.finetune_mode == "linear" else 3e-4
    if args.min_lr is None:
        args.min_lr = 0.1 * args.lr
    if args.weight_decay is None:
        args.weight_decay = 0.0 if args.finetune_mode == "linear" else 0.05
    if args.freeze_bn is None:
        args.freeze_bn = (args.finetune_mode == "linear")

    trainable = [p for p in model_without_ddp.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters (check --finetune_mode/--freeze_blocks)")
    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(
            trainable, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=args.sgd_nesterov)

    world = get_world_size()
    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.batch_size * world))
    total_steps = steps_per_epoch * args.epochs // max(1, args.accum_iter)
    warmup_steps = steps_per_epoch * args.warmup_epochs // max(1, args.accum_iter)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_fn(warmup_steps, total_steps, args.min_lr, args.lr))

    scaler = None
    if args.amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    class_weights = None
    if args.class_weighted:
        class_weights = build_class_weights_from_records(
            label_maps, train_dataset.records, tasks)

    start_epoch = 0
    if args.resume:
        start_epoch = load_resume(args, model_without_ddp, optimizer)

    # -- samplers / loaders ----------------------------------------------------
    if args.distributed:
        sampler_train = DistributedSampler(
            train_dataset, num_replicas=world, rank=get_rank(), shuffle=True)
    else:
        sampler_train = RandomSampler(train_dataset)
    loader_train = DataLoader(
        train_dataset, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_supervised(b, tasks),
        pin_memory=args.pin_mem, drop_last=True)

    log_dir = args.log_dir or args.output_dir
    if args.output_dir and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a",
                  encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "config",
                "args": {k: v for k, v in vars(args).items()},
                "label_maps": {t: list(m) for t, m in label_maps.items()},
                "split_metadata": split_metadata,
            }) + "\n")

    # -- training loop ----------------------------------------------------------
    print(f"Start training for {args.epochs} epochs "
          f"(steps/epoch={steps_per_epoch}, total_steps={total_steps})", flush=True)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, loader_train, optimizer, device, epoch, args,
            tasks, task_weights, class_weights, scaler)
        scheduler.step()

        val_metrics = None
        if (epoch + 1) % args.eval_freq == 0 and is_main_process() and len(val_dataset) > 0:
            val_metrics = evaluate(
                model_without_ddp, val_dataset, tasks, label_maps, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_mem=args.pin_mem)
            print(f"  val: " + " ".join(
                f"{t}(acc={m['accuracy']},f1={m['macro_f1']})"
                for t, m in val_metrics.items()), flush=True)

        if args.output_dir and is_main_process() and \
                ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            save_checkpoint(args, epoch + 1, model_without_ddp, optimizer,
                            label_maps, task_weights, split_metadata, val_metrics)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch}
        if val_metrics:
            for task, m in val_metrics.items():
                log_stats[f"val_{task}_acc"] = m["accuracy"]
                log_stats[f"val_{task}_f1"] = m["macro_f1"]
        if args.output_dir and is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
