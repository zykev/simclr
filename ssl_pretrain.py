"""PyTorch SimCLR continued self-supervised pretraining for dental images.

This entry point intentionally has no label heads and never consumes record
labels.  It initializes a SimCLRv2 ResNet + projection head from the official
ImageNet checkpoint, then optimizes two independently augmented views using
the symmetric NT-Xent objective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import simclrv2_resnet as simclr_resnet
from ssl_dental_dataset import build_ssl_dataset, resolve_compatible_path


class SimCLRModel(nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, images):
        return self.projector(self.backbone.global_features(images))


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return rank() == 0


def setup_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return
    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method=args.dist_url)
    dist.barrier()


def gather_with_grad(tensor):
    """All-gather embeddings while retaining gradients for remote negatives."""
    if not is_distributed():
        return tensor
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(tensor), dim=0)


def nt_xent(z1, z2, temperature):
    """Symmetric SimCLR NT-Xent with global in-batch negatives."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    all_z1, all_z2 = gather_with_grad(z1), gather_with_grad(z2)
    local_batch = z1.shape[0]
    offset = rank() * local_batch
    targets = torch.arange(local_batch, device=z1.device) + offset
    diagonal = torch.arange(local_batch, device=z1.device)

    def direction(anchor, positives, same_view):
        logits_pos = anchor @ positives.T / temperature
        logits_same = anchor @ same_view.T / temperature
        logits_same[diagonal, targets] = torch.finfo(logits_same.dtype).min
        return F.cross_entropy(torch.cat([logits_pos, logits_same], dim=1), targets)

    return 0.5 * (direction(z1, all_z2, all_z1) + direction(z2, all_z1, all_z2))


def cosine_lr(optimizer, base_lr, min_lr, step, total_steps, warmup_steps):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(args, epoch, model, optimizer, records, loss):
    if not is_main():
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DDP) else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "ssl": {"objective": "symmetric_nt_xent", "temperature": args.temperature,
                "projection_dim": args.projection_dim, "categories": args.categories,
                "record_count": len(records), "loss": loss},
    }
    torch.save(payload, os.path.join(args.output_dir, f"checkpoint-{epoch:03d}.pth"))


def parse_args():
    parser = argparse.ArgumentParser("SimCLRv2 dental self-supervised continued pretraining")
    parser.add_argument("--data_path", default=str(resolve_compatible_path(".datasets/intraoral")))
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--categories", default="full",
                        help="SSL source records; default full avoids annotation-derived crops")
    parser.add_argument("--checkpoint", required=True, help="official SimCLRv2 ImageNet checkpoint")
    parser.add_argument("--depth", type=int, choices=(50, 101), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--sk_ratio", type=float, default=None)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32, help="per-GPU batch size")
    # Keep the no-odontify runtime recipe aligned with ft_train.py's full mode.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


def main(args):
    setup_distributed(args)
    seed = args.seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if args.distributed else "cuda")
    torch.backends.cudnn.benchmark = True

    dataset, records = build_ssl_dataset(args.data_path, args.split_json, args.split,
                                         args.categories, args.input_size)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if args.distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0)
    if len(loader) == 0:
        raise ValueError("dataset is smaller than one batch; reduce --batch_size")

    backbone = simclr_resnet.build_backbone_from_ckpt_name(
        args.checkpoint, args.depth, args.width, args.sk_ratio)
    # Persist resolved values rather than only optional CLI overrides.  The
    # representation adapter treats checkpoint args as the source of truth.
    args.depth = backbone.depth
    args.width_multiplier = backbone.width_multiplier
    args.sk_ratio = backbone.sk_ratio
    _, _, init_ckpt = simclr_resnet.load_simclrv2_backbone(backbone, args.checkpoint, strict=True)
    projector = simclr_resnet.ContrastiveHead(backbone.channels_out, args.projection_dim)
    if args.projection_dim == 128 and isinstance(init_ckpt, dict) and isinstance(init_ckpt.get("head"), dict):
        missing, unexpected = projector.load_state_dict(init_ckpt["head"], strict=False)
        if is_main():
            print(f"[projector init] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = SimCLRModel(backbone, projector).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    total_steps = args.epochs * len(loader)
    global_step = 0

    if is_main():
        print(f"[ssl] records={len(records)} batches/epoch={len(loader)} input={args.input_size} "
              f"world={args.world_size}; labels are not loaded into the model", flush=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum, n_steps, started = 0.0, 0, time.time()
        for view1, view2 in loader:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            lr = cosine_lr(optimizer, args.lr, args.min_lr, global_step, total_steps,
                           args.warmup_epochs * len(loader))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                loss = nt_xent(model(view1), model(view2), args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item()
            n_steps += 1
            global_step += 1
        mean_loss = loss_sum / max(1, n_steps)
        if is_main():
            print(json.dumps({"epoch": epoch + 1, "ssl_loss": round(mean_loss, 6),
                              "lr": lr, "seconds": round(time.time() - started, 1)}), flush=True)
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args, epoch + 1, model, optimizer, records, mean_loss)
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
