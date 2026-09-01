"""SimCLRv2 ResNet (PyTorch), vendored from Separius/SimCLRv2-Pytorch.

The architecture (stem / bottleneck blocks / selective kernels / projections /
contrastive head) matches the lightly-ai HuggingFace checkpoints EXACTLY
(they were converted with the Separius conversion script):

  - https://huggingface.co/lightly-ai/simclrv2-imagenet1k-r50_1x_sk1   (r50_1x_sk1.pth)
  - https://huggingface.co/lightly-ai/simclrv2-imagenet1k-r101_1x_sk1  (r101_1x_sk1.pth)

Checkpoint format (``torch.load``)::

    {
      'resnet': state_dict,   # net.0.* stem / net.{1..4}.blocks.* / fc.* (1000-cls, unused)
      'head':   state_dict,   # layers.{0,1,3,4,6,7}.* — 3-layer contrastive head -> 128-d
    }

Notes:
  - ``_sk1`` weights use sk_ratio=0.0625: ResNet-D style stem (3x3x3 convs, no 7x7)
    + SelectiveKernel blocks (``mixing_conv``) + AvgPool shortcut projections.
    Note the stem still ends with a MaxPool2d(3, stride=2), so a 224x224 input
    produces a 7x7 grid at the last stage (stride 32), like a vanilla ResNet.
  - ``resnet.fc`` is the 1000-class supervised head — NOT used for fine-tuning
    (we attach task heads instead); it is skipped by ``load_simclrv2_backbone``.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BATCH_NORM_EPSILON", "BATCH_NORM_DECAY",
    "BatchNormRelu", "conv", "SelectiveKernel", "Projection",
    "BottleneckBlock", "Blocks", "Stem", "ResNet", "ContrastiveHead",
    "get_resnet", "name_to_params", "SimCLRv2Backbone",
    "load_simclrv2_backbone", "build_backbone_from_ckpt_name",
]

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON),
                         nn.ReLU() if relu else nn.Identity())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2, bias=bias)


class SelectiveKernel(nn.Module):
    """SK conv (SimCLRv2, sk_ratio>0). Splits into 2 streams and mixes them
    with softmax weights computed from the global-pooled activations."""

    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(
            conv(in_channels, 2 * out_channels, stride=stride),
            BatchNormRelu(2 * out_channels))
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(
            conv(out_channels, mid_dim, kernel_size=1),
            BatchNormRelu(mid_dim),
            conv(mid_dim, 2 * out_channels, kernel_size=1))

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    """Shortcut projection for the first block of each stage."""

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),          # kernel_size = 2 => padding = 1
                nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                conv(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [conv(in_channels, out_channels, kernel_size=1),
               BatchNormRelu(out_channels)]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


class ResNet(nn.Module):
    def __init__(self, layers, width_multiplier, sk_ratio):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

    def forward(self, x, apply_fc=False):
        h = self.net(x).mean(dim=[2, 3])
        if apply_fc:
            h = self.fc(h)
        return h


class ContrastiveHead(nn.Module):
    """SimCLRv2 3-layer projection head -> out_dim (128). Optional for
    fine-tuning; kept so projection features can be extracted if needed."""

    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


def get_resnet(depth=50, width_multiplier=1, sk_ratio=0):  # sk_ratio=0.0625 is recommended
    layers = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
              152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}[depth]
    resnet = ResNet(layers, width_multiplier, sk_ratio)
    return resnet, ContrastiveHead(resnet.channels_out)


def name_to_params(checkpoint):
    """Infer (depth, width_multiplier, sk_ratio) from a checkpoint file name,
    e.g. 'simclrv2_r50_1x_sk1.pth' -> (50, 1, 0.0625)."""
    sk_ratio = 0.0625 if '_sk1' in checkpoint else 0
    if 'r50_' in checkpoint:
        depth = 50
    elif 'r101_' in checkpoint:
        depth = 101
    elif 'r152_' in checkpoint:
        depth = 152
    else:
        raise NotImplementedError(f"cannot infer resnet depth from: {checkpoint}")
    if '_1x_' in checkpoint:
        width = 1
    elif '_2x_' in checkpoint:
        width = 2
    elif '_3x_' in checkpoint:
        width = 3
    else:
        raise NotImplementedError(f"cannot infer width multiplier from: {checkpoint}")
    return depth, width, sk_ratio


class SimCLRv2Backbone(nn.Module):
    """Backbone used for fine-tuning: ``net`` (stem + 4 blocks) without the
    1000-class ``fc``. ``channels_out`` is 2048 for width_multiplier=1."""

    def __init__(self, depth=50, width_multiplier=1, sk_ratio=0.0625):
        super().__init__()
        self.depth = depth
        self.width_multiplier = width_multiplier
        self.sk_ratio = sk_ratio
        resnet, _head = get_resnet(depth, width_multiplier, sk_ratio)
        self.net = resnet.net
        self.channels_out = resnet.channels_out

    def forward(self, x):
        """Dense last-stage feature map: [B, 2048, H/32, W/32]."""
        return self.net(x)

    def global_features(self, x):
        """Global average-pooled features: [B, 2048] (SimCLR representation)."""
        return self.net(x).mean(dim=[2, 3])


def load_simclrv2_backbone(backbone, ckpt_path, strict=True, log_fn=print):
    """Load pretrained weights into a ``SimCLRv2Backbone`` from a lightly-ai
    checkpoint (``{'resnet': ..., 'head': ...}``). Only the ``net.*`` keys are
    loaded (the 1000-class ``fc.*`` and the contrastive ``head`` are skipped).

    Returns (missing, unexpected, ckpt).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'resnet' in ckpt and isinstance(ckpt['resnet'], dict):
        state = ckpt['resnet']
    else:  # tolerate plain state dicts
        state = ckpt
    # checkpoint keys are 'net.0.0.weight' ... ; backbone.net expects '0.0.weight' ...
    net_state = {k[len('net.'):]: v for k, v in state.items() if k.startswith('net.')}
    missing, unexpected = backbone.net.load_state_dict(net_state, strict=strict)
    if log_fn is not None:
        log_fn(f"[load_simclrv2_backbone] {os.path.basename(str(ckpt_path))}: "
               f"loaded {len(net_state)} tensors, missing={len(missing)}, "
               f"unexpected={len(unexpected)}")
        if unexpected:
            log_fn(f"  unexpected (skipped): {sorted(unexpected)[:8]} ...")
        if missing:
            log_fn(f"  missing: {sorted(missing)[:8]} ...")
    return missing, unexpected, ckpt


def build_backbone_from_ckpt_name(ckpt_path, depth=None, width=None, sk_ratio=None):
    """Infer arch from the file name unless explicit args override it."""
    if depth is None or width is None or sk_ratio is None:
        d, w, s = name_to_params(os.path.basename(str(ckpt_path)))
        depth = d if depth is None else depth
        width = w if width is None else width
        sk_ratio = s if sk_ratio is None else sk_ratio
    return SimCLRv2Backbone(depth=depth, width_multiplier=width, sk_ratio=sk_ratio)
