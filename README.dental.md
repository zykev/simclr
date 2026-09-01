# SimCLRv2 牙科微调（baselines/simclr）

把 [zykev/simclr](https://github.com/zykev/simclr.git)（TF1.x 原始实现）作为 git submodule 引入，
另在 `simclrv2_resnet.py` 中 **vendored 了 Separius/SimCLRv2-Pytorch 的 PyTorch 复刻**（与
lightly-ai HuggingFace 权重逐键对齐），在同一个病例级 split 上做监督微调，供 `compare/` 表征对比使用。

## 文件结构

```
baselines/simclr/
├── simclrv2_resnet.py        # PyTorch 模型：Stem + SelectiveKernel Bottleneck + Projection
│                             #   + SimCLRv2Backbone（net 输出 [B, C, H, W] 网格，stride 32）
│                             #   + load_simclrv2_backbone / build_backbone_from_ckpt_name
├── dental_dataset.py         # 监督微调数据集：full/tooth/sextant → fdi/tooth_class/sextant/view
├── ft_train.py               # 训练入口：linear/full/partial 三种微调模式，AMP、多卡 torchrun
├── ft_eval.py                # 评估入口：加载 checkpoint 在 train/val/test split 上出指标
├── utils/dental_records.py   # vendored 病例级 split/标注读取（与 Swin-MAE 版本一致，自包含）
├── scripts/download_weights.sh
├── scripts/finetune_dental.sh
├── simclr/                   # submodule（TF1.x 原始代码，仅作参考，不参与训练）
└── README.dental.md
```

## 预训练权重

来自 [lightly-ai/simclrv2-pytorch-weights](https://huggingface.co/collections/lightly-ai/simclrv2-pytorch-weights)：

- `lightly-ai/simclrv2-imagenet1k-r50_1x_sk1` → `simclrv2_r50_1x_sk1.pth`（r50, 37.3M 参数）
- `lightly-ai/simclrv2-imagenet1k-r101_1x_sk1` → `simclrv2_r101_1x_sk1.pth`（r101, 66.8M 参数）

下载到 `/disk1/work/zychen/Checkpoints/intraoral/`（可用 `CKPT_DIR` 覆盖）：

```bash
./scripts/download_weights.sh
```

权重格式 `{'resnet': state_dict, 'head': state_dict}`；`resnet` 键带 `net.` 前缀
（`net.0.0.weight` …），`load_simclrv2_backbone` 负责去前缀后严格加载到 backbone（0 missing）。
`resnet.fc`（1000 类监督头）不参与微调；`head`（128 维对比头）同样不用。

## 微调训练

```bash
source /home/zychen/miniconda3/etc/profile.d/conda.sh && conda activate intraoral
cd /home/zychen/Documents/intraoral_code/dentmag/baselines/simclr

# 默认 448 分辨率全量微调 r50（在 lightly 预训练权重上微调，batch 自动 32），6 卡（GPU 2-7）
./scripts/finetune_dental.sh --model r50 --mode full

# 全量微调 r101 @448（batch 自动 24）
./scripts/finetune_dental.sh --model r101 --mode full

# 224 分辨率（linear probe 常用；batch 自动升到 128 / r101 96）
INPUT_SIZE=224 ./scripts/finetune_dental.sh --model r50 --mode linear

# partial：冻结 stem+前 2 个 stage，其余微调
./scripts/finetune_dental.sh --model r50 --mode partial --freeze_blocks 2
```

脚本会**强制校验预训练权重存在**（默认 `/disk1/work/zychen/Checkpoints/intraoral/simclrv2_r{50,101}_1x_sk1.pth`），
缺失时提示先运行 `./scripts/download_weights.sh`，确保始终在 pretrain 基础上微调。

常用环境变量：`GPUS`（默认 `"2 3 4 5 6 7"`）、`INPUT_SIZE`（默认 `448`）、`BATCH_SIZE`、`DATA_ROOT`、
`SPLIT_JSON`、`OUTPUT_DIR`、`CKPT_DIR`、`INIT_CKPT`。额外训练参数（如 `--epochs 150 --lr 5e-4
--class_weighted --accum_iter 2`）直接追加在脚本后面透传给 `ft_train.py`。

输出：`exp/simclrv2_<model>_<mode>_<input_size>/`，每 `--save_freq`（默认 20）epoch 存
`checkpoint-<epoch>.pth`，内容 `{'model', 'optimizer', 'epoch', 'args', 'label_maps',
'task_weights', 'split_metadata', 'val_metrics'}`；`log.txt` 记录每个 epoch 的 train/val 指标。
`args.model_arch = {depth, width_multiplier, sk_ratio}` 与 `args.input_size` 会写入 checkpoint，
供 `ft_eval.py` 与 `compare/adapters/simclr.py` 重建模型。

## 评估

```bash
# 在 test split 上评估（默认 --split test；也可 --split train / --split val）
python ft_eval.py \
    --checkpoint exp/simclrv2_r50_full_448/checkpoint-100.pth \
    --split_json ../../.datasets/intraoral/train_test_seed0_test0p2.json \
    --split test --output_dir exp/simclrv2_r50_full_448/eval
```

输出 `eval_<split>.json`（每任务 accuracy / macro-F1 / per-class recall）与混淆矩阵
（`.npy` + `.json`）。

## 接入 compare/ 表征对比

`compare/configs/models.yaml` 已注册：

- `simclrv2_r50_adapted` → `baselines/simclr/exp/simclrv2_r50_full_448`
- `simclrv2_r101_adapted` → `baselines/simclr/exp/simclrv2_r101_full_448`

`compare/adapters/simclr.py` 自动取目录下 epoch 最大的 `checkpoint-*.pth`，从 `args` 恢复
架构与分辨率，输出末 stage dense grid（`[B, H, W, 2048]`，224→7×7，448→14×14，stride 32）
与 valid-patch-mean global 特征。若你用其它 mode/分辨率训练，改 models.yaml 的 `checkpoint`
指向对应输出目录即可。

训练完成后可直接用统一评估脚本（仓库根目录执行，`--model name=/path/to/checkpoint.pth` 直接
指定训好的 ckpt，无需软链或改 models.yaml）：

```bash
cd /home/zychen/Documents/intraoral_code/dentmag
# 完整流程 check -> prepare -> extract(test+train) -> evaluate -> visualize -> report
./scripts/eval/eval_baseline.sh --model simclrv2_r50_adapted=baselines/simclr/exp/simclrv2_r50_full_448/checkpoint-100.pth --run simclrv2_r50
# 只重跑 evaluate+之后（特征已缓存时）
./scripts/eval/eval_baseline.sh --model simclrv2_r101_adapted=baselines/simclr/exp/simclrv2_r101_full_448/checkpoint-100.pth --run simclrv2_r101 --evaluate-only
```

默认训练输出目录 `exp/simclrv2_r{50,101}_full_448` 与 models.yaml 注册路径一致，因此训练后
直接 `--model simclrv2_r50_adapted`（不带 `=`）也能被 `eval_baseline.sh` 识别。

## 实现要点 / 坑

- **权重前缀**：lightly 权重 `resnet.net.*`，`backbone.net` 期望 `*`；`load_simclrv2_backbone`
  已处理（去 `net.` 前缀，strict 加载 0 missing）。
- **`--min_lr` 默认 `0.1×lr`（随 `--lr` 联动）**：cosine 最终衰减到 `min_lr` 而非 0——微调场景
  学习率降到 0 会浪费末尾 epoch 且可能扰动已学特征；与预训练 `lr_end=0.1×lr` 惯例一致
  （full/partial: lr 3e-4 → min_lr 3e-5；linear: lr 0.01 → min_lr 1e-3）。
- **`torch.load(..., weights_only=False)`**：checkpoint 内含 `args` Namespace，PyTorch 2.6+ 默认
  `weights_only=True` 会拒绝；`ft_train.py`/`ft_eval.py`/adapter 均已显式传 `weights_only=False`。
- **class_weights 设备**：`compute_loss` 中 per-task 权重移到 logits 同设备，否则
  `F.cross_entropy(weight=...)` 报 cuda/cpu 混用。
- **linear 模式**：`--freeze_bn` 默认 True（BN 统计量冻结），与 SimCLRv2 小 batch 微调惯例一致。
- **数据路径**：`finetune_dental.sh` 从 `SIMCLR_ROOT/../..` 定位 workspace 根；`.datasets` 是
  指向 `/disk1/work/zychen/Datasets` 的符号链接，脚本默认 `DATA_ROOT=${WORKSPACE_ROOT}/.datasets/intraoral`。
- **submodule**：`git submodule update --init baselines/simclr` 拉取 TF 参考实现；
  训练只依赖本目录的 vendored PyTorch 代码，与 submodule 内容无关。
