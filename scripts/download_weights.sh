#!/bin/sh
set -e

# 下载 SimCLRv2 预训练权重（lightly-ai 仓库，Separius 架构格式）
# 用法:
#   ./scripts/download_weights.sh [DEST_DIR]
# 默认下载到 /disk1/work/zychen/Checkpoints/intraoral
#
# 权重来源:
#   https://huggingface.co/lightly-ai/simclrv2-imagenet1k-r50_1x_sk1
#   https://huggingface.co/lightly-ai/simclrv2-imagenet1k-r101_1x_sk1
# 每个权重内部格式: {'resnet': state_dict, 'head': state_dict}
#   - _sk1 => sk_ratio=0.0625（ResNet-D stem + SelectiveKernel）
#   - resnet.fc 是 1000 类监督头（微调时丢弃，仅加载 net.*）

DEST_DIR="${1:-/disk1/work/zychen/Checkpoints/intraoral}"
mkdir -p "${DEST_DIR}"

download() {
    # $1 = hf model, $2 = file, $3 = local name
    URL="https://huggingface.co/${1}/resolve/main/${2}"
    OUT="${DEST_DIR}/${3}"
    if [ -f "${OUT}" ]; then
        echo "[skip] ${OUT} already exists"
        return 0
    fi
    echo "[get] ${URL}"
    curl -L --fail --retry 3 -o "${OUT}" "${URL}"
    echo "[ok] ${OUT}"
}

download "lightly-ai/simclrv2-imagenet1k-r50_1x_sk1"  "r50_1x_sk1.pth"   "simclrv2_r50_1x_sk1.pth"
download "lightly-ai/simclrv2-imagenet1k-r101_1x_sk1" "r101_1x_sk1.pth"  "simclrv2_r101_1x_sk1.pth"

echo "---"
ls -la "${DEST_DIR}"/simclrv2_*.pth
