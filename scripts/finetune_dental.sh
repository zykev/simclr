#!/bin/sh
set -e

# SimCLRv2 牙科监督微调（病例级 split；任务与 compare/ 探针对齐：fdi / tooth_class / sextant / view）
# 用法（在 baselines/simclr 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/finetune_dental.sh --model r50 --mode full       # 默认 448 分辨率全量微调 r50
#   ./scripts/finetune_dental.sh --model r101 --mode full      # r101 全量微调 @448
#   INPUT_SIZE=224 ./scripts/finetune_dental.sh --model r50 --mode linear  # linear probe @224
#   ./scripts/finetune_dental.sh --model r50 --mode full --epochs 150 --lr 5e-4
#
# 可调环境变量:
#   GPUS        默认 "2 3 4 5 6 7"（Swin-MAE 同款；GPU 0/1 通常被占用）
#   INPUT_SIZE  默认 448（与 compare/ models.yaml 的 simclrv2_*_adapted 对齐）
#   DATA_ROOT   默认 ${WORKSPACE_ROOT}/.datasets/intraoral
#   SPLIT_JSON  默认 ${DATA_ROOT}/train_test_seed0_test0p2.json
#   OUTPUT_DIR  默认 ${SIMCLR_ROOT}/exp/simclrv2_<model>_<mode>_<input_size>
#   CKPT_DIR    默认 /disk1/work/zychen/Checkpoints/intraoral（预训练权重目录）
#   BATCH_SIZE  默认按模型/分辨率（r50: 224->128, 448->32; r101: 224->96, 448->24）

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 目录定位（脚本位于 <root>/baselines/simclr/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMCLR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SIMCLR_ROOT}/../.." && pwd)"

# ---- 解析 --model / --mode / --input-size，其余透传给 ft_train.py -------------
MODEL="r50"
MODE="full"
INPUT_SIZE="448"
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --input-size) INPUT_SIZE="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

case "${MODEL}" in
    r50|r101) ;;
    *) echo "[error] --model must be r50 or r101 (got ${MODEL})" >&2; exit 1 ;;
esac
case "${MODE}" in
    linear|full|partial) ;;
    *) echo "[error] --mode must be linear/full/partial (got ${MODE})" >&2; exit 1 ;;
esac

# ---- GPU / 环境 ---------------------------------------------------------------
GPUS="${GPUS:-2 3 4 5 6 7}"
CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
for GPU in $GPUS; do
    GPU_ID=${GPU#cuda:}
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID
    else
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,$GPU_ID
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done
export CUDA_VISIBLE_DEVICES

DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/.datasets/intraoral}"
SPLIT_JSON="${SPLIT_JSON:-${DATA_ROOT}/train_test_seed0_test0p2.json}"
CKPT_DIR="${CKPT_DIR:-/disk1/work/zychen/Checkpoints/intraoral}"
if [ "${MODEL}" = "r50" ]; then
    INIT_CKPT="${INIT_CKPT:-${CKPT_DIR}/simclrv2_r50_1x_sk1.pth}"
    if [ "${INPUT_SIZE}" = "448" ]; then
        BATCH_SIZE="${BATCH_SIZE:-32}"
    else
        BATCH_SIZE="${BATCH_SIZE:-128}"
    fi
else
    INIT_CKPT="${INIT_CKPT:-${CKPT_DIR}/simclrv2_r101_1x_sk1.pth}"
    if [ "${INPUT_SIZE}" = "448" ]; then
        BATCH_SIZE="${BATCH_SIZE:-24}"
    else
        BATCH_SIZE="${BATCH_SIZE:-96}"
    fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${SIMCLR_ROOT}/exp/simclrv2_${MODEL}_${MODE}_${INPUT_SIZE}}"

# 保证在预训练权重上微调：权重必须存在
if [ ! -f "${INIT_CKPT}" ]; then
    echo "[error] pretrained weights not found: ${INIT_CKPT}" >&2
    echo "        请先运行 ./scripts/download_weights.sh，或设置 CKPT_DIR/INIT_CKPT" >&2
    exit 1
fi

cd "${SIMCLR_ROOT}"
echo "[finetune_dental] model=${MODEL} mode=${MODE} input_size=${INPUT_SIZE} GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[finetune_dental] INIT_CKPT=${INIT_CKPT}"
echo "[finetune_dental] DATA_ROOT=${DATA_ROOT}"
echo "[finetune_dental] SPLIT_JSON=${SPLIT_JSON}"
echo "[finetune_dental] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune_dental] BATCH_SIZE=${BATCH_SIZE} (accum 1)"

torchrun --nproc_per_node="${NUM_GPUS}" ft_train.py \
    --checkpoint "${INIT_CKPT}" \
    --input_size "${INPUT_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --finetune_mode "${MODE}" \
    --epochs 100 \
    --warmup_epochs 5 \
    --save_freq 20 \
    --eval_freq 1 \
    --data_path "${DATA_ROOT}" \
    --split_json "${SPLIT_JSON}" \
    --split train \
    --categories "full,tooth,sextant" \
    --tasks "fdi,tooth_class,sextant,view" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo "[finetune_dental] done. 评估 test split 用:"
echo "  python ft_eval.py --checkpoint ${OUTPUT_DIR}/checkpoint-100.pth \\
      --split_json ${SPLIT_JSON} --split test --output_dir ${OUTPUT_DIR}/eval"
