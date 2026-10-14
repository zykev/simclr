#!/bin/sh
set -eu

# Genuine SimCLRv2 continued SSL on the dental no-odontify training split.
# This script creates two augmented views and optimizes NT-Xent only; FDI,
# sextant and view labels are never passed to ssl_pretrain.py.
# The runtime recipe deliberately matches finetune_dental_no_odontify.sh's
# full-mode defaults: 50 epochs, 5 warmup epochs, AdamW 3e-4 -> 3e-5,
# weight decay .05, save every 10 epochs, seed 0 and AMP.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMCLR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SIMCLR_ROOT}/../.." && pwd)"

MODEL="r50"
INPUT_SIZE="${INPUT_SIZE:-448}"
EXTRA_ARGS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --input-size) INPUT_SIZE="$2"; shift 2 ;;
        *) EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done
case "${MODEL}" in r50|r101) ;; *) echo "--model must be r50 or r101" >&2; exit 2;; esac

GPUS="${GPUS:-2 3 4 5 6 7}"
CUDA_VISIBLE_DEVICES="$(printf '%s' "${GPUS}" | tr ',; ' '   ' | tr -s ' ' ',' | sed 's/^,*//; s/,*$//; s/cuda://g')"
[ -n "${CUDA_VISIBLE_DEVICES}" ] || { echo "GPUS must not be empty" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES
NUM_GPUS="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')"

resolve_path() {
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}
DATA_ROOT="${DATA_ROOT:-$(resolve_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_path "/train_test_seed0_test0p2_no_odontify.json")}"
CKPT_DIR="${CKPT_DIR:-/disk1/work/zychen/Checkpoints/intraoral}"
INIT_CKPT="${INIT_CKPT:-${CKPT_DIR}/simclrv2_${MODEL}_1x_sk1.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${SIMCLR_ROOT}/exp/simclrv2_${MODEL}_ssl_${INPUT_SIZE}_no_odontify}"
if [ "${MODEL}" = r50 ]; then BATCH_SIZE="${BATCH_SIZE:-32}"; else BATCH_SIZE="${BATCH_SIZE:-24}"; fi
[ -f "${INIT_CKPT}" ] || { echo "pretrained checkpoint missing: ${INIT_CKPT}" >&2; exit 1; }
[ -f "${SPLIT_JSON}" ] || { echo "split missing: ${SPLIT_JSON}" >&2; exit 1; }

cd "${SIMCLR_ROOT}"
echo "[simclr_ssl] model=${MODEL} input=${INPUT_SIZE} gpus=${CUDA_VISIBLE_DEVICES} batch/gpu=${BATCH_SIZE}"
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29533}" ssl_pretrain.py \
    --checkpoint "${INIT_CKPT}" --input_size "${INPUT_SIZE}" --batch_size "${BATCH_SIZE}" \
    --data_path "${DATA_ROOT}" --split_json "${SPLIT_JSON}" --split train \
    --categories "full" --epochs 50 --warmup_epochs 5 --lr 3e-4 --min_lr 3e-5 \
    --weight_decay 0.05 --save_freq 10 --seed 0 --output_dir "${OUTPUT_DIR}" ${EXTRA_ARGS}
