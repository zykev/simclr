#!/bin/sh
set -e

# SimCLRv2 牙科监督微调 —— no-odontify 病例级 split
#（自有采集者 + AIDC 纳入训练；odontify 外部集整体排除）
# 任务与 compare/ 探针对齐：fdi / tooth_class / sextant / view
#
# 独立脚本：自带全部训练超参与 GPU / 端口配置，直接调 torchrun。
#
# 用法（在 baselines/simclr 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/finetune_dental_no_odontify.sh --model r50 --mode full
#   ./scripts/finetune_dental_no_odontify.sh --model r101 --mode full
#   INPUT_SIZE=224 ./scripts/finetune_dental_no_odontify.sh --model r50 --mode linear
#   ./scripts/finetune_dental_no_odontify.sh --model r50 --mode full --epochs 150 --lr 5e-4
#
# 可覆盖环境变量:
#   GPUS        默认 6 卡 GPU 2-7（分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀）
#               GPUS="2 3" / GPUS="2,3" / GPUS="cuda:2 cuda:3" 三者等价
#   INPUT_SIZE  默认 448（与 compare/ models.yaml 的 simclrv2_*_adapted 对齐）
#   DATA_ROOT   默认 <root>/.datasets/intraoral（挂载点缺失时回退 intraoral1）
#   SPLIT_JSON  默认 .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   OUTPUT_DIR  默认 baselines/simclr/exp/simclrv2_<model>_<mode>_<input_size>_no_odontify
#   CKPT_DIR    默认 /disk1/work/zychen/Checkpoints/intraoral（预训练权重目录）
#   INIT_CKPT   默认 ${CKPT_DIR}/simclrv2_<model>_1x_sk1.pth
#   BATCH_SIZE  默认按模型/分辨率（r50: 224->128, 448->32; r101: 224->96, 448->24）
#   MASTER_PORT 默认 29533（torchrun 默认 29500 常被其它任务占用）
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)

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
# 指定用于训练的 GPU：分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀
#   GPUS="2 3"   GPUS="2,3"   GPUS="cuda:2 cuda:3"   GPUS="2, 3"
GPUS="${GPUS:-2 3 4 5 6 7}"

# 解析 GPUS -> torchrun 用的 CUDA_VISIBLE_DEVICES 列表与卡数 NUM_GPUS
# set -f：解析期间关闭通配符展开（GPUS 里误写 * ? [ 时不会被 glob 成文件名）。
CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
set -f
for GPU in $(printf '%s\n' "${GPUS}" | tr ',;' '  '); do
    GPU_ID=${GPU#cuda:}
    case "${GPU_ID}" in
        ''|*[!0-9]*)
            echo "[finetune_dental_no_odontify] ERROR: GPUS 中的 '${GPU}' 不是合法 GPU 序号" >&2
            echo "  合法写法: GPUS=\"2 3\" / GPUS=\"2,3\" / GPUS=\"cuda:2 cuda:3\"" >&2
            exit 1
            ;;
    esac
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},${GPU_ID}
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done
set +f
if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[finetune_dental_no_odontify] ERROR: GPUS 为空，应形如 GPUS=\"2 3\" 或 GPUS=\"2,3\"" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES
MASTER_PORT="${MASTER_PORT:-29533}"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。显式设置 DATA_ROOT / SPLIT_JSON 时以环境变量为准，不回退。
resolve_dental_path() {
    # $1 = 相对 .datasets 层名的后缀（"" 或 "/train_test_seed0_test0p2_no_odontify.json"）
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "/train_test_seed0_test0p2_no_odontify.json")}"
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
# 用独立输出目录，避免覆盖 compare/ models.yaml 仍在引用的常规 split 产物
OUTPUT_DIR="${OUTPUT_DIR:-${SIMCLR_ROOT}/exp/simclrv2_${MODEL}_${MODE}_${INPUT_SIZE}_no_odontify}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[finetune_dental_no_odontify] ERROR: split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

# 保证在预训练权重上微调：权重必须存在
if [ ! -f "${INIT_CKPT}" ]; then
    echo "[error] pretrained weights not found: ${INIT_CKPT}" >&2
    echo "        请先运行 ./scripts/download_weights.sh，或设置 CKPT_DIR/INIT_CKPT" >&2
    exit 1
fi

cd "${SIMCLR_ROOT}"
echo "[finetune_dental_no_odontify] model=${MODEL} mode=${MODE} input_size=${INPUT_SIZE} GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[finetune_dental_no_odontify] INIT_CKPT=${INIT_CKPT}"
echo "[finetune_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[finetune_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[finetune_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune_dental_no_odontify] BATCH_SIZE=${BATCH_SIZE} (accum 1) MASTER_PORT=${MASTER_PORT}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" ft_train.py \
    --checkpoint "${INIT_CKPT}" \
    --input_size "${INPUT_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --finetune_mode "${MODE}" \
    --epochs 50 \
    --warmup_epochs 5 \
    --save_freq 10 \
    --eval_freq 50 \
    --data_path "${DATA_ROOT}" \
    --split_json "${SPLIT_JSON}" \
    --split train \
    --categories "full,tooth,sextant" \
    --tasks "fdi,tooth_class,sextant,view" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo "[finetune_dental_no_odontify] done. 评估 test split 用:"
echo "  python ft_eval.py --checkpoint ${OUTPUT_DIR}/checkpoint-050.pth \\
      --split_json ${SPLIT_JSON} --split test --output_dir ${OUTPUT_DIR}/eval"
