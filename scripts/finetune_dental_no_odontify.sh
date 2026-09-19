#!/bin/sh
set -e

# SimCLRv2 牙科监督微调 —— 新公开 split（自有采集者 + AIDC，排除 odontify）
#
# 本脚本是 scripts/finetune_dental.sh 的「预设包装」：只预设下面 4 个环境变量，
# 训练超参（100 epoch / warmup 5 / finetune_mode / 任务 fdi,tooth_class,sextant,view /
# batch_size 按模型与分辨率 / 预训练权重 …）全部沿用原脚本，不在两处重复声明，
# 避免以后改超参时两边漂移。
#
#   1) SPLIT_JSON  -> <root>/.datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#                     新的病例级 split（把 AIDC 纳入训练、排除 odontify 外部集）
#   2) DATA_ROOT   -> <root>/.datasets/intraoral（intraoral -> intraoral1 回退不变）
#   3) OUTPUT_DIR  -> exp/simclrv2_<model>_<mode>_<input_size>_no_odontify
#                     与旧的 exp/simclrv2_<model>_<mode>_<input_size> 完全隔离：旧产物是
#                     compare/ models.yaml 的 simclrv2_*_adapted 引用对象，绝不能覆盖
#   4) MASTER_PORT -> 29533（与原脚本默认 29501 错开，便于 4 个 baseline 并行起）
#
# ⚠️ OUTPUT_DIR 默认值依赖 --model / --mode / --input-size，故本脚本会只读地扫描一遍
#    这几个参数（不消费 "$@"，参数仍原样透传），支持 `--model r50` 与 `--model=r50` 两种写法。
#
# 用法（从任意目录；参数原样透传给原脚本）:
#   bash baselines/simclr/scripts/finetune_dental_no_odontify.sh --model r50  --mode full
#   bash baselines/simclr/scripts/finetune_dental_no_odontify.sh --model r101 --mode full
#
# 环境变量优先级仍然是「显式传入 > 本脚本预设」，例如:
#   GPUS="0 1" OUTPUT_DIR=/tmp/simclr_try \
#       bash baselines/simclr/scripts/finetune_dental_no_odontify.sh --model r50 --mode full
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)

# 目录定位（脚本位于 <root>/baselines/simclr/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMCLR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SIMCLR_ROOT}/../.." && pwd)"
ORIGINAL_SCRIPT="${SCRIPT_DIR}/finetune_dental.sh"

# 新 split 的文件名（相对 .datasets/<collector-root> 的后缀）
NEW_SPLIT_SUFFIX="/train_test_seed0_test0p2_no_odontify.json"

# 只读扫描 --model / --mode / --input-size（默认值与原脚本一致），用于拼 OUTPUT_DIR 名字。
# 不 shift，不修改 "$@"，后面原样透传。
MODEL_DEFAULT="r50"
MODE_DEFAULT="full"
INPUT_SIZE_DEFAULT="448"
MODEL="${MODEL:-${MODEL_DEFAULT}}"
MODE="${MODE:-${MODE_DEFAULT}}"
INPUT_SIZE="${INPUT_SIZE:-${INPUT_SIZE_DEFAULT}}"
PREV_ARG=""
for CUR_ARG in "$@"; do
    case "${PREV_ARG}" in
        --model) MODEL="${CUR_ARG}" ;;
        --mode) MODE="${CUR_ARG}" ;;
        --input-size) INPUT_SIZE="${CUR_ARG}" ;;
    esac
    case "${CUR_ARG}" in
        --model=*) MODEL="${CUR_ARG#--model=}" ;;
        --mode=*) MODE="${CUR_ARG#--mode=}" ;;
        --input-size=*) INPUT_SIZE="${CUR_ARG#--input-size=}" ;;
    esac
    PREV_ARG="${CUR_ARG}"
done

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。与原脚本同一套规则。
resolve_dental_path() {
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

export DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
export SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "${NEW_SPLIT_SUFFIX}")}"
# 新 split 用独立输出目录，避免覆盖 compare/ models.yaml 还在引用的旧产物
export OUTPUT_DIR="${OUTPUT_DIR:-${SIMCLR_ROOT}/exp/simclrv2_${MODEL}_${MODE}_${INPUT_SIZE}_no_odontify}"
export MASTER_PORT="${MASTER_PORT:-29533}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[finetune_dental_no_odontify] ERROR: 新 split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

echo "[finetune_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[finetune_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[finetune_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune_dental_no_odontify] MASTER_PORT=${MASTER_PORT}"
echo "[finetune_dental_no_odontify] -> exec ${ORIGINAL_SCRIPT} $*"

exec sh "${ORIGINAL_SCRIPT}" "$@"
