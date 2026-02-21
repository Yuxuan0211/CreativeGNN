#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CGNN_DIR="${ROOT_DIR}/external/CreativeGNN"
SPLIT_DIR="${CGNN_DIR}/dataset_splits_fine100"

EDGE_CSV_DEFAULT="${ROOT_DIR}/airfoil2d_archive/naca0012_2d_M0.8_A0_Re3e6_fine100/creativegnn_dataset/edges_knn_k8.csv"
EDGE_CSV="${EDGE_CSV:-$EDGE_CSV_DEFAULT}"
ENV_NAME="${ENV_NAME:-creativegnn}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-120}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
ODE_STEPS="${ODE_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-5e-4}"

TRAIN_NODES="$(python - <<'PY'
from pathlib import Path
p=Path("/root/autodl-tmp/article_jcp_bundle/external/CreativeGNN/dataset_splits_fine100/train_nodes.txt")
print(",".join([x.strip() for x in p.read_text().splitlines() if x.strip()]))
PY
)"
VAL_NODES="$(python - <<'PY'
from pathlib import Path
p=Path("/root/autodl-tmp/article_jcp_bundle/external/CreativeGNN/dataset_splits_fine100/val_nodes.txt")
print(",".join([x.strip() for x in p.read_text().splitlines() if x.strip()]))
PY
)"
TEST_NODES="$(python - <<'PY'
from pathlib import Path
p=Path("/root/autodl-tmp/article_jcp_bundle/external/CreativeGNN/dataset_splits_fine100/test_nodes.txt")
print(",".join([x.strip() for x in p.read_text().splitlines() if x.strip()]))
PY
)"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${CGNN_DIR}/runs/fine100_run_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

cd "${CGNN_DIR}"
conda run -n "${ENV_NAME}" python train.py \
  --train-node-csv "${TRAIN_NODES}" \
  --val-node-csv "${VAL_NODES}" \
  --test-node-csv "${TEST_NODES}" \
  --edge-csv "${EDGE_CSV}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --ode-steps "${ODE_STEPS}" \
  --lr "${LR}" \
  --normalize \
  --seed "${SEED}" \
  --early-stop-patience 20 \
  --output-dir "${OUT_DIR}" \
  --loss-csv "${OUT_DIR}/losses.csv" \
  --ckpt "${OUT_DIR}/model.pt"

