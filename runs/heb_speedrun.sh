#!/bin/bash

# Hebrew speedrun for a 4xGPU node (CUDA).
#
# Trains a nanochat base model on a 50/50 interleaved mix of:
#   - ClimbMix English parquets (karpathy/climbmix-400b-shuffle)
#   - HeDC4 Hebrew parquets    (HeNLP/HeDC4)
# distributed across 4 GPUs via torchrun.
#
# Launch:
#   bash runs/heb_speedrun.sh
#   # or with wandb:
#   WANDB_RUN=heb_speedrun bash runs/heb_speedrun.sh
#
# If you have a different GPU count, override it on the command line:
#   NPROC=8 bash runs/heb_speedrun.sh

set -e

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Number of GPUs (override with NPROC=N)
NPROC=${NPROC:-4}

# Opt in to the 50/50 English/Hebrew mix. This is read by
# nanochat/dataset.py::list_parquet_files and interleaves HeDC4 shards into
# the listing, so the tokenizer trainer *and* the pretraining dataloader
# both see the mixed corpus.
export NANOCHAT_MIX_HEB=1

# -----------------------------------------------------------------------------
# Python venv setup with uv (GPU extras for CUDA wheels)

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report directory and write a fresh header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Data: grab all 10 Hebrew shards + a matching number of English shards.
python -m nanochat.heb_dataset -n 10
python -m nanochat.dataset -n 10

# -----------------------------------------------------------------------------
# Tokenizer: train on a slice of the interleaved EN/HE corpus.
python -m scripts.tok_train --max-chars=500000000 --vocab-size=32768
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) on 4 GPUs via torchrun + DDP.
# d12 is a reasonable laptop-of-the-cloud scale model; adjust --depth as desired.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=16 \
    --target-param-data-ratio=8 \
    --fp8
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --device-batch-size=16 \
    --model-tag=d12

# -----------------------------------------------------------------------------
# SFT (distributed across the same 4 GPUs).
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --model-tag=d12 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- \
    -i sft \
    --model-tag=d12

# chat with the model over CLI (single-process is fine):
# python -m scripts.chat_cli -p "שלום, מה שלומך?"

# -----------------------------------------------------------------------------
# Assemble the report
python -m nanochat.report generate
