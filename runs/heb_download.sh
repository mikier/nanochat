#!/bin/bash

# Download all available Hebrew pretraining text and repackage it as nanochat parquet shards.
#
# Sources (~16B tokens total, measured with o200k_base):
#   - HuggingFaceFW/fineweb-2, subset heb_Hebr   (~15.8B tokens, 14.5M docs)
#   - wikimedia/wikipedia, config 20231101.he     (~0.4B tokens, 334K docs)
# HeDC4 is not used: HF only auto-converted a partial ~1B-token slice of it, and it's
# Common Crawl-derived like FineWeb-2, so it would mostly duplicate it.
#
# Output matches karpathy/climbmix-400b-shuffle: a single "text" column, zstd level 3,
# row groups of 1024 docs, globally shuffled, named shard_XXXXX.parquet. The last shard
# is the FineWeb-2 test split, so nanochat's "last file = val" convention holds.
#
# Shards are sized in tokens, not chars: NANOCHAT_MIX_HEB=1 interleaves EN/HE one shard
# at a time, so each HE shard should match a ClimbMix shard (~52M tokens). Hebrew
# averages ~2.61 chars/token vs ~4.85 for ClimbMix, hence ~136M chars per shard.
#
# Usage:
#   bash runs/heb_download.sh
#   KEEP_RAW=1 bash runs/heb_download.sh      # keep the ~26GB of raw downloads
#
# Needs ~26GB for raw downloads + ~25GB for output shards, and ~200GB RAM for the shuffle.

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
OUT_DIR="${OUT_DIR:-$NANOCHAT_BASE_DIR/base_data_hedc4}"  # the path nanochat/heb_dataset.py reads
RAW_DIR="${RAW_DIR:-$NANOCHAT_BASE_DIR/raw_hebrew}"
CHARS_PER_SHARD="${CHARS_PER_SHARD:-136000000}"
KEEP_RAW="${KEEP_RAW:-0}"

if [ -f "$OUT_DIR/.complete" ]; then
    echo "Hebrew shards already prepared in $OUT_DIR, nothing to do."
    exit 0
fi
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $OUT_DIR is not empty (old HeDC4 shards?). Move it aside first, e.g.:"
    echo "  mv $OUT_DIR ${OUT_DIR}_old"
    exit 1
fi

cd "$(dirname "$0")/.."
[ -d ".venv" ] || uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 1) Download raw parquets (resumable: hf skips files that are already complete)
mkdir -p "$RAW_DIR"
hf download HuggingFaceFW/fineweb-2 --repo-type dataset \
    --include "data/heb_Hebr/*" --local-dir "$RAW_DIR/fineweb2"
hf download wikimedia/wikipedia --repo-type dataset \
    --include "20231101.he/*" --local-dir "$RAW_DIR/wikipedia"

# -----------------------------------------------------------------------------
# 2) Shuffle and repackage into shards
mkdir -p "$OUT_DIR"
RAW_DIR="$RAW_DIR" OUT_DIR="$OUT_DIR" CHARS_PER_SHARD="$CHARS_PER_SHARD" python - <<'EOF'
import glob, os, time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

raw, out = os.environ["RAW_DIR"], os.environ["OUT_DIR"]
chars_per_shard = int(os.environ["CHARS_PER_SHARD"])
row_group_size = 1024
chars_per_token = 2.61  # Hebrew, o200k_base; only used for the printed estimate

def load_text(paths):
    chunks = []
    for p in paths:
        col = pq.read_table(p, columns=["text"]).column("text")
        for chunk in col.chunks:
            chunk = chunk.cast(pa.large_string())
            chunks.append(chunk.filter(pc.greater(pc.utf8_length(pc.fill_null(chunk, "")), 0)))
        print(f"  loaded {p}")
    return pa.concat_arrays(chunks)

def write_shard(texts, index):
    path = os.path.join(out, f"shard_{index:05d}.parquet")
    pq.write_table(
        pa.table({"text": texts.cast(pa.string())}), path,
        row_group_size=row_group_size, use_dictionary=False,
        compression="zstd", compression_level=3, write_statistics=False,
    )
    return path

t0 = time.time()
train_paths = sorted(glob.glob(f"{raw}/fineweb2/data/heb_Hebr/train/*.parquet")) + \
              sorted(glob.glob(f"{raw}/wikipedia/20231101.he/*.parquet"))
val_paths = sorted(glob.glob(f"{raw}/fineweb2/data/heb_Hebr/test/*.parquet"))
assert train_paths and val_paths, f"raw parquets missing under {raw}"

print("Loading train text...")
texts = load_text(train_paths)
n = len(texts)
print(f"Shuffling {n:,} docs...")
texts = texts.take(pa.array(np.random.default_rng(42).permutation(n)))
cum_chars = np.cumsum(pc.utf8_length(texts).to_numpy())

print("Writing train shards...")
shard_index, start = 0, 0
while start < n:
    base = cum_chars[start - 1] if start > 0 else 0
    end = int(np.searchsorted(cum_chars, base + chars_per_shard)) + 1
    count = -(-(end - start) // row_group_size) * row_group_size  # round up to a full row group
    end = min(n, start + count)
    path = write_shard(texts.slice(start, end - start), shard_index)
    chars = cum_chars[end - 1] - base
    print(f"  {path}: {end - start:,} docs, {chars / 1e6:.0f}M chars, ~{chars / chars_per_token / 1e6:.0f}M tokens")
    shard_index, start = shard_index + 1, end

print("Writing val shard...")
val = load_text(val_paths)
path = write_shard(val, shard_index)
print(f"  {path}: {len(val):,} docs (val)")

total_chars = int(cum_chars[-1])
print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {shard_index} train shards + 1 val shard in {out}")
print(f"Train: {n:,} docs, {total_chars / 1e9:.1f}B chars, ~{total_chars / chars_per_token / 1e9:.1f}B tokens")
print(f"For a 50/50 mix, download a matching number of English shards: python -m nanochat.dataset -n {shard_index}")
EOF

touch "$OUT_DIR/.complete"
if [ "$KEEP_RAW" != "1" ]; then
    rm -rf "$RAW_DIR"
fi
