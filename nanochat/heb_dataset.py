"""
Hebrew pretraining dataset, mirroring the format of `nanochat/dataset.py`.

Source: HeNLP/HeDC4 (Hebrew FineWeb / HeDC4) on Hugging Face.
We use the Hugging Face auto-converted parquet files (10 shards total) rather
than the original monolithic CSV, so the on-disk layout and iteration match
the ClimbMix parquet layout used by `nanochat/dataset.py`.

This module also exposes a 50/50 mixed iterator that interleaves row_group
batches between the original ClimbMix (English) parquets and the HeDC4
(Hebrew) parquets.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir
from nanochat import dataset as en_dataset

# -----------------------------------------------------------------------------
# The specifics of the Hebrew pretraining dataset

# HF auto-converts datasets to parquet under refs/convert/parquet. For HeDC4
# this yields 10 shards named 0.parquet .. 9.parquet under default/train/.
BASE_URL = "https://huggingface.co/api/datasets/HeNLP/HeDC4/parquet/default/train"
MAX_SHARD = 9  # the last datashard is 9.parquet (10 shards total, indices 0..9)
index_to_filename = lambda index: f"{index}.parquet"  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_hedc4")

# Column name holding the raw text in HeDC4 parquet files.
TEXT_COLUMN = "text"

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def _iter_parquet_paths(parquet_paths, start=0, step=1, text_column=TEXT_COLUMN):
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column(text_column).to_pylist()
            texts = [t for t in texts if t]  # drop None / empty (HeDC4 has nulls)
            if texts:
                yield texts

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the Hebrew dataset, in batches of underlying row_groups.
    - split can be "train" or "val". the last parquet file is val.
    - start/step are useful for skipping rows in DDP.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    yield from _iter_parquet_paths(parquet_paths, start=start, step=step)

def parquets_iter_batched_mixed(split, start=0, step=1):
    """
    50/50 interleaved iterator across English (ClimbMix) and Hebrew (HeDC4).

    Emits one batch from English, then one batch from Hebrew, and so on.
    Continues until both iterators are exhausted (a drained side is simply
    skipped so the other side can keep going).
    """
    en_iter = en_dataset.parquets_iter_batched(split, start=start, step=step)
    he_iter = parquets_iter_batched(split, start=start, step=step)
    en_done = he_done = False
    while not (en_done and he_done):
        if not en_done:
            try:
                yield next(en_iter)
            except StopIteration:
                en_done = True
        if not he_done:
            try:
                yield next(he_iter)
            except StopIteration:
                he_done = True

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HeDC4 Hebrew pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help=f"Number of train shards to download (default: -1 = all {MAX_SHARD} train shards)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # User specifies number of train shards; the validation shard (last one) is always downloaded.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard

    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
