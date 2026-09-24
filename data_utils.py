"""
TSV I/O matching the PS's exact formats. All reads use sep="\t" explicitly —
the PS calls out that omitting this silently collapses everything into one
column, since addresses and ID lists contain commas.
"""

import os
import pandas as pd
from tqdm import tqdm

from config import CFG
from normalize import normalize_name, normalize_address


def load_source(path: str, use_cache: bool = True) -> pd.DataFrame:
    fname = os.path.basename(path)
    cache_path = path.replace(".tsv", "_normalized.parquet")

    if use_cache and os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(path):
            print(f"Loading cached normalized dataset: {os.path.basename(cache_path)}...")
            return pd.read_parquet(cache_path)

    print(f"Loading {fname}...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    tqdm.pandas(desc=f"  [1/2] Normalizing names ({fname})")
    df["norm_name"] = df["business_name"].progress_apply(normalize_name)

    tqdm.pandas(desc=f"  [2/2] Normalizing addrs ({fname})")
    df["norm_addr"] = df["business_address"].progress_apply(normalize_address)

    if use_cache:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"Saved normalized cache: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"Warning: Failed to save parquet cache ({e})")

    return df


def load_ground_truth(path: str) -> dict:
    """Returns {source1_entity_id: set(matched_entity_ids)}. Empty string -> empty set."""
    print(f"Loading {os.path.basename(path)}...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="  Parsing Ground Truth"):
        ids = row.matched_entity_ids.strip()
        gt[row.source1_entity_id] = set(ids.split(",")) if ids else set()
    return gt


def write_id_list_tsv(mapping: dict, out_path: str, id_col: str, list_col: str):
    """
    Shared writer for matching_results.tsv and candidate_pairs.tsv.
    mapping: {source1_entity_id: iterable of matched/candidate ids}
    Writes exactly one row per key, comma-joined, no quoting, tab-separated.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fname = os.path.basename(out_path)
    with open(out_path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in tqdm(mapping.items(), desc=f"Writing {fname}"):
            ids = sorted(set(ids))  # dedupe defensively
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def write_matching_results(mapping: dict, out_path: str = None):
    write_id_list_tsv(mapping, out_path or CFG.MATCHING_OUT,
                       "source1_entity_id", "matched_entity_ids")


def write_candidate_pairs(mapping: dict, out_path: str = None):
    write_id_list_tsv(mapping, out_path or CFG.CANDIDATES_OUT,
                       "source1_entity_id", "candidate_entity_ids")
