"""
Train the pairwise match/no-match classifier and pick a decision threshold
that maximizes F_0.5 on a held-out split of Source-1 entities.

Usage:
    python train_matcher.py
"""

import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from tqdm import tqdm

from config import CFG
from data_utils import load_source, load_ground_truth
from blocking import generate_candidates, candidate_recall
from sklearn.feature_extraction.text import TfidfVectorizer
from features import build_feature_matrix, df_to_lookup, FEATURE_NAMES

def _fit_tfidf_memory_efficient(*paths) -> TfidfVectorizer:
    """
    Fit TF-IDF without holding all DataFrames in memory simultaneously.
    Reads each source, extracts text, then releases the DataFrame before
    moving to the next one.
    """
    print("  Fitting TF-IDF (memory-efficient streaming across all sources)...")
    all_text = []
    for path in paths:
        print(f"    collecting text from {os.path.basename(path)}...")
        df = load_source(path)
        if len(df) > CFG.TFIDF_SAMPLE_SIZE:
            df = df.sample(n=CFG.TFIDF_SAMPLE_SIZE, random_state=42)
        all_text.extend((df["norm_name"] + " " + df["norm_addr"]).tolist())
        del df   # release immediately
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    vec.fit(all_text)
    del all_text
    print("  TF-IDF fitted.")
    return vec

from evaluate import macro_f0_5, precision_recall_summary


def build_training_pairs(candidates: dict, ground_truth: dict, max_neg_per_pos: int, seed: int):
    """
    Positives: every (s1, matched_id) from ground truth that is reachable in
    candidates (if it's not reachable, blocking already lost it — that's a
    recall problem to fix in blocking.py, not something the classifier can
    recover). Negatives: candidates for the same S1 entity that are NOT true
    matches, capped per positive to keep classes from being wildly imbalanced.
    """
    rng = random.Random(seed)
    pos_pairs, neg_pairs = [], []

    for s1_id, cand_ids in tqdm(candidates.items(), desc="  Building training pairs"):
        truth = ground_truth.get(s1_id, set())
        pos_here = [(s1_id, cid) for cid in cand_ids if cid in truth]
        neg_here = [(s1_id, cid) for cid in cand_ids if cid not in truth]

        pos_pairs.extend(pos_here)
        n_neg = max(len(pos_here), 1) * max_neg_per_pos
        if len(neg_here) > n_neg:
            neg_here = rng.sample(neg_here, n_neg)
        neg_pairs.extend(neg_here)

    return pos_pairs, neg_pairs


def stratified_entity_sample(s1: "pd.DataFrame", gt: dict, n: int, seed: int) -> set:
    """
    Draw a stratified random sample of *n* Source-1 entity IDs that preserves:
      1. Country distribution (US vs India — and any others in the data).
      2. Match-count distribution bucket (0, 1, 2, 3, 4, 5+) to retain the
         ~5.6% zero-match entities seen in EDA rather than inadvertently
         discarding them via pure random sampling.

    This means the sampled training set has the same class structure as the
    full corpus but is ~10x smaller, cutting candidate generation and feature
    extraction time from hours to minutes.

    Args:
        s1:   Source-1 DataFrame (must have columns 'entity_id', 'country').
        gt:   Ground-truth dict {s1_id: set(matched_ids)}.
        n:    Target sample size (will be satisfied approximately per-stratum).
        seed: RNG seed.

    Returns:
        Set of sampled entity_id strings.
    """
    import math
    rng = random.Random(seed)

    # Build per-entity strata key: (country, match_count_bucket)
    strata: dict[tuple, list] = {}
    for row in s1.itertuples(index=False):
        eid = row.entity_id
        country = getattr(row, "country", "unk") or "unk"
        n_matches = len(gt.get(eid, set()))
        bucket = min(n_matches, 5)   # bucket 5 = "5 or more"
        key = (country, bucket)
        strata.setdefault(key, []).append(eid)

    total = len(s1)
    sampled: list[str] = []
    for key, ids in strata.items():
        # Proportional allocation: how many from this stratum?
        quota = max(1, math.floor(len(ids) / total * n))
        if len(ids) <= quota:
            sampled.extend(ids)
        else:
            sampled.extend(rng.sample(ids, quota))

    # Trim or top-up to exactly n (±tiny rounding error is acceptable)
    rng.shuffle(sampled)
    sampled = sampled[:n]
    return set(sampled)


def entity_level_split(s1_ids, val_frac, seed):
    rng = random.Random(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])  # train_ids, val_ids



def tune_threshold(scores, valid_pairs, ground_truth, all_s1_ids_in_split):
    """
    Sweep thresholds, group predictions back into per-entity sets, and pick
    the threshold maximizing macro F_0.5 — matching the real scoring rule,
    not just AUC/accuracy on the pair classifier.
    """
    best_t, best_f0_5 = 0.5, -1.0
    for t in tqdm(np.arange(0.1, 0.95, 0.05), desc="  Sweeping F_0.5 thresholds"):
        preds = {s1_id: set() for s1_id in all_s1_ids_in_split}
        for (s1_id, other_id), score in zip(valid_pairs, scores):
            if score >= t:
                preds[s1_id].add(other_id)
        truth_subset = {k: v for k, v in ground_truth.items() if k in all_s1_ids_in_split}
        result = macro_f0_5(preds, truth_subset)
        if result["macro_f0_5"] > best_f0_5:
            best_f0_5, best_t = result["macro_f0_5"], t
    return best_t, best_f0_5


def main():
    print("Loading sources...")
    s1 = load_source(CFG.TRAIN_S1)
    s2 = load_source(CFG.TRAIN_S2)
    s3 = load_source(CFG.TRAIN_S3)
    gt = load_ground_truth(CFG.TRAIN_GT)

    print("Generating candidates (all Source-1 train entities)...")
    # Use the base cache prefix so it picks up the user's existing 2.2M run caches.
    cache_train_prefix = os.path.join(CFG.CACHE_DIR, "candidates_train")
    cand_all = generate_candidates(s1, s2, s3, cache_prefix=cache_train_prefix)

    # ── B: Stratified entity subsampling ─────────────────────────────────
    # We do this *after* candidate generation so we can fully utilize any existing
    # candidate/embedding caches built for the full 2.2M dataset.
    if CFG.TRAIN_ENTITY_SAMPLE is not None and CFG.TRAIN_ENTITY_SAMPLE < len(s1):
        print(f"Stratified subsampling {CFG.TRAIN_ENTITY_SAMPLE:,} / {len(s1):,} S1 entities "
              f"(stratified by country × match-count bucket)...")
        sampled_ids = stratified_entity_sample(
            s1, gt, CFG.TRAIN_ENTITY_SAMPLE, CFG.TRAIN_ENTITY_SAMPLE_SEED
        )
        s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
        cand_all = {k: cand_all[k] for k in sampled_ids if k in cand_all}
        
        n_zero = sum(1 for eid in sampled_ids if len(gt.get(eid, set())) == 0)
        print(f"  Sampled {len(s1):,} entities — "
              f"zero-match: {100*n_zero/len(s1):.1f}%  "
              f"(full corpus: 5.6%)")
    else:
        print(f"Using all {len(s1):,} S1 entities (TRAIN_ENTITY_SAMPLE=None or >= corpus size).")

    train_ids, val_ids = entity_level_split(s1["entity_id"], CFG.VAL_FRAC, CFG.SEED)
    if len(val_ids) == 0:
        raise RuntimeError(
            f"VAL_FRAC={CFG.VAL_FRAC} on {len(s1)} Source-1 entities rounds down to 0 "
            "validation entities. Increase VAL_FRAC or the dataset size, or all downstream "
            "metrics below are meaningless (empty-set edge cases, not real signal)."
        )
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    print(f"  Train entities: {len(s1_train):,}  Val entities: {len(s1_val):,}")

    cand_train = {k: cand_all[k] for k in train_ids if k in cand_all}
    cand_val = {k: cand_all[k] for k in val_ids if k in cand_all}
    print(f"  candidate recall on train split: {candidate_recall(cand_train, gt):.4f}")
    gt_val_subset = {k: v for k, v in gt.items() if k in val_ids}
    print(f"  candidate recall on val split:   {candidate_recall(cand_val, gt_val_subset):.4f}")
    # If this number is low, fix blocking.py before touching the classifier —
    # it is a hard ceiling on your final F_0.5.


    print("Fitting shared TF-IDF vectorizer (train + test vocabulary)...")
    tfidf_vec = _fit_tfidf_memory_efficient(
        CFG.TRAIN_S1, CFG.TRAIN_S2, CFG.TRAIN_S3,
        CFG.TEST_S1,  CFG.TEST_S2,  CFG.TEST_S3,
    )

    s1_lookup = df_to_lookup(s1)
    other_lookup = {**df_to_lookup(s2), **df_to_lookup(s3)}

    print("Building training pairs...")
    pos_pairs, neg_pairs = build_training_pairs(
        cand_train, gt, CFG.MAX_NEGATIVES_PER_POSITIVE, CFG.SEED
    )
    print(f"  {len(pos_pairs)} positive, {len(neg_pairs)} negative training pairs")

    X_pos, valid_pos = build_feature_matrix(pos_pairs, s1_lookup, other_lookup, tfidf_vec)
    X_neg, valid_neg = build_feature_matrix(neg_pairs, s1_lookup, other_lookup, tfidf_vec)

    if len(valid_neg) == 0:
        print("  WARNING: zero negative training pairs. This usually means your blocking "
              "keys are too tight (only ever surfacing true matches) or the dataset is "
              "too small/clean. The classifier can't learn a decision boundary without "
              "negatives — loosen blocking (e.g. shorter BLOCK_KEY_PREFIX_LEN or higher "
              "TOP_K_EMBEDDING) so some non-matches show up as candidates too.")
    if len(valid_pos) == 0:
        raise RuntimeError(
            "Zero positive training pairs reachable from candidates — blocking recall is "
            "effectively 0 on the train split. Fix blocking.py before training a matcher."
        )

    X = np.vstack([X_pos, X_neg]) if len(valid_neg) else X_pos
    y = np.array([1] * len(valid_pos) + [0] * len(valid_neg))

    print("Training classifier...")
    if CFG.USE_XGBOOST and CFG.DEVICE == "cuda" and HAS_XGBOOST:
        print("  → XGBoost with device='cuda' (GPU-accelerated)")
        model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda",
            scale_pos_weight=(len(valid_neg) / max(len(valid_pos), 1)),
            eval_metric="logloss", random_state=CFG.SEED, verbosity=1,
        )
    else:
        print("  → LightGBM on CPU")
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=CFG.SEED,
            class_weight="balanced", min_child_samples=5, verbose=-1,
        )
    model.fit(X, y)

    print("Scoring validation candidates...")
    val_pairs = [(s1_id, cid) for s1_id, cids in cand_val.items() for cid in cids]
    X_val, valid_val_pairs = build_feature_matrix(val_pairs, s1_lookup, other_lookup, tfidf_vec)
    val_scores = model.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

    print("Tuning decision threshold for F_0.5...")
    best_t, best_f0_5 = tune_threshold(val_scores, valid_val_pairs, gt, set(s1_val["entity_id"]))
    print(f"  best threshold={best_t:.2f}  val macro F_0.5={best_f0_5:.4f}")

    preds = {s1_id: set() for s1_id in s1_val["entity_id"]}
    for (s1_id, other_id), score in zip(valid_val_pairs, val_scores):
        if score >= best_t:
            preds[s1_id].add(other_id)
    diag = precision_recall_summary(preds, {k: v for k, v in gt.items() if k in preds})
    print(f"  micro precision={diag['precision']:.4f}  recall={diag['recall']:.4f}")

    if HAS_XGBOOST and isinstance(model, xgb.XGBClassifier):
        model.save_model("outputs_model.json")
        print("Saved XGBoost model to outputs_model.json")
    else:
        model.booster_.save_model("outputs_model.txt")
        print("Saved LightGBM model to outputs_model.txt")
    with open("outputs_threshold.txt", "w") as f:
        f.write(str(best_t))
    print(f"Saved threshold to outputs_threshold.txt")


if __name__ == "__main__":
    main()
