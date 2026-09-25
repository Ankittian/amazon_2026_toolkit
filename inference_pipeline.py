"""
End-to-end test-set inference: load -> block -> feature -> score -> threshold
-> write both required TSVs.

`candidate_pairs.tsv` here is written as the *actual* input to the matcher
(exactly the candidates the model scored), matching the PS's requirement
that it be the last-stage set, not an earlier looser blocking pass.

Usage:
    python inference_pipeline.py --model outputs_model.txt --threshold_file outputs_threshold.txt
"""

import argparse
import os
import numpy as np
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from tqdm import tqdm

from config import CFG
from data_utils import load_source, write_matching_results, write_candidate_pairs
from blocking import generate_candidates
from features import fit_tfidf_on_all_text, build_feature_matrix, df_to_lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                     help="path to saved model (outputs_model.json for XGBoost, outputs_model.txt for LightGBM)")
    ap.add_argument("--threshold_file", default="outputs_threshold.txt")
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the tuned threshold if you want to trade precision/recall manually")
    args = ap.parse_args()

    threshold = args.threshold
    if threshold is None:
        with open(args.threshold_file) as f:
            threshold = float(f.read().strip())
    print(f"Using decision threshold: {threshold}")

    print("Loading train sources (for a consistent TF-IDF fit) and test sources...")
    s1_train = load_source(CFG.TRAIN_S1)
    s2_train = load_source(CFG.TRAIN_S2)
    s3_train = load_source(CFG.TRAIN_S3)

    s1 = load_source(CFG.TEST_S1)
    s2 = load_source(CFG.TEST_S2)
    s3 = load_source(CFG.TEST_S3)

    # Fit TF-IDF on the union of train+test text so vocabulary covers France
    # (test-only country) rather than only what appeared in training.
    tfidf_vec = fit_tfidf_on_all_text(s1_train, s2_train, s3_train, s1, s2, s3)

    print("Generating candidates for the full test Source-1 set...")
    cache_test_prefix = os.path.join(CFG.CACHE_DIR, "candidates_test")
    candidates = generate_candidates(s1, s2, s3, cache_prefix=cache_test_prefix)

    # Every Source-1 entity must appear, even with an empty candidate/match list.
    all_s1_ids = set(s1["entity_id"])
    for s1_id in all_s1_ids:
        candidates.setdefault(s1_id, set())

    write_candidate_pairs(candidates)
    print(f"Wrote {CFG.CANDIDATES_OUT} ({sum(len(v) for v in candidates.values())} candidate pairs)")

    s1_lookup = df_to_lookup(s1)
    other_lookup = {**df_to_lookup(s2), **df_to_lookup(s3)}

    print("Flattening candidate pairs for scoring...")
    pairs = [(s1_id, cid) for s1_id, cids in tqdm(candidates.items(), desc="  Flattening pairs") for cid in cids]
    print(f"Scoring {len(pairs)} candidate pairs...")

    # Auto-detect model format: XGBoost (.json) vs LightGBM (.txt)
    model_path = args.model
    if model_path is None:
        if os.path.exists("outputs_model.json"):
            model_path = "outputs_model.json"
        else:
            model_path = "outputs_model.txt"

    use_xgb = model_path.endswith(".json")
    if use_xgb:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"Loaded XGBoost model from {model_path}")
    else:
        model = lgb.Booster(model_file=model_path)
        print(f"Loaded LightGBM model from {model_path}")

    matches = {s1_id: set() for s1_id in all_s1_ids}

    if pairs:
        X, valid_pairs = build_feature_matrix(pairs, s1_lookup, other_lookup, tfidf_vec)
        print("Scoring candidate pairs with model...")
        if use_xgb:
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)
        for (s1_id, other_id), score in tqdm(zip(valid_pairs, scores), total=len(valid_pairs), desc="  Filtering matches by threshold"):
            if score >= threshold:
                matches[s1_id].add(other_id)

    write_matching_results(matches)
    n_matched_entities = sum(1 for v in matches.values() if v)
    print(f"Wrote {CFG.MATCHING_OUT} ({n_matched_entities}/{len(matches)} S1 entities have >=1 match)")


if __name__ == "__main__":
    main()
