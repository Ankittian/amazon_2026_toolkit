# Business Entity Resolution Toolkit — Amazon ML Challenge 2026

Purpose-built for the Business Entity Resolution PS: match Source-1
(deduplicated reference) business records to Source-2/Source-3 records,
across an open set of countries (train has US/India; test adds France),
scored by macro-averaged F_0.5.

## Pipeline

```
load_source (TSV, normalize name/address)
        │
        ▼
generate_candidates (blocking.py)      <- determines your recall ceiling
   ├── string-key blocking (fast, brittle to typos)
   └── embedding kNN blocking (slower, catches paraphrase)
        │
        ▼
candidate_pairs.tsv                    <- required output #1
        │
        ▼
build_feature_matrix (features.py)     <- Jaccard/Levenshtein/TF-IDF/country features
        │
        ▼
LightGBM matcher (train_matcher.py)    <- match / no-match classifier
        │
        ▼
threshold tuned on macro F_0.5 (evaluate.py)  <- precision-heavy, not 0.5 by default
        │
        ▼
matching_results.tsv                   <- required output #2, the only one leaderboard-scored
```

## Files

- `config.py` — paths, blocking parameters, threshold. Edit paths to match the real `student_resource/` layout first.
- `normalize.py` — generic (non-country-keyed) name/address cleanup. Deliberately pattern-based so it generalizes to France, which is unseen in training.
- `data_utils.py` — TSV loading with explicit `sep="\t"`, ground-truth parsing, and writers that produce the exact required output format.
- `blocking.py` — candidate generation. Run `candidate_recall()` first, before training anything — it tells you the maximum F_0.5 your matcher could ever reach.
- `features.py` — pairwise similarity features (name/address Jaccard, Levenshtein ratio, token-sort ratio, TF-IDF cosine, country match, token overlap).
- `train_matcher.py` — builds positive/negative training pairs from ground truth + candidates, trains a LightGBM classifier, and sweeps thresholds to maximize **macro F_0.5** on an entity-level validation split (not a row-level split — that would leak).
- `inference_pipeline.py` — runs the full pipeline on the test set and writes both required TSVs.
- `evaluate.py` — exact reimplementation of the PS's F_0.5 formula (macro-averaged per Source-1 entity, singleton handling included) so local validation tracks the leaderboard.

## Quickstart

```bash
pip install -r requirements.txt --break-system-packages

# 1. Train the matcher + tune the threshold (uses train/ files, holds out an entity-level split)
python train_matcher.py

# 2. Run full inference on the test set, writing both required TSVs
python inference_pipeline.py

# 3. Validate output format before submitting (provided by the organizers)
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Things this toolkit deliberately does NOT do (by design, per the PS rules)

- No external API/database lookups (geocoding, business registries, etc.) — everything is derived from the provided TSVs only.
- No country-specific hard-coding — `normalize.py` and `blocking.py` treat `country` as an open string label so France at test time is handled the same way as US/India.
- No model over 8B parameters / non-permissive license — the matcher is LightGBM (not an LLM), and the optional embedding model listed in `config.py` is Apache-2.0.

## Where to spend your time, in priority order

1. **Blocking recall.** Check `candidate_recall()` immediately after generating candidates. If it's below ~0.9, no amount of matcher tuning will fix your ceiling — improve `blocking_key()` or raise `TOP_K_EMBEDDING` first.
2. **Feature engineering on names/addresses**, especially anything robust to transliteration and word-order — the PS explicitly flags these as expected noise patterns.
3. **Threshold tuning**, since F_0.5 punishes false merges 2x — don't leave `MATCH_THRESHOLD` at a naive 0.5 without checking `tune_threshold()`'s output.
4. **Singleton handling** — verify your pipeline correctly leaves `matched_entity_ids` empty when nothing clears the threshold; this is worth a full point per entity in the macro average.
5. Only after 1–4: consider swapping the embedding model, adding a second classifier for ensembling, or hand-tuning per-country blocking prefixes (kept generic, not hard-coded).
