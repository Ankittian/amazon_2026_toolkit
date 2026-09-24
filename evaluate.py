"""
Local F_0.5 evaluator matching the PS's exact scoring rule:
  - computed per Source-1 entity
  - macro-averaged across all Source-1 entities in the eval set
  - singleton (no true matches) scores 1.0 for an empty prediction, 0.0 for
    any predicted match
Use this on your held-out validation split so your local score tracks the
leaderboard.
"""


def f0_5_per_entity(predicted: set, truth: set) -> float:
    if not truth:
        # singleton: full credit for correctly predicting no match, zero otherwise
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0  # recall = 0 -> F_0.5 = 0 regardless of precision

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)

    if precision == 0 and recall == 0:
        return 0.0

    beta_sq = 0.25  # beta = 0.5 -> beta^2 = 0.25
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f0_5(predictions: dict, ground_truth: dict) -> dict:
    """
    predictions / ground_truth: {source1_entity_id: set(matched_ids)}
    Every S1 entity in ground_truth must have a prediction — missing entries
    are scored as an empty prediction (matches how a rejected/incomplete
    submission would be judged, though the real leaderboard would reject it
    outright per the PS's format rules).
    Returns per-entity scores plus the macro-average.
    """
    scores = {}
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        scores[s1_id] = f0_5_per_entity(pred, truth)

    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return {"macro_f0_5": macro, "per_entity": scores}


def precision_recall_summary(predictions: dict, ground_truth: dict) -> dict:
    """Aggregate (micro) precision/recall for diagnostics alongside the macro F_0.5."""
    tp = fp = fn = 0
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
