"""
Pairwise similarity features between a Source-1 record and a candidate
Source-2/3 record. This is the feature set fed into the GBDT matcher.
All features are generic string/set similarity — nothing country-specific,
so they transfer to France at test time.

EDA-driven additions (2026-09-25):
  - number_overlap:     fraction of numeric tokens in addr_a that appear in addr_b.
                        EDA showed +0.74 pos/neg separation — strongest signal in the set.
  - addr_a_missing /
    addr_b_missing:     binary flags for empty addresses (3.3% of S2/S3 rows).
                        Prevents addr_levenshtein from silently treating two empty strings
                        as perfectly similar and helps the model learn a missingness effect.
"""

import re
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import CFG
from normalize import token_set


def levenshtein_ratio(a: str, b: str) -> float:
    """difflib's ratio is a fast, dependency-free stand-in for normalized edit
    distance (2 * matches / total length) — good enough to rank similarity."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def token_sort_ratio(a: str, b: str) -> float:
    """Word-order-invariant similarity — catches 'John Smith LLC' vs 'Smith John LLC'."""
    return levenshtein_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def number_overlap(addr_a: str, addr_b: str) -> float:
    """
    Fraction of numeric tokens in addr_a that also appear in addr_b.

    EDA finding: pos_mean=0.74, neg_mean=0.004 — the single strongest feature.
    Numeric tokens capture house numbers, postal codes, route numbers etc.
    Returns 0.0 (not NaN) when either address is empty so the feature is always
    well-defined for the tree model.
    """
    nums_a = set(t for t in addr_a.split() if re.search(r'\d', t))
    nums_b = set(t for t in addr_b.split() if re.search(r'\d', t))
    if not nums_a:
        return 0.0   # no numbers in a → overlap is undefined; use 0.0 as safe default
    return len(nums_a & nums_b) / len(nums_a)


def fit_tfidf_on_all_text(*dfs) -> TfidfVectorizer:
    """Fit one shared TF-IDF vectorizer across all sources so cosine similarity is comparable."""
    all_text = []
    for df in tqdm(dfs, desc="  Collecting text for TF-IDF"):
        if len(df) > CFG.TFIDF_SAMPLE_SIZE:
            df = df.sample(n=CFG.TFIDF_SAMPLE_SIZE, random_state=42)
        all_text.extend((df["norm_name"] + " " + df["norm_addr"]).tolist())
    print("  Fitting TfidfVectorizer...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    vec.fit(all_text)
    return vec


def tfidf_cosine(vec: TfidfVectorizer, text_a: str, text_b: str) -> float:
    va = vec.transform([text_a])
    vb = vec.transform([text_b])
    num = (va.multiply(vb)).sum()
    denom = np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum())
    return float(num / denom) if denom > 0 else 0.0


FEATURE_NAMES = [
    # ── name similarity ──────────────────────────────────────────────────
    "name_jaccard",           # token-set jaccard (EDA: +0.61 pos/neg separation)
    "name_levenshtein",       # char edit distance ratio (EDA: +0.54)
    "name_token_sort",        # word-order-invariant levenshtein
    # ── address similarity ───────────────────────────────────────────────
    "addr_jaccard",           # token-set jaccard on normalised address
    "addr_levenshtein",       # char edit distance on normalised address (EDA: +0.55)
    "number_overlap",         # ★ NEW (EDA): numeric-token overlap (EDA: +0.74 separation)
    # ── TF-IDF cosine ────────────────────────────────────────────────────
    "name_tfidf_cosine",
    "addr_tfidf_cosine",
    # ── meta / structural ────────────────────────────────────────────────
    "country_match",          # 1.0 if same country code
    "name_len_diff",          # |len(name_a) - len(name_b)|
    "common_token_count",     # |tok_a ∩ tok_b|
    "name_first_token_match", # 1.0 if first word matches
    # ── missingness indicators (EDA: ~3.3% of S2/S3 addresses are empty) ─
    "addr_a_missing",         # ★ NEW: 1.0 if S1 address is empty
    "addr_b_missing",         # ★ NEW: 1.0 if S2/S3 address is empty
]


def _get(row, key: str) -> str:
    """Safely retrieve a field from a namedtuple, dict, or pandas Series.
    Always returns a str — guards against None/NaN in missing fields.
    """
    val = getattr(row, key) if hasattr(row, key) else row[key]
    return str(val) if val is not None else ""


def pair_features(row_a, row_b, tfidf_vec: TfidfVectorizer) -> list:
    # Works with both dicts / Series / NamedTuples
    name_a = _get(row_a, "norm_name")
    name_b = _get(row_b, "norm_name")
    addr_a = _get(row_a, "norm_addr")
    addr_b = _get(row_b, "norm_addr")
    country_a = _get(row_a, "country")
    country_b = _get(row_b, "country")

    tok_a, tok_b = token_set(name_a), token_set(name_b)
    atok_a, atok_b = token_set(addr_a), token_set(addr_b)

    # Missingness flags (EDA: 3.3% of S2/S3 rows have empty addresses).
    addr_a_empty = 1.0 if not addr_a.strip() else 0.0
    addr_b_empty = 1.0 if not addr_b.strip() else 0.0

    # Address similarity: when either address is empty return 0.0 (not 1.0) to
    # avoid falsely rewarding two records that both lack address data.
    either_addr_empty = bool(addr_a_empty or addr_b_empty)
    addr_lev   = 0.0 if either_addr_empty else levenshtein_ratio(addr_a, addr_b)
    addr_jac   = 0.0 if either_addr_empty else jaccard(atok_a, atok_b)
    addr_tfidf = 0.0 if either_addr_empty else tfidf_cosine(tfidf_vec, addr_a, addr_b)

    return [
        # name
        jaccard(tok_a, tok_b),
        levenshtein_ratio(name_a, name_b),
        token_sort_ratio(name_a, name_b),
        # address
        addr_jac,
        addr_lev,
        number_overlap(addr_a, addr_b),      # ★ NEW
        # tfidf
        tfidf_cosine(tfidf_vec, name_a, name_b),
        addr_tfidf,
        # meta
        1.0 if country_a == country_b else 0.0,
        abs(len(name_a) - len(name_b)),
        len(tok_a & tok_b),
        1.0 if (name_a.split()[:1] == name_b.split()[:1] and name_a) else 0.0,
        # missingness
        addr_a_empty,                         # ★ NEW
        addr_b_empty,                         # ★ NEW
    ]


def build_feature_matrix(pairs, s1_lookup: dict, other_lookup: dict, tfidf_vec: TfidfVectorizer):
    """
    pairs: iterable of (s1_entity_id, other_entity_id)
    s1_lookup / other_lookup: {entity_id: row_data} for O(1) row access
    Returns: (X: np.ndarray, valid_pairs: list) — valid_pairs excludes any id
    missing from the lookups (defensive against pipeline bugs).
    """
    X, valid_pairs = [], []
    for s1_id, other_id in tqdm(pairs, desc="  Extracting pairwise features"):
        row_a = s1_lookup.get(s1_id)
        row_b = other_lookup.get(other_id)
        if row_a is None or row_b is None:
            continue
        X.append(pair_features(row_a, row_b, tfidf_vec))
        valid_pairs.append((s1_id, other_id))
    if not X:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float), valid_pairs
    return np.array(X, dtype=float), valid_pairs


def df_to_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="  Indexing lookup table"):
        lookup[row.entity_id] = row
    return lookup
