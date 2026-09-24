"""
Pairwise similarity features between a Source-1 record and a candidate
Source-2/3 record. This is the feature set fed into the GBDT matcher.
All features are generic string/set similarity — nothing country-specific,
so they transfer to France at test time.
"""

import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

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


def fit_tfidf_on_all_text(*dfs) -> TfidfVectorizer:
    """Fit one shared TF-IDF vectorizer across all sources so cosine similarity is comparable."""
    all_text = []
    for df in tqdm(dfs, desc="  Collecting text for TF-IDF"):
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
    "name_jaccard", "name_levenshtein", "name_token_sort",
    "addr_jaccard", "addr_levenshtein",
    "name_tfidf_cosine", "addr_tfidf_cosine",
    "country_match", "name_len_diff", "common_token_count",
    "name_first_token_match",
]


def pair_features(row_a, row_b, tfidf_vec: TfidfVectorizer) -> list:
    # Works with both dicts / Series / NamedTuples
    name_a = getattr(row_a, "norm_name", row_a.get("norm_name") if isinstance(row_a, dict) else row_a["norm_name"])
    name_b = getattr(row_b, "norm_name", row_b.get("norm_name") if isinstance(row_b, dict) else row_b["norm_name"])
    addr_a = getattr(row_a, "norm_addr", row_a.get("norm_addr") if isinstance(row_a, dict) else row_a["norm_addr"])
    addr_b = getattr(row_b, "norm_addr", row_b.get("norm_addr") if isinstance(row_b, dict) else row_b["norm_addr"])
    country_a = getattr(row_a, "country", row_a.get("country") if isinstance(row_a, dict) else row_a["country"])
    country_b = getattr(row_b, "country", row_b.get("country") if isinstance(row_b, dict) else row_b["country"])

    tok_a, tok_b = token_set(name_a), token_set(name_b)
    atok_a, atok_b = token_set(addr_a), token_set(addr_b)

    return [
        jaccard(tok_a, tok_b),
        levenshtein_ratio(name_a, name_b),
        token_sort_ratio(name_a, name_b),
        jaccard(atok_a, atok_b),
        levenshtein_ratio(addr_a, addr_b),
        tfidf_cosine(tfidf_vec, name_a, name_b),
        tfidf_cosine(tfidf_vec, addr_a, addr_b),
        1.0 if country_a == country_b else 0.0,
        abs(len(name_a) - len(name_b)),
        len(tok_a & tok_b),
        1.0 if (name_a.split()[:1] == name_b.split()[:1] and name_a) else 0.0,
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
