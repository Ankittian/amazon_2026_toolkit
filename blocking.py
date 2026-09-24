"""
Candidate generation (blocking). This determines your recall ceiling — the
PS explicitly calls this out as the top priority. Two complementary signals,
unioned together:

  1. String-key blocking: exact match on (country, normalized-name-prefix).
     Cheap, high-precision, but brittle to typos/transliteration/word-order.
  2. Embedding kNN blocking: cosine nearest-neighbours in sentence-embedding
     space over (name + address). Catches paraphrase-level variation the
     string key misses, at the cost of being slower.

Returns candidates as {s1_entity_id: set(s2/s3_entity_ids)}.
"""

from collections import defaultdict
import numpy as np
from tqdm import tqdm

from config import CFG
from normalize import blocking_key


def string_key_candidates(s1_df, other_df, prefix_len=None):
    prefix_len = prefix_len or CFG.BLOCK_KEY_PREFIX_LEN

    other_by_key = defaultdict(list)
    for row in tqdm(other_df.itertuples(index=False), total=len(other_df), desc="  Indexing string keys (target)"):
        key = blocking_key(row.norm_name, row.country, prefix_len)
        other_by_key[key].append(row.entity_id)

    candidates = {}
    for row in tqdm(s1_df.itertuples(index=False), total=len(s1_df), desc="  Matching string keys (source)"):
        key = blocking_key(row.norm_name, row.country, prefix_len)
        candidates[row.entity_id] = set(other_by_key.get(key, []))
    return candidates


def _combined_text(df):
    return (df["norm_name"] + " " + df["norm_addr"]).tolist()


def embedding_knn_candidates(s1_df, other_df, top_k=None, model_name=None, batch_size=None):
    """
    For each S1 record, find the top_k nearest Source-2/3 records by cosine
    similarity on sentence embeddings of (normalized name + address).
    Grouped by country first — a business in France won't be a neighbour of
    one in the US, so this also caps compute.

    When CFG.USE_FAISS_GPU is True (auto-detected), uses FAISS-GPU for
    sub-second kNN search over millions of vectors.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    top_k = top_k or CFG.TOP_K_EMBEDDING
    model_name = model_name or CFG.EMBEDDING_MODEL
    batch_size = batch_size or CFG.EMBEDDING_BATCH_SIZE
    model = SentenceTransformer(model_name, device=CFG.DEVICE)
    if CFG.DEVICE == "cuda":
        model.half()  # FP16 — halves VRAM, doubles throughput on 80 GB GPU

    candidates = defaultdict(set)

    for country, s1_group in tqdm(s1_df.groupby("country"), desc="  Embedding kNN by country"):
        other_group = other_df[other_df["country"] == country]
        if len(other_group) == 0 or len(s1_group) == 0:
            continue

        s1_emb = model.encode(_combined_text(s1_group), batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
        other_emb = model.encode(_combined_text(other_group), batch_size=batch_size,
                                  show_progress_bar=True, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)

        k = min(top_k, len(other_group))
        dim = other_emb.shape[1]

        # Inner-product on L2-normalized vectors = cosine similarity
        cpu_index = faiss.IndexFlatIP(dim)

        if CFG.USE_FAISS_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = cpu_index

        index.add(other_emb)
        _, idx = index.search(s1_emb, k)

        other_ids = other_group["entity_id"].values
        s1_ids = s1_group["entity_id"].values
        for i, s1_id in enumerate(s1_ids):
            candidates[s1_id].update(other_ids[j] for j in idx[i] if j >= 0)

    return dict(candidates)


def generate_candidates(s1_df, s2_df, s3_df, use_embeddings=None):
    """
    Union of string-key blocking and (optionally) embedding kNN, across both
    Source 2 and Source 3. Returns {s1_id: set(candidate_ids)}.
    """
    use_embeddings = CFG.USE_EMBEDDING_BLOCKING if use_embeddings is None else use_embeddings

    print("Generating string-key candidates (Source 2)...")
    key_s2 = string_key_candidates(s1_df, s2_df)
    print("Generating string-key candidates (Source 3)...")
    key_s3 = string_key_candidates(s1_df, s3_df)

    combined = defaultdict(set)
    for s1_id in s1_df["entity_id"]:
        combined[s1_id] |= key_s2.get(s1_id, set())
        combined[s1_id] |= key_s3.get(s1_id, set())

    if use_embeddings:
        print("Generating embedding kNN candidates (Source 2)...")
        emb_s2 = embedding_knn_candidates(s1_df, s2_df)
        print("Generating embedding kNN candidates (Source 3)...")
        emb_s3 = embedding_knn_candidates(s1_df, s3_df)
        for s1_id in s1_df["entity_id"]:
            combined[s1_id] |= emb_s2.get(s1_id, set())
            combined[s1_id] |= emb_s3.get(s1_id, set())

    return dict(combined)


def candidate_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Sanity check before you build the matcher: what fraction of true matches
    are even reachable from your candidate set? This is your recall ceiling
    — the matching model can never exceed it. Run this first.
    """
    hit, total = 0, 0
    for s1_id, true_matches in tqdm(ground_truth.items(), desc="Calculating candidate recall"):
        if s1_id not in candidates or not true_matches:
            continue  # entity not in this candidate set (e.g. a val-only subset) -> not this call's concern
        cand = candidates[s1_id]
        hit += len(true_matches & cand)
        total += len(true_matches)
    return hit / total if total else 1.0
