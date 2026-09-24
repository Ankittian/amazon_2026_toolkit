"""
Central config for the Business Entity Resolution pipeline.
Edit paths here; everything else imports from this file.
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ---- Paths (adjust to the actual student_resource/ layout) ----
    TRAIN_S1: str = "dataset/train/train_source1.tsv"
    TRAIN_S2: str = "dataset/train/train_source2.tsv"
    TRAIN_S3: str = "dataset/train/train_source3.tsv"
    TRAIN_GT: str = "dataset/train/train_ground_truth.tsv"

    TEST_S1: str = "dataset/test/test_source1.tsv"
    TEST_S2: str = "dataset/test/test_source2.tsv"
    TEST_S3: str = "dataset/test/test_source3.tsv"

    OUT_DIR: str = "output"
    MATCHING_OUT: str = "output/matching_results.tsv"
    CANDIDATES_OUT: str = "output/candidate_pairs.tsv"

    # ---- Blocking ----
    BLOCK_KEY_PREFIX_LEN: int = 4        # first-N-chars-of-normalized-name blocking key
    TOP_K_EMBEDDING: int = 15            # nearest neighbours per S1 entity, per source, via embeddings
    USE_EMBEDDING_BLOCKING: bool = True  # set False for a pure string-key baseline (faster, lower recall ceiling)
    EMBEDDING_MODEL: str = "sentence-transformers/LaBSE" #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    # ---- Matching model ----
    MATCH_THRESHOLD: float = 0.5   # tuned on validation to maximize F_0.5 (see train_matcher.py)
    SEED: int = 42
    VAL_FRAC: float = 0.15         # held out at the *S1-entity* level, not row level

    # ---- Negative sampling for training the pairwise classifier ----
    MAX_NEGATIVES_PER_POSITIVE: int = 5

    # ---- GPU acceleration (for 80 GB VRAM Linux server) ----
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FAISS_GPU: bool = torch.cuda.is_available()   # FAISS-GPU kNN instead of sklearn
    EMBEDDING_BATCH_SIZE: int = 2048 if torch.cuda.is_available() else 128
    USE_XGBOOST: bool = True   # XGBoost with device="cuda"; False = LightGBM CPU


CFG = Config()
