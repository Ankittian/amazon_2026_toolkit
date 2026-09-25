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
    CACHE_DIR: str = "output/cache"
    MATCHING_OUT: str = "output/matching_results.tsv"
    CANDIDATES_OUT: str = "output/candidate_pairs.tsv"

    # ---- Blocking ----
    BLOCK_KEY_PREFIX_LEN: int = 4        # first-N-chars-of-normalized-name blocking key
    MAX_STRING_CANDIDATES_PER_KEY: int = 300  # caps mega-buckets (e.g. generic prefixes) to prevent OOM
    TOP_K_EMBEDDING: int = 15            # nearest neighbours per S1 entity, per source, via embeddings
    USE_EMBEDDING_BLOCKING: bool = True  # set False for a pure string-key baseline (faster, lower recall ceiling)
    SAVE_RAW_EMBEDDINGS: bool = True     # saves .npy dense vectors on disk to avoid re-encoding on restarts
    EMBEDDING_MODEL: str = "sentence-transformers/LaBSE" #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    # ---- Matching model ----
    MATCH_THRESHOLD: float = 0.5   # tuned on validation to maximize F_0.5 (see train_matcher.py)
    SEED: int = 42
    VAL_FRAC: float = 0.15         # held out at the *S1-entity* level, not row level

    # ---- Negative sampling for training the pairwise classifier ----
    MAX_NEGATIVES_PER_POSITIVE: int = 4   # raised from 5→4 to tighten imbalance slightly

    # ---- Training entity subsampling (EDA: 2.2M S1 entities is far more than needed) ----
    # Set to None to use all entities; otherwise a random stratified sample of this many
    # S1 entity IDs is drawn (stratified by country + match-count bucket) before
    # generating candidates and building feature matrices.  150k–200k gives ~95% of the
    # signal at ~10% of the compute cost.
    TRAIN_ENTITY_SAMPLE: int = 175_000   # None = use all; int = stratified subsample size
    TRAIN_ENTITY_SAMPLE_SEED: int = 42   # separate seed so changing it doesn't affect other rng
    TFIDF_SAMPLE_SIZE: int = 100_000     # Subsample size per dataframe to avoid memory blowup during TF-IDF fitting


    # ---- GPU acceleration (for 80 GB VRAM Linux server) ----
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FAISS_GPU: bool = torch.cuda.is_available()   # FAISS-GPU kNN instead of sklearn
    EMBEDDING_BATCH_SIZE: int = 2048 if torch.cuda.is_available() else 128
    USE_XGBOOST: bool = True   # XGBoost with device="cuda"; False = LightGBM CPU


CFG = Config()
