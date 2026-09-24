# GPU Migration — All Changes Applied

Every modification has been applied directly to your existing files. Here is a summary of what changed and why, plus setup instructions for the 80GB Linux server.

---

## Files Modified

### 1. [`config.py`](file:///D:/Amazon_ML/amazon_2026_toolkit/config.py)

**Added** `import torch` and 5 new GPU config fields:

```python
# ---- GPU acceleration (for 80 GB VRAM Linux server) ----
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_FAISS_GPU: bool = torch.cuda.is_available()   # FAISS-GPU kNN instead of sklearn
EMBEDDING_BATCH_SIZE: int = 2048 if torch.cuda.is_available() else 128
USE_XGBOOST: bool = True   # XGBoost with device="cuda"; False = LightGBM CPU
```

> [!NOTE]
> Everything auto-detects. On your Windows laptop (no CUDA), it falls back to CPU/128-batch/LightGBM. On the 80GB Linux server, it auto-enables CUDA/2048-batch/XGBoost-GPU/FAISS-GPU.

---

### 2. [`blocking.py`](file:///D:/Amazon_ML/amazon_2026_toolkit/blocking.py)

**Replaced** `sklearn.neighbors.NearestNeighbors` with **FAISS-GPU** for kNN blocking:

| Before | After |
|:---|:---|
| `from sklearn.neighbors import NearestNeighbors` | `import faiss` (lazy, inside function) |
| `NearestNeighbors(n_neighbors=k, metric="cosine").fit(other_emb)` | `faiss.IndexFlatIP(dim)` → `faiss.index_cpu_to_gpu()` |
| `SentenceTransformer(model_name)` | `SentenceTransformer(model_name, device=CFG.DEVICE)` + `.half()` on GPU |
| `batch_size=128` hardcoded | `batch_size=CFG.EMBEDDING_BATCH_SIZE` (2048 on GPU) |

**Why this matters**: sklearn's NearestNeighbors runs in CPU RAM. With 5M+ S2/S3 vectors, it's extremely slow or OOMs. FAISS-GPU indexes and searches ~5M vectors in **under 1 second** in GPU VRAM.

---

### 3. [`train_matcher.py`](file:///D:/Amazon_ML/amazon_2026_toolkit/train_matcher.py)

**Replaced** the broken `LGBMClassifier(device_type="cuda")` (which requires a custom CUDA build of LightGBM that you don't have) with a proper GPU/CPU branch:

```python
if CFG.USE_XGBOOST and CFG.DEVICE == "cuda":
    model = xgb.XGBClassifier(
        tree_method="hist", device="cuda", ...
    )
else:
    model = lgb.LGBMClassifier(...)  # CPU fallback
```

**Model save** now branches too:
- XGBoost → `outputs_model.json`
- LightGBM → `outputs_model.txt`

> [!IMPORTANT]
> XGBoost's pip package ships with native CUDA support out-of-the-box (`pip install xgboost`). No custom compilation needed, unlike LightGBM's `device_type="cuda"` which requires building from source with `-DUSE_CUDA=1`.

---

### 4. [`inference_pipeline.py`](file:///D:/Amazon_ML/amazon_2026_toolkit/inference_pipeline.py)

**Auto-detects** which model format to load:

```python
# Looks for outputs_model.json (XGBoost) first, falls back to outputs_model.txt (LightGBM)
if model_path.endswith(".json"):
    model = xgb.XGBClassifier(); model.load_model(model_path)
    scores = model.predict_proba(X)[:, 1]
else:
    model = lgb.Booster(model_file=model_path)
    scores = model.predict(X)
```

---

### 5. [`requirements.txt`](file:///D:/Amazon_ML/amazon_2026_toolkit/requirements.txt)

**Added**: `xgboost`, `faiss-cpu` (safe default), with comments for GPU variants.

---

## Linux Server Setup

Run these on the 80GB GPU server:

```bash
# 1. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install FAISS-GPU (replace faiss-cpu)
pip uninstall faiss-cpu -y
pip install faiss-gpu-cu12

# 3. Install rest of dependencies
pip install -r requirements.txt

# 4. Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import faiss; print('FAISS GPU:', faiss.get_num_gpus())"
```

## Running the Pipeline

```bash
# Train (auto-uses XGBoost GPU + FAISS-GPU + FP16 embeddings)
python train_matcher.py

# Inference
python inference_pipeline.py
```

## What Gets GPU-Accelerated

| Stage | Component | GPU Acceleration |
|:---|:---|:---|
| **Embedding** | SentenceTransformer | `device="cuda"` + `.half()` FP16 + batch 2048 |
| **kNN Blocking** | FAISS | `index_cpu_to_gpu()` — IndexFlatIP in VRAM |
| **Tree Training** | XGBoost | `device="cuda"`, `tree_method="hist"` |
| **Data Loading** | pandas `read_csv` | No change (CPU — fast enough at ~2s per file) |
| **Feature Engineering** | TF-IDF / Jaccard / Levenshtein | No change (CPU — string ops, not worth GPU overhead) |

> [!TIP]
> No changes needed to `normalize.py`, `evaluate.py`, `data_utils.py`, or `features.py`. Those are string-processing and evaluation logic where GPU won't help.
