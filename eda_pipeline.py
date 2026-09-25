"""
EDA Pipeline — Business Entity Resolution Dataset
===================================================
Covers every metric listed in EDA.md:

Per-field:     missing %, unique count, duplicate rate, min/median/max length,
               top/rare values, country distribution
Names:         avg token count, legal-suffix %, char lengths, punctuation freq
Addresses:     numeric-token freq, avg token count, common abbreviations, missingness
Pair-level:    name/address/token/number similarity across positive vs. negative pairs

Outputs a structured Markdown report to  eda_report.md  and prints a
summary to stdout.  Designed to run without GPU; uses a 50k-row sample from
source2/3 for the heavy pair-level analysis to keep runtime manageable.
"""

import os
import re
import sys
import time
import random
import itertools
import collections
import string

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from tqdm import tqdm
import difflib

# ── project imports ──────────────────────────────────────────────────────────
from config import CFG
from normalize import normalize_name, normalize_address, token_set
from data_utils import load_source

# ── output paths ─────────────────────────────────────────────────────────────
REPORT_PATH = "eda_report.md"
PAIR_SAMPLE_SIZE = 50_000   # pairs to analyse for pos/neg distributions

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pct(num, denom):
    return 0.0 if denom == 0 else 100.0 * num / denom


def levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def number_overlap(a: str, b: str) -> float:
    """Fraction of numeric tokens in a that also appear in b."""
    nums_a = set(t for t in a.split() if re.search(r'\d', t))
    nums_b = set(t for t in b.split() if re.search(r'\d', t))
    if not nums_a:
        return float('nan')
    return len(nums_a & nums_b) / len(nums_a)


LEGAL_SUFFIXES = re.compile(
    r'\b(inc|corp|ltd|llc|llp|pvt|co|company|incorporated|limited|plc|gmbh|sarl|sa)\b',
    re.IGNORECASE
)

ADDR_ABBREVS = ['rd', 'st', 'ave', 'blvd', 'apt', 'ste', 'dr', 'ln', 'ct', 'pl', 'hwy']


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 – per-source field stats
# ═══════════════════════════════════════════════════════════════════════════

def field_stats(df: pd.DataFrame, col: str, label: str) -> dict:
    """Compute EDA.md metrics for a single string column."""
    vals = df[col]
    n = len(vals)
    missing_mask = vals.isin(["", None]) | vals.isna()
    missing_pct = _pct(missing_mask.sum(), n)
    non_empty = vals[~missing_mask]

    lengths = non_empty.str.len()
    unique_cnt = vals.nunique()
    dup_rate = _pct(n - vals.nunique(), n)

    token_counts = non_empty.str.split().apply(len)
    top_values = vals.value_counts().head(10).to_dict()
    rare_values = vals.value_counts().tail(10).to_dict()

    return {
        "label": label,
        "n": n,
        "missing_pct": missing_pct,
        "unique_count": unique_cnt,
        "duplicate_rate_pct": dup_rate,
        "len_min": int(lengths.min()) if len(lengths) else 0,
        "len_median": float(lengths.median()) if len(lengths) else 0,
        "len_max": int(lengths.max()) if len(lengths) else 0,
        "avg_token_count": float(token_counts.mean()) if len(token_counts) else 0,
        "top_values": top_values,
        "rare_values": rare_values,
    }


def country_distribution(df: pd.DataFrame) -> dict:
    return df['country'].value_counts().to_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 – name-specific metrics
# ═══════════════════════════════════════════════════════════════════════════

def name_metrics(df: pd.DataFrame) -> dict:
    names = df['business_name'].fillna("").astype(str)
    non_empty = names[names.str.strip() != ""]

    legal_pct = _pct(non_empty.str.contains(LEGAL_SUFFIXES, regex=True).sum(), len(non_empty))
    punct_freq = non_empty.apply(
        lambda s: sum(1 for c in s if c in string.punctuation)
    ).mean()
    avg_tokens = non_empty.str.split().apply(len).mean()

    return {
        "legal_suffix_pct": legal_pct,
        "avg_char_len": float(non_empty.str.len().mean()),
        "avg_token_count": float(avg_tokens),
        "avg_punctuation_per_name": float(punct_freq),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 – address-specific metrics
# ═══════════════════════════════════════════════════════════════════════════

def address_metrics(df: pd.DataFrame) -> dict:
    addrs = df['business_address'].fillna("").astype(str)
    non_empty = addrs[addrs.str.strip() != ""]
    missing_pct = _pct((addrs.str.strip() == "").sum(), len(addrs))

    numeric_token_pct = non_empty.apply(
        lambda s: sum(1 for t in s.split() if re.search(r'\d', t)) / max(len(s.split()), 1)
    ).mean() * 100

    abbrev_counts = {}
    for abbr in ADDR_ABBREVS:
        cnt = non_empty.str.contains(r'\b' + abbr + r'\.?\b', case=False, regex=True).sum()
        abbrev_counts[abbr] = int(cnt)

    avg_tokens = non_empty.str.split().apply(len).mean()

    return {
        "missing_pct": missing_pct,
        "avg_numeric_token_pct": float(numeric_token_pct),
        "avg_token_count": float(avg_tokens),
        "abbreviation_counts": abbrev_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 – positive/negative pair distributions
# ═══════════════════════════════════════════════════════════════════════════

def load_gt(path: str) -> dict:
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        ids = row.matched_entity_ids.strip()
        gt[row.source1_entity_id] = set(ids.split(',')) if ids else set()
    return gt


def build_pair_sample(s1_df, other_df, gt: dict, source_prefix: str, sample: int):
    """
    Returns two lists of (row_s1, row_other) — positives and negatives — each
    capped at  sample/2  pairs.  Negative pairs are drawn from within the same
    blocking key (hard negatives) or randomly.
    """
    s1_idx = {r.entity_id: r for r in s1_df.itertuples(index=False)}
    oth_idx = {r.entity_id: r for r in other_df.itertuples(index=False)}
    oth_ids_from_source = [eid for eid in oth_idx if eid.startswith(source_prefix)]

    positives, negatives = [], []
    half = sample // 2

    s1_keys = list(gt.keys())
    random.shuffle(s1_keys)

    neg_pool = oth_ids_from_source.copy()
    random.shuffle(neg_pool)
    neg_iter = itertools.cycle(neg_pool)

    for s1_id in s1_keys:
        if len(positives) >= half and len(negatives) >= half:
            break
        pos_ids = {eid for eid in gt.get(s1_id, set()) if eid.startswith(source_prefix)}
        row_a = s1_idx.get(s1_id)
        if row_a is None:
            continue
        # positives
        if len(positives) < half:
            for eid in list(pos_ids)[:3]:
                row_b = oth_idx.get(eid)
                if row_b and len(positives) < half:
                    positives.append((row_a, row_b))
        # negatives (pick ids not in pos_ids)
        if len(negatives) < half:
            attempts = 0
            while len(negatives) < half and attempts < 10:
                neg_id = next(neg_iter)
                if neg_id not in pos_ids:
                    row_b = oth_idx.get(neg_id)
                    if row_b:
                        negatives.append((row_a, row_b))
                attempts += 1

    return positives, negatives


def compute_pair_stats(pairs: list) -> dict:
    """Vectorised similarity stats over a list of (row_a, row_b) pairs."""
    name_jac, name_lev, tok_ol, num_ol, addr_lev = [], [], [], [], []

    for row_a, row_b in tqdm(pairs, desc="  Computing pair similarities", leave=False):
        na = normalize_name(getattr(row_a, 'business_name', '') or '')
        nb = normalize_name(getattr(row_b, 'business_name', '') or '')
        aa = normalize_address(getattr(row_a, 'business_address', '') or '')
        ab = normalize_address(getattr(row_b, 'business_address', '') or '')

        ta, tb = token_set(na), token_set(nb)
        name_jac.append(jaccard(ta, tb))
        name_lev.append(levenshtein_ratio(na, nb))
        tok_ol.append(len(ta & tb))
        addr_lev.append(levenshtein_ratio(aa, ab))
        no = number_overlap(aa, ab)
        if not np.isnan(no):
            num_ol.append(no)

    def _stats(lst):
        if not lst:
            return {"mean": float('nan'), "median": float('nan'), "p25": float('nan'), "p75": float('nan')}
        a = np.array(lst, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    return {
        "name_jaccard": _stats(name_jac),
        "name_levenshtein": _stats(name_lev),
        "token_overlap_count": _stats(tok_ol),
        "addr_levenshtein": _stats(addr_lev),
        "number_overlap": _stats(num_ol),
        "n_pairs": len(pairs),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Section 5 – GT structure stats
# ═══════════════════════════════════════════════════════════════════════════

def gt_stats(gt: dict) -> dict:
    match_counts = [len(v) for v in gt.values()]
    s2_counts = [sum(1 for eid in v if eid.startswith('S2')) for v in gt.values()]
    s3_counts = [sum(1 for eid in v if eid.startswith('S3')) for v in gt.values()]
    no_match = sum(1 for v in gt.values() if len(v) == 0)

    return {
        "total_s1_entities": len(gt),
        "no_match_pct": _pct(no_match, len(gt)),
        "avg_matches_per_s1": float(np.mean(match_counts)) if match_counts else 0,
        "median_matches": float(np.median(match_counts)) if match_counts else 0,
        "max_matches": int(max(match_counts)) if match_counts else 0,
        "avg_s2_matches": float(np.mean(s2_counts)) if s2_counts else 0,
        "avg_s3_matches": float(np.mean(s3_counts)) if s3_counts else 0,
        "s2_match_counts_hist": dict(collections.Counter(s2_counts).most_common(10)),
        "s3_match_counts_hist": dict(collections.Counter(s3_counts).most_common(10)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Markdown report builder
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_dict(d: dict, indent=4) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- **{k}**: {v}" for k, v in d.items())


def _stats_table(s: dict) -> str:
    return (f"mean={s['mean']:.4f}  median={s['median']:.4f}  "
            f"p25={s['p25']:.4f}  p75={s['p75']:.4f}")


def write_report(
    sources_train: list,
    sources_test: list,
    gt_train: dict,
    pair_results: dict,
    report_path: str,
):
    lines = ["# EDA Report — Business Entity Resolution\n"]
    lines.append(f"_Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # ── Ground truth overview ─────────────────────────────────────────────
    lines.append("---\n## 1  Ground Truth Structure (Train)\n")
    g = gt_train
    lines.append(_fmt_dict(g) + "\n")

    # ── Per-source field stats ─────────────────────────────────────────────
    lines.append("---\n## 2  Per-Source Field Statistics\n")
    for info in sources_train + sources_test:
        src = info['source']
        lines.append(f"### {src}\n")
        lines.append(f"**Rows**: {info['n_rows']:,}\n")

        for col_label, stats in info['field_stats'].items():
            lines.append(f"\n#### `{col_label}`\n")
            lines.append(
                f"- Missing: **{stats['missing_pct']:.2f}%**\n"
                f"- Unique: **{stats['unique_count']:,}**\n"
                f"- Duplicate rate: **{stats['duplicate_rate_pct']:.2f}%**\n"
                f"- Length (min/median/max): "
                f"**{stats['len_min']} / {stats['len_median']:.0f} / {stats['len_max']}**\n"
                f"- Avg token count: **{stats['avg_token_count']:.2f}**\n"
            )
            top5 = list(stats['top_values'].items())[:5]
            if top5:
                lines.append("- **Top 5 values**:\n")
                for val, cnt in top5:
                    short = str(val)[:80].replace('\n', ' ')
                    lines.append(f"  - `{short}` ({cnt})\n")

        # country dist
        lines.append(f"\n#### Country Distribution\n")
        for ctry, cnt in list(info['country_dist'].items())[:10]:
            lines.append(f"- {ctry}: {cnt:,}\n")

        # name metrics
        lines.append(f"\n#### Name-Specific Metrics\n")
        nm = info['name_metrics']
        lines.append(
            f"- Legal suffix present: **{nm['legal_suffix_pct']:.2f}%**\n"
            f"- Avg char length: **{nm['avg_char_len']:.1f}**\n"
            f"- Avg token count: **{nm['avg_token_count']:.2f}**\n"
            f"- Avg punctuation chars per name: **{nm['avg_punctuation_per_name']:.2f}**\n"
        )

        # address metrics
        lines.append(f"\n#### Address-Specific Metrics\n")
        am = info['address_metrics']
        lines.append(
            f"- Missing: **{am['missing_pct']:.2f}%**\n"
            f"- Avg numeric-token %: **{am['avg_numeric_token_pct']:.2f}%**\n"
            f"- Avg token count: **{am['avg_token_count']:.2f}**\n"
        )
        lines.append("- Common abbreviation counts:\n")
        for abbr, cnt in sorted(am['abbreviation_counts'].items(), key=lambda x: -x[1])[:8]:
            lines.append(f"  - `{abbr}`: {cnt:,}\n")

    # ── Pair-level distributions ──────────────────────────────────────────
    lines.append("---\n## 3  Positive vs Negative Pair Distributions\n")
    lines.append(
        "> These stats are computed on a random sample of pairs from each source.\n"
        "> **Key insight**: the gap between pos/neg in each metric tells us which\n"
        "> features are most discriminative for the classifier.\n\n"
    )

    for src, pr in pair_results.items():
        lines.append(f"### {src}\n")
        lines.append(
            f"- Positive pairs sampled: {pr['positive']['n_pairs']:,}\n"
            f"- Negative pairs sampled: {pr['negative']['n_pairs']:,}\n\n"
        )
        metrics_to_show = ['name_jaccard', 'name_levenshtein', 'token_overlap_count',
                           'addr_levenshtein', 'number_overlap']
        header = "| Metric | Pos mean | Pos median | Neg mean | Neg median | Delta (mean) |"
        sep    = "|--------|----------|------------|----------|------------|--------------|"
        lines.append(header + "\n" + sep + "\n")
        for m in metrics_to_show:
            ps = pr['positive'][m]
            ns = pr['negative'][m]
            delta = (ps['mean'] - ns['mean']) if not (np.isnan(ps['mean']) or np.isnan(ns['mean'])) else float('nan')
            lines.append(
                f"| {m} | {ps['mean']:.4f} | {ps['median']:.4f} | "
                f"{ns['mean']:.4f} | {ns['median']:.4f} | **{delta:+.4f}** |\n"
            )
        lines.append("\n")

    # ── Actionable findings ───────────────────────────────────────────────
    lines.append("---\n## 4  Actionable Findings & Recommendations\n\n")
    lines.append(
        "_Auto-generated from the metrics above. Review and adapt._\n\n"
    )

    # We'll add a placeholder — the real findings are appended after analysis
    lines.append("_(See `eda_findings` section appended at the end of this report after full run.)_\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✅  Report written to: {report_path}")


def append_findings(report_path: str, findings: list):
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n---\n## 5  Detailed Actionable Findings\n\n")
        for finding in findings:
            f.write(f"- {finding}\n")
    print(f"✅  Findings appended to: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-derive findings from the computed stats
# ═══════════════════════════════════════════════════════════════════════════

def derive_findings(sources_train, sources_test, gt_train_stats, pair_results) -> list:
    findings = []

    # GT structure
    g = gt_train_stats
    if g['no_match_pct'] > 5:
        findings.append(
            f"**{g['no_match_pct']:.1f}% of S1 entities have zero matches** → consider "
            f"adding a 'no-match' class or ensuring negatives cover these during training."
        )
    if g['max_matches'] > 20:
        findings.append(
            f"**Max matches per S1 entity = {g['max_matches']}** → the match-count distribution is "
            f"skewed; highly matched entities may dominate positive pair sampling. "
            f"Cap or weight during training."
        )
    findings.append(
        f"Ground truth: avg {g['avg_s2_matches']:.2f} S2 matches + {g['avg_s3_matches']:.2f} S3 matches per S1 entity."
    )

    # Per-source findings
    for info in sources_train + sources_test:
        src = info['source']
        am = info['address_metrics']
        nm = info['name_metrics']

        if am['missing_pct'] > 10:
            findings.append(
                f"**{src}: {am['missing_pct']:.1f}% missing addresses** → "
                f"address-based features will be noisy for this source. "
                f"Consider fallback to name-only similarity or a missingness indicator feature."
            )
        if nm['legal_suffix_pct'] < 20:
            findings.append(
                f"**{src}: Only {nm['legal_suffix_pct']:.1f}% of names contain legal suffixes** → "
                f"suffix normalization may add little value here; check if this is a non-business source."
            )
        if nm['avg_punctuation_per_name'] > 3:
            findings.append(
                f"**{src}: High punctuation density ({nm['avg_punctuation_per_name']:.1f} chars/name)** → "
                f"aggressive punctuation stripping in `normalize_name` may be removing meaningful tokens. "
                f"Consider keeping hyphens/apostrophes."
            )
        if am['avg_numeric_token_pct'] > 40:
            findings.append(
                f"**{src}: {am['avg_numeric_token_pct']:.1f}% of address tokens are numeric** → "
                f"numeric-token overlap is likely a strong feature for this source. "
                f"Add `number_overlap` to feature set if not already present."
            )

    # Pair-level findings
    for src, pr in pair_results.items():
        for metric in ['name_jaccard', 'name_levenshtein', 'addr_levenshtein']:
            pm = pr['positive'].get(metric, {})
            nm_s = pr['negative'].get(metric, {})
            if all(k in pm for k in ['mean']) and all(k in nm_s for k in ['mean']):
                delta = pm['mean'] - nm_s['mean']
                if delta < 0.1:
                    findings.append(
                        f"**{src} / {metric}: Low pos-neg separation (Δ={delta:.3f})** → "
                        f"this feature may be a weak discriminator for {src}. "
                        f"Check if normalization is too aggressive or source has high noise."
                    )
                elif delta > 0.4:
                    findings.append(
                        f"**{src} / {metric}: Strong pos-neg separation (Δ={delta:.3f})** → "
                        f"highly discriminative; ensure this feature is in the model."
                    )

        # Name jaccard for fast negative filtering
        neg_jac = pr['negative']['name_jaccard']['p75']
        if not np.isnan(neg_jac) and neg_jac < 0.05:
            findings.append(
                f"**{src}: 75th-percentile negative name_jaccard = {neg_jac:.3f}** → "
                f"a Jaccard threshold filter (≥ 0.05) could pre-filter >75% of hard negatives cheaply "
                f"before running the full classifier."
            )

    return findings


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Load source files (with normalisation) ───────────────────────────
    print("\n=== Loading train sources ===")
    train_paths = {
        'Train-S1': CFG.TRAIN_S1,
        'Train-S2': CFG.TRAIN_S2,
        'Train-S3': CFG.TRAIN_S3,
    }
    test_paths = {
        'Test-S1': CFG.TEST_S1,
        'Test-S2': CFG.TEST_S2,
        'Test-S3': CFG.TEST_S3,
    }

    def _load_and_describe(label, path):
        print(f"\n  Loading {label} from {path}...")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        return df

    sources_train_info, sources_test_info = [], []
    dfs_train = {}

    for label, path in train_paths.items():
        df = _load_and_describe(label, path)
        dfs_train[label] = df
        info = {
            'source': label,
            'n_rows': len(df),
            'field_stats': {
                'business_name': field_stats(df, 'business_name', 'business_name'),
                'business_address': field_stats(df, 'business_address', 'business_address'),
                'country': field_stats(df, 'country', 'country'),
            },
            'country_dist': country_distribution(df),
            'name_metrics': name_metrics(df),
            'address_metrics': address_metrics(df),
        }
        sources_train_info.append(info)

    dfs_test = {}
    for label, path in test_paths.items():
        df = _load_and_describe(label, path)
        dfs_test[label] = df
        info = {
            'source': label,
            'n_rows': len(df),
            'field_stats': {
                'business_name': field_stats(df, 'business_name', 'business_name'),
                'business_address': field_stats(df, 'business_address', 'business_address'),
                'country': field_stats(df, 'country', 'country'),
            },
            'country_dist': country_distribution(df),
            'name_metrics': name_metrics(df),
            'address_metrics': address_metrics(df),
        }
        sources_test_info.append(info)

    # ── Ground truth ─────────────────────────────────────────────────────
    print("\n=== Loading ground truth ===")
    gt = load_gt(CFG.TRAIN_GT)
    gt_s = gt_stats(gt)

    print("\nGround Truth Stats:")
    for k, v in gt_s.items():
        print(f"  {k}: {v}")

    # ── Pair-level analysis ───────────────────────────────────────────────
    print(f"\n=== Pair-level analysis (sample={PAIR_SAMPLE_SIZE:,} per source) ===")
    s1_df = dfs_train['Train-S1']
    pair_results = {}

    for src_label, prefix in [('Train-S2', 'S2'), ('Train-S3', 'S3')]:
        print(f"\n  Building pairs for {src_label}...")
        other_df = dfs_train[src_label]
        pos_pairs, neg_pairs = build_pair_sample(
            s1_df, other_df, gt, prefix, PAIR_SAMPLE_SIZE
        )
        print(f"  Positives: {len(pos_pairs):,}  Negatives: {len(neg_pairs):,}")
        pair_results[src_label] = {
            'positive': compute_pair_stats(pos_pairs),
            'negative': compute_pair_stats(neg_pairs),
        }

    # ── Print pair summary ────────────────────────────────────────────────
    print("\n=== Pair Similarity Summary ===")
    for src, pr in pair_results.items():
        print(f"\n--- {src} ---")
        for metric in ['name_jaccard', 'name_levenshtein', 'addr_levenshtein', 'number_overlap']:
            pm = pr['positive'][metric]
            nm = pr['negative'][metric]
            delta = pm['mean'] - nm['mean']
            print(f"  {metric:30s}  pos_mean={pm['mean']:.4f}  neg_mean={nm['mean']:.4f}  diff={delta:+.4f}")

    # ── Write report ─────────────────────────────────────────────────────
    print("\n=== Writing Markdown report ===")
    write_report(
        sources_train_info,
        sources_test_info,
        gt_s,
        pair_results,
        REPORT_PATH,
    )

    # ── Derive and append findings ────────────────────────────────────────
    findings = derive_findings(sources_train_info, sources_test_info, gt_s, pair_results)
    append_findings(REPORT_PATH, findings)

    print(f"\nTotal EDA time: {time.time() - t0:.1f}s")
    print(f"Report: {os.path.abspath(REPORT_PATH)}")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
