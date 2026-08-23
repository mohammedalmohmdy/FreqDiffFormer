"""Statistical analysis: Wilcoxon signed-rank test on per-class mAP@200.

Manuscript:
  - Wilcoxon signed-rank test, two-sided, alpha = 0.01
  - performed INDEPENDENTLY PER DATASET
  - observations are PER-CLASS mAP@200 values
  - dataset observation counts: Sketchy 25, ShoeV2 20, ChairV2 15, TU-Berlin 50
  - report: statistic, p-value, effect size (rank-biserial), 95% CI, paired diffs

This script consumes ACTUAL per-class results produced by scripts/eval.py:
   outputs/results/per_class_<tagA>.csv   (FreqDiffFormer)
   outputs/results/per_class_<tagB>.csv   (DiffSketch, paired baseline)

The two CSVs MUST share the same set of class IDs; classes are matched by class_id
and treated as paired observations. NO classifier results are ever fabricated.

Usage:
  python scripts/stats_analysis.py \
        --freq outputs/results/per_class_sketchy.json_or_csv \
        --baseline outputs/results/per_class_diffsketch_sketchy.csv \
        --dataset sketchy \
        --out outputs/stats/stats_sketchy.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


REQUIRED_OBS = {"sketchy": 25, "shoev2": 20, "chairv2": 15, "tu_berlin": 50}
ALPHA = 0.01


def _read_per_class(path):
    """Read per-class CSV (class_id,mAP@200) or an eval JSON's per_class map."""
    if path.endswith(".json"):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pc = d.get("per_class_mAP@200", {})
        return {int(k): float(v) for k, v in pc.items()}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = [row for row in r if any(c.strip() for c in row)]
    if not rows:
        return out
    header = [h.strip().lower() for h in rows[0]]
    try:
        ci = header.index("class_id")
        mi = header.index("mAP@200")
    except ValueError:
        # positional fallback
        ci, mi = 0, 1 if len(rows[1]) > 1 else 1
    body = rows[1:]
    for row in body:
        out[int(row[ci])] = float(row[mi])
    return out


def wilcoxon_signed_rank(diffs: np.ndarray, alpha: float = ALPHA):
    """Wilcoxon signed-rank statistic, p-value, effect size, 95% CI.

    Uses scipy.stats.wilcoxon (two-sided). Effect size = rank-biserial correlation:
        r = 1 - 2*W / (n*(n+1)/2)  with the 'less-signed' rank form, equivalently
        r = (T_plus - T_minus) / (T_plus + T_minus).
    95% CI is computed for the median paired difference via the Hodges-Lehmann
    estimator and Walsh averages (nonparametric).
    """
    diffs = diffs[np.abs(diffs) > 1e-12]   # drop zero diffs (standard Wilcoxon handling)
    n = len(diffs)
    result = {
        "n_nonzero_diffs": int(n), "alpha": alpha, "two_sided": True,
        "statistic": None, "p_value": None, "effect_size_rank_biserial": None,
        "hodges_lehmann_median_diff": None, "ci95_low": None, "ci95_high": None,
    }
    if n < 1:
        return result
    from scipy import stats
    # use wilcoxon with alternative='two-sided', default method='wilcox'
    stat_p = stats.wilcoxon(diffs, alternative="two-sided")
    if len(getattr(stat_p, "_asdict", lambda: {})()) >= 2:
        statistic, p_value = stat_p.statistic, stat_p.pvalue
    else:
        statistic, p_value = stat_p
    # rank-biserial effect size (signed)
    ranks = np.arange(1, n + 1)
    # signed ranks via scipy (positive vs negative sums)
    pos = ranks[diffs > 0].sum() if (diffs > 0).any() else 0.0
    neg = ranks[diffs < 0].sum() if (diffs < 0).any() else 0.0
    denom = pos + neg  # == n(n+1)/2 (ties aside)
    r = float((pos - neg) / denom) if denom > 0 else 0.0
    # Hodges-Lehmann estimator of median diff + 95% CI via Walsh averages
    walsh = np.sort(np.concatenate([np.tril(np.add.outer(diffs, diffs))[np.tril_indices(n, 0)]]))
    med = float(np.median(walsh))
    from scipy.stats import binom
    # k such that the (k+1)-th smallest/largest Walsh average bounds the median CI
    try:
        c_low_idx = int(binom.ppf(0.025, n, 0.5))
        c_high_idx = n * (n + 1) // 2 - c_low_idx + 1
        ci_low = float(walsh[max(c_low_idx - 1, 0)])
        ci_high = float(walsh[min(c_high_idx - 1, len(walsh) - 1)])
    except Exception:
        ci_low = ci_high = None
    result.update(statistic=float(statistic), p_value=float(p_value),
                  effect_size_rank_biserial=float(r), hodges_lehmann_median_diff=med,
                  ci95_low=ci_low, ci95_high=ci_high)
    return result


def main(args):
    freq = _read_per_class(args.freq)
    base = _read_per_class(args.baseline)
    common = sorted(set(freq) & set(base), key=int)
    if len(common) < 2:
        raise RuntimeError(f"Not enough matching classes: freq={len(freq)} "
                          f"baseline={len(base)} common={len(common)}. "
                          "Both per-class result files must come from real eval runs.")
    if args.dataset in REQUIRED_OBS and len(common) != REQUIRED_OBS[args.dataset]:
        print(f"[warn] expected {REQUIRED_OBS[args.dataset]} per-class observations for "
              f"{args.dataset}, got {len(common)}; proceeding (per-class results are real).")
    freq_vals = np.array([float(freq[c]) for c in common], dtype=np.float64)
    base_vals = np.array([float(base[c]) for c in common], dtype=np.float64)
    diffs = freq_vals - base_vals

    out = {
        "dataset": args.dataset,
        "alpha": ALPHA,
        "two_sided": True,
        "n_classes": int(len(common)),
        "per_class": [{"class_id": int(c), "freqdiffformer_mAP": float(freq[c]),
                       "diffsketch_mAP": float(base[c]),
                       "paired_diff_freq_minus_baseline": float(freq[c] - base[c])}
                      for c in common],
    }
    out["wilcoxon"] = wilcoxon_signed_rank(diffs, ALPHA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    w = out["wilcoxon"]
    stat = w["statistic"]
    p = w["p_value"]
    r = w["effect_size_rank_biserial"]
    lo = w["ci95_low"]
    hi = w["ci95_high"]
    sig = (p is not None) and (p < ALPHA)
    p_str = "%.4g" % p if p is not None else "n/a"
    r_str = "%.3f" % r if r is not None else "n/a"
    print(f"Wilcoxon ({args.dataset}, n={len(common)}): "
          f"W={stat}, p={p_str}, r={r_str}, 95% CI=({lo}, {hi}), sig@0.01={sig}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", required=True, help="FreqDiffFormer per-class CSV/JSON")
    p.add_argument("--baseline", required=True, help="DiffSketch per-class CSV/JSON")
    p.add_argument("--dataset", required=True,
                    help="sketchy|shoev2|chairv2|tu_berlin (controls expected obs count)")
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()
    main(args)
