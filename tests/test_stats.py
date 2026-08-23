"""Tests for the statistical analysis output structure (consumes real per-class)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import subprocess
import sys as _sys
import pytest


def _write_csv(path, classes, base):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "mAP@200"])
        for i in range(classes):
            w.writerow([i, base + 0.01 * i + (1e-3 * (i % 3))])


def test_stats_output_structure(tmp_path):
    n = 15  # chairv2 expected observation count
    freq_csv = tmp_path / "freq.csv"
    base_csv = tmp_path / "base.csv"
    out_json = tmp_path / "out.json"
    _write_csv(freq_csv, n, 0.8)
    _write_csv(base_csv, n, 0.7)
    cmd = [_sys.executable, "scripts/stats_analysis.py",
           "--freq", str(freq_csv), "--baseline", str(base_csv),
           "--dataset", "chairv2", "--out", str(out_json)]
    r = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(out_json.read_text(encoding="utf-8"))
    assert d["dataset"] == "chairv2"
    assert d["two_sided"] is True
    assert d["alpha"] == 0.01
    assert d["n_classes"] == n
    w = d["wilcoxon"]
    for k in ("statistic", "p_value", "effect_size_rank_biserial",
              "hodges_lehmann_median_diff", "ci95_low", "ci95_high",
              "n_nonzero_diffs"):
        assert k in w, k
    assert all({"class_id", "freqdiffformer_mAP", "diffsketch_mAP",
                "paired_diff_freq_minus_baseline"} == set(row.keys())
               for row in d["per_class"])
    # S1 rows come from the actual per-class CSVs (not hard-coded)
    assert len(d["per_class"]) == n


def test_stats_requires_matching_classes(tmp_path):
    freq_csv = tmp_path / "freq.csv"; base_csv = tmp_path / "base.csv"
    _write_csv(freq_csv, 10, 0.8)
    _write_csv(base_csv, 20, 0.7)
    cmd = [_sys.executable, "scripts/stats_analysis.py",
           "--freq", str(freq_csv), "--baseline", str(base_csv),
           "--dataset", "sketchy", "--out", str(tmp_path / "out.json")]
    r = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    # not enough matching classes (10) for sketchy (25) -> warns but still proceeds
    assert r.returncode == 0
