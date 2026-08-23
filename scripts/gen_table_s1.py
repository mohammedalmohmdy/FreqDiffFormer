"""Generate Supplementary Table S1 from ACTUAL per-class evaluation results.

Required columns (per manuscript):
  - Dataset
  - Class ID
  - FreqDiffFormer mAP@200
  - DiffSketch mAP@200
  - Paired Difference (FreqDiffFormer - DiffSketch)

Source files per dataset (real eval outputs):
  <results_dir>/per_class_<freq_tag>.csv    -> FreqDiffFormer
  <results_dir>/per_class_<diffsketch_tag>.csv -> DiffSketch (paired baseline)

If a DiffSketch per-class file is not provided/present, the DiffSketch column is
left EMPTY with a clear status marker "DIFFSKETCH_RESULTS_NOT_AVAILABLE"; we do
NOT fabricate values.

Usage:
  python scripts/gen_table_s1.py \
        --datasets sketchy shoev2 chairv2 tu_berlin \
        --freq-tag freqdiffformer --diffsketch-tag diffsketch \
        --results-dir outputs/results \
        --out outputs/supplementary/Table_S1.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_per_class(path):
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return out
    header = [h.strip().lower() for h in rows[0]]
    try:
        ci = header.index("class_id"); mi = header.index("mAP@200")
    except ValueError:
        ci, mi = 0, 1
    for row in rows[1:]:
        out[int(row[ci])] = float(row[mi])
    return out


def main(args):
    rows = []
    missing_any = False
    for ds in args.datasets:
        freq_path = os.path.join(args.results_dir, f"per_class_{args.freq_tag}_{ds}.csv")
        if not os.path.isfile(freq_path):
            freq_path = os.path.join(args.results_dir, f"per_class_{args.freq_tag}.csv")
        ds_path = os.path.join(args.results_dir, f"per_class_{args.diffsketch_tag}_{ds}.csv")
        if not os.path.isfile(ds_path):
            ds_path = os.path.join(args.results_dir, f"per_class_{args.diffsketch_tag}.csv")
        freq = _read_per_class(freq_path)
        if not freq:
            print(f"[skip] no FreqDiffFormer per-class results for {ds} "
                  f"(looked for {freq_path}); skipping.")
            missing_any = True
            continue
        diff = _read_per_class(ds_path)
        diff_present = bool(diff)
        for cid in sorted(freq, key=int):
            fv = freq[cid]
            dv = diff.get(cid, None)
            if diff_present and dv is not None:
                paired = fv - dv
                ds_str = "%.4f" % dv
                pair_str = "%.4f" % paired
            else:
                ds_str = "DIFFSKETCH_RESULTS_NOT_AVAILABLE"
                pair_str = "DIFFSKETCH_RESULTS_NOT_AVAILABLE"
                missing_any = True
            rows.append({
                "Dataset": ds, "Class ID": cid,
                "FreqDiffFormer mAP@200": "%.4f" % fv,
                "DiffSketch mAP@200": ds_str,
                "Paired Difference": pair_str,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Dataset", "Class ID",
                                            "FreqDiffFormer mAP@200",
                                            "DiffSketch mAP@200", "Paired Difference"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved Supplementary Table S1 -> {out_path}  ({len(rows)} rows)")
    if missing_any:
        print("[note] Some DiffSketch per-class results were not available; "
              "those entries are marked DIFFSKETCH_RESULTS_NOT_AVAILABLE (values "
              "are never fabricated). Run a real DiffSketch evaluation to fill them.")
    print("[s1] Table S1 was generated from actual per-class evaluation outputs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["sketchy", "shoev2", "chairv2", "tu_berlin"])
    p.add_argument("--freq-tag", default="freqdiffformer")
    p.add_argument("--diffsketch-tag", default="diffsketch")
    p.add_argument("--results-dir", default="outputs/results")
    p.add_argument("--out", default="outputs/supplementary/Table_S1.csv")
    args = p.parse_args()
    main(args)
