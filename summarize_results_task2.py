#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Task II results into a single CSV.

Expected input files:
  results_task2/{wk}_{imu}_{group}_tau{tau}.json
e.g.
  w3_imu_qwen_tau0.5.json

Each JSON is expected to contain:
  {
    "meta": {
      "wk": "...", "imu": "...", "tau": ...,
      "models": [...],
      "min_model": "...", "max_model": "...",
      "n_all_aligned": ..., "n_sol": ..., "rho_sol": ...,
      "Lmin": ..., "Lmax": ...,
      "auc_oracle": ..., "auc_min": ..., "auc_max": ..., "auc_rand": ...,
      "score_min": ..., "score_max": ..., "score_rand": ...
    },
    ...
  }

Output:
  summary_task2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


FNAME_RE = re.compile(
    r"^(w[345])_(imu|noimu)_([A-Za-z0-9\-]+)_tau([0-9]*\.?[0-9]+)\.json$"
)


def parse_from_filename(fname: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    """
    Returns (wk, imu, group, tau) parsed from filename, or (None, None, None, None) if not matched.
    """
    m = FNAME_RE.match(fname)
    if not m:
        return None, None, None, None
    wk, imu, group, tau_s = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        tau = float(tau_s)
    except ValueError:
        tau = None
    return wk, imu, group, tau


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="results_task2", help="Directory of task2 json outputs")
    ap.add_argument("--out_csv", type=str, default="summary_task2.csv", help="Output CSV path")
    ap.add_argument("--glob", type=str, default="*.json", help="Glob pattern inside in_dir (default: *.json)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} under {in_dir}")

    rows = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        meta = obj.get("meta", {})
        wk_f, imu_f, group_f, tau_f = parse_from_filename(fp.name)

        # Prefer meta fields; fallback to filename if missing
        wk = safe_get(meta, "wk", wk_f)
        imu = safe_get(meta, "imu", imu_f)
        tau = safe_get(meta, "tau", tau_f)
        group = group_f  # group is not necessarily in meta; keep from filename

        models = safe_get(meta, "models", [])
        min_model = safe_get(meta, "min_model", "")
        max_model = safe_get(meta, "max_model", "")

        row = {
            "file": fp.name,
            "wk": wk,
            "imu": imu,
            "group": group,
            "tau": tau,

            "n_all_aligned": safe_get(meta, "n_all_aligned", None),
            "n_sol": safe_get(meta, "n_sol", None),
            "rho_sol": safe_get(meta, "rho_sol", None),
            "Lmin": safe_get(meta, "Lmin", None),
            "Lmax": safe_get(meta, "Lmax", None),

            "auc_oracle": safe_get(meta, "auc_oracle", None),
            "auc_min": safe_get(meta, "auc_min", None),
            "auc_max": safe_get(meta, "auc_max", None),
            "auc_rand": safe_get(meta, "auc_rand", None),

            "score_min": safe_get(meta, "score_min", None),
            "score_max": safe_get(meta, "score_max", None),
            "score_rand": safe_get(meta, "score_rand", None),

            "min_model": min_model,
            "max_model": max_model,
            "num_models": len(models) if isinstance(models, list) else None,
            "models": ",".join(models) if isinstance(models, list) else "",
            "switch_only": safe_get(meta, "switch_only", None),
        }
        rows.append(row)

    # Stable column order
    fieldnames = [
        "file", "wk", "imu", "group", "tau",
        "n_all_aligned", "n_sol", "rho_sol", "Lmin", "Lmax",
        "auc_oracle", "auc_min", "auc_max", "auc_rand",
        "score_min", "score_max", "score_rand",
        "min_model", "max_model", "num_models", "models", "switch_only",
    ]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote {len(rows)} rows -> {out_csv.resolve()}")


if __name__ == "__main__":
    main()
