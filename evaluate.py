"""Block 5: measure the engine over many random datasets.

  --mode null     Dead Salmon Test: datasets where NOTHING is true (every column is shuffled
                  independently, which destroys all relationships but keeps each column's
                  distribution). Every reported finding is a false discovery.
  --mode planted  Benchmark with planted truths: measures recall and false discovery rate.

Usage:
    python evaluate.py --mode null --seeds 10 --n 6000
    python evaluate.py --mode planted --seeds 10 --n 6000
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import confounders as cf
import generate_benchmark as gb
import verification as vf


def null_dataset(n, seed):
    df, _ = gb.make_benchmark(n, seed)
    rng = np.random.default_rng(seed + 1000)
    return pd.DataFrame({c: rng.permutation(df[c].to_numpy()) for c in df.columns})


def run_one(df, seed):
    res, disc, hold = vf.discover(df, vf.generate_hypotheses(df), seed=seed, verbose=False)
    cf.screen(res, disc, hold)
    return res


def stages(res):
    return {
        "naive (p < 0.05, no correction)": [r for r in res if r["p_disc"] < 0.05],
        "+ FDR control (BH)": [r for r in res if r["passed_bh"]],
        "+ replication on unseen data": [r for r in res if r["replicated"]],
        "+ confounder screening (full)": [r for r in res if r["reported"]],
    }


def match(finding, truth):
    """The planted item a finding refers to (all its variables belong to that item), or None."""
    v = set(finding["vars"])
    for t in truth:
        if v <= set(t["variables"]):
            return t
    return None


def score(findings, truth):
    core = [t for t in truth if t["is_real"] and not t.get("obvious") and t["kind"] != "confounder_arm"]
    hit, tp, fp = set(), 0, 0
    for f in findings:
        t = match(f, truth)
        if t is not None and t["is_real"]:
            tp += 1
            hit.add(t["id"])
        else:
            fp += 1
    n = tp + fp
    return dict(reported=n, false=fp, recall=len([t for t in core if t["id"] in hit]) / len(core))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["null", "planted"], default="null")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()

    names = list(stages([]).keys())
    rows = {k: [] for k in names}
    t0 = time.time()
    for s in range(a.seeds):
        if a.mode == "null":
            df, truth = null_dataset(a.n, s), None
        else:
            df, truth = gb.make_benchmark(a.n, s)
        res = run_one(df, s)
        for k, fs in stages(res).items():
            rows[k].append(len(fs) if truth is None else score(fs, truth))
        print(f"  dataset {s + 1}/{a.seeds} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{a.mode.upper()} | {a.seeds} datasets of {a.n:,} rows")
    summary = {}
    if a.mode == "null":
        print("Every reported finding is a false discovery.\n")
        print(f"  {'stage':<34}{'mean per dataset':>17}{'min-max':>10}{'datasets with >= 1':>20}")
        for k in names:
            v = np.array(rows[k])
            summary[k] = dict(mean=float(v.mean()), min=int(v.min()), max=int(v.max()), datasets_with_any=int((v > 0).sum()))
            print(f"  {k:<34}{v.mean():>17.1f}{f'{v.min()}-{v.max()}':>10}{int((v > 0).sum()):>14} / {a.seeds}")
    else:
        print(f"  {'stage':<34}{'reported':>9}{'false':>7}{'FDR (pooled)':>14}{'recall (mean)':>15}{'recall (min)':>13}")
        for k in names:
            r = rows[k]
            rep, fal = sum(x["reported"] for x in r), sum(x["false"] for x in r)
            rec = [x["recall"] for x in r]
            summary[k] = dict(reported=rep, false=fal, fdr_pooled=fal / rep if rep else 0, recall_mean=float(np.mean(rec)), recall_min=float(min(rec)))
            print(f"  {k:<34}{rep:>9}{fal:>7}{(fal / rep if rep else 0):>14.1%}{np.mean(rec):>15.0%}{min(rec):>13.0%}")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(mode=a.mode, seeds=a.seeds, n=a.n, summary=summary), open(f"results/evaluation_{a.mode}.json", "w"), indent=1)
    print(f"\nSaved results/evaluation_{a.mode}.json")


if __name__ == "__main__":
    main()