"""Does Muse Spark, used alone as a data analyst, report 'discoveries' in data where nothing is true?
Every claim that asserts an effect is then checked against the verification engine (the seed of Audit mode).

A false DISCOVERY is a claim that asserts an effect that is not real. A claim saying "there is no
relationship" is a denial: it is not a discovery, so it is never counted as a false discovery here.

Usage:
    python salmon_muse.py --mode null --datasets 3
    python salmon_muse.py --mode planted --datasets 3 [--careful]
Needs MODEL_API_KEY in the environment.
"""
import argparse
import json
import os
import tempfile
import time

import analyst
import evaluate as ev
import generate_benchmark as gb


def asserts_effect(claim):
    return bool(claim.get("asserts_effect", True))


def supported(claim, reported):
    """A claim is supported if the engine, exploring on its own with full accounting, found it too."""
    v = set(claim.get("variables", []))
    return bool(v) and any(v <= set(f["vars"]) for f in reported)


def is_false(claim, truth):
    """False discovery: the claim asserts an effect that is not real."""
    if not asserts_effect(claim):
        return False
    if truth is None:                       # null data: nothing is true
        return True
    t = ev.match({"vars": claim.get("variables", [])}, truth)
    return t is None or not t["is_real"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["null", "planted"], default="null")
    ap.add_argument("--datasets", type=int, default=3)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--careful", action="store_true", help="warn the model about multiple comparisons")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp()
    rows = []
    for s in range(a.datasets):
        if a.mode == "null":
            df, truth = ev.null_dataset(a.n, s), None
        else:
            df, truth = gb.make_benchmark(a.n, s)
        path = os.path.join(tmp, f"d{s}.csv")
        df.to_csv(path, index=False)
        t0 = time.time()
        claims, _ = analyst.analyst(path, careful=a.careful)
        reported = [r for r in ev.run_one(df, s) if r["reported"]]
        for c in claims:
            c["false"] = is_false(c, truth)
            c["supported_by_engine"] = asserts_effect(c) and supported(c, reported)
        rows.append(claims)
        eff = [c for c in claims if asserts_effect(c)]
        kept = [c for c in eff if c["supported_by_engine"]]
        print(f"\nDataset {s + 1}/{a.datasets} ({time.time() - t0:.0f}s): Muse Spark alone made {len(claims)} claims, "
              f"{len(eff)} asserting an effect, {sum(c['false'] for c in eff)} of them false.  "
              f"After the engine check: {len(kept)} kept, {sum(c['false'] for c in kept)} false.")
        for c in claims:
            if asserts_effect(c):
                tag = ("FALSE" if c["false"] else "true ") + (" kept    " if c["supported_by_engine"] else " rejected")
            else:
                tag = "denial        "
            print(f"   [{tag}] {str(c.get('claim', ''))[:105]}")

    eff = [c for r in rows for c in r if asserts_effect(c)]
    kept = [c for c in eff if c["supported_by_engine"]]
    print(f"\nSUMMARY ({a.mode}, {a.datasets} datasets, {'careful' if a.careful else 'plain'} prompt)")
    print(f"  Muse Spark alone : {sum(len(r) for r in rows)} claims, {len(eff)} asserting an effect, "
          f"{sum(c['false'] for c in eff)} false discoveries")
    print(f"  After the check  : {len(kept)} effect claims kept, {sum(c['false'] for c in kept)} false")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(f"results/muse_{a.mode}{'_careful' if a.careful else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()