"""Block 3: control of false discoveries (Benjamini-Hochberg) and replication on unseen data.

Pipeline:
  1. Split the rows in two halves: DISCOVERY and HOLDOUT (the holdout is never used to choose anything).
  2. Test every hypothesis on the discovery half.
  3. Control the false discovery rate over ALL hypotheses tested (Benjamini-Hochberg).
  4. Re-test the survivors on the holdout half; keep those that replicate (same direction).
"""
import itertools
import time

import numpy as np
import pandas as pd

import stat_tests as st

Q = 0.05                                                    # target false discovery rate
MIN_EFFECT = {"assoc": 0.05, "nonlinear": 0.08, "interaction": 0.05}   # practical significance


def bh(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]            # enforce monotonicity
    q = np.empty(m)
    q[order] = np.minimum(adj, 1.0)
    return q


def generate_hypotheses(df):
    """Every hypothesis the engine will try. The ledger of ALL of them is what makes the correction valid."""
    cols = list(df.columns)
    kinds = {c: st.kind(df[c]) for c in cols}
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    hyps = [dict(type="assoc", x=a, y=b) for a, b in itertools.combinations(cols, 2)]
    hyps += [dict(type="nonlinear", x=x, y=y) for x in cols if kinds[x] == "num" for y in numeric if y != x]
    for a, b in itertools.combinations(cols, 2):
        if "cat" in (kinds[a], kinds[b]):
            hyps += [dict(type="interaction", x=a, b=b, y=y) for y in numeric if y not in (a, b)]
    for i, h in enumerate(hyps):
        h["id"] = f"H{i:05d}"
        h["vars"] = [h[k] for k in ("x", "b", "y") if k in h]
    return hyps


def run_test(df, h):
    if h["type"] == "assoc":
        return st.assoc_test(df[h["x"]], df[h["y"]])
    if h["type"] == "nonlinear":
        return st.nonlinear_test(df[h["x"]], df[h["y"]])
    return st.interaction_test(df[h["x"]], df[h["b"]], df[h["y"]])


def discover(df, hyps, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    disc, hold = df.iloc[idx[: len(df) // 2]], df.iloc[idx[len(df) // 2:]]

    # 2-3. discovery half + false discovery control over ALL hypotheses
    t0 = time.time()
    res = []
    for h in hyps:
        r = run_test(disc, h)
        res.append(dict(h, p_disc=r["p"], effect_disc=r["effect"], sign_disc=r["sign"], test=r["test"]))
    q = bh([r["p_disc"] for r in res])
    for r, qi in zip(res, q):
        r["q_disc"] = float(qi)
        r["passed_bh"] = bool(qi <= Q and r["effect_disc"] >= MIN_EFFECT[r["type"]])
        r["replicated"] = False
    if verbose:
        print(f"  tested {len(res):,} hypotheses in {time.time() - t0:.0f}s")

    # 4. replication on unseen data (BH again, among the survivors only)
    surv = [r for r in res if r["passed_bh"]]
    for r in surv:
        out = run_test(hold, r)
        r.update(p_hold=out["p"], effect_hold=out["effect"], sign_hold=out["sign"])
    if surv:
        qh = bh([r["p_hold"] for r in surv])
        for r, qi in zip(surv, qh):
            same_direction = r["sign_disc"] == 0 or r["sign_hold"] == r["sign_disc"]
            r["q_hold"] = float(qi)
            r["replicated"] = bool(qi <= Q and same_direction and r["effect_hold"] >= MIN_EFFECT[r["type"]])
    return res, disc, hold


if __name__ == "__main__":
    import generate_benchmark as gb

    df, _ = gb.make_benchmark(6000, 42)
    hyps = generate_hypotheses(df)
    res, disc, hold = discover(df, hyps, seed=0)

    naive = [r for r in res if r["p_disc"] < 0.05]
    passed = [r for r in res if r["passed_bh"]]
    rep = [r for r in res if r["replicated"]]
    print("\nFunnel")
    print(f"  hypotheses tested                      {len(res):>6}")
    print(f"  'significant' at p < 0.05 (no fix)     {len(naive):>6}")
    print(f"  after Benjamini-Hochberg               {len(passed):>6}")
    print(f"  after replication on unseen data       {len(rep):>6}")
    on_noise = [r for r in rep if any(v.startswith("noise_") for v in r["vars"])]
    print(f"  of which involve a pure-noise column   {len(on_noise):>6}   <- should be 0 or very close")
    print("\nTop replicated findings (by effect size)")
    for r in sorted(rep, key=lambda r: -r["effect_hold"])[:10]:
        print(f"  {r['effect_hold']:.2f}  {r['type']:<11} {' ~ '.join(r['vars'])}")