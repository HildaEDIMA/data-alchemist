"""Block 4: confounder screening.

A finding "x ~ y" can be real only because a third variable c drives both
(ice-cream sales and drownings both rise with temperature). The check:
  1. Find candidate confounders c: variables clearly tied to BOTH x and y (measured on the DISCOVERY half).
  2. On the HOLDOUT half, compare the effect of x on y with and without adjusting for c
     ("all else being equal": regression y ~ x + c).
  3. Verdict:  robust      the effect survives the adjustment
               confounded  the effect vanishes (or shrinks to less than 30%)
               reversed    the sign flips (Simpson's paradox): the raw trend was a trap
We report associations, never proven causes: the screening can only use variables that were measured.
"""
import numpy as np
import pandas as pd

import stat_tests as st
import verification as vf

CONF_P, CONF_EFFECT = 1e-6, 0.15           # a confounder must be clearly tied to BOTH variables
ADJ_P, KEEP_RATIO, MIN_BETA = 0.01, 0.30, 0.05
RANK = {"robust": 0, "confounded": 1, "reversed": 2}


def adjusted_effect(hold, x, y, c):
    """Standardised effect of x on y without and with adjustment for c, on the holdout half."""
    d = hold[[x, y, c]].dropna()
    yv = d[y].to_numpy(float)
    xz, C = st._block(d[x]), st._block(d[c])
    sd_y = yv.std()
    raw = np.linalg.lstsq(np.column_stack([np.ones(len(yv)), xz]), yv, rcond=None)[0][1] / sd_y
    p_adj, beta = st._wald_hc3(np.column_stack([xz, C]), yv, [0])    # test the coefficient of x
    return raw, beta[1] / sd_y, p_adj


def screen(res, disc, hold):
    assoc = {frozenset((r["x"], r["y"])): r for r in res if r["type"] == "assoc"}
    for r in res:
        r["verdict"], r["confounding"], r["reported"] = None, [], False
        if not r["replicated"]:
            continue
        if r["type"] != "assoc":
            r["verdict"], r["reported"] = "robust", True
            continue
        x, y = r["x"], r["y"]
        if not pd.api.types.is_numeric_dtype(hold[y]):               # the outcome must be numeric
            x, y = y, x
        if not pd.api.types.is_numeric_dtype(hold[y]) or st._block(hold[x]).shape[1] != 1:
            r["verdict"], r["reported"] = "robust", True              # adjustment not applicable here
            continue
        cands = []
        for c in disc.columns:
            if c in (x, y):
                continue
            a1, a2 = assoc.get(frozenset((c, x))), assoc.get(frozenset((c, y)))
            if a1 and a2 and max(a1["p_disc"], a2["p_disc"]) < CONF_P \
                    and min(a1["effect_disc"], a2["effect_disc"]) >= CONF_EFFECT:
                cands.append((min(a1["effect_disc"], a2["effect_disc"]), c))
        verdict = "robust"
        for _, c in sorted(cands, reverse=True)[:3]:
            raw, adj, p_adj = adjusted_effect(hold, x, y, c)
            if np.sign(adj) != np.sign(raw) and p_adj < ADJ_P and abs(adj) >= MIN_BETA:
                v = "reversed"
            elif p_adj >= ADJ_P or abs(adj) < KEEP_RATIO * abs(raw):
                v = "confounded"
            else:
                v = "robust"
            r["confounding"].append(dict(confounder=c, raw=float(raw), adjusted=float(adj), p_adj=float(p_adj), verdict=v))
            if RANK[v] > RANK[verdict]:
                verdict = v
        r["verdict"], r["reported"] = verdict, verdict != "confounded"
    return res


if __name__ == "__main__":
    import generate_benchmark as gb

    df, _ = gb.make_benchmark(6000, 42)
    res, disc, hold = vf.discover(df, vf.generate_hypotheses(df), seed=0)
    screen(res, disc, hold)
    rep = [r for r in res if r["replicated"]]
    kept = [r for r in rep if r["reported"]]
    print(f"\nafter replication: {len(rep)}   ->   after confounder screening: {len(kept)}")
    for title, v in (("KILLED: explained by something else", "confounded"),
                     ("REVERSED: the sign flips once adjusted (Simpson-type trap)", "reversed")):
        print(f"\n{title}")
        for r in rep:
            if r["verdict"] == v:
                w = max(r["confounding"], key=lambda c: RANK[c["verdict"]])
                print(f"  {' ~ '.join(r['vars']):<34} raw {w['raw']:+.2f}  ->  adjusted for {w['confounder']}: {w['adjusted']:+.2f}")
    print("\nKEPT")
    for r in sorted(kept, key=lambda r: -r["effect_hold"]):
        print(f"  {r['effect_hold']:.2f}  {r['type']:<11} {' ~ '.join(r['vars'])}")