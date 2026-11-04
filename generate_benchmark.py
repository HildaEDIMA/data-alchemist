"""Synthetic marketplace benchmark with planted ground truth.

Every relationship is planted on purpose, so a discovery system can be scored:
  - recall: how many REAL effects it finds
  - false discovery rate: how many of its findings are NOT real

Usage:
    python generate_benchmark.py --n 20000 --seed 42 --out data/
Writes: benchmark.csv and ground_truth.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_benchmark(n=20000, seed=42, n_noise=12):
    rng = np.random.default_rng(seed)  # fixed seed = reproducible data
    df = pd.DataFrame()

    # ---- base columns -------------------------------------------------
    df["region"] = rng.choice(["North", "South", "East", "West"], n, p=[0.25, 0.30, 0.25, 0.20])
    df["channel"] = rng.choice(["web", "app", "store"], n, p=[0.40, 0.35, 0.25])
    df["loyalty_member"] = rng.binomial(1, 0.30, n)
    df["weekend"] = rng.binomial(1, 2 / 7, n)
    df["age"] = np.clip(rng.normal(48, 14, n), 18, 85).round(0)
    df["price"] = np.exp(rng.normal(3.6, 0.6, n)).round(2)
    df["discount_pct"] = rng.uniform(0, 30, n).round(1)
    df["delivery_days"] = (1 + rng.gamma(3.0, 1.1, n)).round(1)

    truth = []

    def add(kind, variables, description, is_real, **extra):
        truth.append(dict(id=f"T{len(truth) + 1:02d}", kind=kind, variables=variables,
                          description=description, is_real=is_real, **extra))

    # 1. MAIN EFFECT: slow delivery lowers rating --------------------------
    df["rating"] = np.clip(4.7 - 0.22 * df["delivery_days"] + rng.normal(0, 0.8, n), 1, 5).round(2)
    add("main_effect", ["delivery_days", "rating"],
        "Each extra delivery day lowers rating by about 0.22.", True, effect=-0.22)

    # 2. MAIN EFFECT (binary outcome): discount raises repeat purchase -----
    df["repeat_purchase"] = rng.binomial(1, _sigmoid(-1.3 + 0.06 * df["discount_pct"]))
    add("main_effect", ["discount_pct", "repeat_purchase"],
        "Higher discounts increase repeat purchase probability.", True, effect=0.06)

    # 3. HIDDEN SUBGROUP: loyalty raises spend ONLY in region North --------
    north = (df["region"] == "North").astype(int)
    df["spend"] = 80 + rng.normal(0, 25, n) + 18 * df["loyalty_member"] * north
    add("subgroup_effect", ["loyalty_member", "spend", "region"],
        "Loyalty members spend about 18 more, but only in region North; no effect elsewhere.",
        True, effect=18.0, subgroup="region == North")

    # 4. PURE INTERACTION: weekend x app drives conversion (no main effects)
    app = (df["channel"] == "app").astype(int)
    df["converted"] = rng.binomial(1, _sigmoid(-1.0 + 1.3 * df["weekend"] * app))
    add("interaction", ["weekend", "channel", "converted"],
        "Conversion jumps only when weekend AND app coincide; neither alone matters.",
        True, effect=1.3)

    # 5. NON-LINEAR: U-shaped return rate vs age (linear correlation ~ 0) --
    df["return_rate"] = np.clip(0.05 + 0.00035 * (df["age"] - 48) ** 2 + rng.normal(0, 0.04, n), 0, 1).round(3)
    add("nonlinear", ["age", "return_rate"],
        "Return rate is U-shaped in age; a linear correlation test sees almost nothing.", True)

    # 6. SIMPSON'S PARADOX: contacts vs satisfaction ------------------------
    seg_budget = rng.binomial(1, 0.5, n)
    df["segment"] = np.where(seg_budget == 1, "budget", "premium")
    df["support_contacts"] = np.where(seg_budget == 1, rng.poisson(6, n), rng.poisson(2, n))
    base = np.where(seg_budget == 1, 60, 85)
    df["satisfaction"] = np.clip(base + 1.5 * df["support_contacts"] + rng.normal(0, 6, n), 0, 100).round(1)
    add("simpson", ["support_contacts", "satisfaction", "segment"],
        "Overall, more support contacts look linked to LOWER satisfaction; within each segment the "
        "effect is POSITIVE (+1.5 per contact). The overall trend is a trap.",
        True, effect=1.5, trap="overall sign is reversed")

    # 7. CONFOUNDER: temperature drives both ad_spend and weekly_sales -----
    t = rng.uniform(0, 1, n)
    df["temperature"] = (15 + 12 * np.sin(2 * np.pi * t) + rng.normal(0, 2, n)).round(1)
    df["ad_spend"] = (50 + 3 * df["temperature"] + rng.normal(0, 8, n)).round(1)
    df["weekly_sales"] = (100 + 4 * df["temperature"] + rng.normal(0, 15, n)).round(1)
    add("confounded", ["ad_spend", "weekly_sales"],
        "Correlated only because temperature drives both. True causal effect of ad_spend = 0.",
        False, confounder="temperature")
    add("confounder_arm", ["temperature", "ad_spend"],
        "Temperature really does drive ad_spend (a real arm of the confounding structure).", True)
    add("confounder_arm", ["temperature", "weekly_sales"],
        "Temperature really does drive weekly_sales (a real arm of the confounding structure).", True)

    # 8. TRIVIAL (obvious, should rank low on novelty) ---------------------
    df["revenue"] = (df["price"] * (1 - df["discount_pct"] / 100)).round(2)
    add("trivial", ["price", "discount_pct", "revenue"],
        "Revenue is price net of discount: true but obvious, should rank low on novelty.", True,
        obvious=True)

    # 9. NOISE: independent columns -> any finding on them is a false discovery
    for i in range(n_noise):
        if i % 3 == 0:
            df[f"noise_cat_{i:02d}"] = rng.choice(list("ABCD"), n)
        else:
            df[f"noise_num_{i:02d}"] = rng.normal(0, 1, n).round(3)
    add("noise", [c for c in df.columns if c.startswith("noise_")],
        "Independent noise columns. Any significant finding involving them is a false discovery.", False)

    return df, truth


def sanity_check(df):
    """Quick proof that the planted effects really exist in the data."""
    c = df[["delivery_days", "rating"]].corr().iloc[0, 1]
    print(f"[delivery_days, rating]       corr = {c:+.3f}   (expected clearly negative, about -0.45)")
    n_ = df[df.region == "North"].groupby("loyalty_member").spend.mean()
    o_ = df[df.region != "North"].groupby("loyalty_member").spend.mean()
    print(f"[loyalty spend gap]           North {n_[1] - n_[0]:+.1f} | others {o_[1] - o_[0]:+.1f}   (expected ~+18 | ~0)")
    g = df.groupby(["weekend", "channel"]).converted.mean().unstack()
    print(f"[conversion weekend x app]    {g.loc[1, 'app']:.2f} vs {g.loc[0, 'app']:.2f} and {g.loc[1, 'web']:.2f}")
    print(f"[age, return_rate] linear     corr = {df[['age', 'return_rate']].corr().iloc[0, 1]:+.3f}   (expected ~0 despite the U shape)")
    print(f"[contacts, satisfaction]      overall = {df[['support_contacts', 'satisfaction']].corr().iloc[0, 1]:+.3f}   (expected negative)")
    for s, g in df.groupby("segment"):
        print(f"      within {s:<8}          corr = {g[['support_contacts', 'satisfaction']].corr().iloc[0, 1]:+.3f}   (expected positive)")
    print(f"[ad_spend, weekly_sales]      corr = {df[['ad_spend', 'weekly_sales']].corr().iloc[0, 1]:+.3f}   (spurious, via temperature)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=int, default=12)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    df, truth = make_benchmark(a.n, a.seed, a.noise)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "benchmark.csv"), index=False)
    with open(os.path.join(a.out, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns, {len(truth)} planted items -> {a.out}/\n")
    sanity_check(df)