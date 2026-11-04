"""Statistical tests of the verification engine (Block 2).

Every test returns a dict: p (p-value), effect (r-like size between 0 and 1),
sign (-1, 0, +1), test (name of the test actually used), n (rows used).
Rows with missing values are dropped test by test (pairwise deletion).
"""
import numpy as np
import pandas as pd
from scipy import stats

MIN_GROUP = 30      # smallest group size before switching to rank-based tests
SKEW_LIMIT = 1.0    # beyond this skewness, Pearson is replaced by Spearman


def kind(s):
    """'cat' for text columns or columns with at most 6 distinct values, otherwise 'num'."""
    return "cat" if s.dtype == object or s.nunique() <= 6 else "num"


def _out(p, effect, sign, test, n):
    return dict(p=float(p), effect=float(effect), sign=int(sign), test=test, n=int(n))


# ---------------------------------------------------------------- association
def assoc_test(x, y):
    """Is x associated with y? The test depends on the types of x and y."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n, kx, ky = len(d), kind(d["x"]), kind(d["y"])

    if kx == "num" and ky == "num":
        skewed = max(abs(stats.skew(d["x"])), abs(stats.skew(d["y"]))) > SKEW_LIMIT
        r, p = stats.spearmanr(d["x"], d["y"]) if skewed else stats.pearsonr(d["x"], d["y"])
        return _out(p, abs(r), np.sign(r), "Spearman" if skewed else "Pearson", n)

    if kx == "cat" and ky == "cat":
        tab = pd.crosstab(d["x"], d["y"])
        chi2, p, _, expected = stats.chi2_contingency(tab, correction=False)
        name = "chi-square"
        if tab.shape == (2, 2) and (expected < 5).any():
            _, p = stats.fisher_exact(tab)
            name = "Fisher exact"
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        return _out(p, v, 0, name, n)

    cat, num = ("x", "y") if kx == "cat" else ("y", "x")
    groups = [g.to_numpy(float) for _, g in d.groupby(cat)[num]]
    k, small = len(groups), min(len(g) for g in groups) < MIN_GROUP
    if k == 2:
        sign = np.sign(groups[1].mean() - groups[0].mean())
        if small:
            u, p = stats.mannwhitneyu(groups[0], groups[1])
            return _out(p, abs(1 - 2 * u / (len(groups[0]) * len(groups[1]))), sign, "Mann-Whitney U", n)
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        return _out(p, abs(t) / np.sqrt(t * t + n - 2), sign, "Welch t", n)
    if small:
        H, p = stats.kruskal(*groups)
        return _out(p, np.sqrt(max((H - k + 1) / (n - k), 0)), 0, "Kruskal-Wallis", n)
    F, p = stats.f_oneway(*groups)
    return _out(p, np.sqrt(F * (k - 1) / (F * (k - 1) + n - k)), 0, "ANOVA", n)


# ------------------------------------------------- regression-based tests (HC3)
def _block(s):
    """Design columns of one variable: dummies for categories, z-score for numbers."""
    if kind(s) == "cat":
        return pd.get_dummies(s, drop_first=True, dtype=float).to_numpy()
    v = s.to_numpy(float)
    return ((v - v.mean()) / v.std())[:, None]


def _rss(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return float(r @ r)


def _wald_hc3(X, y, cols):
    """Fit OLS and test that the coefficients in `cols` are all zero, using a
    heteroskedasticity-robust (HC3) covariance. Returns (p, coefficients)."""
    X = np.column_stack([np.ones(len(y)), X])
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    h = np.einsum("ij,jk,ik->i", X, inv, X)                    # leverage
    e = (y - X @ beta) / np.maximum(1 - h, 1e-6)
    V = inv @ ((X * (e ** 2)[:, None]).T @ X) @ inv
    idx = [c + 1 for c in cols]                                 # +1 for the intercept
    b, Vs = beta[idx], V[np.ix_(idx, idx)]
    W = float(b @ np.linalg.pinv(Vs) @ b)
    return float(stats.chi2.sf(W, len(idx))), beta


def nonlinear_test(x, y):
    """Does y depend on x beyond a straight line? Tests the squared term of x."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    z, yv = _block(d["x"])[:, 0], d["y"].to_numpy(float)
    X1 = np.column_stack([z, z ** 2])
    p, beta = _wald_hc3(X1, yv, [1])
    rss0, rss1 = _rss(z[:, None], yv), _rss(X1, yv)
    return _out(p, np.sqrt(max(rss0 - rss1, 0) / rss0), np.sign(beta[-1]), "quadratic term (HC3 Wald)", len(d))


def interaction_test(a, b, y):
    """Does the effect of a on y depend on b? Tests the a x b product terms."""
    d = pd.DataFrame({"a": a, "b": b, "y": y}).dropna()
    A, B, yv = _block(d["a"]), _block(d["b"]), d["y"].to_numpy(float)
    X0 = np.column_stack([A, B])
    inter = np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1]) for j in range(B.shape[1])])
    X1 = np.column_stack([X0, inter])
    p, _ = _wald_hc3(X1, yv, list(range(X0.shape[1], X1.shape[1])))
    rss0, rss1 = _rss(X0, yv), _rss(X1, yv)
    return _out(p, np.sqrt(max(rss0 - rss1, 0) / rss0), 0, "interaction terms (HC3 Wald)", len(d))


if __name__ == "__main__":
    import generate_benchmark as gb
    df, _ = gb.make_benchmark(20000, 42)
    cases = [
        ("assoc        delivery_days ~ rating            (real)", assoc_test(df.delivery_days, df.rating)),
        ("assoc        discount_pct ~ repeat_purchase    (real)", assoc_test(df.discount_pct, df.repeat_purchase)),
        ("assoc        ad_spend ~ weekly_sales    (confounded)", assoc_test(df.ad_spend, df.weekly_sales)),
        ("assoc        noise_num_01 ~ spend              (noise)", assoc_test(df.noise_num_01, df.spend)),
        ("assoc        region ~ channel            (independent)", assoc_test(df.region, df.channel)),
        ("nonlinear    age -> return_rate                (real)", nonlinear_test(df.age, df.return_rate)),
        ("nonlinear    delivery_days -> rating    (straight line)", nonlinear_test(df.delivery_days, df.rating)),
        ("interaction  loyalty x region -> spend         (real)", interaction_test(df.loyalty_member, df.region, df.spend)),
        ("interaction  weekend x channel -> converted    (real)", interaction_test(df.weekend, df.channel, df.converted)),
        ("interaction  loyalty x channel -> spend       (noise)", interaction_test(df.loyalty_member, df.channel, df.spend)),
    ]
    for label, r in cases:
        print(f"{label:<58}{r['test']:<28}p = {r['p']:.2e}   effect = {r['effect']:.3f}")