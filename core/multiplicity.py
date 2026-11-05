"""Multiple-testing adjustments with explicit assumptions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Adjustment:
    method: str
    alpha: float
    adjusted_p: tuple[float, ...]
    rejected: tuple[bool, ...]
    guarantee: str


def _validate(p_values: Sequence[float], alpha: float) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite one-dimensional sequence in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return p


def adjust(p_values: Sequence[float], method: str, alpha: float = 0.05) -> Adjustment:
    p = _validate(p_values, alpha)
    m = len(p)
    if not m:
        return Adjustment(method, alpha, (), (), _guarantee(method, alpha))
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted = np.empty(m, dtype=np.float64)
    if method == "bonferroni":
        adjusted[:] = np.minimum(1.0, p * m)
    elif method in ("holm",):
        holm_adjusted = sorted_p * (m - np.arange(m))
        holm_adjusted = np.maximum.accumulate(holm_adjusted)
        adjusted[order] = np.minimum(holm_adjusted, 1)
    elif method in ("bh", "by"):
        factor = float(np.sum(1 / np.arange(1, m + 1))) if method == "by" else 1.0
        bh_adjusted = sorted_p * m * factor / np.arange(1, m + 1)
        bh_adjusted = np.minimum.accumulate(bh_adjusted[::-1])[::-1]
        adjusted[order] = np.minimum(bh_adjusted, 1)
    else:
        raise ValueError(f"Unknown p-value adjustment: {method}")
    rejected = adjusted <= alpha
    return Adjustment(
        method,
        alpha,
        tuple(map(float, adjusted)),
        tuple(map(bool, rejected)),
        _guarantee(method, alpha),
    )


def e_bh(p_values: Sequence[float], alpha: float = 0.05, gamma: float = 0.5) -> Adjustment:
    """e-BH using e=gamma*p**(gamma-1); valid under arbitrary dependence
    for a fixed family of valid null p-values.
    """
    p = _validate(p_values, alpha)
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    m = len(p)
    if not m:
        return Adjustment("e_bh", alpha, (), (), _guarantee("e_bh", alpha))
    e_values = gamma * np.maximum(p, np.finfo(float).tiny) ** (gamma - 1)
    order = np.argsort(-e_values, kind="stable")
    sorted_e = e_values[order]
    sorted_adjusted = m / (np.arange(1, m + 1) * sorted_e)
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    adjusted = np.ones(m, dtype=float)
    adjusted[order] = np.minimum(1.0, sorted_adjusted)
    rejected = adjusted <= alpha
    return Adjustment(
        "e_bh",
        alpha,
        tuple(map(float, adjusted)),
        tuple(map(bool, rejected)),
        _guarantee("e_bh", alpha),
    )


def _guarantee(method: str, alpha: float) -> str:
    rate = f"{alpha:.1%}"
    if method == "bonferroni":
        return f"FWER at most {rate} under valid marginal p-values"
    if method == "holm":
        return f"FWER at most {rate} under valid marginal p-values"
    if method == "bh":
        return f"FDR at most {rate} under independence or PRDS and valid p-values"
    if method == "by":
        return f"FDR at most {rate} under arbitrary dependence and valid p-values"
    if method == "e_bh":
        return (
            f"FDR at most {rate} under arbitrary dependence for valid e-values and a fixed family"
        )
    raise ValueError(f"Unknown method: {method}")
