"""Instrumented statistical test API and immutable audit records."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(frame[c].dtype)) for c in frame.columns]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _series_hash(series: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(str(series.name).encode())
    digest.update(str(series.dtype).encode())
    values = series.to_numpy(copy=False)
    if values.dtype.kind in "biufcMm":
        digest.update(np.ascontiguousarray(values).view(np.uint8).tobytes())
    else:
        digest.update(pd.util.hash_pandas_object(series, index=False).values.tobytes())
    if isinstance(series.index, pd.RangeIndex):
        digest.update(repr(series.index).encode())
    else:
        digest.update(pd.util.hash_pandas_object(series.index, index=False).values.tobytes())
    return digest.hexdigest()


def _column_hashes(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): _series_hash(frame[column]) for column in frame.columns}


def _stable_levels(values: pd.Series) -> list[Any]:
    return sorted(pd.unique(values), key=lambda value: (type(value).__name__, repr(value)))


@dataclass(frozen=True)
class TransformationRecord:
    id: str
    timestamp: float
    operation: str
    input_hash: str
    output_hash: str
    filter_expression: str | None = None


@dataclass(frozen=True)
class ExtractionRecord:
    id: str
    dataset_id: str
    parent_extraction_id: str | None
    timestamp: float
    sample_hash: str
    n: int
    filter_expression: str | None


@dataclass(frozen=True)
class TestRecord:
    id: str
    timestamp: float
    test: str
    variables: tuple[str, ...]
    extraction_id: str
    sample_hash: str
    filter_expression: str | None
    code: str
    statistic: float
    p_value: float
    effect_size: float
    n: int
    null_hypothesis: str
    direction: Literal["two-sided", "greater", "less"] = "two-sided"
    adjusted_for: tuple[str, ...] = ()
    error: str | None = None
    procedure: str = ""


@dataclass
class DatasetHandle:
    """A dataset admitted to the registry; analyses must use logged extractions."""

    registry: Registry
    dataset_id: str
    frame: pd.DataFrame
    data_hash: str

    def extract(
        self,
        *,
        filter_expression: str | None = None,
        predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> TrackedFrame:
        if (filter_expression is None) != (predicate is None):
            raise ValueError("A filter expression and its predicate must be supplied together")
        if _frame_hash(self.frame) != self.data_hash:
            raise ValueError("Dataset changed after it was registered")
        selected = self.frame.copy(deep=True)
        if predicate is not None:
            mask = predicate(selected)
            if not isinstance(mask, pd.Series) or not mask.index.equals(selected.index):
                raise ValueError("Filter must return a Series aligned with the dataset")
            selected = selected.loc[mask.astype(bool)].copy()
        extraction_id = self.registry._record_extraction(
            self.dataset_id, selected, filter_expression
        )
        return TrackedFrame(self.registry, extraction_id, selected, filter_expression)


@dataclass
class TrackedFrame:
    """A logged, dataset-bound view used exclusively by the statistical API."""

    registry: Registry
    extraction_id: str
    _frame: pd.DataFrame = field(repr=False)
    filter_expression: str | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._frame.columns)

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    def to_pandas(self) -> pd.DataFrame:
        """Return a defensive copy; use this only for descriptive inspection."""
        return self._frame.copy(deep=True)

    def _assert_unchanged(self, columns: Sequence[str] | None = None) -> None:
        if columns is None:
            expected_hash = self.registry._extraction_hashes.get(self.extraction_id)
            if expected_hash is None or _frame_hash(self._frame) != expected_hash:
                raise ValueError("Extraction changed after it was logged")
            return
        expected_hashes = self.registry._extraction_column_hashes.get(self.extraction_id)
        if expected_hashes is None or any(
            column not in expected_hashes
            or column not in self._frame.columns
            or _series_hash(self._frame[column]) != expected_hashes[column]
            for column in columns
        ):
            raise ValueError("Extraction changed after it was logged")

    def filter(
        self, expression: str, predicate: Callable[[pd.DataFrame], pd.Series]
    ) -> TrackedFrame:
        self._assert_unchanged()
        mask = predicate(self._frame.copy(deep=True))
        if not isinstance(mask, pd.Series) or not mask.index.equals(self._frame.index):
            raise ValueError("Filter must return a Series aligned with the extraction")
        child = self._frame.loc[mask.astype(bool)].copy()
        combined = (
            f"({self.filter_expression}) AND ({expression})"
            if self.filter_expression
            else expression
        )
        extraction_id = self.registry._record_transformation(
            self.extraction_id, "filter", child, combined
        )
        return TrackedFrame(self.registry, extraction_id, child, combined)

    def recode(
        self, operation: str, transform: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> TrackedFrame:
        self._assert_unchanged()
        if not operation.strip():
            raise ValueError("A recoding operation description is required")
        output = transform(self._frame.copy(deep=True))
        if not isinstance(output, pd.DataFrame) or len(output) != len(self._frame):
            raise ValueError("A recode must return a DataFrame with the same row count")
        extraction_id = self.registry._record_transformation(
            self.extraction_id, operation, output, self.filter_expression
        )
        return TrackedFrame(self.registry, extraction_id, output, self.filter_expression)


class Registry:
    """Append-only ledger. Statistical calls accept only registered extractions."""

    def __init__(self) -> None:
        self.tests: list[TestRecord] = []
        self.transformations: list[TransformationRecord] = []
        self.extractions: list[ExtractionRecord] = []
        self.promotions: list[dict[str, Any]] = []
        self._extractions: dict[str, tuple[pd.DataFrame, str | None]] = {}
        self._extraction_hashes: dict[str, str] = {}
        self._extraction_column_hashes: dict[str, dict[str, str]] = {}
        self._extraction_dataset: dict[str, str] = {}
        self._association_profiles: dict[tuple[str, str], tuple[bool, float, bool]] = {}
        self._datasets: dict[str, str] = {}
        self._dataset_meta: dict[str, dict[str, Any]] = {}

    @property
    def m(self) -> int:
        return len(self.tests)

    def register(self, frame: pd.DataFrame) -> DatasetHandle:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Register a non-empty pandas DataFrame")
        data = frame.copy(deep=True).reset_index(drop=True)
        dataset_id = str(uuid.uuid4())
        digest = _frame_hash(data)
        self._datasets[dataset_id] = digest
        self._dataset_meta[dataset_id] = {
            "id": dataset_id,
            "data_hash": digest,
            "n": len(data),
            "columns": list(map(str, data.columns)),
        }
        return DatasetHandle(self, dataset_id, data, digest)

    def _record_extraction(
        self, dataset_id: str, frame: pd.DataFrame, expression: str | None
    ) -> str:
        if dataset_id not in self._datasets:
            raise ValueError("Dataset is not registered in this registry")
        extraction_id = str(uuid.uuid4())
        copied = frame.copy(deep=True)
        digest = _frame_hash(copied)
        self._extractions[extraction_id] = (copied, expression)
        self._extraction_hashes[extraction_id] = digest
        self._extraction_column_hashes[extraction_id] = _column_hashes(copied)
        self._extraction_dataset[extraction_id] = dataset_id
        self.extractions.append(
            ExtractionRecord(
                extraction_id, dataset_id, None, time.time(), digest, len(copied), expression
            )
        )
        if expression is not None:
            self.transformations.append(
                TransformationRecord(
                    str(uuid.uuid4()),
                    time.time(),
                    "split" if "half" in expression or "holdout" in expression else "filter",
                    self._datasets[dataset_id],
                    digest,
                    expression,
                )
            )
        return extraction_id

    def _association_profile(
        self, extraction_id: str, column: str, values: pd.Series
    ) -> tuple[bool, float, bool]:
        key = extraction_id, column
        cached = self._association_profiles.get(key)
        if cached is not None:
            return cached
        numeric = pd.api.types.is_numeric_dtype(values)
        categorical = not numeric or values.nunique() <= 6
        has_missing = bool(values.isna().any())
        skew = (
            abs(float(stats.skew(values)))
            if numeric and not categorical and not has_missing
            else 0.0
        )
        profile = categorical, skew, has_missing
        self._association_profiles[key] = profile
        return profile

    def _record_transformation(
        self, parent_id: str, operation: str, output: pd.DataFrame, expression: str | None
    ) -> str:
        if parent_id not in self._extractions:
            raise ValueError("Transformation input is not a registered extraction")
        extraction_id = str(uuid.uuid4())
        child = output.copy(deep=True)
        child_hash = _frame_hash(child)
        parent_hash = self._extraction_hashes[parent_id]
        self._extractions[extraction_id] = (child, expression)
        self._extraction_hashes[extraction_id] = child_hash
        self._extraction_column_hashes[extraction_id] = _column_hashes(child)
        dataset_id = self._extraction_dataset[parent_id]
        self._extraction_dataset[extraction_id] = dataset_id
        self.extractions.append(
            ExtractionRecord(
                extraction_id,
                dataset_id,
                parent_id,
                time.time(),
                child_hash,
                len(child),
                expression,
            )
        )
        self.transformations.append(
            TransformationRecord(
                str(uuid.uuid4()),
                time.time(),
                operation,
                parent_hash,
                child_hash,
                expression,
            )
        )
        return extraction_id

    def _run(
        self,
        data: TrackedFrame,
        test: str,
        variables: Sequence[str],
        null: str,
        calculator: Callable[[pd.DataFrame], tuple[float, float, float, int, str]],
        *,
        direction: Literal["two-sided", "greater", "less"] = "two-sided",
        adjusted_for: Sequence[str] = (),
        code: str | None = None,
    ) -> TestRecord:
        if not isinstance(data, TrackedFrame) or data.registry is not self:
            raise TypeError("Tests require an extraction logged by this Registry")
        if data.extraction_id not in self._extractions:
            raise ValueError("Extraction is not present in the registry")
        source, expression = self._extractions[data.extraction_id]
        sample_hash = self._extraction_hashes[data.extraction_id]
        names = tuple(variables)
        required = tuple(dict.fromkeys((*names, *adjusted_for)))
        checked_columns = (
            required if all(name in data._frame.columns for name in required) else None
        )
        data._assert_unchanged(checked_columns)
        try:
            missing = [name for name in required if name not in source.columns]
            if missing:
                raise ValueError(f"Unknown variable(s): {', '.join(missing)}")
            statistic, p_value, effect, n, actual_test = calculator(source)
            if not (math.isfinite(p_value) and 0 <= p_value <= 1):
                raise ValueError("Test returned an invalid p-value")
            error = None
        except Exception as exc:
            statistic, p_value, effect, n, actual_test = 0.0, 1.0, 0.0, 0, test
            error = f"{type(exc).__name__}: {exc}"
        record = TestRecord(
            str(uuid.uuid4()),
            time.time(),
            actual_test,
            names,
            data.extraction_id,
            sample_hash,
            expression,
            code or _caller_code(),
            float(statistic),
            float(p_value),
            float(effect),
            int(n),
            null,
            direction,
            tuple(adjusted_for),
            error,
            test,
        )
        self.tests.append(record)
        return record

    def association(
        self,
        data: TrackedFrame,
        x: str,
        y: str,
        *,
        method: Literal["auto", "pearson", "spearman"] = "auto",
        direction: Literal["two-sided", "greater", "less"] = "two-sided",
        code: str | None = None,
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            profiles = (
                self._association_profile(data.extraction_id, x, frame[x]),
                self._association_profile(data.extraction_id, y, frame[y]),
            ) if len(frame) == data.n_rows else None
            if profiles is not None and not (profiles[0][2] or profiles[1][2]):
                a, b = frame[x], frame[y]
                (cat_a, skew_a, _), (cat_b, skew_b, _) = profiles
            else:
                d = frame[[x, y]].dropna()
                if len(d) < 3:
                    raise ValueError("At least three complete rows are required")
                a, b = d[x], d[y]
                cat_a = not pd.api.types.is_numeric_dtype(a) or a.nunique() <= 6
                cat_b = not pd.api.types.is_numeric_dtype(b) or b.nunique() <= 6
                skew_a = abs(float(stats.skew(a))) if not cat_a else 0.0
                skew_b = abs(float(stats.skew(b))) if not cat_b else 0.0
            sample_n = len(a)
            if not cat_a and not cat_b:
                skewed = max(skew_a, skew_b) > 1
                chosen = (
                    "spearman"
                    if method == "spearman" or (method == "auto" and skewed)
                    else "pearson"
                )
                if chosen == "spearman":
                    result = stats.spearmanr(a, b, alternative=direction)
                else:
                    result = stats.pearsonr(a, b, alternative=direction)
                return (
                    float(result.statistic),
                    float(result.pvalue),
                    abs(float(result.statistic)),
                    sample_n,
                    chosen.capitalize(),
                )
            if cat_a and cat_b:
                table = pd.crosstab(a, b)
                if min(table.shape) < 2:
                    raise ValueError("Categorical association requires two levels per variable")
                statistic, p_value, _, _ = stats.chi2_contingency(table, correction=False)
                effect = math.sqrt(statistic / (sample_n * (min(table.shape) - 1)))
                return float(statistic), float(p_value), effect, sample_n, "chi-square"
            cat, num = (a, b) if cat_a else (b, a)
            levels = _stable_levels(cat)
            groups = [
                num.loc[cat == level].to_numpy(dtype=float)
                for level in levels
            ]
            if len(groups) < 2:
                raise ValueError("Group comparison requires at least two groups")
            if len(groups) == 2:
                result = stats.ttest_ind(
                    groups[0], groups[1], equal_var=False, alternative=direction
                )
                effect = abs(float(np.mean(groups[1]) - np.mean(groups[0]))) / max(
                    float(np.std(num, ddof=1)), np.finfo(float).eps
                )
                return float(result.statistic), float(result.pvalue), effect, sample_n, "Welch t"
            if direction != "two-sided":
                raise ValueError(
                    "Directional tests are supported only for two groups or numeric associations"
                )
            result = stats.f_oneway(*groups)
            effect = max(
                0.0,
                float(result.statistic)
                * (len(groups) - 1)
                / (float(result.statistic) * (len(groups) - 1) + sample_n - len(groups)),
            )
            return (
                float(result.statistic),
                float(result.pvalue),
                math.sqrt(effect),
                sample_n,
                "ANOVA",
            )

        return self._run(
            data,
            "association",
            (x, y),
            f"{x} is independent of {y} (association test)",
            calculate,
            direction=direction,
            code=code,
        )

    def mann_whitney(
        self,
        data: TrackedFrame,
        group: str,
        outcome: str,
        *,
        direction: Literal["two-sided", "greater", "less"] = "two-sided",
        code: str | None = None,
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            d = frame[[group, outcome]].dropna()
            levels = _stable_levels(d[group])
            if len(levels) != 2:
                raise ValueError("Mann-Whitney requires exactly two groups")
            a = d.loc[d[group] == levels[0], outcome].to_numpy(dtype=float)
            b = d.loc[d[group] == levels[1], outcome].to_numpy(dtype=float)
            result = stats.mannwhitneyu(a, b, alternative=direction)
            return (
                float(result.statistic),
                float(result.pvalue),
                abs(1 - 2 * result.statistic / (len(a) * len(b))),
                len(d),
                "Mann-Whitney U",
            )

        return self._run(
            data,
            "mann_whitney",
            (group, outcome),
            f"{group} has no distributional effect on {outcome}",
            calculate,
            direction=direction,
            code=code,
        )

    def regression(
        self,
        data: TrackedFrame,
        outcome: str,
        predictor: str,
        *,
        adjust_for: Sequence[str] = (),
        code: str | None = None,
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            from core.registry import _design_matrix

            columns = [predictor, *adjust_for]
            d = frame[[outcome, *columns]].dropna()
            y = pd.to_numeric(d[outcome], errors="raise").to_numpy(dtype=float)
            x_parts = [_design_matrix(d[name]) for name in columns]
            widths = [part.shape[1] for part in x_parts]
            X = np.column_stack([np.ones(len(d)), *x_parts])
            if len(d) <= X.shape[1] or widths[0] == 0:
                raise ValueError("Regression requires more complete rows than coefficients")
            inverse = np.linalg.pinv(X.T @ X)
            beta = inverse @ X.T @ y
            residual = y - X @ beta
            leverage = np.einsum("ij,jk,ik->i", X, inverse, X)
            scaled = residual / np.maximum(1 - leverage, 1e-8)
            covariance = inverse @ ((X * (scaled**2)[:, None]).T @ X) @ inverse
            start, end = 1, 1 + widths[0]
            coefficients = beta[start:end]
            cov = covariance[start:end, start:end]
            wald = float(coefficients @ np.linalg.pinv(cov) @ coefficients)
            p_value = float(stats.chi2.sf(wald, end - start))
            sse_full = float(residual @ residual)
            base = np.column_stack([np.ones(len(d)), *x_parts[1:]])
            residual_base = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
            sse_base = float(residual_base @ residual_base)
            partial_r2 = max(0.0, min(1.0, (sse_base - sse_full) / max(sse_base, 1e-12)))
            return wald, p_value, partial_r2, len(d), "OLS HC3 Wald"

        return self._run(
            data,
            "ols_hc3_wald",
            (predictor, outcome),
            f"coefficients for {predictor} are zero conditional on {tuple(adjust_for)}",
            calculate,
            adjusted_for=adjust_for,
            code=code,
        )

    def nonlinear(
        self, data: TrackedFrame, predictor: str, outcome: str, *, code: str | None = None
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            d = frame[[predictor, outcome]].dropna()
            x = pd.to_numeric(d[predictor], errors="raise").to_numpy(dtype=float)
            y = pd.to_numeric(d[outcome], errors="raise").to_numpy(dtype=float)
            scale = x.std()
            if scale == 0:
                raise ValueError("Non-linearity test requires a varying numeric predictor")
            z = (x - x.mean()) / scale
            base = np.column_stack([np.ones(len(z)), z])
            full = np.column_stack([base, z**2])
            statistic, p_value = _hc3_wald(full, y, [2])
            residual_base = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
            residual_full = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
            effect = max(
                0.0,
                min(
                    1.0,
                    float(
                        (residual_base @ residual_base - residual_full @ residual_full)
                        / max(residual_base @ residual_base, 1e-12)
                    ),
                ),
            )
            return statistic, p_value, effect, len(d), "quadratic term (HC3 Wald)"

        return self._run(
            data,
            "quadratic_hc3_wald",
            (predictor, outcome),
            f"quadratic coefficient for {predictor} in {outcome} is zero",
            calculate,
            code=code,
        )

    def interaction(
        self, data: TrackedFrame, first: str, second: str, outcome: str, *, code: str | None = None
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            d = frame[[first, second, outcome]].dropna()
            y = pd.to_numeric(d[outcome], errors="raise").to_numpy(dtype=float)
            a, b = _design_matrix(d[first]), _design_matrix(d[second])
            if not a.shape[1] or not b.shape[1]:
                raise ValueError("Interaction requires varying predictors")
            base = np.column_stack([np.ones(len(d)), a, b])
            products = np.column_stack(
                [a[:, i] * b[:, j] for i in range(a.shape[1]) for j in range(b.shape[1])]
            )
            full = np.column_stack([base, products])
            indexes = list(range(base.shape[1], full.shape[1]))
            statistic, p_value = _hc3_wald(full, y, indexes)
            residual_base = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
            residual_full = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
            effect = max(
                0.0,
                min(
                    1.0,
                    float(
                        (residual_base @ residual_base - residual_full @ residual_full)
                        / max(residual_base @ residual_base, 1e-12)
                    ),
                ),
            )
            return statistic, p_value, effect, len(d), "interaction terms (HC3 Wald)"

        return self._run(
            data,
            "interaction_hc3_wald",
            (first, second, outcome),
            f"interaction coefficients for {first} and {second} in {outcome} are zero",
            calculate,
            code=code,
        )

    def anova(
        self, data: TrackedFrame, group: str, outcome: str, *, code: str | None = None
    ) -> TestRecord:
        return self._group_test(data, group, outcome, "anova", code)

    def _group_test(
        self,
        data: TrackedFrame,
        group: str,
        outcome: str,
        method: Literal["anova", "mann_whitney"],
        code: str | None,
    ) -> TestRecord:
        def calculate(frame: pd.DataFrame) -> tuple[float, float, float, int, str]:
            d = frame[[group, outcome]].dropna()
            groups = [g.to_numpy(dtype=float) for _, g in d.groupby(group, observed=True)[outcome]]
            if len(groups) < 2:
                raise ValueError("Group test requires at least two groups")
            result = stats.f_oneway(*groups)
            df_between = len(groups) - 1
            eta_squared = (
                1.0
                if np.isposinf(result.statistic)
                else float(
                    result.statistic
                    * df_between
                    / (result.statistic * df_between + len(d) - len(groups))
                )
            )
            return (
                float(result.statistic),
                float(result.pvalue),
                math.sqrt(max(0.0, eta_squared)),
                len(d),
                "ANOVA",
            )

        return self._run(
            data,
            method,
            (group, outcome),
            f"all {outcome} group means are equal across {group}",
            calculate,
            code=code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "datasets": list(self._dataset_meta.values()),
            "extractions": [asdict(item) for item in self.extractions],
            "promotions": list(self.promotions),
            "tests": [asdict(test) for test in self.tests],
            "transformations": [asdict(item) for item in self.transformations],
        }

    def get_test(self, test_id: str) -> TestRecord | None:
        return next((test for test in self.tests if test.id == test_id), None)


def _design_matrix(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 6:
        values = cast(np.ndarray, np.asarray(series.to_numpy(dtype=float), dtype=float))
        scale = values.std()
        if scale == 0:
            raise ValueError("Cannot regress on a constant predictor")
        return cast(
            np.ndarray, np.asarray(((values - values.mean()) / scale)[:, None], dtype=np.float64)
        )
    return np.asarray(
        pd.get_dummies(series.astype("category"), drop_first=True, dtype=float).to_numpy(),
        dtype=float,
    )


def _hc3_wald(X: np.ndarray, y: np.ndarray, indexes: Sequence[int]) -> tuple[float, float]:
    inverse = np.linalg.pinv(X.T @ X)
    beta = inverse @ X.T @ y
    leverage = np.einsum("ij,jk,ik->i", X, inverse, X)
    residual = (y - X @ beta) / np.maximum(1 - leverage, 1e-8)
    covariance = inverse @ ((X * (residual**2)[:, None]).T @ X) @ inverse
    coefficients = beta[list(indexes)]
    selected_covariance = covariance[np.ix_(indexes, indexes)]
    statistic = float(coefficients @ np.linalg.pinv(selected_covariance) @ coefficients)
    return statistic, float(stats.chi2.sf(statistic, len(indexes)))


def _caller_code() -> str:
    frame = inspect.currentframe()
    if frame is None:
        return "unknown"
    for _ in range(3):
        if frame.f_back is None:
            break
        frame = frame.f_back
    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    del frame
    return location
