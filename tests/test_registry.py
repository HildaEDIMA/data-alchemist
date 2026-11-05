import numpy as np
import pandas as pd
import pytest

from core.registry import Registry


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(size=300)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(size=300),
            "group": np.where(x > 0, "yes", "no"),
            "z": rng.normal(size=300),
        }
    )


def test_association_logs_test_and_extraction() -> None:
    registry = Registry()
    view = registry.register(frame()).extract()
    record = registry.association(view, "x", "y")
    assert registry.m == 1
    assert record.p_value < 0.05
    assert record.variables == ("x", "y")
    assert record.sample_hash
    assert record.extraction_id
    assert record.null_hypothesis


def test_extraction_filter_and_recode_are_logged() -> None:
    registry = Registry()
    view = registry.register(frame()).extract(
        filter_expression="x > 0", predicate=lambda data: data["x"] > 0
    )
    recoded = view.recode("z-score x", lambda data: data.assign(x=data.x / data.x.std()))
    assert len(registry.transformations) == 2
    assert registry.transformations[-1].operation == "z-score x"
    assert registry.association(recoded, "x", "y").n < 300


def test_unregistered_or_foreign_extraction_is_refused() -> None:
    registry, other = Registry(), Registry()
    frame_ = frame()
    with pytest.raises(TypeError, match="logged"):
        registry.association(frame_, "x", "y")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="logged"):
        registry.association(other.register(frame_).extract(), "x", "y")


def test_mutated_extraction_is_refused() -> None:
    registry = Registry()
    view = registry.register(frame()).extract()
    view._frame.loc[0, "x"] = 100
    with pytest.raises(ValueError, match="changed"):
        registry.association(view, "x", "y")


def test_invalid_filter_shape_is_refused() -> None:
    registry = Registry()
    handle = registry.register(frame())
    with pytest.raises(ValueError, match="aligned"):
        handle.extract(filter_expression="bad", predicate=lambda data: pd.Series([True]))
    with pytest.raises(ValueError, match="supplied together"):
        handle.extract(filter_expression="missing predicate")


def test_categorical_anova_and_mann_whitney() -> None:
    data = pd.DataFrame(
        {
            "group": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            "value": np.r_[np.zeros(40), np.ones(40), np.full(40, 2)],
        }
    )
    registry = Registry()
    view = registry.register(data).extract()
    assert registry.anova(view, "group", "value").p_value < 0.05
    two = registry.register(data[data.group != "c"]).extract()
    assert registry.mann_whitney(two, "group", "value").p_value < 0.05


def test_regression_adjustment_and_error_is_journaled() -> None:
    rng = np.random.default_rng(2)
    confounder = rng.normal(size=500)
    predictor = confounder + rng.normal(scale=0.1, size=500)
    outcome = 2 * confounder + rng.normal(size=500)
    registry = Registry()
    view = registry.register(
        pd.DataFrame({"x": predictor, "y": outcome, "confounder": confounder})
    ).extract()
    result = registry.regression(view, "y", "x", adjust_for=["confounder"])
    assert result.adjusted_for == ("confounder",)
    assert result.p_value > 0.05
    failed = registry.association(view, "absent", "y")
    assert failed.error
    assert registry.m == 2


def test_json_serialization_shape() -> None:
    registry = Registry()
    registry.association(registry.register(frame()).extract(), "x", "y")
    payload = registry.to_dict()
    assert payload["m"] == 1
    assert len(payload["tests"]) == 1
    assert len(payload["extractions"]) == 1


def test_registration_and_transform_validation() -> None:
    registry = Registry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(pd.DataFrame())
    with pytest.raises(ValueError, match="non-empty"):
        registry.register({"x": [1]})  # type: ignore[arg-type]
    handle = registry.register(frame())
    handle.frame.loc[0, "x"] = 100
    with pytest.raises(ValueError, match="changed"):
        handle.extract()
    fresh = registry.register(frame())
    view = fresh.extract()
    with pytest.raises(ValueError, match="description"):
        view.recode("", lambda data: data)
    with pytest.raises(ValueError, match="row count"):
        view.recode("drop rows", lambda data: data.iloc[:-1])
    with pytest.raises(ValueError, match="registered extraction"):
        registry._record_transformation("invalid", "recode", frame(), None)


def test_filter_and_recode_reject_misaligned_or_bad_outputs() -> None:
    registry = Registry()
    handle = registry.register(frame())
    with pytest.raises(ValueError, match="aligned"):
        handle.extract(
            filter_expression="bad index", predicate=lambda data: pd.Series([True], index=[-1])
        )
    view = handle.extract()
    with pytest.raises(ValueError, match="aligned"):
        view.filter("bad index", lambda data: pd.Series([True], index=[-1]))
    with pytest.raises(ValueError, match="same row count"):
        view.recode("bad output", lambda data: pd.Series([1]))


def test_association_dispatches_categorical_types_and_direction_errors() -> None:
    registry = Registry()
    categorical = pd.DataFrame(
        {
            "a": ["x", "y"] * 50,
            "b": ["x", "x", "y", "y"] * 25,
            "group": ["a"] * 40 + ["b"] * 40 + ["c"] * 20,
            "value": np.arange(100, dtype=float),
        }
    )
    view = registry.register(categorical).extract()
    assert registry.association(view, "a", "b").test == "chi-square"
    assert registry.association(view, "group", "value").test == "ANOVA"
    unsupported = registry.association(view, "group", "value", direction="greater")
    assert unsupported.error and unsupported.p_value == 1
    constant = registry.association(
        registry.register(pd.DataFrame({"x": [1, 1, 1], "y": [2, 2, 2]})).extract(), "x", "y"
    )
    assert constant.error


def test_directional_group_test_is_invariant_to_row_order() -> None:
    group = ["B"] * 20 + ["A"] * 20
    outcome = list(range(20)) + list(range(20, 40))
    data = pd.DataFrame({"group": group, "outcome": outcome})
    records = []
    for frame in (data, data.iloc[::-1].reset_index(drop=True)):
        registry = Registry()
        view = registry.register(frame).extract()
        records.append(registry.mann_whitney(view, "group", "outcome", direction="greater"))

    assert records[0].p_value == records[1].p_value
    assert records[0].direction == records[1].direction == "greater"


def test_invalid_p_value_and_missing_variables_are_recorded() -> None:
    registry = Registry()
    view = registry.register(frame()).extract()
    unknown = registry.association(view, "missing", "y")
    assert unknown.error and unknown.p_value == 1
    invalid = registry._run(
        view, "test", ("x", "y"), "null", lambda _: (0.0, float("nan"), 0.0, 3, "bad")
    )
    assert invalid.error == "ValueError: Test returned an invalid p-value"


def test_regression_rejects_constant_predictor_and_adjustment_column() -> None:
    registry = Registry()
    view = registry.register(
        pd.DataFrame({"constant": [1.0] * 30, "y": np.arange(30, dtype=float)})
    ).extract()
    result = registry.regression(view, "y", "constant")
    assert result.error
    missing = registry.regression(view, "y", "constant", adjust_for=["absent"])
    assert missing.error


def test_nonlinear_and_interaction_hc3_tests() -> None:
    rng = np.random.default_rng(12)
    x = rng.uniform(-2, 2, 1000)
    binary = rng.binomial(1, 0.5, 1000)
    outcome = x**2 + rng.normal(scale=0.2, size=1000)
    interaction_outcome = x * binary + rng.normal(scale=0.2, size=1000)
    registry = Registry()
    view = registry.register(
        pd.DataFrame(
            {
                "x": x,
                "binary": binary,
                "y": outcome,
                "z": interaction_outcome,
            }
        )
    ).extract()
    nonlinear = registry.nonlinear(view, "x", "y")
    interaction = registry.interaction(view, "x", "binary", "z")
    assert nonlinear.p_value < 0.05 and "quadratic" in nonlinear.test
    assert interaction.p_value < 0.05 and "interaction" in interaction.test
