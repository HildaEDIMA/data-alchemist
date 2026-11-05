import numpy as np
import pandas as pd
from scipy import stats

from benchmark.run import _null_frame, _score
from core.multiplicity import e_bh
from core.registry import Registry
from core.split import ConfirmationSplit
from generate_benchmark import make_benchmark


def test_e_bh_adjusted_values_are_monotone_in_e_rank() -> None:
    result = e_bh([1e-8, 0.01, 0.7, 0.8])
    ranked = sorted(zip([1e-8, 0.01, 0.7, 0.8], result.adjusted_p, strict=True))
    assert [value for _, value in ranked] == sorted(value for _, value in ranked)
    assert result.rejected == tuple(value <= 0.05 for value in result.adjusted_p)


def test_benchmark_scoring_uses_ground_truth_keys_not_random_record_ids() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    registry = Registry()
    view = registry.register(frame).extract()
    true_record = registry.association(view, "x", "y")
    false_record = registry.association(view, "x", "y")

    score = _score([("T01", true_record), ("noise", false_record)], [True, True], {"T01"})

    assert score == {
        "true_positives": 1,
        "false_discoveries": 1,
        "discoveries": 2,
        "recall": 1.0,
        "fdp": 0.5,
    }


def test_null_generator_preserves_marginals_and_uses_blinded_names() -> None:
    source, _ = make_benchmark(n=80, seed=13)
    null = _null_frame(seed=13, n=80)

    assert null.columns.tolist() == [f"var_{index + 1}" for index in range(len(source.columns))]
    for index, column in enumerate(source.columns):
        assert sorted(null[f"var_{index + 1}"].astype(str)) == sorted(source[column].astype(str))


def test_split_confirmation_fdr_monte_carlo_under_global_null() -> None:
    simulations = 1000
    alpha = 0.05
    false_discovery_runs = 0
    rng = np.random.default_rng(0)
    for seed in range(simulations):
        frame = pd.DataFrame(
            {
                name: rng.normal(size=120)
                for index in range(10)
                for name in (f"x{index}", f"y{index}")
            }
        )
        registry = Registry()
        split = ConfirmationSplit(frame, registry, seed=seed)
        exploration = [
            (index, registry.association(split.exploration, f"x{index}", f"y{index}"))
            for index in range(10)
        ]
        promoted = [
            {
                "test": "association",
                "variables": [f"x{index}", f"y{index}"],
            }
            for index, record in exploration
            if record.p_value < alpha
        ]
        if promoted:
            split.promote(promoted)
            confirmed = split.confirm(registry)
            false_discovery_runs += any(item["confirmed"] for item in confirmed)

    interval = stats.beta.interval(
        0.95,
        false_discovery_runs + 1,
        simulations - false_discovery_runs,
    )
    upper = float(interval[1])
    assert upper <= alpha
