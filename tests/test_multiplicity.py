import pytest

from core.multiplicity import adjust, e_bh


@pytest.mark.parametrize("method", ["bonferroni", "holm", "bh", "by"])
def test_adjustment_contract(method: str) -> None:
    result = adjust([0.001, 0.02, 0.2, 0.7], method)
    assert len(result.adjusted_p) == 4
    assert result.rejected == tuple(value <= 0.05 for value in result.adjusted_p)
    assert "valid" in result.guarantee


def test_bh_monotonic_and_by_more_conservative() -> None:
    p = [0.01, 0.04, 0.2]
    bh = adjust(p, "bh")
    by = adjust(p, "by")
    assert all(a <= b for a, b in zip(bh.adjusted_p, by.adjusted_p, strict=True))
    assert bh.adjusted_p[0] <= bh.adjusted_p[1] <= bh.adjusted_p[2]


def test_empty_and_bad_values() -> None:
    assert adjust([], "holm").rejected == ()
    with pytest.raises(ValueError):
        adjust([0.1, float("nan")], "bh")
    with pytest.raises(ValueError):
        adjust([1.1], "bh")
    with pytest.raises(ValueError):
        adjust([0.1], "invented")


def test_e_bh_is_arbitrary_dependence_and_rejects_small_p() -> None:
    result = e_bh([1e-8, 0.01, 0.7], alpha=0.05)
    assert result.rejected[0]
    assert not result.rejected[2]
    assert "arbitrary dependence" in result.guarantee
    with pytest.raises(ValueError):
        e_bh([0.1], gamma=1.0)
