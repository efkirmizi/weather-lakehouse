"""When a challenger replaces the champion. Runs inside dag-pytorch-model-training."""
import numpy as np
import pytest

from promotion import promotion_verdict

STRIDE = 96  # seq_len + pred_len: one sample per non-overlapping horizon


def _errors(n=120_000, level=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(level, 0.3, n))


def test_a_real_improvement_is_promoted():
    """The margin an incremental run actually produces: small, consistent, real."""
    champion = _errors()
    challenger = champion * 0.99          # 1% lower MSE everywhere
    verdict = promotion_verdict(champion, challenger, stride=STRIDE)
    assert verdict.promote


def test_an_identical_challenger_is_rejected():
    champion = _errors()
    verdict = promotion_verdict(champion, champion.copy(), stride=STRIDE)
    assert not verdict.promote


def test_a_worse_challenger_is_rejected():
    champion = _errors()
    verdict = promotion_verdict(champion, champion * 1.05, stride=STRIDE)
    assert not verdict.promote


def test_a_consistent_but_negligible_gain_is_rejected_on_effect_size():
    """Perfectly reliable, and far too small to be worth a deployment."""
    champion = _errors()
    challenger = champion * 0.9999998      # ~0.00001% better RMSE
    verdict = promotion_verdict(champion, challenger, stride=STRIDE)
    assert not verdict.promote
    assert "improvement" in verdict.reason


def test_a_gain_buried_in_noise_is_rejected_on_significance():
    """Better on average and past the effect-size floor, but the per-window
    differences are dominated by noise - exactly the case a fixed percentage
    threshold waves through."""
    rng = np.random.default_rng(7)
    champion = _errors(seed=1)
    challenger = np.clip(champion - 0.004 + rng.normal(0, 0.15, champion.size), 1e-6, None)

    verdict = promotion_verdict(champion, challenger, stride=STRIDE)

    assert verdict.relative_improvement > 0.001    # clears the floor
    assert verdict.p_value > 0.05                  # but is not reliable
    assert not verdict.promote
    assert "reliable" in verdict.reason


def test_overlapping_windows_are_subsampled_away():
    """113k overlapping windows are nowhere near 113k independent observations."""
    champion = _errors(n=113_852)
    verdict = promotion_verdict(champion, champion * 0.99, stride=STRIDE)
    assert verdict.samples == len(range(0, 113_852, STRIDE))
    assert verdict.samples < 1500


def test_the_verdict_reports_the_relative_improvement():
    champion = _errors()
    verdict = promotion_verdict(champion, champion * 0.9801, stride=STRIDE)  # 1% on RMSE
    assert verdict.relative_improvement == pytest.approx(0.01, abs=1e-4)
