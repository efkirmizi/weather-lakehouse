"""Naive forecasts the trained models are measured against.

Runs inside dag-pytorch-model-training (PYTHONPATH=/app).
"""
import pytest
import torch

from baselines import persistence_forecast, seasonal_naive_forecast

SEQ, PRED, FEATURES = 72, 24, 4


def _context(batch=2):
    """Each hour carries its own index as its value, so a wrong slice is visible."""
    return torch.arange(SEQ, dtype=torch.float32).reshape(1, SEQ, 1).repeat(batch, 1, FEATURES)


def test_persistence_repeats_the_last_observed_hour():
    out = persistence_forecast(_context(), PRED)
    assert out.shape == (2, PRED, FEATURES)
    assert torch.all(out == SEQ - 1)


def test_seasonal_naive_replays_the_same_hour_one_day_earlier():
    """Target hour t+k is predicted by hour t+k-24, which sits inside the context."""
    out = seasonal_naive_forecast(_context(), PRED)
    assert out.shape == (2, PRED, FEATURES)
    # first predicted hour (t+1) <- t+1-24, i.e. context index SEQ-24
    assert torch.all(out[:, 0, :] == SEQ - PRED)
    # last predicted hour (t+24) <- t, i.e. the newest context hour
    assert torch.all(out[:, -1, :] == SEQ - 1)


def test_the_two_baselines_are_not_the_same_forecast():
    """If they agreed there would be no point logging both."""
    x = _context()
    assert not torch.equal(persistence_forecast(x, PRED), seasonal_naive_forecast(x, PRED))


def test_seasonal_naive_refuses_a_context_shorter_than_the_horizon():
    with pytest.raises(ValueError, match="context"):
        seasonal_naive_forecast(torch.zeros(1, 10, FEATURES), PRED)
