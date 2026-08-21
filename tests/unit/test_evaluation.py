"""Scoring the promotion benchmark. Runs inside dag-pytorch-model-training
(PYTHONPATH=/app)."""
import torch
import torch.nn as nn

from evaluate_and_promote import score_predictors

PRED = 24


class CountingLoader:
    """A DataLoader stand-in that records how often it was iterated.

    The test block is ~113k windows, so how many times it is walked is not an
    implementation detail - it is the cost of the whole evaluation task.
    """

    def __init__(self, batches):
        self._batches = batches
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _loader(n_batches=3, batch=4, seq=72, feats=4, seed=0):
    """Batches whose target is the last PRED hours of their own context.

    That makes seasonal-naive the exactly-correct forecast, so a predictor's error
    is known in closed form instead of having to be recomputed by the test.
    """
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch, seq, feats, generator=generator)
        batches.append((x, x[:, -PRED:, :].clone()))
    return CountingLoader(batches)


def _perfect(x):
    return x[:, -PRED:, :]


def _off_by_one(x):
    return x[:, -PRED:, :] + 1.0


def test_each_predictor_is_scored_independently():
    """One shared loop must not let one predictor's errors leak into another's.

    Scored alone these are trivial; scored together they are the whole risk of
    walking the loader once - a single accumulator, or a window_mse list appended
    to by every predictor, produces numbers that look plausible and are wrong.
    """
    scores = score_predictors(
        {"perfect": _perfect, "off_by_one": _off_by_one}, _loader(), nn.SmoothL1Loss()
    )

    assert scores["perfect"].rmse == 0.0
    assert scores["perfect"].mae == 0.0
    assert scores["perfect"].loss == 0.0

    assert scores["off_by_one"].rmse == 1.0
    assert scores["off_by_one"].mae == 1.0


def test_the_loader_is_walked_once_for_every_predictor():
    """The reason this function exists.

    Champion, challenger and two naive baselines used to be four separate passes
    over the same 113k windows, inside a task bounded at 15 minutes.
    """
    loader = _loader()

    score_predictors(
        {"a": _perfect, "b": _off_by_one, "c": _perfect, "d": _off_by_one},
        loader, nn.SmoothL1Loss(),
    )

    assert loader.passes == 1


def test_window_errors_are_one_per_window_and_stay_aligned():
    """promotion_verdict pairs the two arrays positionally, so order and length
    have to match across predictors or the paired test compares unrelated hours."""
    scores = score_predictors(
        {"perfect": _perfect, "off_by_one": _off_by_one}, _loader(), nn.SmoothL1Loss()
    )

    assert scores["perfect"].window_mse.shape == (12,)   # 3 batches x 4 windows
    assert scores["off_by_one"].window_mse.shape == (12,)
    assert (scores["perfect"].window_mse == 0.0).all()
    assert (scores["off_by_one"].window_mse == 1.0).all()


def test_an_empty_predictor_set_still_consumes_nothing():
    """Defensive: NAIVE_FORECASTS is configurable, and an empty mapping must not
    walk 113k windows to produce nothing."""
    loader = _loader()

    assert score_predictors({}, loader, nn.SmoothL1Loss()) == {}
    assert loader.passes == 0
