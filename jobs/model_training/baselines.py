"""Naive forecasts to measure the trained models against.

Champion-versus-challenger answers which model is better. It never answers whether
either is worth running at all, which is the question a stakeholder actually asks - and
without an answer an RMSE of 0.81 is just a number. These two supply it on the same
test block, in the same metric, at the cost of no training.

Both are pure functions of the context window, so neither needs a registry entry, an
artifact, or a forward pass.
"""


def persistence_forecast(x, pred_len):
    """Every future hour equals the last observed hour.

    The classic "no skill" reference. On an hourly weather series it is a weak one -
    it throws the daily cycle away - but it is the floor any forecaster must clear.
    """
    return x[:, -1:, :].expand(-1, pred_len, -1)


def seasonal_naive_forecast(x, pred_len):
    """Every future hour equals the same hour one day earlier.

    Target hour t+k is predicted by t+k-24, and for a 24-hour horizon all of those sit
    inside a 72-hour context. This is the baseline that matters on a strongly diurnal
    series: it reproduces the daily cycle for free, so beating it is the real test.
    """
    if x.shape[1] < pred_len:
        raise ValueError(
            f"seasonal-naive needs at least {pred_len} hours of context, got {x.shape[1]}."
        )
    return x[:, -pred_len:, :]


# Logged alongside every promotion decision. Order is the order they are reported in.
NAIVE_FORECASTS = {
    "persistence": persistence_forecast,
    "seasonal_naive": seasonal_naive_forecast,
}
