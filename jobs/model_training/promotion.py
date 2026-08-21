"""When a challenger replaces the champion.

The original gate demanded a 1% RMSE improvement. That was unreachable in practice: a
warm-started incremental run improves by ~0.2%, so once warm-starting began working the
gate would have rejected every challenger it ever saw and @champion would have frozen
at whatever the last from-scratch run produced.

A fixed percentage is the wrong instrument anyway. It asks "is the gap big?" when the
question is "is the gap real?" - and answers both wrongly, waving through a large gap
that is pure noise and blocking a small one that is perfectly consistent. This asks the
two questions separately: the improvement must clear a floor worth deploying for, and
it must survive a paired test on the same windows.

The subsampling is not optional. Consecutive test windows share all but one of their 96
hours, so the ~113k windows carry nowhere near 113k independent observations; feeding
them all to a t-test would report overwhelming significance for anything at all. One
sample per horizon leaves windows that share no timestep.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy.stats import ttest_rel

# Below this the difference is not worth a deployment, however reliable it is.
MIN_RELATIVE_IMPROVEMENT = 0.001   # 0.1% of the champion's RMSE
SIGNIFICANCE_LEVEL = 0.05


class Verdict(NamedTuple):
    promote: bool
    reason: str
    relative_improvement: float
    p_value: float
    samples: int


def promotion_verdict(champion_window_mse, challenger_window_mse, *, stride,
                      alpha=SIGNIFICANCE_LEVEL,
                      min_improvement=MIN_RELATIVE_IMPROVEMENT) -> Verdict:
    """Whether the challenger has earned the @champion alias.

    Both arrays hold one mean-squared-error per test window, in the same order, so the
    comparison is paired: every difference is two models scored on identical data.
    """
    champion = np.asarray(champion_window_mse, dtype=np.float64)
    challenger = np.asarray(challenger_window_mse, dtype=np.float64)
    if champion.shape != challenger.shape:
        raise ValueError(
            f"paired comparison needs matching windows, got {champion.shape} and {challenger.shape}."
        )

    champion_rmse = math.sqrt(champion.mean())
    challenger_rmse = math.sqrt(challenger.mean())
    improvement = 1.0 - (challenger_rmse / champion_rmse) if champion_rmse else 0.0

    disjoint = slice(None, None, stride)
    a, b = champion[disjoint], challenger[disjoint]
    samples = a.size

    if improvement < min_improvement:
        return Verdict(False, f"improvement {improvement:+.3%} is below the {min_improvement:.1%} floor",
                       improvement, float("nan"), samples)

    # One-sided: the champion's error must be the larger one.
    result = ttest_rel(a, b, alternative="greater")
    p_value = float(result.pvalue)

    if not math.isfinite(p_value) or p_value >= alpha:
        return Verdict(False, f"improvement {improvement:+.3%} is not statistically reliable "
                              f"(p={p_value:.4f} over {samples} disjoint windows)",
                       improvement, p_value, samples)

    return Verdict(True, f"improvement {improvement:+.3%} is reliable "
                         f"(p={p_value:.4f} over {samples} disjoint windows)",
                   improvement, p_value, samples)
