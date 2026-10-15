import numpy as np

from video_recsys.experiments import analyze_ab_test, cuped_adjust


def test_cuped_reduces_variance() -> None:
    random = np.random.default_rng(42)
    pre = random.normal(size=2000)
    metric = 0.8 * pre + random.normal(scale=0.3, size=2000)
    assert np.var(cuped_adjust(metric, pre)) < np.var(metric)


def test_ab_test_detects_clear_uplift() -> None:
    random = np.random.default_rng(7)
    control = random.normal(10, 1, size=1500)
    treatment = random.normal(10.25, 1, size=1500)
    result = analyze_ab_test(control, treatment)
    assert result.significant
    assert result.absolute_uplift > 0
    assert result.confidence_interval_95[0] > 0

