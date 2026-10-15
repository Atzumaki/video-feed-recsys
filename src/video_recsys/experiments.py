from __future__ import annotations

import numpy as np
from scipy import stats

from video_recsys.schemas import ABTestResult


def cuped_adjust(metric: np.ndarray, pre_experiment_metric: np.ndarray) -> np.ndarray:
    """Reduce experimental variance using a pre-period covariate."""
    variance = float(np.var(pre_experiment_metric, ddof=1))
    if variance == 0:
        return metric.copy()
    theta = float(np.cov(metric, pre_experiment_metric, ddof=1)[0, 1] / variance)
    return metric - theta * (pre_experiment_metric - np.mean(pre_experiment_metric))


def analyze_ab_test(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    control_pre: np.ndarray | None = None,
    treatment_pre: np.ndarray | None = None,
) -> ABTestResult:
    if min(len(control), len(treatment)) < 2:
        raise ValueError("each experiment group needs at least two observations")
    control_values = np.asarray(control, dtype=float)
    treatment_values = np.asarray(treatment, dtype=float)
    if control_pre is not None and treatment_pre is not None:
        all_metric = np.concatenate([control_values, treatment_values])
        all_pre = np.concatenate([control_pre, treatment_pre]).astype(float)
        adjusted = cuped_adjust(all_metric, all_pre)
        control_values = adjusted[: len(control_values)]
        treatment_values = adjusted[len(control_values) :]
    difference = float(np.mean(treatment_values) - np.mean(control_values))
    standard_error = np.sqrt(
        np.var(control_values, ddof=1) / len(control_values)
        + np.var(treatment_values, ddof=1) / len(treatment_values)
    )
    test = stats.ttest_ind(treatment_values, control_values, equal_var=False)
    control_mean = float(np.mean(control_values))
    return ABTestResult(
        control_mean=round(control_mean, 6),
        treatment_mean=round(float(np.mean(treatment_values)), 6),
        absolute_uplift=round(difference, 6),
        relative_uplift_percent=round(100 * difference / max(abs(control_mean), 1e-12), 3),
        confidence_interval_95=(
            round(difference - 1.96 * standard_error, 6),
            round(difference + 1.96 * standard_error, 6),
        ),
        p_value=round(float(test.pvalue), 8),
        significant=bool(test.pvalue < 0.05),
    )

