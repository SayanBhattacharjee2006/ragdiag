"""Latency statistical calculations and summarization."""

import math
import statistics
from collections.abc import Sequence

from ragdiag.metrics.models import LatencySummary


def calculate_percentile(values: Sequence[float], percentile: float) -> float:
    """Calculate the p-th percentile of a dataset using linear interpolation.

    Uses standard linear interpolation between the two nearest ranks (NIST Method 7 /
    NumPy default). Given sorted values x_0, ..., x_{n-1} and percentile p in [0, 100]:
        index = (n - 1) * (p / 100)
        fraction = index - floor(index)
        value = x[floor(index)] + fraction * (x[ceil(index)] - x[floor(index)])

    Args:
        values: Sequence of numeric values (must not be empty).
        percentile: Percentile rank to compute, in the range [0.0, 100.0].

    Returns:
        The computed percentile value as a float.

    Raises:
        ValueError: If values is empty or percentile is outside [0.0, 100.0].
    """
    if not values:
        raise ValueError("Cannot calculate percentile of an empty sequence.")

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be between 0.0 and 100.0, got {percentile}.")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    idx = (n - 1) * (percentile / 100.0)
    lower_idx = int(math.floor(idx))
    upper_idx = int(math.ceil(idx))

    if lower_idx == upper_idx:
        return float(sorted_vals[lower_idx])

    fraction = idx - lower_idx
    return float(
        sorted_vals[lower_idx] + fraction * (sorted_vals[upper_idx] - sorted_vals[lower_idx])
    )


def calculate_latency_summary(latencies_ms: Sequence[float]) -> LatencySummary:
    """Calculate summary statistics (mean, p50, p95, p99, min, max) for latency measurements.

    Args:
        latencies_ms: Sequence of latency values in milliseconds.

    Returns:
        A validated `LatencySummary` instance. If `latencies_ms` is empty,
        returns a summary with count=0 and all statistics set to 0.0.
    """
    if not latencies_ms:
        return LatencySummary(
            count=0,
            mean_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
        )

    vals = [float(v) for v in latencies_ms]
    return LatencySummary(
        count=len(vals),
        mean_ms=statistics.mean(vals),
        p50_ms=calculate_percentile(vals, 50.0),
        p95_ms=calculate_percentile(vals, 95.0),
        p99_ms=calculate_percentile(vals, 99.0),
        min_ms=min(vals),
        max_ms=max(vals),
    )
