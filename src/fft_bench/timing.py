"""Timing utility for benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable


def benchmark_single(
    execute_fn: Callable[[], None],
    warmup: int,
    repetitions: int,
) -> list[float]:
    """Run warmup iterations then collect timing measurements.

    Parameters
    ----------
    execute_fn : Callable[[], None]
        Function to time (a single FFT execution).
    warmup : int
        Number of untimed warmup iterations.
    repetitions : int
        Number of timed repetitions.

    Returns
    -------
    list[float]
        Individual timing measurements in seconds.
    """
    for _ in range(warmup):
        execute_fn()

    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        execute_fn()
        end = time.perf_counter()
        timings.append(end - start)

    return timings
