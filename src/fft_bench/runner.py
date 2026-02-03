"""Benchmark orchestration loop."""

from __future__ import annotations

import logging

from . import backends
from .config import SingleBenchmarkConfig
from .hardware import capture_hardware_info
from .results import BenchmarkResult, BenchmarkSuite
from .timing import benchmark_single

logger = logging.getLogger(__name__)


def run_benchmarks(
    configs: list[SingleBenchmarkConfig],
    progress: bool = True,
) -> BenchmarkSuite:
    """Run all benchmark configurations and collect results.

    Each configuration is run independently. Failures are recorded as
    unsuccessful results rather than aborting the entire suite.

    Parameters
    ----------
    configs : list[SingleBenchmarkConfig]
        List of benchmark configurations to run.
    progress : bool
        Whether to print progress information.

    Returns
    -------
    BenchmarkSuite
        Complete suite with all results and hardware info.
    """
    hardware = capture_hardware_info()
    results: list[BenchmarkResult] = []
    total = len(configs)

    for i, config in enumerate(configs, 1):
        label = (
            f"[{i}/{total}] {config.backend} "
            f"shape={config.shape} dtype={config.dtype} "
            f"threads={config.threads}"
        )
        if progress:
            print(f"  {label} ...", end="", flush=True)

        result = _run_single(config)
        results.append(result)

        if progress:
            if result.success:
                print(f" {result.mean:.6f}s (mean)")
            else:
                print(f" FAILED: {result.error}")

    return BenchmarkSuite(results=results, hardware=hardware)


def _run_single(config: SingleBenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Parameters
    ----------
    config : SingleBenchmarkConfig
        Configuration to run.

    Returns
    -------
    BenchmarkResult
    """
    try:
        backend_cls = backends.get(config.backend)
        backend = backend_cls()

        backend.setup(
            shape=config.shape,
            dtype=config.dtype,
            threads=config.threads,
        )

        try:
            timings = benchmark_single(
                execute_fn=backend.execute,
                warmup=config.warmup,
                repetitions=config.repetitions,
            )
        finally:
            backend.teardown()

        return BenchmarkResult(
            config=config,
            timings=timings,
            success=True,
        )

    except MemoryError:
        logger.error(
            "MemoryError for %s shape=%s dtype=%s",
            config.backend, config.shape, config.dtype,
        )
        return BenchmarkResult(
            config=config,
            timings=[],
            success=False,
            error="MemoryError: insufficient memory for this configuration",
        )
    except Exception as e:
        logger.error(
            "Error for %s shape=%s dtype=%s: %s",
            config.backend, config.shape, config.dtype, e,
        )
        return BenchmarkResult(
            config=config,
            timings=[],
            success=False,
            error=f"{type(e).__name__}: {e}",
        )
