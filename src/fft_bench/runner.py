"""Benchmark orchestration loop."""

from __future__ import annotations

import gc
import logging
import sys

from . import backends
from .config import SingleBenchmarkConfig
from .hardware import capture_hardware_info
from .progress import progress_context, resolve_progress_mode
from .results import BenchmarkResult, BenchmarkSuite
from .timing import benchmark_single

logger = logging.getLogger(__name__)


def _try_gpu_cleanup() -> None:
    """Best-effort GPU memory cleanup.

    Frees CuPy memory pools and plan caches. Only runs when CuPy is
    already imported. Safe to call at any time -- never raises.
    """
    if "cupy" not in sys.modules:
        return
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass


def run_benchmarks(
    configs: list[SingleBenchmarkConfig],
    progress: bool | str = "auto",
) -> BenchmarkSuite:
    """Run all benchmark configurations and collect results.

    Each configuration is run independently. Failures are recorded as
    unsuccessful results rather than aborting the entire suite.

    Parameters
    ----------
    configs : list[SingleBenchmarkConfig]
        List of benchmark configurations to run.
    progress : bool | str
        Progress display mode. ``True`` maps to ``"auto"``, ``False`` to
        ``"silent"``. String values: ``"auto"``, ``"bar"``, ``"plain"``,
        ``"silent"``.

    Returns
    -------
    BenchmarkSuite
        Complete suite with all results and hardware info.
    """
    mode = resolve_progress_mode(progress)
    hardware = capture_hardware_info()
    results: list[BenchmarkResult] = []

    with progress_context(configs, mode) as reporter:
        for i, config in enumerate(configs, 1):
            reporter.on_start(i, config)
            result = _run_single(config)
            results.append(result)
            reporter.on_finish(i, config, result)
            gc.collect()
            _try_gpu_cleanup()

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

        if hasattr(backend, "run_timed"):
            timings = backend.run_timed(
                shape=config.shape,
                dtype=config.dtype,
                threads=config.threads,
                warmup=config.warmup,
                repetitions=config.repetitions,
            )
        else:
            try:
                backend.setup(
                    shape=config.shape,
                    dtype=config.dtype,
                    threads=config.threads,
                )
                timings = benchmark_single(
                    execute_fn=backend.execute,
                    warmup=config.warmup,
                    repetitions=config.repetitions,
                )
            finally:
                try:
                    backend.teardown()
                except Exception:
                    logger.debug("Backend teardown failed", exc_info=True)

        return BenchmarkResult(
            config=config,
            timings=timings,
            success=True,
        )

    except MemoryError:
        gc.collect()
        _try_gpu_cleanup()
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
        gc.collect()
        _try_gpu_cleanup()
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
