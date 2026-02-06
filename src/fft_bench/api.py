"""Programmatic API for running and loading benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import expand_parameter_grid
from .results import BenchmarkSuite
from .runner import run_benchmarks


def run(
    backends: Sequence[str] | None = None,
    sizes: Sequence[int] | None = None,
    ndims: Sequence[int] | None = None,
    dtypes: Sequence[str] | None = None,
    threads: Sequence[int] | None = None,
    warmup: int = 3,
    repetitions: int = 10,
    progress: bool | str = "auto",
) -> BenchmarkSuite:
    """Run FFT benchmarks and return results as a :class:`BenchmarkSuite`.

    This mirrors the ``fft-bench run`` CLI but returns the suite directly
    instead of writing it to disk.

    Parameters
    ----------
    backends : Sequence[str] | None
        Backend names to benchmark. *None* means all available backends.
    sizes : Sequence[int] | None
        FFT sizes along each dimension. Defaults to
        ``[64, 128, 256, 512, 1024]``.
    ndims : Sequence[int] | None
        Number of dimensions to benchmark. Defaults to ``[1]``.
    dtypes : Sequence[str] | None
        Data type strings. Defaults to ``["float64", "complex128"]``.
    threads : Sequence[int] | None
        Thread counts. Defaults to ``[1]``.
    warmup : int
        Number of warmup iterations before timing.
    repetitions : int
        Number of timed repetitions.
    progress : bool | str
        Progress display mode. ``True`` maps to ``"auto"``, ``False`` to
        ``"silent"``. String values: ``"auto"``, ``"bar"``, ``"plain"``,
        ``"silent"``.

    Returns
    -------
    BenchmarkSuite
        The completed benchmark suite.

    Raises
    ------
    ValueError
        If unknown backends are requested or no valid configurations
        remain after filtering.
    """
    # Lazy import to avoid shadowing the module-level `backends` import
    from . import backends as _backends_mod

    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]
    if ndims is None:
        ndims = [1]
    if dtypes is None:
        dtypes = ["float64", "complex128"]
    if threads is None:
        threads = [1]

    discovered = _backends_mod.discover_backends()
    backend_names = backends if backends is not None else discovered

    missing = set(backend_names) - set(discovered)
    if missing:
        raise ValueError(
            f"Unknown backends: {', '.join(sorted(missing))}. "
            f"Available: {', '.join(sorted(discovered))}"
        )

    caps = {
        name: _backends_mod.get_capabilities(name) for name in backend_names
    }

    configs = expand_parameter_grid(
        backends=caps,
        sizes=sizes,
        ndims=ndims,
        dtypes=dtypes,
        threads=threads,
        warmup=warmup,
        repetitions=repetitions,
    )

    if not configs:
        raise ValueError(
            "No valid configurations after filtering. Nothing to run."
        )

    return run_benchmarks(configs, progress=progress)


def load(path: str | Path) -> BenchmarkSuite:
    """Load a benchmark suite from a JSON file.

    Convenience wrapper around :meth:`BenchmarkSuite.load`.

    Parameters
    ----------
    path : str | Path
        Path to the JSON results file.

    Returns
    -------
    BenchmarkSuite
    """
    return BenchmarkSuite.load(path)
