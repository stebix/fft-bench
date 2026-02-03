"""Plot generation from benchmark results."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fileutil import OverwritePolicy, resolve_output_path
from .results import BenchmarkResult, BenchmarkSuite


def generate_all_plots(
    suite: BenchmarkSuite,
    output_dir: str | Path,
    fmt: str = "png",
    dpi: int = 150,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Generate all plot types from a benchmark suite.

    Parameters
    ----------
    suite : BenchmarkSuite
        Benchmark results to plot.
    output_dir : str | Path
        Directory to save plots to.
    fmt : str
        Image format ('png', 'pdf', 'svg').
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle already-existing plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in suite.results if r.success]
    if not successful:
        print("No successful results to plot.")
        return

    plot_runtime_vs_size(successful, output_dir, fmt, dpi, overwrite_policy)
    plot_backend_comparison(successful, output_dir, fmt, dpi, overwrite_policy)
    plot_threading_scaling(successful, output_dir, fmt, dpi, overwrite_policy)
    plot_dtype_impact(successful, output_dir, fmt, dpi, overwrite_policy)

    print(f"Plots saved to {output_dir}/")


def _save_fig(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    fmt: str,
    dpi: int,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Save a figure and close it.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_dir : Path
        Output directory.
    name : str
        Filename without extension.
    fmt : str
        Image format.
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle an already-existing file.
    """
    path = output_dir / f"{name}.{fmt}"
    path = resolve_output_path(path, overwrite_policy)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_runtime_vs_size(
    results: list[BenchmarkResult],
    output_dir: Path,
    fmt: str,
    dpi: int,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Plot runtime vs FFT size, per ndim.

    Lines are colored by backend and styled by dtype. Uses log-log scale
    with error bars showing standard deviation.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Successful benchmark results.
    output_dir : Path
        Output directory.
    fmt : str
        Image format.
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle already-existing plot files.
    """
    # Group by ndim
    by_ndim: dict[int, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.config.threads == 1:
            by_ndim[r.config.ndim].append(r)

    dtype_styles = {"float32": "-", "float64": "--", "complex64": ":", "complex128": "-."}
    backend_colors = _get_backend_colors(results)

    for ndim, ndim_results in sorted(by_ndim.items()):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by (backend, dtype)
        groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
        for r in ndim_results:
            groups[(r.config.backend, r.config.dtype)].append(r)

        for (backend, dtype), group in sorted(groups.items()):
            group.sort(key=lambda r: r.config.size)
            sizes = [r.config.size for r in group]
            means = [r.mean for r in group]
            stds = [r.std for r in group]

            style = dtype_styles.get(dtype, "-")
            color = backend_colors.get(backend, None)
            ax.errorbar(
                sizes, means, yerr=stds,
                label=f"{backend} ({dtype})",
                linestyle=style,
                color=color,
                marker="o",
                markersize=4,
                capsize=3,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FFT Size")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Runtime vs Size ({ndim}D FFT, threads=1)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        _save_fig(fig, output_dir, f"runtime_vs_size_{ndim}d", fmt, dpi, overwrite_policy)


def plot_backend_comparison(
    results: list[BenchmarkResult],
    output_dir: Path,
    fmt: str,
    dpi: int,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Plot grouped bar chart comparing backends.

    One plot per (ndim, dtype) combination. Groups are sizes, bars are backends.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Successful benchmark results.
    output_dir : Path
        Output directory.
    fmt : str
        Image format.
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle already-existing plot files.
    """
    # Group by (ndim, dtype)
    groups: dict[tuple[int, str], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.config.threads == 1:
            groups[(r.config.ndim, r.config.dtype)].append(r)

    backend_colors = _get_backend_colors(results)

    for (ndim, dtype), group in sorted(groups.items()):
        # Get unique sizes and backends
        sizes = sorted({r.config.size for r in group})
        backend_names = sorted({r.config.backend for r in group})

        # Build lookup: (backend, size) -> result
        lookup: dict[tuple[str, int], BenchmarkResult] = {}
        for r in group:
            lookup[(r.config.backend, r.config.size)] = r

        fig, ax = plt.subplots(figsize=(10, 6))
        n_backends = len(backend_names)
        bar_width = 0.8 / max(n_backends, 1)
        x = np.arange(len(sizes))

        for j, backend in enumerate(backend_names):
            means = []
            stds = []
            for size in sizes:
                r = lookup.get((backend, size))
                means.append(r.mean if r else 0)
                stds.append(r.std if r else 0)

            offset = (j - (n_backends - 1) / 2) * bar_width
            color = backend_colors.get(backend, None)
            ax.bar(
                x + offset, means, bar_width,
                yerr=stds,
                label=backend,
                color=color,
                capsize=3,
            )

        ax.set_xlabel("FFT Size")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Backend Comparison ({ndim}D, {dtype}, threads=1)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        _save_fig(
            fig, output_dir,
            f"backend_comparison_{ndim}d_{dtype}", fmt, dpi,
            overwrite_policy,
        )


def plot_threading_scaling(
    results: list[BenchmarkResult],
    output_dir: Path,
    fmt: str,
    dpi: int,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Plot threading scaling for backends that support it.

    One plot per (backend, ndim, dtype). X=threads, Y=runtime, lines per size.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Successful benchmark results.
    output_dir : Path
        Output directory.
    fmt : str
        Image format.
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle already-existing plot files.
    """
    # Only include results where threads > 1 exists for that backend
    thread_counts: dict[str, set[int]] = defaultdict(set)
    for r in results:
        thread_counts[r.config.backend].add(r.config.threads)

    threaded_backends = {
        b for b, tc in thread_counts.items() if len(tc) > 1
    }
    if not threaded_backends:
        return

    # Group by (backend, ndim, dtype)
    groups: dict[tuple[str, int, str], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.config.backend in threaded_backends:
            groups[(r.config.backend, r.config.ndim, r.config.dtype)].append(r)

    for (backend, ndim, dtype), group in sorted(groups.items()):
        sizes = sorted({r.config.size for r in group})
        threads_list = sorted({r.config.threads for r in group})

        if len(threads_list) < 2:
            continue

        # Build lookup: (size, threads) -> result
        lookup: dict[tuple[int, int], BenchmarkResult] = {}
        for r in group:
            lookup[(r.config.size, r.config.threads)] = r

        fig, ax = plt.subplots(figsize=(10, 6))

        for size in sizes:
            means = []
            valid_threads = []
            for t in threads_list:
                r = lookup.get((size, t))
                if r:
                    means.append(r.mean)
                    valid_threads.append(t)

            if valid_threads:
                ax.plot(
                    valid_threads, means,
                    marker="o", label=f"size={size}",
                )

        ax.set_xlabel("Threads")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Threading Scaling ({backend}, {ndim}D, {dtype})")
        ax.set_xticks(threads_list)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        _save_fig(
            fig, output_dir,
            f"threading_{backend}_{ndim}d_{dtype}", fmt, dpi,
            overwrite_policy,
        )


def plot_dtype_impact(
    results: list[BenchmarkResult],
    output_dir: Path,
    fmt: str,
    dpi: int,
    overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> None:
    """Plot dtype impact on runtime.

    One plot per (backend, ndim). X=size, Y=runtime, lines per dtype.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Successful benchmark results.
    output_dir : Path
        Output directory.
    fmt : str
        Image format.
    dpi : int
        DPI for raster formats.
    overwrite_policy : OverwritePolicy
        How to handle already-existing plot files.
    """
    # Group by (backend, ndim)
    groups: dict[tuple[str, int], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.config.threads == 1:
            groups[(r.config.backend, r.config.ndim)].append(r)

    for (backend, ndim), group in sorted(groups.items()):
        dtypes = sorted({r.config.dtype for r in group})
        if len(dtypes) < 2:
            continue

        # Build lookup: (dtype, size) -> result
        lookup: dict[tuple[str, int], BenchmarkResult] = {}
        for r in group:
            lookup[(r.config.dtype, r.config.size)] = r

        sizes = sorted({r.config.size for r in group})

        fig, ax = plt.subplots(figsize=(10, 6))

        for dtype in dtypes:
            means = []
            valid_sizes = []
            for size in sizes:
                r = lookup.get((dtype, size))
                if r:
                    means.append(r.mean)
                    valid_sizes.append(size)

            if valid_sizes:
                ax.plot(
                    valid_sizes, means,
                    marker="o", label=dtype,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FFT Size")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Dtype Impact ({backend}, {ndim}D, threads=1)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        _save_fig(
            fig, output_dir,
            f"dtype_impact_{backend}_{ndim}d", fmt, dpi,
            overwrite_policy,
        )


def _get_backend_colors(
    results: list[BenchmarkResult],
) -> dict[str, str]:
    """Assign consistent colors to backends.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Results to extract backend names from.

    Returns
    -------
    dict[str, str]
        Mapping of backend name to color.
    """
    backend_names = sorted({r.config.backend for r in results})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]
    return {
        name: colors[i % len(colors)]
        for i, name in enumerate(backend_names)
    }
