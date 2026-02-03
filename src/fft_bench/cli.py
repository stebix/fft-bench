"""Command-line interface for fft-bench."""

from __future__ import annotations

import argparse
import logging
import sys

from . import backends
from .config import expand_parameter_grid
from .results import BenchmarkSuite
from .runner import run_benchmarks


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="fft-bench",
        description="Benchmark FFT backends in Python",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run", help="Run FFT benchmarks"
    )
    run_parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to benchmark (default: all available)",
    )
    run_parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="FFT sizes along each dimension (default: 64 128 256 512 1024)",
    )
    run_parser.add_argument(
        "--ndims",
        nargs="+",
        type=int,
        default=[1],
        help="Number of dimensions (default: 1)",
    )
    run_parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float64", "complex128"],
        help="Data types to benchmark (default: float64 complex128)",
    )
    run_parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1],
        help="Thread counts to test (default: 1)",
    )
    run_parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of timed repetitions (default: 10)",
    )
    run_parser.add_argument(
        "-o", "--output",
        default="results.json",
        help="Output JSON file path (default: results.json)",
    )

    # --- plot subcommand ---
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from benchmark results"
    )
    plot_parser.add_argument(
        "input",
        help="Input JSON results file",
    )
    plot_parser.add_argument(
        "-o", "--output",
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    plot_parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    plot_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    discovered = backends.discover_backends()
    print(f"Available backends: {', '.join(discovered)}")

    backend_names = args.backends or discovered
    missing = set(backend_names) - set(discovered)
    if missing:
        print(f"Error: unknown backends: {', '.join(sorted(missing))}")
        sys.exit(1)

    # Build capabilities map for requested backends
    caps = {name: backends.get_capabilities(name) for name in backend_names}

    configs = expand_parameter_grid(
        backends=caps,
        sizes=args.sizes,
        ndims=args.ndims,
        dtypes=args.dtypes,
        threads=args.threads,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    if not configs:
        print("No valid configurations after filtering. Nothing to run.")
        sys.exit(1)

    print(f"Running {len(configs)} benchmark configurations...")
    suite = run_benchmarks(configs, progress=True)

    suite.save(args.output)
    print(f"\nResults saved to {args.output}")

    # Summary
    successful = sum(1 for r in suite.results if r.success)
    failed = sum(1 for r in suite.results if not r.success)
    print(f"  {successful} successful, {failed} failed")


def cmd_plot(args: argparse.Namespace) -> None:
    """Execute the 'plot' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    from .plotting import generate_all_plots

    suite = BenchmarkSuite.load(args.input)
    print(f"Loaded {len(suite.results)} results from {args.input}")

    generate_all_plots(
        suite=suite,
        output_dir=args.output,
        fmt=args.format,
        dpi=args.dpi,
    )


def main() -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)
