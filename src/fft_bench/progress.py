"""Progress reporting for benchmark runs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Literal

from .config import SingleBenchmarkConfig
from .results import BenchmarkResult

ProgressMode = Literal["auto", "bar", "plain", "silent"]

_VALID_MODES: set[str] = {"auto", "bar", "plain", "silent"}


def resolve_progress_mode(progress: bool | str) -> ProgressMode:
    """Resolve a user-supplied progress value to a concrete mode string.

    Parameters
    ----------
    progress : bool | str
        ``True`` maps to ``"auto"``, ``False`` maps to ``"silent"``,
        or one of ``"auto"``, ``"bar"``, ``"plain"``, ``"silent"``.

    Returns
    -------
    ProgressMode
        The resolved mode.

    Raises
    ------
    ValueError
        If *progress* is a string not in the valid set.
    """
    if progress is True:
        mode = "auto"
    elif progress is False:
        mode = "silent"
    elif isinstance(progress, str):
        mode = progress.lower()
    else:
        raise ValueError(
            f"Invalid progress value: {progress!r}. "
            f"Expected bool or one of {sorted(_VALID_MODES)}."
        )

    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid progress mode: {mode!r}. "
            f"Must be one of {sorted(_VALID_MODES)}."
        )

    if mode == "auto":
        env = os.environ.get("FFT_BENCH_PROGRESS", "").lower()
        if env and env in _VALID_MODES and env != "auto":
            mode = env

    return mode  # type: ignore[return-value]


def _format_shape(shape: tuple[int, ...]) -> str:
    """Format a shape tuple compactly (e.g. ``(512, 512)`` -> ``"512x512"``)."""
    return "x".join(str(s) for s in shape)


class _TqdmReporter:
    """Progress reporter using tqdm bars."""

    def __init__(self, total: int) -> None:
        from tqdm.auto import tqdm

        self._bar = tqdm(total=total, unit="cfg")

    def on_start(self, index: int, config: SingleBenchmarkConfig) -> None:
        """Update the bar description for the current config."""
        parts = [config.backend, _format_shape(config.shape), config.dtype]
        if config.threads > 1:
            parts.append(f"t={config.threads}")
        self._bar.set_description(" ".join(parts))

    def on_finish(
        self,
        index: int,
        config: SingleBenchmarkConfig,
        result: BenchmarkResult,
    ) -> None:
        """Update postfix and advance the bar."""
        if result.success:
            self._bar.set_postfix_str(f"{result.mean:.4f}s")
        else:
            self._bar.set_postfix_str("FAILED")
        self._bar.update(1)

    def close(self) -> None:
        """Clean up the tqdm bar."""
        self._bar.close()


class _PlainReporter:
    """Print-based reporter matching the original output format."""

    def __init__(self, total: int) -> None:
        self._total = total

    def on_start(self, index: int, config: SingleBenchmarkConfig) -> None:
        """Print the config label (no newline)."""
        label = (
            f"[{index}/{self._total}] {config.backend} "
            f"shape={config.shape} dtype={config.dtype} "
            f"threads={config.threads}"
        )
        print(f"  {label} ...", end="", flush=True)

    def on_finish(
        self,
        index: int,
        config: SingleBenchmarkConfig,
        result: BenchmarkResult,
    ) -> None:
        """Print the result on the same line."""
        if result.success:
            print(f" {result.mean:.6f}s (mean)")
        else:
            print(f" FAILED: {result.error}")

    def close(self) -> None:
        """Nothing to clean up."""


class _SilentReporter:
    """No-op reporter."""

    def __init__(self, total: int) -> None:
        pass

    def on_start(self, index: int, config: SingleBenchmarkConfig) -> None:
        pass

    def on_finish(
        self,
        index: int,
        config: SingleBenchmarkConfig,
        result: BenchmarkResult,
    ) -> None:
        pass

    def close(self) -> None:
        pass


_REPORTERS = {
    "auto": _TqdmReporter,
    "bar": _TqdmReporter,
    "plain": _PlainReporter,
    "silent": _SilentReporter,
}


@contextmanager
def progress_context(
    configs: list[SingleBenchmarkConfig],
    mode: ProgressMode,
):
    """Context manager that yields the appropriate progress reporter.

    Parameters
    ----------
    configs : list[SingleBenchmarkConfig]
        The list of configs (used for ``total`` count).
    mode : ProgressMode
        Resolved progress mode.

    Yields
    ------
    _TqdmReporter | _PlainReporter | _SilentReporter
        The active reporter instance.
    """
    reporter_cls = _REPORTERS[mode]
    reporter = reporter_cls(total=len(configs))
    try:
        yield reporter
    finally:
        reporter.close()
