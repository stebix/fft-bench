"""Native cuFFT backends via the cufft-bench binary."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

from . import register
from .base import BackendCapabilities, BackendKind

logger = logging.getLogger(__name__)


def _find_binary() -> str | None:
    """Locate the cufft-bench binary.

    Checks ``CUFFT_BENCH_PATH`` environment variable first, then falls back
    to ``shutil.which("cufft-bench")``.

    Returns
    -------
    str or None
        Absolute path to the binary, or ``None`` if not found.
    """
    env_path = os.environ.get("CUFFT_BENCH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    return shutil.which("cufft-bench")


class _CuFFTNativeBase:
    """Base class for native cuFFT backends.

    Parameters
    ----------
    mode : str
        Timing mode passed to the binary (``"kernel"`` or ``"e2e"``).
    binary_path : str
        Path to the ``cufft-bench`` executable.
    """

    _mode: str
    _binary_path: str
    _backend_name: str

    def __init__(self, mode: str, binary_path: str, backend_name: str) -> None:
        self._mode = mode
        self._binary_path = binary_path
        self._backend_name = backend_name

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._backend_name

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return BackendCapabilities(
            supported_dtypes=frozenset(
                {"float32", "float64", "complex64", "complex128"}
            ),
            supports_threading=False,
            max_ndim=3,
            kind=BackendKind.NATIVE,
            device="cuda",
        )

    def run_timed(
        self,
        shape: tuple[int, ...],
        dtype: str,
        threads: int,
        warmup: int,
        repetitions: int,
    ) -> list[float]:
        """Run the native binary and return per-iteration timings in seconds.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string (e.g. ``"float32"``).
        threads : int
            Ignored for native CUDA backends.
        warmup : int
            Number of warmup iterations.
        repetitions : int
            Number of timed iterations.

        Returns
        -------
        list[float]
            Per-iteration timings in seconds.

        Raises
        ------
        RuntimeError
            If the binary exits with a non-zero return code.
        """
        cmd = self._build_command(shape, dtype, warmup, repetitions)
        logger.debug("Running native cuFFT: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"cufft-bench failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )

        data = json.loads(result.stdout)
        timings_ms = data["timings_ms"]
        return [t / 1000.0 for t in timings_ms]

    def _build_command(
        self,
        shape: tuple[int, ...],
        dtype: str,
        warmup: int,
        repetitions: int,
    ) -> list[str]:
        """Build the CLI command for cufft-bench.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string.
        warmup : int
            Number of warmup iterations.
        repetitions : int
            Number of timed iterations.

        Returns
        -------
        list[str]
            Command and arguments.
        """
        ndim = len(shape)
        cmd = [
            self._binary_path,
            "--dtype", dtype,
            "--dim", str(ndim),
            "--mode", self._mode,
            "--warmup", str(warmup),
            "--iters", str(repetitions),
            "--format", "json",
        ]

        if ndim == 1:
            cmd.extend(["--nx", str(shape[0])])
        elif ndim == 2:
            cmd.extend(["--nx", str(shape[0]), "--ny", str(shape[1])])
        else:
            cmd.extend([
                "--nx", str(shape[0]),
                "--ny", str(shape[1]),
                "--nz", str(shape[2]),
            ])

        return cmd

    def setup(self, shape: tuple[int, ...], dtype: str, threads: int) -> None:
        """Not used for native backends."""
        raise NotImplementedError("Native backends use run_timed()")

    def execute(self) -> None:
        """Not used for native backends."""
        raise NotImplementedError("Native backends use run_timed()")

    def teardown(self) -> None:
        """Not used for native backends."""
        raise NotImplementedError("Native backends use run_timed()")


class CuFFTKernelBackend(_CuFFTNativeBase):
    """Native cuFFT backend with kernel-only timing.

    Times only the ``cufftExec*`` call, excluding memory transfers.
    """

    def __init__(self, *, binary_path: str) -> None:
        super().__init__(mode="kernel", binary_path=binary_path, backend_name="cufft")


class CuFFTE2EBackend(_CuFFTNativeBase):
    """Native cuFFT backend with end-to-end timing.

    Times the full cycle including memory allocation, host-to-device copy,
    FFT execution, device-to-host copy, and deallocation.
    """

    def __init__(self, *, binary_path: str) -> None:
        super().__init__(mode="e2e", binary_path=binary_path, backend_name="cufft-e2e")


# Conditional registration
_bin = _find_binary()
if _bin:
    # Use lambda factories to capture the binary path
    _kernel_cls = type(
        "CuFFTKernelBackend", (CuFFTKernelBackend,),
        {"__init__": lambda self: CuFFTKernelBackend.__init__(self, binary_path=_bin)},
    )
    _e2e_cls = type(
        "CuFFTE2EBackend", (CuFFTE2EBackend,),
        {"__init__": lambda self: CuFFTE2EBackend.__init__(self, binary_path=_bin)},
    )
    register("cufft", _kernel_cls)
    register("cufft-e2e", _e2e_cls)
    logger.debug("Registered native cuFFT backends (binary: %s)", _bin)
else:
    logger.debug("cufft-bench binary not found; native cuFFT backends not available")
