"""Backend protocol and supporting types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class BackendKind(Enum):
    """Kind of backend implementation.

    Attributes
    ----------
    PYTHON : str
        Pure Python backend (timing via perf_counter).
    NATIVE : str
        Native backend (may use subprocess, timing may be internal).
    CUDA : str
        CUDA-accelerated backend (requires GPU, execute must synchronize).
    """

    PYTHON = "python"
    NATIVE = "native"
    CUDA = "cuda"


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes what a backend supports.

    Parameters
    ----------
    supported_dtypes : frozenset[str]
        Set of supported dtype strings.
    supports_threading : bool
        Whether the backend supports multi-threaded execution.
    max_ndim : int
        Maximum number of dimensions supported.
    kind : BackendKind
        Whether this is a Python, native, or CUDA backend.
    device : str
        Target device for computation ('cpu' or 'cuda').
    """

    supported_dtypes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"float32", "float64", "complex64", "complex128"}
        )
    )
    supports_threading: bool = False
    max_ndim: int = 3
    kind: BackendKind = BackendKind.PYTHON
    device: str = "cpu"


@runtime_checkable
class Backend(Protocol):
    """Protocol for FFT benchmark backends.

    Backends follow a three-phase lifecycle:
    1. ``setup`` -- prepare input data, create plans, compile code
    2. ``execute`` -- run a single FFT (this is what gets timed)
    3. ``teardown`` -- release resources
    """

    @property
    def name(self) -> str:
        """Human-readable name of this backend."""
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities of this backend."""
        ...

    def setup(self, shape: tuple[int, ...], dtype: str, threads: int) -> None:
        """Prepare the backend for benchmarking.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string (e.g. 'float64').
        threads : int
            Number of threads to use.
        """
        ...

    def execute(self) -> None:
        """Run a single FFT. This is what gets timed."""
        ...

    def teardown(self) -> None:
        """Release resources after benchmarking."""
        ...
