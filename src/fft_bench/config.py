"""Benchmark configuration and parameter grid expansion."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.base import BackendCapabilities

logger = logging.getLogger(__name__)

# Mapping from user-facing dtype strings to numpy dtype strings
DTYPE_MAP: dict[str, str] = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
}


@dataclass(frozen=True)
class SingleBenchmarkConfig:
    """Configuration for a single benchmark run.

    Parameters
    ----------
    backend : str
        Name of the FFT backend to use.
    shape : tuple[int, ...]
        Shape of the input array.
    dtype : str
        Data type string (e.g. 'float64', 'complex128').
    threads : int
        Number of threads to use.
    warmup : int
        Number of warmup iterations before timing.
    repetitions : int
        Number of timed repetitions.
    """

    backend: str
    shape: tuple[int, ...]
    dtype: str
    threads: int
    warmup: int
    repetitions: int

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Size along each dimension (assumes cubic/square shape)."""
        return self.shape[0]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "backend": self.backend,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "threads": self.threads,
            "warmup": self.warmup,
            "repetitions": self.repetitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SingleBenchmarkConfig:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with config fields.

        Returns
        -------
        SingleBenchmarkConfig
        """
        return cls(
            backend=data["backend"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            threads=data["threads"],
            warmup=data["warmup"],
            repetitions=data["repetitions"],
        )


def expand_parameter_grid(
    backends: dict[str, BackendCapabilities],
    sizes: list[int],
    ndims: list[int],
    dtypes: list[str],
    threads: list[int],
    warmup: int,
    repetitions: int,
) -> list[SingleBenchmarkConfig]:
    """Expand a parameter grid into a list of benchmark configurations.

    Produces the Cartesian product of all parameter axes, filtered against
    each backend's capabilities. Incompatible combinations are skipped with
    a logged warning.

    Parameters
    ----------
    backends : dict[str, BackendCapabilities]
        Mapping of backend name to its capabilities.
    sizes : list[int]
        FFT sizes along each dimension.
    ndims : list[int]
        Number of dimensions to benchmark.
    dtypes : list[str]
        Data type strings to benchmark.
    threads : list[int]
        Thread counts to benchmark.
    warmup : int
        Number of warmup iterations.
    repetitions : int
        Number of timed repetitions.

    Returns
    -------
    list[SingleBenchmarkConfig]
        List of valid benchmark configurations.
    """
    configs: list[SingleBenchmarkConfig] = []

    for backend_name, caps in backends.items():
        for size, ndim, dtype, nthreads in itertools.product(
            sizes, ndims, dtypes, threads
        ):
            # Check ndim support
            if ndim > caps.max_ndim:
                logger.warning(
                    "Skipping %s with ndim=%d (max supported: %d)",
                    backend_name, ndim, caps.max_ndim,
                )
                continue

            # Check dtype support
            if dtype not in caps.supported_dtypes:
                logger.warning(
                    "Skipping %s with dtype=%s (not supported)",
                    backend_name, dtype,
                )
                continue

            # Check threading support
            actual_threads = nthreads
            if nthreads > 1 and not caps.supports_threading:
                if nthreads == threads[0]:
                    # Only warn once per backend, not for every combo
                    logger.warning(
                        "Backend %s does not support threading; "
                        "using threads=1 instead",
                        backend_name,
                    )
                actual_threads = 1
                # Skip if we already have a threads=1 config for this combo
                if 1 in threads:
                    continue

            shape = (size,) * ndim
            config = SingleBenchmarkConfig(
                backend=backend_name,
                shape=shape,
                dtype=dtype,
                threads=actual_threads,
                warmup=warmup,
                repetitions=repetitions,
            )
            if config not in configs:
                configs.append(config)

    return configs
