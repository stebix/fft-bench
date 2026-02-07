"""CuPy FFT backend (CUDA-accelerated via cuFFT)."""

from __future__ import annotations

import numpy as np
import cupy as cp

from .base import BackendCapabilities, BackendKind
from . import register


class CuPyBackend:
    """FFT backend using CuPy.

    CuPy dispatches to NVIDIA cuFFT on the GPU. Data is transferred to the
    device during ``setup()`` and each ``execute()`` call includes a device
    synchronization so that timing reflects the actual kernel completion.
    """

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "cupy"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return BackendCapabilities(
            supported_dtypes=frozenset(
                {"float32", "float64", "complex64", "complex128"}
            ),
            supports_threading=False,
            max_ndim=3,
            kind=BackendKind.CUDA,
            device="cuda",
        )

    def setup(self, shape: tuple[int, ...], dtype: str, threads: int) -> None:
        """Generate input data on CPU and transfer to GPU.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string.
        threads : int
            Ignored for CuPy (GPU execution).
        """
        host_data = _generate_input(shape, dtype)
        self._data = cp.asarray(host_data)
        ndim = len(shape)
        if ndim == 1:
            self._fft_func = cp.fft.fft
        elif ndim == 2:
            self._fft_func = cp.fft.fft2
        else:
            self._fft_func = cp.fft.fftn
        # Warm-up: run once and synchronize to trigger cuFFT plan creation
        self._fft_func(self._data)
        cp.cuda.Stream.null.synchronize()

    def execute(self) -> None:
        """Run a single FFT and synchronize the device."""
        self._fft_func(self._data)
        cp.cuda.Stream.null.synchronize()

    def teardown(self) -> None:
        """Release GPU arrays and free cached memory."""
        self._data = None
        self._fft_func = None
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _generate_input(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Generate random input data on the host.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        Data type string.

    Returns
    -------
    np.ndarray
        Random input array.
    """
    if "complex" in dtype:
        data = (
            np.random.standard_normal(shape)
            + 1j * np.random.standard_normal(shape)
        )
        return data.astype(dtype)
    else:
        return np.random.standard_normal(shape).astype(dtype)


register("cupy", CuPyBackend)
