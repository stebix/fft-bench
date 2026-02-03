"""pyFFTW FFT backend."""

from __future__ import annotations

import numpy as np
import pyfftw

from .base import Backend, BackendCapabilities, BackendKind
from . import register


class PyFFTWBackend:
    """FFT backend using pyFFTW.

    pyFFTW creates an FFTW plan during ``setup()`` which is then
    executed in ``execute()``. This ensures planning cost is excluded
    from timing. Thread count is set during plan creation.
    """

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "pyfftw"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return BackendCapabilities(
            supported_dtypes=frozenset(
                {"float32", "float64", "complex64", "complex128"}
            ),
            supports_threading=True,
            max_ndim=3,
            kind=BackendKind.PYTHON,
        )

    def setup(self, shape: tuple[int, ...], dtype: str, threads: int) -> None:
        """Generate input data and create FFTW plan.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string.
        threads : int
            Number of threads for FFTW.
        """
        self._data = _generate_input(shape, dtype)
        ndim = len(shape)
        if ndim == 1:
            builder = pyfftw.builders.fft
        elif ndim == 2:
            builder = pyfftw.builders.fft2
        else:
            builder = pyfftw.builders.fftn
        self._fft_obj = builder(self._data, threads=threads)

    def execute(self) -> None:
        """Run a single FFT using the pre-created plan."""
        self._fft_obj(self._data)

    def teardown(self) -> None:
        """Release plan and input data."""
        self._fft_obj = None
        self._data = None


def _generate_input(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Generate random input data as a pyFFTW-aligned array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        Data type string.

    Returns
    -------
    np.ndarray
        Aligned array suitable for pyFFTW.
    """
    if "complex" in dtype:
        data = (
            np.random.standard_normal(shape)
            + 1j * np.random.standard_normal(shape)
        )
    else:
        data = np.random.standard_normal(shape)

    aligned = pyfftw.empty_aligned(shape, dtype=dtype)
    aligned[:] = data.astype(dtype)
    return aligned


register("pyfftw", PyFFTWBackend)
