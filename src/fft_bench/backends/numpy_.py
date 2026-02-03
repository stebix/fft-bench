"""NumPy FFT backend."""

from __future__ import annotations

import numpy as np
import numpy.fft

from .base import Backend, BackendCapabilities, BackendKind
from . import register


class NumPyBackend:
    """FFT backend using NumPy.

    NumPy's FFT does not support threading. It selects the appropriate
    FFT function (fft, fft2, fftn) based on the number of dimensions.
    """

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "numpy"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return BackendCapabilities(
            supported_dtypes=frozenset(
                {"float32", "float64", "complex64", "complex128"}
            ),
            supports_threading=False,
            max_ndim=3,
            kind=BackendKind.PYTHON,
        )

    def setup(self, shape: tuple[int, ...], dtype: str, threads: int) -> None:
        """Generate input data and select FFT function.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input array.
        dtype : str
            Data type string.
        threads : int
            Ignored for NumPy.
        """
        self._data = _generate_input(shape, dtype)
        ndim = len(shape)
        if ndim == 1:
            self._fft_func = np.fft.fft
        elif ndim == 2:
            self._fft_func = np.fft.fft2
        else:
            self._fft_func = np.fft.fftn

    def execute(self) -> None:
        """Run a single FFT."""
        self._fft_func(self._data)

    def teardown(self) -> None:
        """Release input data."""
        self._data = None
        self._fft_func = None


def _generate_input(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Generate random input data.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        Data type string.

    Returns
    -------
    np.ndarray
    """
    if "complex" in dtype:
        data = (
            np.random.standard_normal(shape)
            + 1j * np.random.standard_normal(shape)
        )
        return data.astype(dtype)
    else:
        return np.random.standard_normal(shape).astype(dtype)


register("numpy", NumPyBackend)
