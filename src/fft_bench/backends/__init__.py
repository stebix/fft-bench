"""Backend registry for FFT benchmark backends."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Backend, BackendCapabilities

logger = logging.getLogger(__name__)

_registry: dict[str, type[Backend]] = {}

# Modules to import during discovery
_BACKEND_MODULES = [
    "fft_bench.backends.numpy_",
    "fft_bench.backends.scipy_",
    "fft_bench.backends.pyfftw_",
    "fft_bench.backends.cupy_",
    "fft_bench.backends.cufft_native",
]


def register(name: str, backend_cls: type[Backend]) -> None:
    """Register a backend class.

    Parameters
    ----------
    name : str
        Name to register the backend under.
    backend_cls : type[Backend]
        Backend class to register.
    """
    _registry[name] = backend_cls
    logger.debug("Registered backend: %s", name)


def get(name: str) -> type[Backend]:
    """Get a registered backend class by name.

    Parameters
    ----------
    name : str
        Name of the backend.

    Returns
    -------
    type[Backend]
        The backend class.

    Raises
    ------
    KeyError
        If the backend is not registered.
    """
    if name not in _registry:
        raise KeyError(
            f"Unknown backend: {name!r}. "
            f"Available: {', '.join(sorted(_registry))}"
        )
    return _registry[name]


def get_capabilities(name: str) -> BackendCapabilities:
    """Get the capabilities of a registered backend.

    Parameters
    ----------
    name : str
        Name of the backend.

    Returns
    -------
    BackendCapabilities
    """
    backend_cls = get(name)
    return backend_cls().capabilities


def discover_backends() -> list[str]:
    """Import all backend modules to trigger registration.

    Returns
    -------
    list[str]
        Names of successfully registered backends.
    """
    for module_name in _BACKEND_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.debug("Could not import %s: %s", module_name, e)
    return list(_registry)


def available_backends() -> list[str]:
    """Return names of all registered backends.

    Returns
    -------
    list[str]
        Sorted list of registered backend names.
    """
    return sorted(_registry)
