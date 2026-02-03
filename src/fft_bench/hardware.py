"""Hardware and software environment information capture."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HardwareInfo:
    """Snapshot of the hardware and software environment.

    Parameters
    ----------
    cpu_model : str
        CPU model name.
    cpu_cores_physical : int
        Number of physical CPU cores.
    cpu_cores_logical : int
        Number of logical CPU cores.
    ram_total_gb : float
        Total RAM in gigabytes.
    os_name : str
        Operating system name and version.
    python_version : str
        Python version string.
    library_versions : dict[str, str]
        Versions of relevant libraries.
    """

    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    os_name: str
    python_version: str
    library_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "cpu_model": self.cpu_model,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "ram_total_gb": self.ram_total_gb,
            "os_name": self.os_name,
            "python_version": self.python_version,
            "library_versions": self.library_versions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HardwareInfo:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with hardware info fields.

        Returns
        -------
        HardwareInfo
        """
        return cls(
            cpu_model=data["cpu_model"],
            cpu_cores_physical=data["cpu_cores_physical"],
            cpu_cores_logical=data["cpu_cores_logical"],
            ram_total_gb=data["ram_total_gb"],
            os_name=data["os_name"],
            python_version=data["python_version"],
            library_versions=data.get("library_versions", {}),
        )


def _get_cpu_model() -> str:
    """Read CPU model name from /proc/cpuinfo on Linux, fallback to platform."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _get_ram_total_gb() -> float:
    """Read total RAM from /proc/meminfo on Linux, fallback to 0."""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemTotal"):
                # Value is in kB
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 2)
    return 0.0


def _get_library_versions() -> dict[str, str]:
    """Collect versions of relevant libraries."""
    versions: dict[str, str] = {}

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass

    try:
        import pyfftw
        versions["pyfftw"] = pyfftw.__version__
    except ImportError:
        pass

    try:
        import matplotlib
        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass

    return versions


def capture_hardware_info() -> HardwareInfo:
    """Capture current hardware and software environment information.

    Returns
    -------
    HardwareInfo
        Snapshot of the current environment.
    """
    return HardwareInfo(
        cpu_model=_get_cpu_model(),
        cpu_cores_physical=os.cpu_count() or 1,
        cpu_cores_logical=os.cpu_count() or 1,
        ram_total_gb=_get_ram_total_gb(),
        os_name=f"{platform.system()} {platform.release()}",
        python_version=sys.version,
        library_versions=_get_library_versions(),
    )
