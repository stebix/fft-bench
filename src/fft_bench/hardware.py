"""Hardware and software environment information capture."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GpuInfo:
    """Information about a single GPU device.

    Parameters
    ----------
    name : str
        GPU device name.
    memory_total_gb : float
        Total GPU memory in gigabytes.
    driver_version : str
        GPU driver version string.
    cuda_version : str
        CUDA runtime version string.
    """

    name: str
    memory_total_gb: float
    driver_version: str
    cuda_version: str

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "memory_total_gb": self.memory_total_gb,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GpuInfo:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with GPU info fields.

        Returns
        -------
        GpuInfo
        """
        return cls(
            name=data["name"],
            memory_total_gb=data["memory_total_gb"],
            driver_version=data["driver_version"],
            cuda_version=data["cuda_version"],
        )


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
    gpu_devices : list[GpuInfo]
        List of available GPU devices.
    """

    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    os_name: str
    python_version: str
    library_versions: dict[str, str] = field(default_factory=dict)
    gpu_devices: list[GpuInfo] = field(default_factory=list)

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
            "gpu_devices": [g.to_dict() for g in self.gpu_devices],
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
            gpu_devices=[
                GpuInfo.from_dict(g) for g in data.get("gpu_devices", [])
            ],
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

    try:
        import cupy
        versions["cupy"] = cupy.__version__
    except ImportError:
        pass

    return versions


def _get_gpu_devices() -> list[GpuInfo]:
    """Detect available CUDA GPU devices via CuPy.

    Returns
    -------
    list[GpuInfo]
        Information for each detected GPU, empty if no CUDA available.
    """
    try:
        import cupy
    except ImportError:
        return []

    devices: list[GpuInfo] = []
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        return []

    cuda_version_int = cupy.cuda.runtime.runtimeGetVersion()
    cuda_major = cuda_version_int // 1000
    cuda_minor = (cuda_version_int % 1000) // 10
    cuda_version = f"{cuda_major}.{cuda_minor}"

    driver_version_int = cupy.cuda.runtime.driverGetVersion()
    driver_major = driver_version_int // 1000
    driver_minor = (driver_version_int % 1000) // 10
    driver_version = f"{driver_major}.{driver_minor}"

    for i in range(device_count):
        with cupy.cuda.Device(i):
            props = cupy.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            mem_bytes = props["totalGlobalMem"]
            mem_gb = round(mem_bytes / (1024**3), 2)
            devices.append(
                GpuInfo(
                    name=name,
                    memory_total_gb=mem_gb,
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                )
            )

    return devices


def cuda_is_available() -> bool:
    """Check whether CUDA is available on this system.

    Returns
    -------
    bool
        ``True`` if at least one CUDA device is detected.
    """
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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
        gpu_devices=_get_gpu_devices(),
    )
