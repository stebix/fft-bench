"""Benchmark result data models and JSON serialization."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import SingleBenchmarkConfig
from .fileutil import OverwritePolicy, resolve_output_path
from .hardware import HardwareInfo

_VERSION = "0.1.0"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark configuration run.

    Parameters
    ----------
    config : SingleBenchmarkConfig
        The configuration used for this benchmark.
    timings : list[float]
        Individual timing measurements in seconds.
    success : bool
        Whether the benchmark completed successfully.
    error : str | None
        Error message if the benchmark failed.
    """

    config: SingleBenchmarkConfig
    timings: list[float]
    success: bool
    error: str | None = None

    @property
    def mean(self) -> float:
        """Mean timing in seconds."""
        return statistics.mean(self.timings) if self.timings else 0.0

    @property
    def std(self) -> float:
        """Standard deviation of timings in seconds."""
        if len(self.timings) < 2:
            return 0.0
        return statistics.stdev(self.timings)

    @property
    def median(self) -> float:
        """Median timing in seconds."""
        return statistics.median(self.timings) if self.timings else 0.0

    @property
    def min(self) -> float:
        """Minimum timing in seconds."""
        return min(self.timings) if self.timings else 0.0

    @property
    def max(self) -> float:
        """Maximum timing in seconds."""
        return max(self.timings) if self.timings else 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        d: dict = {
            "config": self.config.to_dict(),
            "timings": self.timings,
            "success": self.success,
            "stats": {
                "mean": self.mean,
                "std": self.std,
                "median": self.median,
                "min": self.min,
                "max": self.max,
            },
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkResult:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with result fields.

        Returns
        -------
        BenchmarkResult
        """
        return cls(
            config=SingleBenchmarkConfig.from_dict(data["config"]),
            timings=data["timings"],
            success=data["success"],
            error=data.get("error"),
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with metadata.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Individual benchmark results.
    hardware : HardwareInfo
        Hardware and software environment information.
    timestamp : str
        ISO-format timestamp of when the suite was run.
    version : str
        Version of fft-bench used.
    """

    results: list[BenchmarkResult]
    hardware: HardwareInfo
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: str = _VERSION

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "hardware": self.hardware.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkSuite:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with suite fields.

        Returns
        -------
        BenchmarkSuite
        """
        return cls(
            results=[BenchmarkResult.from_dict(r) for r in data["results"]],
            hardware=HardwareInfo.from_dict(data["hardware"]),
            timestamp=data["timestamp"],
            version=data.get("version", "unknown"),
        )

    def save(
        self,
        path: str | Path,
        overwrite_policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
    ) -> Path:
        """Save the suite to a JSON file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        overwrite_policy : OverwritePolicy
            How to handle an already-existing file.

        Returns
        -------
        Path
            The actual path written to (may differ under ``AUTO_RENAME``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = resolve_output_path(path, overwrite_policy)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkSuite:
        """Load a suite from a JSON file.

        Parameters
        ----------
        path : str | Path
            Input file path.

        Returns
        -------
        BenchmarkSuite
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
