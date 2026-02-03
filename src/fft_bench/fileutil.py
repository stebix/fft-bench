"""File output utilities with overwrite protection."""

from __future__ import annotations

import enum
import sys
from pathlib import Path


class OverwritePolicy(enum.Enum):
    """Policy for handling existing output files.

    Attributes
    ----------
    AUTO_RENAME : str
        Append ``_1``, ``_2``, etc. before the extension and print a notice.
    FORCE : str
        Overwrite the file silently.
    ERROR : str
        Exit with an error if the file already exists.
    """

    AUTO_RENAME = "auto-rename"
    FORCE = "force"
    ERROR = "error"


def resolve_output_path(
    path: str | Path,
    policy: OverwritePolicy = OverwritePolicy.AUTO_RENAME,
) -> Path:
    """Apply the overwrite policy and return the actual path to write to.

    Parameters
    ----------
    path : str | Path
        Desired output file path.
    policy : OverwritePolicy
        How to handle an already-existing file.

    Returns
    -------
    Path
        The resolved path (may differ from *path* under ``AUTO_RENAME``).

    Raises
    ------
    SystemExit
        If *policy* is ``ERROR`` and the file exists.
    """
    path = Path(path)

    if not path.exists():
        return path

    if policy is OverwritePolicy.FORCE:
        return path

    if policy is OverwritePolicy.ERROR:
        print(f"Error: output file already exists: {path}", file=sys.stderr)
        sys.exit(1)

    # AUTO_RENAME
    new_path = _next_available_path(path)
    print(f"Notice: {path} exists, writing to {new_path} instead")
    return new_path


def _next_available_path(path: Path) -> Path:
    """Find the next available ``stem_N.ext`` path.

    Parameters
    ----------
    path : Path
        Original file path that already exists.

    Returns
    -------
    Path
        A path that does not yet exist, e.g. ``results_1.json``.
    """
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
