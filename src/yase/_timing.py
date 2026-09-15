"""Small helpers for opt-in backend phase instrumentation.

Timing is deliberately kept private and dependency-free.  Backend adapters
only pay for the clock reads when ``record_timings=True``; the default path
keeps the historical metadata contract and has no instrumentation overhead.
"""

from collections.abc import Mapping
from typing import Any


def add_phase_timings(
    metadata: dict[str, Any],
    record: bool,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    """Return metadata with JSON-friendly phase durations when requested."""
    if record:
        metadata["timings_seconds"] = {
            str(name): max(0.0, float(seconds)) for name, seconds in timings.items()
        }
    return metadata


__all__ = ["add_phase_timings"]
