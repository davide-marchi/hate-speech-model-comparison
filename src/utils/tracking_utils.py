from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Tuple

from .io_utils import ensure_dir

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - optional dependency
    EmissionsTracker = None  # type: ignore


@dataclass
class ExecutionStats:
    task_name: str
    duration_seconds: float
    emissions_kg: Optional[float]
    emissions_file: Optional[Path]
    tracker_available: bool


def _build_tracker(task_name: str, output_dir: Path) -> Optional[EmissionsTracker]:
    if EmissionsTracker is None:
        return None
    ensure_dir(output_dir)
    return EmissionsTracker(
        project_name=task_name,
        output_dir=str(output_dir),
        output_file=f"{task_name}_emissions.csv",
        save_to_file=True,
        log_level="error",
    )


def run_with_tracking(
    task_name: str,
    output_dir: Path,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, ExecutionStats]:
    """Execute a callable while measuring time and emissions."""
    tracker = _build_tracker(task_name, output_dir)
    if tracker is not None:
        tracker.start()
    start = perf_counter()
    result = func(*args, **kwargs)
    duration = perf_counter() - start
    emissions = tracker.stop() if tracker is not None else None
    emissions_file = (
        Path(output_dir) / f"{task_name}_emissions.csv" if tracker is not None else None
    )
    stats = ExecutionStats(
        task_name=task_name,
        duration_seconds=duration,
        emissions_kg=emissions,
        emissions_file=emissions_file,
        tracker_available=tracker is not None,
    )
    return result, stats
