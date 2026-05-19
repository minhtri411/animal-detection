"""Project path utilities."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT_DEFAULT = PROJECT_ROOT / "data" / "processed" / "yolo_dataset_processed"
RUNS_ROOT = (PROJECT_ROOT/ "runs"/ "object_detection")


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_dataset_root() -> Path:
    """Return the default processed dataset root."""
    return DATASET_ROOT_DEFAULT

def get_runs_root() -> Path:
    """Return the experiment output root."""
    return RUNS_ROOT