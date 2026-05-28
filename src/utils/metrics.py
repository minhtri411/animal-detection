"""Metrics export helpers."""

import json
from pathlib import Path
from typing import Dict, Any


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
