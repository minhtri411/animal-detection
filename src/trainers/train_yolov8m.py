"""YOLOv8m training pipeline."""

from pathlib import Path
from typing import Dict, Any, List

from ultralytics import YOLO

from utils.metrics import save_metrics


def build_model(weights: str = "yolov8m.pt") -> YOLO:
    """Initialize YOLOv8m model with pretrained weights."""
    return YOLO(weights)


def train_model(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    seed: int = 42,
) -> Any:
    """Train the model with predefined settings."""
    return model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=8,
        amp=True,
        seed=seed,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
    )


def evaluate_model(model: YOLO, data_yaml: Path, project_dir: Path, name: str) -> Any:
    """Evaluate the trained model on the validation set."""
    return model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=4,
        device=0,
        project=str(project_dir),
        name=name,
        exist_ok=True,
    )


def run_inference(
    model: YOLO,
    sample_images: List[Path],
    project_dir: Path,
    name: str,
) -> Any:
    """Run inference on sample images."""
    sources = [str(p) for p in sample_images]
    return model.predict(
        source=sources,
        project=str(project_dir),
        name=f"{name}/predict",
        exist_ok=True,
        save=True,
        conf=0.25,
    )


def export_metrics(val_results: Any, output_dir: Path) -> Path:
    """Export evaluation metrics to JSON."""
    metrics: Dict[str, Any] = {
        "map50": float(getattr(val_results.box, "map50", 0.0)),
        "map50_95": float(getattr(val_results.box, "map", 0.0)),
    }
    output_path = output_dir / "metrics.json"
    save_metrics(metrics, output_path)
    return output_path
