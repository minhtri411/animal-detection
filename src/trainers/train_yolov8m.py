from pathlib import Path
from typing import Any, Dict, List

from ultralytics import YOLO

from utils.metrics import save_metrics


# ======================
# 1. BUILD MODEL
# ======================
def build_model(weights: str = "yolov8m.pt") -> YOLO:
    """Load pretrained YOLOv8 model."""
    return YOLO(weights)


# ======================
# 2. TRAIN
# ======================
def train_model(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    epochs: int = 10,
    batch: int = 16,
    imgsz: int = 640,
    seed: int = 42,
) -> Any:
    """
    Train YOLOv8 model.
    """
    return model.train(
        data=str(data_yaml),
        # epochs=100,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        device=0,
        # workers=2,
        workers=4,        
        amp=True,
        seed=seed,
        project=str(project_dir),
        name=name,
        exist_ok=True,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,         
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
    )


# ======================
# 3. VALIDATION
# ======================
def evaluate_model(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    imgsz: int = 640,
    batch: int = 8,
) -> Any:
    """Run validation."""
    return model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(project_dir),
        name=name,
        exist_ok=True,
    )


# ======================
# 4. INFERENCE
# ======================
def run_inference(
    model: YOLO,
    sample_images: List[Path],
    project_dir: Path,
    name: str,
):
    """Run prediction on images."""
    return model.predict(
        source=[str(p) for p in sample_images],
        conf=0.25,
        save=True,
        project=str(project_dir),
        name=f"{name}/predict",
        exist_ok=True,
    )


# ======================
# 5. SAVE METRICS
# ======================
def extract_metrics(results: Any) -> Dict[str, Any]:
    """Extract precision/recall/mAP metrics from a YOLO results object."""
    return {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }


def export_metrics(val_results: Any, output_dir: Path) -> Path:
    """Save validation metrics."""
    metrics = extract_metrics(val_results)
    path = output_dir / "metrics.json"
    save_metrics(metrics, path)
    return path


def evaluate_test_set(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    output_path: Path,
    imgsz: int = 640,
    batch: int = 8,
) -> Dict[str, Any]:
    """Evaluate the model on the test set and save metrics."""
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(project_dir),
        name=f"{name}/test",
        exist_ok=True,
        split="test",
    )
    metrics = extract_metrics(results)
    save_metrics(metrics, output_path)
    return metrics