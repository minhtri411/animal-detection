from pathlib import Path
from typing import Any, Dict, List
import argparse
import json

from ultralytics import YOLO


# ======================
# 1. CONFIG
# ======================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/processed/yolo_dataset_processed"
RUNS_ROOT = PROJECT_ROOT / "outputs"


# ======================
# 2. UTILS
# ======================

def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def ensure_paths_exist(paths, label: str):
    missing = [str(p) for p in paths if not p.exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing {label}: {', '.join(missing)}"
        )


# ======================
# 3. BUILD MODEL
# ======================

def build_model(weights: str = "yolov8m.pt") -> YOLO:
    return YOLO(weights)


# ======================
# 4. TRAIN
# ======================

def train_model(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    seed: int = 42,
):

    return model.train(
        data=str(data_yaml),

        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        patience=20,
        device=0,
        workers=4,

        amp=True,
        seed=seed,

        project=str(project_dir),
        name=name,
        exist_ok=True,

        # augmentation
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
# 5. VALIDATION
# ======================

def evaluate_model(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    imgsz: int = 640,
    batch: int = 8,
):

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
# 6. TEST
# ======================

def evaluate_test_set(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    name: str,
    imgsz: int = 640,
    batch: int = 8,
):

    return model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=0,

        split="test",

        project=str(project_dir),
        name=f"{name}/test",
        exist_ok=True,
    )


# ======================
# 7. INFERENCE
# ======================

def run_inference(
    model: YOLO,
    sample_images: List[Path],
    project_dir: Path,
    name: str,
):

    return model.predict(
        source=[str(p) for p in sample_images],

        conf=0.25,
        save=True,

        project=str(project_dir),
        name=f"{name}/predict",
        exist_ok=True,
    )


# ======================
# 8. METRICS
# ======================

def extract_metrics(results: Any) -> Dict[str, Any]:

    return {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }


# ======================
# 9. MAIN
# ======================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolov8m",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not args.train and not args.test:
        args.train = True
        args.test = True

    data_yaml = DATASET_ROOT / "data.yaml"

    ensure_paths_exist([data_yaml], "data.yaml")

    # ======================
    # TRAIN
    # ======================

    if args.train:

        model = build_model(args.model)

        train_model(
            model=model,
            data_yaml=data_yaml,
            project_dir=RUNS_ROOT,
            name=args.name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )

        val_results = evaluate_model(
            model=model,
            data_yaml=data_yaml,
            project_dir=RUNS_ROOT,
            name=args.name,
            imgsz=args.imgsz,
            batch=max(1, args.batch // 2),
        )

        metrics = extract_metrics(val_results)

        save_metrics(
            metrics,
            RUNS_ROOT / args.name / "metrics.json",
        )

        print("\nValidation Metrics")
        print(metrics)

        # sample inference
        sample_dir = DATASET_ROOT / "test/images"

        sample_images = sorted(sample_dir.glob("*"))[:5]

        if sample_images:

            run_inference(
                model=model,
                sample_images=sample_images,
                project_dir=RUNS_ROOT,
                name=args.name,
            )

    # ======================
    # TEST
    # ======================

    if args.test:

        best_weights = (
            RUNS_ROOT
            / args.name
            / "weights"
            / "best.pt"
        )

        ensure_paths_exist([best_weights], "best.pt")

        model = build_model(str(best_weights))

        test_results = evaluate_test_set(
            model=model,
            data_yaml=data_yaml,
            project_dir=RUNS_ROOT,
            name=args.name,
            imgsz=args.imgsz,
            batch=max(1, args.batch // 2),
        )

        metrics = extract_metrics(test_results)

        save_metrics(
            metrics,
            RUNS_ROOT / args.name / "test_metrics.json",
        )

        print("\nTest Metrics")
        print(metrics)


if __name__ == "__main__":
    main()