"""Entry point for the YOLO training pipeline."""

import argparse
from pathlib import Path
from typing import Iterable

from trainers.train_yolov8m import (
    build_model,
    evaluate_model,
    evaluate_test_set,
    export_metrics,
    run_inference,
    train_model,
)
from utils.paths import get_dataset_root, get_runs_root
from utils.seed import set_seed


def ensure_paths_exist(paths: Iterable[Path], label: str) -> None:
    """Ensure all given paths exist."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: {', '.join(missing)}")


def ensure_data_yaml_has_test(data_yaml: Path) -> None:
    """Ensure data.yaml declares a test split."""
    content = data_yaml.read_text(encoding="utf-8")
    has_test = any(line.strip().startswith("test:") for line in content.splitlines())
    if not has_test:
        raise ValueError("data.yaml is missing the 'test:' split entry.")


def resolve_best_weights(
    runs_root: Path,
    experiment_name: str,
    weights_override: str | None,
) -> Path:
    """Resolve best.pt path for test evaluation."""
    if weights_override:
        return Path(weights_override)
    return runs_root / experiment_name / "weights" / "best.pt"


def run_training_flow(
    model_weights: str,
    experiment_name: str,
    data_yaml: Path,
    runs_root: Path,
    epochs: int,
    batch: int,
    imgsz: int,
) -> None:
    """Train, validate, export metrics, and run sample inference."""
    model = build_model(weights=model_weights)

    train_model(
        model=model,
        data_yaml=data_yaml,
        project_dir=runs_root,
        name=experiment_name,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=42,
    )

    val_results = evaluate_model(
        model=model,
        data_yaml=data_yaml,
        project_dir=runs_root,
        name=experiment_name,
        imgsz=imgsz,
        batch=max(1, batch // 2),
    )

    export_metrics(
        val_results,
        runs_root / experiment_name,
    )

    sample_dir = data_yaml.parent / "test" / "images"
    sample_images = sorted(sample_dir.glob("*"))[:5]

    if sample_images:
        run_inference(
            model=model,
            sample_images=sample_images,
            project_dir=runs_root,
            name=experiment_name,
        )


def run_test_flow(
    weights_path: Path,
    experiment_name: str,
    data_yaml: Path,
    runs_root: Path,
    batch: int,
    imgsz: int,
) -> Path:
    """Evaluate the test set and export metrics."""
    model = build_model(weights=str(weights_path))
    metrics_path = runs_root / experiment_name / "test_metrics.json"

    metrics = evaluate_test_set(
        model=model,
        data_yaml=data_yaml,
        project_dir=runs_root,
        name=experiment_name,
        output_path=metrics_path,
        imgsz=imgsz,
        batch=max(1, batch // 2),
    )

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    print(f"Best weights: {weights_path}")
    print(f"Metrics JSON: {metrics_path}")
    return metrics_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set using best.pt",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Initial model weights for training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to best.pt for test evaluation",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolov8m_baseline",
        help="Experiment name",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""

    args = parse_args()

    if not args.train and not args.test:
        raise ValueError("Please specify --train, --test, or both.")

    set_seed(42, deterministic=True)

    dataset_root = get_dataset_root()
    runs_root = get_runs_root()
    data_yaml = dataset_root / "data.yaml"

    ensure_paths_exist([data_yaml], "data.yaml")

    if args.test:
        ensure_data_yaml_has_test(data_yaml)

        test_dirs = [
            dataset_root / "test" / "images",
            dataset_root / "test" / "labels",
        ]
        ensure_paths_exist(test_dirs, "test dataset directories")

    if args.train:
        run_training_flow(
            model_weights=args.model,
            experiment_name=args.name,
            data_yaml=data_yaml,
            runs_root=runs_root,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )

    if args.test:
        weights_path = resolve_best_weights(
            runs_root,
            args.name,
            None if args.train else args.weights,
        )
        ensure_paths_exist([weights_path], "best.pt")

        run_test_flow(
            weights_path=weights_path,
            experiment_name=args.name,
            data_yaml=data_yaml,
            runs_root=runs_root,
            batch=args.batch,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()