"""Entry point for the YOLO training pipeline."""

import argparse

from trainers.train_yolov8m import (
    build_model,
    evaluate_model,
    export_metrics,
    run_inference,
    train_model,
)
from utils.paths import get_dataset_root, get_runs_root
from utils.seed import set_seed


def run_pipeline(model_name: str) -> None:
    """Run YOLO training, evaluation, and inference."""

    set_seed(42, deterministic=True)

    dataset_root = get_dataset_root()
    runs_root = get_runs_root()

    data_yaml = dataset_root / "data.yaml"

    experiment_name = f"{model_name}_baseline"
    weights = f"{model_name}.pt"

    model = build_model(weights=weights)

    train_model(
        model=model,
        data_yaml=data_yaml,
        project_dir=runs_root,
        name=experiment_name,
        seed=42,
    )

    val_results = evaluate_model(
        model=model,
        data_yaml=data_yaml,
        project_dir=runs_root,
        name=experiment_name,
    )

    export_metrics(
        val_results,
        runs_root / experiment_name,
    )

    sample_dir = dataset_root / "test" / "images"
    sample_images = sorted(sample_dir.glob("*"))[:5]

    if sample_images:
        run_inference(
            model=model,
            sample_images=sample_images,
            project_dir=runs_root,
            name=experiment_name,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m",
        help="YOLO model name",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""

    args = parse_args()

    run_pipeline(model_name=args.model)


if __name__ == "__main__":
    main()