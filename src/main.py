import argparse
import subprocess
import sys


def run_script(script_path):
    subprocess.run([sys.executable, script_path], check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cnn", "transformer", "yolo", "all"],
        required=True,
    )

    args = parser.parse_args()

    if args.model == "cnn":
        run_script("models/cnn.py")

    elif args.model == "transformer":
        run_script("models/transformer.py")

    elif args.model == "yolo":
        run_script("models/yolov8m.py")

    elif args.model == "all":
        run_script("models/cnn.py")
        run_script("models/transformer.py")
        run_script("models/yolov8m.py")


if __name__ == "__main__":
    main()