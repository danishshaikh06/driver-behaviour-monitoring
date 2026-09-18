"""Train a YOLO model with explicit, portable command-line options."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a driver-monitor object detector.")
    parser.add_argument("data", help="Path to a YOLO dataset YAML file")
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        cache=False,
        device=args.device,
    )


if __name__ == "__main__":
    main()
