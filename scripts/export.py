"""Export a trained Ultralytics model."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()
    from ultralytics import YOLO
    YOLO(args.model).export(format=args.format)


if __name__ == "__main__":
    main()
