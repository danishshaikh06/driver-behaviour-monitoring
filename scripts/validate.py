"""Validate a trained Ultralytics model against a dataset configuration."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to a YOLO dataset YAML file")
    parser.add_argument("--model", default="models/best.pt")
    args = parser.parse_args()
    from ultralytics import YOLO
    YOLO(args.model).val(data=args.data)


if __name__ == "__main__":
    main()
