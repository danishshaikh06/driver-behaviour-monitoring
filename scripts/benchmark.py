"""Benchmark YOLO person and phone detection on a reproducible COCO subset."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import psutil
import yaml
from pycocotools.coco import COCO
from ultralytics import YOLO


ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "https://images.cocodataset.org/val2017/"


def download(url: str, destination: Path) -> None:
    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)


def prepare_subset(root: Path, sample_size: int, model: YOLO) -> Path:
    """Build an annotated COCO subset focused on person and cell-phone images."""
    archive = root / "annotations_trainval2017.zip"
    annotation = root / "annotations" / "instances_val2017.json"
    download(ANNOTATIONS_URL, archive)
    if not annotation.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract("annotations/instances_val2017.json", root)

    coco = COCO(str(annotation))
    category_ids = coco.getCatIds(catNms=["person", "cell phone"])
    image_ids = sorted(set().union(*(coco.getImgIds(catIds=[category]) for category in category_ids)))[:sample_size]
    images_dir, labels_dir = root / "images" / "val", root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_by_name = {name: index for index, name in model.names.items()}

    for image in coco.loadImgs(image_ids):
        image_path = images_dir / image["file_name"]
        download(IMAGES_URL + image["file_name"], image_path)
        lines = []
        for annotation_item in coco.loadAnns(coco.getAnnIds(imgIds=[image["id"]], catIds=category_ids, iscrowd=None)):
            category = coco.loadCats([annotation_item["category_id"]])[0]["name"]
            x, y, width, height = annotation_item["bbox"]
            lines.append(
                f"{class_by_name[category]} {(x + width / 2) / image['width']:.6f} "
                f"{(y + height / 2) / image['height']:.6f} {width / image['width']:.6f} {height / image['height']:.6f}"
            )
        (labels_dir / f"{Path(image['file_name']).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    config = {"path": str(root), "train": "images/val", "val": "images/val", "names": model.names}
    data_file = root / "data.yaml"
    data_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/coco-person-phone"))
    args = parser.parse_args()

    model = YOLO(args.model)
    data_file = prepare_subset(args.output, args.samples, model)
    image_paths = sorted((args.output / "images" / "val").glob("*.jpg"))
    for path in image_paths[:5]:
        model(str(path), imgsz=args.image_size, verbose=False)

    process = psutil.Process()
    start = time.perf_counter()
    for path in image_paths:
        model(str(path), imgsz=args.image_size, verbose=False)
    elapsed = time.perf_counter() - start
    metrics = model.val(data=str(data_file), imgsz=args.image_size, device="cpu", verbose=False)
    report = {
        "dataset": "COCO 2017 val, person/cell phone-focused subset",
        "samples": len(image_paths),
        "model": args.model,
        "device": "CPU",
        "mean_latency_ms": elapsed / len(image_paths) * 1000,
        "throughput_fps": len(image_paths) / elapsed,
        "process_rss_mb": process.memory_info().rss / 1024**2,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    report["f1"] = 2 * report["precision"] * report["recall"] / max(report["precision"] + report["recall"], 1e-12)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
