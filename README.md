# Driver Behaviour Monitoring

Webcam-based prototypes for monitoring head pose, eye movement, body posture,
behaviour changes, and phone presence.

## Layout

- `src/driver_monitor/`: installable application package.
- `scripts/`: training, validation, and model-export utilities.
- `configs/`: checked-in defaults for inference and training.
- `models/`: local weights only; ignored by Git.
- `tests/` and `docs/`: automated checks and project documentation.

The project no longer stores model weights, generated logs, caches, or nested
repositories. Ultralytics downloads a named model on first use, or you can
place a local weight file in `models/` and point the relevant module to it.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

## Run a monitor

```bash
driver-monitor head
driver-monitor eyes
driver-monitor body
driver-monitor behaviour
driver-monitor phone
```

Each monitor opens the default webcam. Press `q` to stop it.

## Training utilities

```bash
python scripts/train.py path/to/data.yaml
python scripts/validate.py path/to/data.yaml --model models/best.pt
python scripts/export.py --model models/best.pt --format onnx
```

All training inputs are passed on the command line; no machine-specific dataset
paths are stored in the repository.

## Benchmarking

The included benchmark runner measures model-only CPU inference and evaluates
person and cell-phone detection on a reproducible 100-image subset of the COCO
2017 validation set. It downloads the COCO annotations and selected images to
the ignored `benchmarks/` directory.

```bash
pip install ultralytics pycocotools psutil pyyaml
python scripts/benchmark.py --model yolov8n.pt --samples 100
python scripts/benchmark.py --model yolov8s.pt --samples 100
```

Baseline results on an Intel Core i7-14700HX CPU at 640px input:

| Model | FPS | Mean latency | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 28.52 | 35.07 ms | 58.22% | 37.46% | 45.59% | 46.76% | 33.47% |
| YOLOv8s | 3.29 | 304.35 ms | 66.56% | 63.78% | 65.14% | 61.42% | 41.72% |

These are combined metrics for the two classes and a small sampled subset; use
the full COCO validation set and target deployment hardware for final model
selection. Full webcam-pipeline FPS will also include MediaPipe, camera I/O,
display, and alerting overhead.
