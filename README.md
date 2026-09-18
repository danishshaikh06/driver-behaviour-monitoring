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
