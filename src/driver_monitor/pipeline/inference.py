"""Command-line dispatch for the available webcam monitors."""

from __future__ import annotations

import argparse
from collections.abc import Callable


def available_monitors() -> dict[str, Callable[[], None]]:
    """Import monitor runners lazily so CLI help needs no CV dependencies."""
    from driver_monitor.detection.phone_detector import main as phone
    from driver_monitor.perception.behaviour import main as behaviour
    from driver_monitor.perception.body_pose import main as body
    from driver_monitor.perception.eye_tracking import main as eyes
    from driver_monitor.perception.head_pose import main as head

    return {"head": head, "eyes": eyes, "body": body, "behaviour": behaviour, "phone": phone}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a driver-monitor webcam module.")
    parser.add_argument("monitor", choices=("head", "eyes", "body", "behaviour", "phone"))
    args = parser.parse_args(argv)
    available_monitors()[args.monitor]()
