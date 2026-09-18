"""Central defaults for model selection and webcam inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime defaults. Model names let Ultralytics download weights on demand."""

    person_model: str = "yolov8n.pt"
    phone_model: str = "yolov8s.pt"
    camera_index: int = 0


settings = Settings()
