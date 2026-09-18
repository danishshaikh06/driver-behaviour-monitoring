"""Reusable YOLO person detector."""

from driver_monitor.config import settings


class PersonDetector:
    """Returns the highest-confidence person bounding box in a frame."""

    def __init__(self, model_name: str = settings.person_model) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_name)

    def detect(self, frame):
        best_box, best_confidence = None, 0.0
        for result in self.model(frame, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if int(box.cls[0]) == 0 and confidence > best_confidence:
                    best_box = box.xyxy[0].cpu().numpy()
                    best_confidence = confidence
        return best_box, best_confidence
