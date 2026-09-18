"""Small reusable calibration primitive for frame-based measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev


@dataclass
class Baseline:
    """Collect a fixed number of scalar samples and expose summary statistics."""

    sample_count: int = 90
    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if not self.complete:
            self.samples.append(value)

    @property
    def complete(self) -> bool:
        return len(self.samples) >= self.sample_count

    @property
    def mean(self) -> float | None:
        return fmean(self.samples) if self.samples else None

    @property
    def standard_deviation(self) -> float | None:
        return pstdev(self.samples) if len(self.samples) > 1 else None
