from __future__ import annotations

from dataclasses import asdict, dataclass
import time


@dataclass
class RunMetrics:
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    actions: int = 0
    successful_actions: int = 0
    repairs: int = 0
    rollbacks: int = 0
    renders: int = 0
    observations: int = 0
    observation_information_gain: float = 0.0
    observation_cost: float = 0.0
    vlm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def start(self) -> None:
        self.started_at = time.perf_counter()

    def stop(self) -> None:
        self.elapsed_seconds = time.perf_counter() - self.started_at

    def to_dict(self) -> dict:
        return asdict(self)
