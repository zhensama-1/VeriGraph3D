from __future__ import annotations

import math

from .models import CameraState, SceneState, Vec3


class ActiveViewSelector:
    def candidates(self, state: SceneState, center: Vec3 = (0, 0, 0), radius: float = 7.0, count: int = 8) -> list[CameraState]:
        z = max(2.0, radius * 0.55)
        return [CameraState(f"Candidate_{i}", (center[0] + radius * math.cos(2 * math.pi * i / count), center[1] + radius * math.sin(2 * math.pi * i / count), z), center) for i in range(count)]

    def select(self, state: SceneState, uncertainties: list[dict], observation_cost: float = 0.1, scorer=None) -> tuple[CameraState | None, float]:
        if not uncertainties:
            return None, 0.0
        names = {u["subject"] for u in uncertainties}
        center_objects = [state.objects[n] for n in names if n in state.objects]
        center = tuple(sum(o.location[i] for o in center_objects) / len(center_objects) for i in range(3)) if center_objects else (0, 0, 0)
        candidates = self.candidates(state, center)  # geometric proxy for expected information gain
        if callable(scorer):
            scores = [(float(scorer(c, names)) - observation_cost, c) for c in candidates]
        else:
            scores = [(sum(1 for o in center_objects if self._front_facing(o.location, c)) / max(1, len(center_objects)) - observation_cost, c) for c in candidates]
        score, camera = max(scores, key=lambda pair: pair[0])
        return (camera, score) if score > 0 else (None, score)

    @staticmethod
    def _front_facing(point: Vec3, camera: CameraState) -> bool:
        return sum((point[i] - camera.location[i]) * (camera.target[i] - camera.location[i]) for i in range(3)) > 0
