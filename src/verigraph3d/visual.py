from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import GoalSpec, SceneState
from .verification import RuleBasedVisualVerifier


@dataclass(frozen=True, slots=True)
class ImageSimilarityWeights:
    pixels: float = 0.55
    histogram: float = 0.20
    edges: float = 0.10
    layout: float = 0.15

    def __post_init__(self) -> None:
        total = self.pixels + self.histogram + self.edges + self.layout
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Image similarity weights must sum to 1.0")


class ReferenceImageVerifier:
    """Deterministic image verifier using pixels, color, edges and coarse layout.

    This is a reproducible non-VLM visual baseline. The candidate render path is
    read from ``SceneState.metadata['render_path']``.
    """

    def __init__(self, weights: ImageSimilarityWeights | None = None, size: int = 256) -> None:
        self.weights = weights or ImageSimilarityWeights()
        self.size = size
        self.attributes = RuleBasedVisualVerifier()

    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str]):
        attribute_score, attribute_differences, recommendations = self.attributes.verify(state, goal, [])
        if not reference_images:
            return attribute_score, attribute_differences, recommendations
        render_path = state.metadata.get("render_path")
        if not render_path:
            return 0.0, ["candidate render is missing"], ["render_scene"]
        candidate = self._load(render_path)
        component_sets = [self._components(candidate, self._load(path)) for path in reference_images]
        best = max(component_sets, key=lambda item: item["combined"])
        image_score = best["combined"]
        differences = list(attribute_differences)
        suggestions = list(recommendations)
        if best["histogram"] < 0.75:
            differences.append("color distribution differs from reference")
            suggestions.append("adjust_material_or_lighting")
        if best["edges"] < 0.70:
            differences.append("object contours differ from reference")
            suggestions.append("adjust_geometry_or_camera")
        if best["layout"] < 0.75:
            differences.append("coarse composition differs from reference")
            suggestions.append("adjust_layout_or_camera")
        checks_attributes = bool(goal.visual)
        score = 0.7 * image_score + 0.3 * attribute_score if checks_attributes else image_score
        return score, differences, list(dict.fromkeys(suggestions))

    def _load(self, path: str | Path):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("ReferenceImageVerifier requires NumPy") from exc
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        try:
            from PIL import Image
            image = Image.open(source).convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.float32) / 255.0
        except ImportError:
            return self._load_with_blender(source, np)

    def _load_with_blender(self, source: Path, np):
        """Load pixels through bpy when Blender's bundled Python has no Pillow."""
        try:
            import bpy  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install Pillow or run inside Blender to load images") from exc
        image = bpy.data.images.load(str(source), check_existing=False)
        try:
            width, height = image.size
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., :3]
            y = np.linspace(0, height - 1, self.size).astype(int)
            x = np.linspace(0, width - 1, self.size).astype(int)
            return pixels[y[:, None], x[None, :], :]
        finally:
            bpy.data.images.remove(image)

    def _components(self, candidate, reference) -> dict[str, float]:
        import numpy as np

        absolute_error = np.abs(candidate - reference)
        error_map = np.max(absolute_error, axis=2).reshape(-1)
        salient_count = max(1, error_map.size // 100)
        salient_error = float(np.partition(error_map, -salient_count)[-salient_count:].mean())
        mean_error = float(absolute_error.mean())
        pixel_score = float(1.0 - (0.25 * mean_error + 0.75 * salient_error))
        histogram_scores = []
        for channel in range(3):
            a, _ = np.histogram(candidate[..., channel], bins=32, range=(0, 1), density=False)
            b, _ = np.histogram(reference[..., channel], bins=32, range=(0, 1), density=False)
            histogram_scores.append(float(np.minimum(a, b).sum() / max(1, a.sum())))
        histogram_score = sum(histogram_scores) / len(histogram_scores)
        edge_a, edge_b = self._edges(candidate), self._edges(reference)
        edge_score = float(1.0 - np.mean(np.abs(edge_a - edge_b)))
        layout_a, layout_b = self._layout(candidate), self._layout(reference)
        layout_score = float(1.0 - np.mean(np.abs(layout_a - layout_b)))
        w = self.weights
        combined = w.pixels * pixel_score + w.histogram * histogram_score + w.edges * edge_score + w.layout * layout_score
        return {"pixels": pixel_score, "histogram": histogram_score, "edges": edge_score, "layout": layout_score, "combined": float(combined)}

    @staticmethod
    def _edges(image):
        import numpy as np

        gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        dx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
        dy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
        return np.clip(dx + dy, 0, 1)

    @staticmethod
    def _layout(image):
        import numpy as np

        height, width = image.shape[:2]
        rows = np.array_split(image, 4, axis=0)
        cells = [cell.mean(axis=(0, 1)) for row in rows for cell in np.array_split(row, 4, axis=1)]
        return np.asarray(cells)


class EnsembleVisualVerifier:
    """Combines reproducible image metrics and semantic VLM judgments."""

    def __init__(self, verifiers: list, weights: list[float] | None = None) -> None:
        if not verifiers:
            raise ValueError("At least one visual verifier is required")
        self.verifiers = verifiers
        self.weights = weights or [1.0 / len(verifiers)] * len(verifiers)
        if len(self.weights) != len(verifiers) or any(weight < 0 for weight in self.weights):
            raise ValueError("Verifier weights must be non-negative and match verifier count")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("Verifier weights must have a positive sum")
        self.weights = [weight / total for weight in self.weights]

    def verify(self, state: SceneState, goal: GoalSpec, reference_images: list[str]):
        results = [verifier.verify(state, goal, reference_images) for verifier in self.verifiers]
        score = sum(weight * result[0] for weight, result in zip(self.weights, results))
        differences = list(dict.fromkeys(item for result in results for item in result[1]))
        recommendations = list(dict.fromkeys(item for result in results for item in result[2]))
        return float(score), differences, recommendations
