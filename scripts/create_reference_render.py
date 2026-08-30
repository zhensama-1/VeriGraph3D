"""Create the expected reference render from the deterministic fixture scene."""

from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from verigraph3d.backends import BlenderBackend  # noqa: E402
from verigraph3d.execution import SafeExecutor  # noqa: E402
from verigraph3d.models import ActionType, SemanticAction  # noqa: E402


def main() -> None:
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = BlenderBackend()
    executor = SafeExecutor(backend)
    placed = executor.execute(SemanticAction("reference_place", ActionType.PLACE_ON, "Cup", "Table"))
    colored = executor.execute(SemanticAction("reference_color", ActionType.SET_MATERIAL, "Cup", parameters={"color": (1, 0, 0, 1)}))
    if placed.status != "success" or colored.status != "success":
        raise RuntimeError(f"Could not create reference: {placed.error or colored.error}")
    backend.render(str(output), "Camera")
    print("VERIGRAPH_REFERENCE", output)


if __name__ == "__main__":
    main()
