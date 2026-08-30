from verigraph3d.models import SceneObject, SceneState
from verigraph3d.state import RelationCalculator


def test_on_top_and_no_intersection():
    state = SceneState(objects={
        "Table": SceneObject("Table", location=(0, 0, 0.5), dimensions=(4, 3, 1)),
        "Cup": SceneObject("Cup", location=(0, 0, 1.25), dimensions=(0.5, 0.5, 0.5)),
    })
    facts = RelationCalculator().facts(state)
    values = {(f.subject, f.predicate, f.object): f.value for f in facts}
    assert values[("Cup", "on_top_of", "Table")]
    assert not values[("Cup", "intersecting", "Table")]

