from verigraph3d.graph import ExecutableSceneGraph
from verigraph3d.graph_visualization import graph_to_dot
from verigraph3d.models import (
    ActionType, Constraint, GoalSpec, SceneObject, SceneState, SemanticAction,
)
from verigraph3d.state import RelationCalculator


def graph_scene():
    return SceneState(objects={
        "Parent": SceneObject("Parent", location=(0, 0, 0.5), dimensions=(2, 2, 1), material="Wood"),
        "Child": SceneObject("Child", location=(0, -3, 0.5), dimensions=(0.5, 0.5, 0.5), parent="Parent"),
    })


def test_missing_relations_material_action_and_constraint_nodes_are_present():
    state = graph_scene()
    facts = {(f.subject, f.predicate, f.object): f.value for f in RelationCalculator().facts(state)}
    assert facts[("Child", "in_front_of", "Parent")]
    assert facts[("Child", "attached_to", "Parent")]
    graph = ExecutableSceneGraph()
    graph.rebuild(state)
    graph.set_goal(GoalSpec(required=[Constraint("Child", "attached_to", "Parent")]))
    graph.record_action(SemanticAction("a1", ActionType.MOVE_OBJECT, "Child", parameters={"location": (0, -2, 0.5)}), "success")
    assert graph.nodes["material:Wood"]["type"] == "material"
    assert graph.nodes["constraint:required:001"]["type"] == "constraint"
    assert graph.nodes["action:a1"]["type"] == "action"
    assert graph.query("Parent", "uses_material", "material:Wood").value


def test_graph_diff_and_dot_export():
    before = ExecutableSceneGraph()
    before.rebuild(graph_scene())
    after_state = graph_scene()
    after_state.objects["Child"].location = (1, -3, 0.5)
    after_state.timestamp = 1
    after = ExecutableSceneGraph()
    after.rebuild(after_state)
    diff = after.diff(before)
    assert "Child" in diff["changed_nodes"]
    assert diff["changed_facts"]
    dot = graph_to_dot(after)
    assert "digraph VeriGraph3D" in dot
    assert "in_front_of" in dot
