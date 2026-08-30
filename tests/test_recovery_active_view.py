from verigraph3d.agent import AgentConfig, VeriGraph3DAgent
from verigraph3d.backends import FaultInjectingBackend, FaultRule, MemoryBackend
from verigraph3d.cli import demo_scene
from verigraph3d.models import Constraint, GoalSpec


def place_goal():
    return GoalSpec(
        required=[Constraint("Cup", "on_top_of", "Table")],
        forbidden=[Constraint("Cup", "intersecting", "Table")],
    )


def test_injected_penetration_is_attributed_and_locally_repaired():
    backend = FaultInjectingBackend(
        MemoryBackend(demo_scene()),
        [FaultRule("action_001", "z_offset", -0.05)],
    )
    result = VeriGraph3DAgent(backend).run(goal=place_goal())
    assert result["accepted"]
    assert result["trace"][0]["repairs"] == 1
    assert result["metrics"]["actions"] == 2


def test_active_view_executes_one_cost_aware_observation():
    config = AgentConfig(use_active_view=True)
    uncertainties = [{
        "subject": "Cup", "predicate": "visible_from", "confidence": 0.3,
        "reason": "occlusion",
    }]
    result = VeriGraph3DAgent(MemoryBackend(demo_scene()), config=config).run(
        goal=GoalSpec(), actions=[], uncertainties=uncertainties,
    )
    assert result["accepted"]
    assert result["metrics"]["observations"] == 1
    assert result["metrics"]["observation_information_gain"] > 0
    assert result["metrics"]["observation_cost"] == config.observation_cost
    assert result["trace"][0]["type"] == "active_observation"


def test_active_view_skips_observation_when_cost_exceeds_gain():
    config = AgentConfig(use_active_view=True, observation_cost=2.0)
    result = VeriGraph3DAgent(MemoryBackend(demo_scene()), config=config).run(
        goal=GoalSpec(), actions=[],
        uncertainties=[{"subject": "Cup", "predicate": "visible_from"}],
    )
    assert result["metrics"]["observations"] == 0
