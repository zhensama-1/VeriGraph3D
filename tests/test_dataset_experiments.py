from pathlib import Path

from verigraph3d.dataset import TaskDataset
from verigraph3d.experiments import ExperimentRunner, ExperimentVariant


DATASET = Path(__file__).parents[1] / "examples" / "tasks.json"
EXPANDED_DATASET = Path(__file__).parents[1] / "examples" / "tasks_expanded.json"


def test_dataset_loads_scene_and_goal():
    dataset = TaskDataset.load(DATASET)
    task = dataset.tasks[0]
    state = dataset.initial_state(task)
    assert task.id == "place_and_color_001"
    assert set(state.objects) == {"Cup", "Table"}
    assert task.goal.required[0].predicate == "on_top_of"


def test_full_ablation_variant_succeeds():
    runner = ExperimentRunner(TaskDataset.load(DATASET))
    result = runner.run([ExperimentVariant("full")])
    assert result["summary"]["full"]["task_success_rate"] == 1.0
    assert result["runs"][0]["metrics"]["actions"] == 2


def test_without_deterministic_verifier_exposes_visual_only_false_positive():
    runner = ExperimentRunner(TaskDataset.load(DATASET))
    variant = ExperimentVariant("visual_only", False, False, False, False)
    result = runner.run([variant])
    assert result["runs"][0]["accepted"]
    assert not result["runs"][0]["ground_truth_success"]
    assert result["summary"]["visual_only"]["false_acceptance_rate"] > 0.0
    cup = result["runs"][0]["final_state"]["objects"]["Cup"]
    assert cup["location"] == (2, 0, 0.5)


def test_fault_injection_reports_attribution_and_repair_metrics():
    result = ExperimentRunner(TaskDataset.load(DATASET)).run([
        ExperimentVariant("constraint_only", True, True, True, False),
        ExperimentVariant("verigraph3d_full", True, True, True, True),
    ])

    full = result["summary"]["verigraph3d_full"]
    assert full["failure_attribution_accuracy"] == 1.0
    assert full["local_repair_success_rate"] == 1.0
    assert full["mean_repairs"] > 0
    comparison = result["comparisons"]["constraint_only"]
    assert comparison["success_rate_delta"] > 0
    assert 0 <= comparison["mcnemar_exact_p"] <= 1


def test_paper_tables_are_exported(tmp_path):
    runner = ExperimentRunner(TaskDataset.load(DATASET))
    result = runner.run([ExperimentVariant("verigraph3d_full")])
    csv_path, markdown_path = tmp_path / "summary.csv", tmp_path / "summary.md"
    category_path = tmp_path / "categories.csv"
    runner.save_summary_csv(result, csv_path)
    runner.save_summary_markdown(result, markdown_path)
    runner.save_category_csv(result, category_path)

    assert "task_success_rate" in csv_path.read_text(encoding="utf-8-sig")
    assert "| Method | Success |" in markdown_path.read_text(encoding="utf-8")
    assert "variant,category" in category_path.read_text(encoding="utf-8-sig")


def test_expanded_dataset_covers_ten_task_categories():
    dataset = TaskDataset.load(EXPANDED_DATASET)
    categories = {task.metadata["category"] for task in dataset.tasks}

    assert len(dataset.tasks) == 64
    assert categories == {
        "move", "rotate", "scale", "place_on", "material", "camera", "light",
        "create", "delete", "compound",
    }
    assert sum(bool(task.actions) for task in dataset.tasks) == 10
    create = next(task for task in dataset.tasks if task.id == "create_001")
    assert create.actions[0].type.value == "create_object"


def test_expanded_dataset_full_variant_passes_all_tasks():
    result = ExperimentRunner(TaskDataset.load(EXPANDED_DATASET)).run([
        ExperimentVariant("verigraph3d_full")
    ])

    assert result["summary"]["verigraph3d_full"]["tasks"] == 64
    assert result["summary"]["verigraph3d_full"]["task_success_rate"] == 1.0
    assert result["category_summary"]["verigraph3d_full"]["compound"]["tasks"] == 4
    assert not [run for run in result["runs"] if not run["ground_truth_success"]]
