from pathlib import Path

import pytest

from verigraph3d.abws_experiment_cli import select_tasks, summarize
from verigraph3d.dataset import TaskDataset


DATASET = Path(__file__).parents[1] / "examples" / "blender_tasks_expanded.json"


def test_real_blender_subset_has_24_stratified_tasks():
    dataset = TaskDataset.load(DATASET)
    selected = select_tasks(dataset, [], ["move", "create"], None)

    assert len(dataset.tasks) == 24
    assert len(selected) == 5
    assert {task.metadata["category"] for task in selected} == {"move", "create"}


def test_task_selection_rejects_unknown_ids():
    with pytest.raises(ValueError, match="Unknown"):
        select_tasks(TaskDataset.load(DATASET), ["missing"], [], None)


def test_real_blender_summary_is_category_stratified():
    summary = summarize([
        {"task_id": "a", "category": "move", "accepted": True,
         "elapsed_seconds": 2, "actions": 1},
        {"task_id": "b", "category": "move", "accepted": False,
         "elapsed_seconds": 4, "actions": 1},
    ])

    assert summary["task_success_rate"] == 0.5
    assert summary["categories"]["move"]["mean_seconds"] == 3
    assert summary["failed_task_ids"] == ["b"]
