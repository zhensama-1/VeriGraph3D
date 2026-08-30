from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import TaskDataset
from .experiments import ABLATIONS, ACTIVE_VIEW_EXPERIMENTS, ExperimentRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible VeriGraph3D ablation experiments")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_results.json"))
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-markdown", type=Path)
    parser.add_argument("--category-csv", type=Path)
    parser.add_argument("--active-view", action="store_true", help="Also run the active-view comparison")
    args = parser.parse_args(argv)
    runner = ExperimentRunner(TaskDataset.load(args.dataset))
    variants = ABLATIONS + ACTIVE_VIEW_EXPERIMENTS if args.active_view else ABLATIONS
    result = runner.run(variants)
    runner.save(result, args.output)
    runner.save_summary_csv(
        result, args.summary_csv or args.output.with_suffix(".summary.csv")
    )
    runner.save_summary_markdown(
        result, args.summary_markdown or args.output.with_suffix(".summary.md")
    )
    runner.save_category_csv(
        result, args.category_csv or args.output.with_suffix(".categories.csv")
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
