#!/usr/bin/env python3
"""Evaluation benchmark runner generating test performance reports."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.evaluation.evaluation import Evaluator
from kidney_ai.evaluation.model_comparison import generate_comparison_report


def main():
    evaluator = Evaluator()
    eval_df = evaluator.evaluate_all()

    print("=" * 70)
    print("             KIDNEY-AI SOTA BENCHMARK EVALUATION                ")
    print("=" * 70)
    print(eval_df.to_string(index=False))
    print("=" * 70)

    report_path = PROJECT_ROOT / "reports" / "evaluation" / "benchmark_summary.md"
    generate_comparison_report(eval_df, report_path)
    csv_path = PROJECT_ROOT / "reports" / "evaluation" / "model_benchmark_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(csv_path, index=False)
    print(f"Benchmark summary written to {report_path}")


if __name__ == "__main__":
    main()
