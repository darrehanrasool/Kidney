"""Model comparison table generation and export."""

import pandas as pd
from pathlib import Path


def generate_comparison_report(eval_df: pd.DataFrame, output_path: Path = None) -> str:
    """Formats benchmark results into publication Markdown table."""
    markdown_table = eval_df.to_markdown(index=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# Kidney-AI Benchmark Comparison Matrix\n\n")
            f.write(markdown_table)
            f.write("\n")
    return markdown_table
