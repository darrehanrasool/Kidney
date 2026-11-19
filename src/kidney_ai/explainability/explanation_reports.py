"""Generates human-readable clinical case attribution reports."""

from typing import Dict, List, Any
import pandas as pd


def generate_patient_risk_report(patient_id: str, risk_score: float, top_contributors: List[Dict[str, Any]]) -> str:
    """Formats patient individual diagnostic explanation into clinical memo."""
    lines = [
        f"=== Kidney-AI Clinical Decision Support Memo ===",
        f"Patient Identifier: {patient_id}",
        f"Predicted CKD Progression Risk: {risk_score * 100:.1f}%",
        f"Risk Category: {'HIGH' if risk_score > 0.5 else 'LOW/MONITOR'}",
        "\nKey Biomarker Contributing Factors:"
    ]
    for c in top_contributors:
        lines.append(f"  • {c['feature']}: impact = {c['impact']:+.3f}")
    return "\n".join(lines)
