"""Clinical explainability reporting and documentation generation."""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from kidneycare.utils.file_utils import ensure_dir, atomic_write_text


class ExplainabilityReporter:
    """Generates structured clinical explanation reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def generate_report(
        self,
        model_name: str,
        importance_df: pd.DataFrame,
        top_k: int = 10,
    ) -> Path:
        """Write explanation report in Markdown."""
        top_features = importance_df.head(top_k)

        lines = [
            f"# Clinical Model Interpretability Report: {model_name}",
            "",
            "> **CLINICAL NOTICE:** Feature importance rankings reflect associative statistical "
            "weights learned from the training distribution. They do **not** establish biological causality "
            "or direct pathogenic mechanisms.",
            "",
            f"## Top {top_k} Discriminative Biomarkers",
            "",
            "| Rank | Feature Name | Attribution Score | Clinical Interpretation |",
            "| :--- | :--- | :--- | :--- |",
        ]

        # Clinical descriptions for well-known features
        descriptions = {
            "sc": "Serum Creatinine: Primary biomarker of glomerular filtration filtration decline.",
            "hemo": "Hemoglobin: Anemia of chronic kidney disease due to impaired erythropoietin production.",
            "pcv": "Packed Cell Volume (Hematocrit): Closely tied to chronic anemia and renal disease progression.",
            "al": "Albuminuria: Glomerular barrier permeability defect and proteinuria indicator.",
            "sg": "Specific Gravity: Loss of tubular concentrating ability (isosthenuria).",
            "bgr": "Blood Glucose Random: Indicator of diabetic nephropathy etiology.",
            "bu": "Blood Urea: Retention of nitrogenous waste metabolites.",
            "egfr_mdrd": "Estimated GFR (MDRD): Standard guideline filtration metric.",
            "htn": "Hypertension: Key etiology and driver of secondary nephrosclerosis.",
            "dm": "Diabetes Mellitus: Leading worldwide contributor to end-stage renal disease.",
        }

        for idx, row in top_features.iterrows():
            feat = row["feature"]
            score = row.get("importance", row.get("mean_importance", 0.0))
            desc = descriptions.get(feat, "Clinical laboratory or observational biomarker.")
            lines.append(f"| {idx + 1} | `{feat}` | {score:.4f} | {desc} |")

        lines.extend([
            "",
            "## Methodological Considerations",
            "- **Collinearity**: Correlated features (e.g. Hemoglobin and PCV) can share feature importance across tree splits.",
            "- **Population Specificity**: Attributions reflect cohort demographics (Apollo Hospitals dataset, India) and may vary in diverse global populations.",
        ])

        out_path = self.output_dir / f"explainability_{model_name.lower().replace('-', '_')}.md"
        atomic_write_text(out_path, "\n".join(lines))
        return out_path
