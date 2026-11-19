"""Main CLI entrypoint for Kidney-AI Clinical Research Platform."""

import argparse
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from kidney_ai.core.config import load_config
from kidney_ai.core.reproducibility import set_seed
from kidney_ai.monitoring.logging_config import setup_logging
from kidney_ai.hardware.hardware_detector import detect_hardware
from kidney_ai.data.dataset_downloader import DatasetDownloader
from kidney_ai.data.data_validator import DataValidator
from kidney_ai.data.data_splitting import DataSplitter
from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.training.train import train_all_models
from kidney_ai.evaluation.evaluation import Evaluator
from kidney_ai.visualization.visualization import Visualizer

logger = logging.getLogger("kidney_ai")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kidney-AI: Advanced Deep Learning & Machine Learning Framework for Nephrology Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=False, help="Subcommand to execute")

    # Command: verify-hardware
    subparsers.add_parser("verify-hardware", help="Inspect and display compute accelerator hardware (MPS / CUDA / CPU)")

    # Command: download
    dl_parser = subparsers.add_parser("download", help="Download or synthesize verified clinical benchmark cohorts")
    dl_parser.add_argument("--mode", type=str, choices=["sample", "full"], default="sample")

    # Command: verify-datasets
    subparsers.add_parser("verify-datasets", help="Audit integrity, schema, and biological bounds of raw datasets")

    # Command: prepare
    subparsers.add_parser("prepare", help="Validate, engineer renal features, and create train/val/test splits")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train baseline classical and deep learning models")
    train_parser.add_argument("--config", type=str, default="configs/training.yaml")

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Benchmark all trained models on holdout test set")

    # Command: visualize
    subparsers.add_parser("visualize", help="Generate all publication-quality ROC, PR, and feature importance plots")

    # Command: analyze-data
    subparsers.add_parser("analyze-data", help="Run comprehensive data exploration, quality audits, and missingness checks")

    # Command: pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run end-to-end clinical risk prediction on sample input")
    pipe_parser.add_argument("--input", type=str, default="sample", help="Input clinical CSV path or 'sample'")

    return parser.parse_args(), parser


def print_welcome_guide(parser):
    print("=" * 80)
    print("        KIDNEY-AI: CLINICAL RESEARCH & DEEP LEARNING PLATFORM                  ")
    print("        Centre of Artificial Intelligence (CAI), IUST, Awantipora              ")
    print("=" * 80)
    print("Quick Start Commands:")
    print("   • Audit Hardware Acceleration: python src/kidney_ai/main.py verify-hardware")
    print("   • Download / Prep Cohort:      python src/kidney_ai/main.py download")
    print("   • Prepare & Split Data:        python src/kidney_ai/main.py prepare")
    print("   • Train Baseline Models:       python src/kidney_ai/main.py train")
    print("   • Evaluate Test Sets:          python src/kidney_ai/main.py evaluate")
    print("   • Generate Visualizations:     python src/kidney_ai/main.py visualize")
    print("=" * 80)
    print("\nAvailable Subcommands:\n")
    parser.print_help()


def main():
    setup_logging()
    args, parser = parse_args()

    if not args.command:
        print_welcome_guide(parser)
        sys.exit(0)

    cfg = load_config()
    set_seed(cfg.get("environment.seed", 42))

    if args.command == "verify-hardware":
        profile = detect_hardware()
        print("================================================================================")
        print("                     KIDNEY-AI COMPUTE HARDWARE AUDIT                           ")
        print("================================================================================")
        print(f"OS: {profile.os_name} {profile.os_version} ({profile.cpu_architecture})")
        print(f"CPUs: {profile.cpu_count_physical} Physical, {profile.cpu_count_logical} Logical")
        print(f"RAM: {profile.total_ram_gb} GB Total, {profile.available_ram_gb} GB Available")
        print(f"NVIDIA CUDA: {profile.has_cuda} (Device: {profile.cuda_device_name})")
        print(f"Apple Silicon MPS: {profile.has_mps}")
        print(f"Recommended Compute Device: '{profile.recommended_device}'")
        print("================================================================================")

    elif args.command == "download":
        downloader = DatasetDownloader()
        downloader.download_all(mode=args.mode)
        print(f"Clinical cohort generated successfully in {downloader.raw_dir}")

    elif args.command == "verify-datasets":
        from scripts.verify_datasets import main as run_verify
        run_verify()

    elif args.command == "prepare":
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    elif args.command == "train":
        train_all_models()
        print("All models successfully trained and serialized to outputs/trained_models/")

    elif args.command == "evaluate":
        from scripts.evaluate_all import main as run_eval
        run_eval()

    elif args.command == "visualize":
        from scripts.generate_visualizations import main as run_viz
        run_viz()

    elif args.command == "analyze-data":
        from kidney_ai.analysis.statistical_analyzer import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        summary = analyzer.run_full_audit()
        print("================================================================================")
        print("               CLINICAL DATA QUALITY AUDIT COMPLETED                            ")
        print("================================================================================")
        print(f"Total Records: {summary.get('total_records')}")
        print(f"Overall Completeness: {summary.get('overall_completeness_pct')}%")
        print("Audit report written to: reports/data_quality/data_quality_report.md")
        print("================================================================================")

    elif args.command == "pipeline":
        from scripts.run_unified_pipeline import main as run_pipe
        run_pipe()


if __name__ == "__main__":
    main()
