"""Data acquisition, validation, quality auditing, preprocessing, and splitting."""

from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.data.dataset_registry import DatasetRegistry
from kidneycare.data.dataset_manifest import DatasetManifest

_LAZY_IMPORTS = {
    "DataValidator": "kidneycare.data.data_validator",
    "DataQualityAuditor": "kidneycare.data.data_quality",
    "Deduplicator": "kidneycare.data.deduplication",
    "MissingValueHandler": "kidneycare.data.missing_values",
    "ClinicalFeatureEngineer": "kidneycare.data.feature_engineering",
    "ClinicalPreprocessor": "kidneycare.data.preprocessing",
    "DataSplitter": "kidneycare.data.data_splitting",
}

__all__ = [
    "DatasetDownloader",
    "DatasetRegistry",
    "DatasetManifest",
    "DataValidator",
    "DataQualityAuditor",
    "Deduplicator",
    "MissingValueHandler",
    "ClinicalFeatureEngineer",
    "ClinicalPreprocessor",
    "DataSplitter",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
