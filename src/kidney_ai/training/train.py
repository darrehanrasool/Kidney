"""Training master runner across classical ML and deep learning model zoo."""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from kidney_ai.data.dataset_downloader import DatasetDownloader
from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.data.data_validator import DataValidator
from kidney_ai.data.preprocessing import Preprocessor
from kidney_ai.data.data_splitting import DataSplitter
from kidney_ai.models.model_factory import ModelFactory
from kidney_ai.models.ensemble import build_voting_ensemble
from kidney_ai.training.trainer import PyTorchTrainer
from kidney_ai.core.config import load_config


def train_all_models() -> Dict[str, Any]:
    """Prepares data and trains all baseline classical & neural architectures."""
    cfg = load_config()
    raw_path = Path("data/raw/clinical/chronic_kidney_disease.csv")
    if not raw_path.exists():
        downloader = DatasetDownloader()
        downloader.download_all()

    df = pd.read_csv(raw_path)
    validator = DataValidator()
    fe = FeatureEngineer()
    df_aug = fe.transform(df)
    y = validator.clean_target(df_aug["classification"])
    X_raw = df_aug.drop(columns=["classification"])

    num_cols = cfg.get("data.features.numerical", [])
    cat_cols = cfg.get("data.features.categorical", [])

    preprocessor = Preprocessor(numerical_features=num_cols, categorical_features=cat_cols)
    X = preprocessor.fit_transform(X_raw)

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    df_with_target = X_raw.copy()
    df_with_target["classification"] = y
    train_df, val_df, test_df = splitter.split(df_with_target)
    splitter.save_splits(train_df, val_df, test_df)

    y_train = train_df["classification"].values
    y_val = val_df["classification"].values
    y_test = test_df["classification"].values

    X_train = preprocessor.transform(train_df.drop(columns=["classification"]))
    X_val = preprocessor.transform(val_df.drop(columns=["classification"]))
    X_test = preprocessor.transform(test_df.drop(columns=["classification"]))

    # Save fitted preprocessor
    out_dir = Path("outputs/trained_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    trained_models = {}

    # 1. Classical Models
    for model_name in ["logistic_regression", "random_forest", "gradient_boosting", "svm"]:
        model = ModelFactory.create_model(model_name)
        model.fit(X_train, y_train)
        joblib.dump(model, out_dir / f"{model_name}.joblib")
        trained_models[model_name] = model

    # 2. Ensemble
    ensemble = build_voting_ensemble([
        ("rf", trained_models["random_forest"]),
        ("gb", trained_models["gradient_boosting"]),
        ("lr", trained_models["logistic_regression"])
    ])
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, out_dir / "voting_ensemble.joblib")
    trained_models["voting_ensemble"] = ensemble

    # 3. Deep Learning (Tabular ResNet)
    input_dim = X_train.shape[1]
    resnet = ModelFactory.create_model("tabular_resnet", {"input_dim": input_dim, "hidden_dim": 64, "num_blocks": 2})
    dl_trainer = PyTorchTrainer(resnet, epochs=15, batch_size=32)
    dl_trainer.fit(X_train, y_train, X_val, y_val, model_name="tabular_resnet")
    trained_models["tabular_resnet"] = resnet

    return trained_models
