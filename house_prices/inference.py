from __future__ import annotations
from typing import Dict
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from house_prices.preprocess import transform_features

MODELS_DIR = Path("models")

def _load_artifacts() -> Dict[str, object]:
    model = joblib.load(MODELS_DIR / "model.joblib")
    num_imp = joblib.load(MODELS_DIR / "num_imputer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    cat_imp = joblib.load(MODELS_DIR / "cat_imputer.joblib")
    ohe = joblib.load(MODELS_DIR / "ohe.joblib")
    meta_path = MODELS_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text())
    return {
        "model": model,
        "num_imputer": num_imp,
        "scaler": scaler,
        "cat_imputer": cat_imp,
        "ohe": ohe,
        "num_cols": meta["num_cols"],
        "cat_cols": meta["cat_cols"],
        "feature_order": meta["feature_order"],
    }

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    art = _load_artifacts()
    pp = {
        "num_imputer": art["num_imputer"],
        "scaler": art["scaler"],
        "cat_imputer": art["cat_imputer"],
        "ohe": art["ohe"],
    }
    X_proc = transform_features(
        input_data,
        art["num_cols"],
        art["cat_cols"],
        pp,
    )
    X_proc = X_proc.reindex(
        columns=art["feature_order"],
        fill_value=0.0,
    )
    preds = art["model"].predict(X_proc)
    return preds
