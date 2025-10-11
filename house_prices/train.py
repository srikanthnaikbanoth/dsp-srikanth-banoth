from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

from house_prices.preprocess import (
    split_features_target,
    get_column_groups,
    create_preprocessors,
    fit_preprocessors,
    transform_features,
)

MODELS_DIR = Path("models")


def _persist_artifacts(
    model,
    pp: Dict[str, object],
    num_cols: List[str],
    cat_cols: List[str],
    feature_order: List[str],
) -> None:
    """Save trained model and preprocessors."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(pp["num_imputer"], MODELS_DIR / "num_imputer.joblib")
    joblib.dump(pp["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(pp["cat_imputer"], MODELS_DIR / "cat_imputer.joblib")
    joblib.dump(pp["ohe"], MODELS_DIR / "ohe.joblib")
    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_order": feature_order,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta))


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """Train model, evaluate, and persist artifacts."""
    X_full, y_full = split_features_target(data, "SalePrice")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    num_cols, cat_cols = get_column_groups(X_tr)
    pp = fit_preprocessors(
        X_tr,
        num_cols,
        cat_cols,
        create_preprocessors(),
    )

    Xtr_proc = transform_features(X_tr, num_cols, cat_cols, pp)
    Xte_proc = transform_features(X_te, num_cols, cat_cols, pp)

    model = RandomForestRegressor(
        random_state=42,
        n_estimators=400,
    )
    model.fit(Xtr_proc, y_tr)

    y_pred = model.predict(Xte_proc)
    rmsle = float(np.sqrt(mean_squared_log_error(y_te, y_pred)))

    _persist_artifacts(
        model,
        pp,
        num_cols,
        cat_cols,
        Xtr_proc.columns.tolist(),
    )

    return {"rmsle": rmsle}
