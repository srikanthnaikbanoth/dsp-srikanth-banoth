from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split df into X (features) and y (target)."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def get_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical column lists."""
    num_cols = df.select_dtypes(include=["number"]) \
                 .columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]) \
                 .columns.tolist()
    return num_cols, cat_cols


def create_preprocessors() -> Dict[str, object]:
    """Create unfitted preprocessing objects."""
    try:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )
    except TypeError:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False,
        )
    return {
        "num_imputer": SimpleImputer(strategy="mean"),
        "scaler": StandardScaler(),
        "cat_imputer": SimpleImputer(strategy="most_frequent"),
        "ohe": ohe,
    }


def fit_preprocessors(
    X_train: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    pp: Dict[str, object],
) -> Dict[str, object]:
    """Fit preprocessors on training data."""
    pp["num_imputer"].fit(X_train[num_cols])
    Xn = pp["num_imputer"].transform(X_train[num_cols])
    pp["scaler"].fit(Xn)
    pp["cat_imputer"].fit(X_train[cat_cols])
    Xc_in = pp["cat_imputer"].transform(X_train[cat_cols])
    pp["ohe"].fit(Xc_in)
    return pp


def transform_features(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    pp: Dict[str, object],
) -> pd.DataFrame:
    """Transform features using fitted preprocessors."""
    Xn = pp["num_imputer"].transform(X[num_cols])
    Xn = pp["scaler"].transform(Xn)
    Xc_in = pp["cat_imputer"].transform(X[cat_cols])
    Xc = pp["ohe"].transform(Xc_in)
    cat_names = pp["ohe"].get_feature_names_out(
        cat_cols
    ).tolist()
    return pd.DataFrame(
        np.hstack([Xn, Xc]),
        columns=num_cols + cat_names,
        index=X.index,
    )
