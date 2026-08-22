"""Entrenamiento XGBoost y regla champion/challenger."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from hvfhs.constants import BEST_XGB_PARAMS


def squeeze_target(y):
    """El ETL guarda y como DataFrame de una columna."""
    if hasattr(y, "squeeze"):
        return y.squeeze()
    return np.asarray(y).ravel()


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_xgb(X_train, y_train, params: dict | None = None) -> XGBRegressor:
    model = XGBRegressor(**(params or BEST_XGB_PARAMS))
    model.fit(X_train, squeeze_target(y_train))
    return model


def should_promote(challenger_r2: float, champion_r2: float | None) -> bool:
    """Promueve si no hay champion o si el challenger es igual o mejor en R²."""
    if champion_r2 is None:
        return True
    return challenger_r2 >= champion_r2
