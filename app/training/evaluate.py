"""
Model evaluation metrics for sports betting models.
Includes log loss, Brier score, calibration, and ROI metrics.
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


def evaluate_classification_model(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    y_pred: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Evaluate a classification model (moneyline, spread, totals).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted binary labels (if None, uses 0.5 threshold)
    
    Returns:
        Dict with evaluation metrics
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Remove NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred_proba))
    y_true_clean = y_true[mask]
    y_pred_proba_clean = y_pred_proba[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            "log_loss": np.nan,
            "brier_score": np.nan,
            "accuracy": np.nan,
            "roc_auc": np.nan,
        }
    
    metrics = {
        "log_loss": log_loss(y_true_clean, y_pred_proba_clean),
        "brier_score": brier_score_loss(y_true_clean, y_pred_proba_clean),
        "accuracy": accuracy_score(y_true_clean, y_pred_clean),
    }
    
    # ROC AUC (only if both classes present)
    if len(y_true_clean.unique()) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_clean, y_pred_proba_clean)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    
    return metrics


def evaluate_regression_model(
    y_true: pd.Series,
    y_pred: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a regression model (score projection).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dict with evaluation metrics
    """
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }
    
    return {
        "rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        "mae": mean_absolute_error(y_true_clean, y_pred_clean),
        "r2": 1 - (np.sum((y_true_clean - y_pred_clean) ** 2) / np.sum((y_true_clean - y_true_clean.mean()) ** 2)),
    }


def evaluate_calibration(y_true: pd.Series, y_pred_proba: pd.Series) -> Dict:
    """Evaluate probability calibration."""
    # Simplified calibration metrics
    mask = ~(pd.isna(y_true) | pd.isna(y_pred_proba))
    y_true_clean = y_true[mask]
    y_pred_proba_clean = y_pred_proba[mask]
    
    if len(y_true_clean) == 0:
        return {"calibration_error": np.nan}
    
    # Brier score is a calibration metric
    return {
        "calibration_error": brier_score_loss(y_true_clean, y_pred_proba_clean),
    }


def calculate_roi_by_edge_bucket(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    edges: pd.Series,
    odds: pd.Series
) -> pd.DataFrame:
    """Calculate ROI by edge bucket."""
    # Simplified ROI calculation
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "edge": edges,
        "odds": odds,
    })
    
    # Bucket by edge
    df["edge_bucket"] = pd.cut(df["edge"], bins=[0, 0.05, 0.10, 0.15, 1.0], labels=["0-5%", "5-10%", "10-15%", "15%+"])
    
    results = []
    for bucket in df["edge_bucket"].unique():
        if pd.isna(bucket):
            continue
        bucket_df = df[df["edge_bucket"] == bucket]
        if len(bucket_df) == 0:
            continue
        
        # Simplified ROI calculation
        wins = bucket_df["y_true"].sum()
        total = len(bucket_df)
        roi = (wins / total - 0.5) * 2 if total > 0 else 0
        
        results.append({
            "edge_bucket": bucket,
            "n_bets": total,
            "win_rate": wins / total if total > 0 else 0,
            "roi": roi,
        })
    
    return pd.DataFrame(results)


def evaluate_model_comprehensive(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    y_pred: Optional[pd.Series] = None,
    model_type: str = "classification",
    edges: Optional[pd.Series] = None,
    odds: Optional[pd.Series] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels/values
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels/values
        model_type: "classification" or "regression"
        edges: Calculated edges (for ROI analysis)
        odds: American odds (for ROI analysis)
    
    Returns:
        Dict with all evaluation metrics
    """
    results = {}
    
    if model_type == "classification":
        results.update(evaluate_classification_model(y_true, y_pred_proba, y_pred))
        results.update(evaluate_calibration(y_true, y_pred_proba))
        
        if edges is not None:
            roi_by_bucket = calculate_roi_by_edge_bucket(y_true, y_pred_proba, edges, odds)
            results["roi_by_bucket"] = roi_by_bucket
    else:
        if y_pred is None:
            y_pred = y_pred_proba
        results.update(evaluate_regression_model(y_true, y_pred))
    
    return results
