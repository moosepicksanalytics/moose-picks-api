"""
Model evaluation metrics for sports betting models.
Includes log loss, Brier score, calibration (ECE), and ROI metrics.
"""
from typing import Dict, Optional, Tuple
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
from sklearn.calibration import calibration_curve
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def evaluate_classification_model(
    y_true,
    y_pred_proba,
    y_pred = None
) -> Dict[str, float]:
    """
    Evaluate a classification model (moneyline, spread, totals).
    
    Args:
        y_true: True binary labels (pandas Series or numpy array)
        y_pred_proba: Predicted probabilities (pandas Series or numpy array)
        y_pred: Predicted binary labels (if None, uses 0.5 threshold)
    
    Returns:
        Dict with evaluation metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true_arr = y_true.values
    else:
        y_true_arr = np.asarray(y_true)
    
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba_arr = y_pred_proba.values
    else:
        y_pred_proba_arr = np.asarray(y_pred_proba)
    
    if y_pred is None:
        y_pred_arr = (y_pred_proba_arr >= 0.5).astype(int)
    else:
        if isinstance(y_pred, pd.Series):
            y_pred_arr = y_pred.values
        else:
            y_pred_arr = np.asarray(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_proba_arr))
    y_true_clean = y_true_arr[mask]
    y_pred_proba_clean = y_pred_proba_arr[mask]
    y_pred_clean = y_pred_arr[mask]
    
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
    unique_classes = np.unique(y_true_clean)
    if len(unique_classes) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_clean, y_pred_proba_clean)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    
    return metrics


def evaluate_regression_model(
    y_true,
    y_pred
) -> Dict[str, float]:
    """
    Evaluate a regression model (score projection).
    
    Args:
        y_true: True values (pandas Series or numpy array)
        y_pred: Predicted values (pandas Series or numpy array)
    
    Returns:
        Dict with evaluation metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true_arr = y_true.values
    else:
        y_true_arr = np.asarray(y_true)
    
    if isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values
    else:
        y_pred_arr = np.asarray(y_pred)
    
    # Handle NaN values
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_clean = y_true_arr[mask]
    y_pred_clean = y_pred_arr[mask]
    
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


def calculate_ece(
    y_true,
    y_pred_proba,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures how well-calibrated predicted probabilities are.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        y_true: True binary labels (pandas Series or numpy array)
        y_pred_proba: Predicted probabilities (pandas Series or numpy array)
        n_bins: Number of bins for calibration (default 10)
    
    Returns:
        ECE value (0-1)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values
    
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_true_clean = y_true[mask]
    y_pred_proba_clean = y_pred_proba[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    # Bin predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_proba_clean > bin_lower) & (y_pred_proba_clean <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy in this bin
            accuracy_in_bin = y_true_clean[in_bin].mean()
            # Average predicted probability in this bin
            avg_confidence_in_bin = y_pred_proba_clean[in_bin].mean()
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_calibration(
    y_true,
    y_pred_proba,
    n_bins: int = 10
) -> Dict:
    """
    Evaluate probability calibration with comprehensive metrics.
    
    Args:
        y_true: True binary labels (pandas Series or numpy array)
        y_pred_proba: Predicted probabilities (pandas Series or numpy array)
        n_bins: Number of bins for calibration
    
    Returns:
        Dict with calibration metrics (ECE, Brier Score, Log Loss)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true_arr = y_true.values
    else:
        y_true_arr = np.asarray(y_true)
    
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba_arr = y_pred_proba.values
    else:
        y_pred_proba_arr = np.asarray(y_pred_proba)
    
    # Handle NaN values
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_proba_arr))
    y_true_clean = y_true_arr[mask]
    y_pred_proba_clean = y_pred_proba_arr[mask]
    
    if len(y_true_clean) == 0:
        return {
            "ece": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan
        }
    
    return {
        "ece": calculate_ece(y_true_clean, y_pred_proba_clean, n_bins),
        "brier_score": brier_score_loss(y_true_clean, y_pred_proba_clean),
        "log_loss": log_loss(y_true_clean, y_pred_proba_clean),
    }


def plot_calibration_curve(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model (for plot title)
        n_bins: Number of bins for calibration
        save_path: Optional path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Skipping calibration plot.")
        print("   Install with: pip install matplotlib")
        return
    mask = ~(pd.isna(y_true) | pd.isna(y_pred_proba))
    y_true_clean = y_true[mask].values
    y_pred_proba_clean = y_pred_proba[mask].values
    
    if len(y_true_clean) == 0:
        return
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_clean, y_pred_proba_clean, n_bins=n_bins, strategy='uniform'
    )
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


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
    y_true,
    y_pred_proba,
    y_pred = None,
    model_type: str = "classification",
    edges = None,
    odds = None
) -> Dict:
    """
    Comprehensive model evaluation with priority on ROI and calibration.
    
    Priority order:
    1. ROI (most important for profitability)
    2. Calibration (ECE) - essential for bankroll management
    3. Value bet accuracy - accuracy on high-confidence edge bets
    4. Overall accuracy - standard metric but less important
    5. Precision/recall - standard classification metrics
    
    Args:
        y_true: True labels/values (pandas Series or numpy array)
        y_pred_proba: Predicted probabilities (pandas Series or numpy array)
        y_pred: Predicted labels/values (pandas Series or numpy array, optional)
        model_type: "classification" or "regression"
        edges: Calculated edges (for ROI analysis, pandas Series or numpy array)
        odds: American odds (for ROI analysis, pandas Series or numpy array)
    
    Returns:
        Dict with all evaluation metrics (prioritized)
    """
    # Convert to pandas Series for easier indexing (if needed)
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred_proba, pd.Series):
        y_pred_proba = pd.Series(y_pred_proba)
    if y_pred is not None and not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)
    
    results = {}
    
    if model_type == "classification":
        # Standard classification metrics
        class_metrics = evaluate_classification_model(y_true, y_pred_proba, y_pred)
        results.update(class_metrics)
        
        # Calibration metrics (HIGH PRIORITY)
        cal_metrics = evaluate_calibration(y_true, y_pred_proba)
        results.update(cal_metrics)
        
        # Value bet accuracy (filter for high edge bets)
        if edges is not None:
            if not isinstance(edges, pd.Series):
                edges = pd.Series(edges)
            value_bet_mask = edges >= 0.05  # 5% edge threshold
            if value_bet_mask.sum() > 0:
                value_y_true = y_true[value_bet_mask]
                value_y_pred_proba = y_pred_proba[value_bet_mask]
                value_y_pred = (value_y_pred_proba >= 0.5).astype(int) if y_pred is None else y_pred[value_bet_mask]
                value_accuracy = accuracy_score(value_y_true, value_y_pred)
                results["value_bet_accuracy"] = value_accuracy
                results["value_bet_count"] = int(value_bet_mask.sum())
            else:
                results["value_bet_accuracy"] = np.nan
                results["value_bet_count"] = 0
        
        # ROI analysis if edges and odds provided
        if edges is not None and odds is not None:
            if not isinstance(edges, pd.Series):
                edges = pd.Series(edges)
            if not isinstance(odds, pd.Series):
                odds = pd.Series(odds)
            roi_by_bucket = calculate_roi_by_edge_bucket(y_true, y_pred_proba, edges, odds)
            results["roi_by_bucket"] = roi_by_bucket
    else:
        # Regression metrics
        if y_pred is None:
            y_pred = y_pred_proba
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
        results.update(evaluate_regression_model(y_true, y_pred))
    
    return results
