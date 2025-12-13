"""
Recalibrate model confidence using Platt scaling.

This script:
1. Loads a trained model
2. Splits data into training, calibration, and test sets
3. Applies Platt scaling (logistic regression calibration)
4. Tests calibration on test set
5. Saves calibrated model
6. Reports ECE improvement
"""
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Game
from app.data_loader import load_games_for_sport
from app.training.features import build_features, get_feature_columns
from app.training.pipeline import prepare_labels
from app.training.evaluate import calculate_ece
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> dict:
    """Load a trained model from file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    return model_data


def prepare_calibration_data(
    sport: str,
    market: str,
    model_data: dict,
    test_weeks: int = 2
) -> tuple:
    """
    Prepare data for calibration.
    
    Splits data into:
    - Training: all data before last (test_weeks + 2) weeks
    - Calibration: last (test_weeks + 2) to test_weeks weeks
    - Test: last test_weeks weeks (unseen during calibration)
    
    Returns:
        (X_train, y_train, X_cal, y_cal, X_test, y_test, feature_cols)
    """
    logger.info(f"Loading game data for {sport} {market}")
    
    # Load all games (load_games_for_sport already filters for final games with scores)
    games_df = load_games_for_sport(sport, status='final')
    
    if len(games_df) == 0:
        raise ValueError(f"No games found for {sport}")
    
    # Ensure we have scores (should already be filtered, but double-check)
    games_df = games_df[
        (games_df['home_score'].notna()) &
        (games_df['away_score'].notna())
    ].copy()
    
    if len(games_df) == 0:
        raise ValueError(f"No completed games found for {sport}")
    
    logger.info(f"Found {len(games_df)} completed games")
    
    # Sort by date
    games_df = games_df.sort_values('date')
    
    # Split by date
    cutoff_date = games_df['date'].max() - timedelta(weeks=test_weeks)
    cal_cutoff_date = cutoff_date - timedelta(weeks=2)
    
    train_df = games_df[games_df['date'] < cal_cutoff_date].copy()
    cal_df = games_df[
        (games_df['date'] >= cal_cutoff_date) &
        (games_df['date'] < cutoff_date)
    ].copy()
    test_df = games_df[games_df['date'] >= cutoff_date].copy()
    
    logger.info(f"Split: {len(train_df)} train, {len(cal_df)} calibration, {len(test_df)} test")
    
    if len(cal_df) == 0:
        raise ValueError("No calibration data available (need at least 2 weeks)")
    if len(test_df) == 0:
        raise ValueError("No test data available (need at least 2 weeks)")
    
    # Build features
    logger.info("Building features...")
    feature_cols = get_feature_columns(sport, market)
    
    train_features = build_features(train_df, sport, market)
    cal_features = build_features(cal_df, sport, market)
    test_features = build_features(test_df, sport, market)
    
    # Prepare labels
    train_labels = prepare_labels(train_df, sport, market)
    cal_labels = prepare_labels(cal_df, sport, market)
    test_labels = prepare_labels(test_df, sport, market)
    
    # Filter out NaN labels
    train_mask = train_labels.notna()
    cal_mask = cal_labels.notna()
    test_mask = test_labels.notna()
    
    X_train = train_features[train_mask]
    y_train = train_labels[train_mask].astype(int)
    
    X_cal = cal_features[cal_mask]
    y_cal = cal_labels[cal_mask].astype(int)
    
    X_test = test_features[test_mask]
    y_test = test_labels[test_mask].astype(int)
    
    # Ensure feature columns match
    available_cols = model_data.get('feature_columns', feature_cols)
    
    # Align columns - ensure all dataframes have the same columns
    all_cols = set(available_cols)
    
    # Add missing columns and reorder
    missing_cols_train = all_cols - set(X_train.columns)
    for col in missing_cols_train:
        X_train[col] = 0.0
    X_train = X_train[[col for col in available_cols if col in X_train.columns]]
    
    missing_cols_cal = all_cols - set(X_cal.columns)
    for col in missing_cols_cal:
        X_cal[col] = 0.0
    X_cal = X_cal[[col for col in available_cols if col in X_cal.columns]]
    
    missing_cols_test = all_cols - set(X_test.columns)
    for col in missing_cols_test:
        X_test[col] = 0.0
    X_test = X_test[[col for col in available_cols if col in X_test.columns]]
    
    logger.info(f"Final split: {len(X_train)} train, {len(X_cal)} calibration, {len(X_test)} test")
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test, available_cols


def apply_platt_scaling(base_model, X_cal, y_cal):
    """
    Apply Platt scaling to calibrate model probabilities.
    
    Platt scaling fits a logistic regression to map raw probabilities to calibrated probabilities.
    """
    logger.info("Applying Platt scaling...")
    
    # Get raw probabilities from base model
    raw_proba = base_model.predict_proba(X_cal)[:, 1]
    
    # Fit logistic regression on raw probabilities
    # Use cross-validation to avoid overfitting
    calibrator = CalibratedClassifierCV(
        base_estimator=LogisticRegression(),
        method='sigmoid',  # Platt scaling
        cv=5
    )
    
    # Reshape for sklearn
    raw_proba_reshaped = raw_proba.reshape(-1, 1)
    
    # Fit calibrator
    calibrator.fit(raw_proba_reshaped, y_cal)
    
    return calibrator


def evaluate_calibration(model, X, y, name: str = "Model"):
    """Evaluate model calibration and return metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    ece = calculate_ece(y.values, y_pred_proba, n_bins=10)
    
    # Calculate accuracy
    y_pred = (y_pred_proba >= 0.5).astype(int)
    accuracy = (y_pred == y.values).mean()
    
    # Calculate average confidence
    avg_confidence = y_pred_proba.mean()
    
    logger.info(f"{name} - ECE: {ece:.4f}, Accuracy: {accuracy:.3f}, Avg Confidence: {avg_confidence:.3f}")
    
    return {
        'ece': ece,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': y_pred_proba
    }


def create_calibrated_wrapper(base_model, calibrator):
    """Create a wrapper that applies calibration to predictions."""
    class CalibratedModel:
        def __init__(self, base_model, calibrator):
            self.base_model = base_model
            self.calibrator = calibrator
        
        def predict_proba(self, X):
            """Get calibrated probabilities."""
            raw_proba = self.base_model.predict_proba(X)[:, 1]
            raw_proba_reshaped = raw_proba.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(raw_proba_reshaped)[:, 1]
            # Ensure probabilities are in [0, 1] range
            calibrated = np.clip(calibrated, 0.0, 1.0)
            # Return in sklearn format [prob_class_0, prob_class_1]
            return np.column_stack([1 - calibrated, calibrated])
        
        def predict(self, X):
            """Get binary predictions."""
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
    
    return CalibratedModel(base_model, calibrator)


def main():
    parser = argparse.ArgumentParser(description='Recalibrate model using Platt scaling')
    parser.add_argument('--sport', type=str, required=True, help='Sport code (NFL, NHL, NBA, MLB)')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pkl)')
    parser.add_argument('--test-weeks', type=int, default=2, help='Number of weeks to use for test set')
    
    args = parser.parse_args()
    sport = args.sport.upper()
    model_path = args.model
    
    try:
        # Load model
        model_data = load_model(model_path)
        base_model = model_data['model']
        market = model_data.get('market', 'unknown')
        
        logger.info(f"Model: {sport} {market}")
        logger.info(f"Algorithm: {model_data.get('algorithm', 'unknown')}")
        
        # Prepare data
        X_train, y_train, X_cal, y_cal, X_test, y_test, feature_cols = prepare_calibration_data(
            sport, market, model_data, test_weeks=args.test_weeks
        )
        
        # Evaluate base model
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING BASE MODEL")
        logger.info("=" * 80)
        base_metrics = evaluate_calibration(base_model, X_test, y_test, "Base Model")
        
        # Apply Platt scaling
        logger.info("\n" + "=" * 80)
        logger.info("APPLYING PLATT SCALING")
        logger.info("=" * 80)
        calibrator = apply_platt_scaling(base_model, X_cal, y_cal)
        
        # Create calibrated model
        calibrated_model = create_calibrated_wrapper(base_model, calibrator)
        
        # Evaluate calibrated model
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING CALIBRATED MODEL")
        logger.info("=" * 80)
        calibrated_metrics = evaluate_calibration(calibrated_model, X_test, y_test, "Calibrated Model")
        
        # Print comparison
        print("\n" + "=" * 80)
        print("CALIBRATION COMPARISON")
        print("=" * 80)
        print(f"Before Calibration:")
        print(f"  ECE: {base_metrics['ece']:.4f}")
        print(f"  Accuracy: {base_metrics['accuracy']:.3f}")
        print(f"  Avg Confidence: {base_metrics['avg_confidence']:.3f}")
        print(f"\nAfter Calibration:")
        print(f"  ECE: {calibrated_metrics['ece']:.4f}")
        print(f"  Accuracy: {calibrated_metrics['accuracy']:.3f}")
        print(f"  Avg Confidence: {calibrated_metrics['avg_confidence']:.3f}")
        print(f"\nImprovement:")
        print(f"  ECE: {base_metrics['ece'] - calibrated_metrics['ece']:+.4f} ({'✓' if calibrated_metrics['ece'] < base_metrics['ece'] else '✗'})")
        print(f"  Confidence: {calibrated_metrics['avg_confidence'] - base_metrics['avg_confidence']:+.3f}")
        
        if calibrated_metrics['ece'] < 0.10:
            print(f"\n✓ Calibration is good (ECE < 0.10)")
        else:
            print(f"\n⚠️  Calibration still needs improvement (ECE >= 0.10)")
        
        # Save calibrated model
        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibrated_model_path = model_dir / f"{sport}_{market}_CALIBRATED_{timestamp}.pkl"
        
        calibrated_model_data = {
            **model_data,  # Keep all original metadata
            'model': calibrated_model,
            'calibrator': calibrator,
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'platt_scaling',
            'base_ece': float(base_metrics['ece']),
            'calibrated_ece': float(calibrated_metrics['ece']),
            'ece_improvement': float(base_metrics['ece'] - calibrated_metrics['ece']),
        }
        
        joblib.dump(calibrated_model_data, calibrated_model_path)
        logger.info(f"\n✓ Calibrated model saved to {calibrated_model_path}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

