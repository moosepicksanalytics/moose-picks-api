"""
Model training pipeline for all market types.
Supports moneyline, spread/puck line, totals, and score projection models.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost (optional dependency)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from app.data_loader import (
    load_games_for_sport,
    split_by_season,
    split_by_week,
    split_random,
    get_season_date_range,
)
from app.training.features import build_features, get_feature_columns
from app.training.evaluate import evaluate_model_comprehensive
from app.config import settings


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return {}


def get_model_class(algorithm: str, model_type: str = "classification"):
    """
    Get the appropriate model class based on algorithm name.
    
    Args:
        algorithm: Algorithm name (e.g., "xGBoost", "xgboost", "gradient_boosting")
        model_type: "classification" or "regression"
    
    Returns:
        Model class
    """
    algorithm_lower = algorithm.lower().strip()
    
    # Normalize algorithm names
    if algorithm_lower in ["xgboost", "xgb"]:
        if not XGBOOST_AVAILABLE:
            print(f"Warning: XGBoost requested but not available. Falling back to GradientBoosting.")
            algorithm_lower = "gradient_boosting"
    
    if algorithm_lower in ["xgboost", "xgb"]:
        if model_type == "classification":
            return xgb.XGBClassifier
        else:
            return xgb.XGBRegressor
    elif algorithm_lower in ["gradient_boosting", "gradientboosting", "sklearn"]:
        if model_type == "classification":
            return GradientBoostingClassifier
        else:
            return GradientBoostingRegressor
    else:
        # Default to gradient boosting if unknown algorithm
        print(f"Warning: Unknown algorithm '{algorithm}'. Using GradientBoosting.")
        if model_type == "classification":
            return GradientBoostingClassifier
        else:
            return GradientBoostingRegressor


def prepare_labels(df: pd.DataFrame, sport: str, market: str) -> pd.Series:
    """
    Prepare labels for a given market.
    
    Args:
        df: DataFrame with game data
        sport: Sport code
        market: Market type
    
    Returns:
        Series with labels
    """
    if market == "moneyline":
        # 1 if home wins, 0 if away wins
        return (df["home_score"] > df["away_score"]).astype(int)
    
    elif market == "spread":
        # 1 if home covers spread, 0 otherwise
        if "spread" not in df.columns or df["spread"].isna().all():
            # If no spread data, use point differential
            return (df["home_score"] - df["away_score"] > 0).astype(int)
        return (df["home_score"] - df["away_score"] > df["spread"]).astype(int)
    
    elif market == "totals":
        # 1 if over, 0 if under
        if "over_under" not in df.columns or df["over_under"].isna().all():
            # Estimate total from averages - use windowed version (default is 10)
            # Try different window sizes, starting with the default
            home_col = None
            away_col = None
            for window in [10, 15, 5, 3]:  # Try most common windows first
                if f"home_points_for_avg_{window}" in df.columns:
                    home_col = f"home_points_for_avg_{window}"
                    away_col = f"away_points_for_avg_{window}"
                    break
            
            if home_col and away_col:
                total = df[home_col].fillna(0) + df[away_col].fillna(0)
            else:
                # Fallback: use actual scores if available, otherwise default to 0
                # This should rarely happen if features were built correctly
                total = pd.Series([0] * len(df))
                if "home_score" in df.columns and "away_score" in df.columns:
                    # Use a simple average if we have some score data
                    avg_home = df["home_score"].mean() if not df["home_score"].isna().all() else 0
                    avg_away = df["away_score"].mean() if not df["away_score"].isna().all() else 0
                    total = pd.Series([avg_home + avg_away] * len(df))
        else:
            total = df["over_under"]
        return (df["home_score"] + df["away_score"] > total).astype(int)
    
    else:
        raise ValueError(f"Unknown market type: {market}")


def train_model_for_market(
    sport: str,
    market: str,
    config: Optional[Dict] = None,
    model_version: Optional[str] = None
) -> Dict:
    """
    Train a model for a specific sport and market.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals, score_projection)
        config: Training configuration dict
        model_version: Version tag for model (if None, uses timestamp)
    
    Returns:
        Dict with training results and metrics
    """
    if config is None:
        config = load_config()
    
    # Load data
    training_seasons = config.get("training_seasons", {}).get(sport, [2020, 2021, 2022, 2023, 2024])
    min_games_per_team = config.get("min_games_per_team", 10)
    
    # Load all games for training seasons
    all_games = []
    for season in training_seasons:
        start_date, end_date = get_season_date_range(sport, season)
        season_games = load_games_for_sport(
            sport,
            start_date=start_date,
            end_date=end_date,
            min_games_per_team=min_games_per_team
        )
        if not season_games.empty:
            all_games.append(season_games)
    
    if not all_games:
        return {"success": False, "error": "No games found for training"}
    
    df = pd.concat(all_games, ignore_index=True)
    
    if len(df) < settings.SPORTS_CONFIG.get(sport, {}).get("min_training_games", 100):
        return {"success": False, "error": f"Not enough games: {len(df)}"}
    
    # Build features
    rolling_window = config.get("features", {}).get("rolling_window_games", 10)
    include_rest_days = config.get("features", {}).get("include_rest_days", True)
    include_h2h = config.get("features", {}).get("include_head_to_head", True)
    
    print(f"Building features for {sport} {market}...")
    df = build_features(
        df,
        sport=sport,
        market=market,
        rolling_window=rolling_window,
        include_rest_days=include_rest_days,
        include_h2h=include_h2h
    )
    
    # Prepare labels
    print(f"Preparing labels for {sport} {market}...")
    y = prepare_labels(df, sport, market)
    
    # Get feature columns
    feature_cols = get_feature_columns(sport, market)
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        return {"success": False, "error": "No features available"}
    
    X = df[available_cols].copy()
    
    # Remove rows with NaN in features or labels
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        return {"success": False, "error": "No valid samples after cleaning"}
    
    # Train/validation split
    split_strategy = config.get("split_strategy", "season")
    validation_seasons = config.get("validation_seasons", 1)
    
    if split_strategy == "season":
        train_df, val_df = split_by_season(df[mask], validation_seasons)
    elif split_strategy == "week":
        train_df, val_df = split_by_week(df[mask], validation_seasons)
    else:
        train_df, val_df = split_random(df[mask], test_size=0.2)
    
    train_idx = train_df.index
    val_idx = val_df.index
    
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model_config = config.get("models", {}).get(market, {})
    algorithm = model_config.get("algorithm", "gradient_boosting")
    n_estimators = model_config.get("n_estimators", 100)
    max_depth = model_config.get("max_depth", 5)
    learning_rate = model_config.get("learning_rate", 0.1)
    
    print(f"Training {algorithm} model...")
    
    if market == "score_projection":
        # Regression model
        ModelClass = get_model_class(algorithm, model_type="regression")
        model = ModelClass(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_proba = None  # Not applicable for regression
    else:
        # Classification model
        ModelClass = get_model_class(algorithm, model_type="classification")
        model = ModelClass(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Evaluate
    print("Evaluating model...")
    model_type = "regression" if market == "score_projection" else "classification"
    eval_results = evaluate_model_comprehensive(
        y_val,
        y_val_pred_proba if y_val_pred_proba is not None else y_val_pred,
        y_val_pred,
        model_type=model_type
    )
    
    # Save model
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    
    model_name = f"{sport}_{market}_{model_version}.pkl"
    model_path = model_dir / model_name
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": available_cols,
        "sport": sport,
        "market": market,
        "algorithm": algorithm,
        "model_version": model_version,
        "training_date": datetime.now().isoformat(),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "eval_metrics": eval_results,
    }
    
    joblib.dump(model_data, model_path)
    print(f"✓ Model saved to {model_path}")
    
    return {
        "success": True,
        "model_path": str(model_path),
        "model_version": model_version,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "eval_metrics": eval_results,
    }


def train_score_projection_models(
    sport: str,
    config: Optional[Dict] = None,
    model_version: Optional[str] = None
) -> Dict:
    """
    Train separate score projection models for home and away teams.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        config: Training configuration
        model_version: Version tag
    
    Returns:
        Dict with results for both models
    """
    if config is None:
        config = load_config()
    
    # Load and prepare data (same as moneyline)
    training_seasons = config.get("training_seasons", {}).get(sport, [2020, 2021, 2022, 2023, 2024])
    min_games_per_team = config.get("min_games_per_team", 10)
    
    all_games = []
    for season in training_seasons:
        start_date, end_date = get_season_date_range(sport, season)
        season_games = load_games_for_sport(
            sport,
            start_date=start_date,
            end_date=end_date,
            min_games_per_team=min_games_per_team
        )
        if not season_games.empty:
            all_games.append(season_games)
    
    if not all_games:
        return {"success": False, "error": "No games found"}
    
    df = pd.concat(all_games, ignore_index=True)
    
    # Build features
    rolling_window = config.get("features", {}).get("rolling_window_games", 10)
    include_rest_days = config.get("features", {}).get("include_rest_days", True)
    include_h2h = config.get("features", {}).get("include_head_to_head", True)
    
    df = build_features(
        df,
        sport=sport,
        market="score_projection",
        rolling_window=rolling_window,
        include_rest_days=include_rest_days,
        include_h2h=include_h2h
    )
    
    feature_cols = get_feature_columns(sport, "score_projection")
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        return {"success": False, "error": "No features available"}
    
    X = df[available_cols].copy()
    
    # Split
    split_strategy = config.get("split_strategy", "season")
    validation_seasons = config.get("validation_seasons", 1)
    
    if split_strategy == "season":
        train_df, val_df = split_by_season(df, validation_seasons)
    elif split_strategy == "week":
        train_df, val_df = split_by_week(df, validation_seasons)
    else:
        train_df, val_df = split_random(df, test_size=0.2)
    
    train_idx = train_df.index
    val_idx = val_df.index
    
    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train home and away models
    model_config = config.get("models", {}).get("score_projection", {})
    algorithm = model_config.get("algorithm", "gradient_boosting")
    n_estimators = model_config.get("n_estimators", 100)
    max_depth = model_config.get("max_depth", 5)
    learning_rate = model_config.get("learning_rate", 0.1)
    
    print(f"Training score projection models with {algorithm}...")
    
    results = {}
    
    for team_type in ["home", "away"]:
        print(f"Training {team_type} score projection model...")
        
        y_train = df.loc[train_idx, f"{team_type}_score"].astype(float)
        y_val = df.loc[val_idx, f"{team_type}_score"].astype(float)
        
        # Remove NaN
        mask_train = ~y_train.isna()
        mask_val = ~y_val.isna()
        
        X_train_clean = X_train_scaled[mask_train]
        y_train_clean = y_train[mask_train]
        X_val_clean = X_val_scaled[mask_val]
        y_val_clean = y_val[mask_val]
        
        ModelClass = get_model_class(algorithm, model_type="regression")
        model = ModelClass(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train_clean, y_train_clean)
        
        y_val_pred = model.predict(X_val_clean)
        eval_results = evaluate_model_comprehensive(
            y_val_clean,
            y_val_pred,
            y_val_pred,
            model_type="regression"
        )
        
        # Save
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(exist_ok=True)
        
        model_name = f"{sport}_score_{team_type}_{model_version}.pkl"
        model_path = model_dir / model_name
        
        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_columns": available_cols,
            "sport": sport,
            "team_type": team_type,
            "algorithm": algorithm,
            "model_version": model_version,
            "training_date": datetime.now().isoformat(),
            "n_train_samples": len(X_train_clean),
            "n_val_samples": len(X_val_clean),
            "eval_metrics": eval_results,
        }
        
        joblib.dump(model_data, model_path)
        print(f"✓ {team_type} model saved to {model_path}")
        
        results[f"{team_type}_model"] = {
            "model_path": str(model_path),
            "n_train_samples": len(X_train_clean),
            "n_val_samples": len(X_val_clean),
            "eval_metrics": eval_results,
        }
    
    results["success"] = True
    results["model_version"] = model_version
    return results
