"""
Advanced training pipeline with multiple algorithms, ensemble, and Optuna optimization.
Supports XGBoost, CatBoost, LightGBM, and stacking ensemble.
"""
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from app.data_loader import split_temporal, get_season_date_range, load_games_for_sport
from app.training.features import build_features, get_feature_columns
from app.training.pipeline import prepare_labels
from app.training.evaluate import (
    evaluate_model_comprehensive,
    calculate_ece,
    plot_calibration_curve
)
from app.utils.betting import (
    calculate_edge,
    calculate_roi,
    american_odds_to_implied_prob
)
from app.config import settings


def get_model_class(algorithm: str, model_type: str = "classification"):
    """
    Get model class for algorithm.
    
    Args:
        algorithm: Algorithm name (xgb, catboost, lightgbm, gradient_boosting)
        model_type: "classification" or "regression"
    
    Returns:
        Model class
    """
    algorithm_lower = algorithm.lower().strip()
    
    if algorithm_lower in ["xgboost", "xgb"]:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        if model_type == "classification":
            return xgb.XGBClassifier
        else:
            return xgb.XGBRegressor
    
    elif algorithm_lower in ["lightgbm", "lgb"]:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        if model_type == "classification":
            return lgb.LGBMClassifier
        else:
            return lgb.LGBMRegressor
    
    elif algorithm_lower in ["catboost", "cat"]:
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        if model_type == "classification":
            return cb.CatBoostClassifier
        else:
            return cb.CatBoostRegressor
    
    elif algorithm_lower in ["gradient_boosting", "sklearn"]:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if model_type == "classification":
            return GradientBoostingClassifier
        else:
            return GradientBoostingRegressor
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    algorithm: str,
    model_type: str,
    hyperparams: Optional[Dict] = None
) -> Tuple[object, Dict]:
    """
    Train a single model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        algorithm: Algorithm name
        model_type: "classification" or "regression"
        hyperparams: Hyperparameters dict
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    ModelClass = get_model_class(algorithm, model_type)
    
    # Default hyperparameters
    default_params = {
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    # CatBoost has verbose parameter
    if algorithm.lower() in ["catboost", "cat"]:
        default_params["verbose"] = False
    
    model = ModelClass(**default_params)
    model.fit(X_train, y_train)
    
    # Predictions
    if model_type == "classification":
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
    else:
        y_pred_proba = model.predict(X_val)
        y_pred = y_pred_proba
    
    # Evaluate
    metrics = evaluate_model_comprehensive(
        pd.Series(y_val),
        pd.Series(y_pred_proba),
        pd.Series(y_pred),
        model_type=model_type
    )
    
    return model, metrics


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    algorithm: str,
    model_type: str,
    n_trials: int = 100
) -> Dict:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        algorithm: Algorithm name
        model_type: "classification" or "regression"
        n_trials: Number of Optuna trials
    
    Returns:
        Best hyperparameters dict
    """
    def objective(trial):
        # Suggest hyperparameters
        if algorithm.lower() in ["xgboost", "xgb"]:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
            }
        elif algorithm.lower() in ["lightgbm", "lgb"]:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "verbosity": -1,
            }
        elif algorithm.lower() in ["catboost", "cat"]:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "verbose": False,
            }
        else:
            # Gradient Boosting
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "random_state": 42,
            }
        
        # Train model
        ModelClass = get_model_class(algorithm, model_type)
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        if model_type == "classification":
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            # Optimize for log loss (calibration focus)
            from sklearn.metrics import log_loss
            score = -log_loss(y_val, y_pred_proba)
        else:
            y_pred = model.predict(X_val)
            from sklearn.metrics import mean_squared_error
            score = -np.sqrt(mean_squared_error(y_val, y_pred))
        
        return score
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    algorithms: List[str],
    model_type: str,
    use_optuna: bool = False,
    n_trials: int = 50
) -> Tuple[Dict, Dict]:
    """
    Train ensemble of models and stacking meta-learner.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        algorithms: List of algorithm names
        model_type: "classification" or "regression"
        use_optuna: Whether to use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials per algorithm
    
    Returns:
        Tuple of (base_models_dict, ensemble_metrics)
    """
    base_models = {}
    base_predictions_train = []
    base_predictions_val = []
    
    # Train base models
    for algorithm in algorithms:
        print(f"Training {algorithm}...")
        
        try:
            if use_optuna:
                print(f"  Optimizing hyperparameters for {algorithm}...")
                hyperparams = optimize_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    algorithm, model_type, n_trials=n_trials
                )
                print(f"  Best params: {hyperparams}")
            else:
                hyperparams = None
            
            model, metrics = train_single_model(
                X_train, y_train, X_val, y_val,
                algorithm, model_type, hyperparams
            )
            
            base_models[algorithm] = {
                "model": model,
                "metrics": metrics,
                "hyperparams": hyperparams
            }
            
            # Get predictions for stacking
            if model_type == "classification":
                train_pred = model.predict_proba(X_train)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
            else:
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
            
            base_predictions_train.append(train_pred)
            base_predictions_val.append(val_pred)
            
            print(f"  {algorithm} - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, "
                  f"ECE: {metrics.get('ece', 'N/A'):.4f}")
        
        except Exception as e:
            print(f"  Error training {algorithm}: {e}")
            continue
    
    if len(base_models) == 0:
        raise ValueError("No base models trained successfully")
    
    # Stacking: train meta-learner on base model predictions
    print("Training stacking meta-learner...")
    X_meta_train = np.column_stack(base_predictions_train)
    X_meta_val = np.column_stack(base_predictions_val)
    
    if model_type == "classification":
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        from sklearn.linear_model import LinearRegression
        meta_model = LinearRegression()
    
    meta_model.fit(X_meta_train, y_train)
    
    # Evaluate ensemble
    if model_type == "classification":
        y_pred_proba = meta_model.predict_proba(X_meta_val)[:, 1]
        y_pred = meta_model.predict(X_meta_val)
    else:
        y_pred_proba = meta_model.predict(X_meta_val)
        y_pred = y_pred_proba
    
    ensemble_metrics = evaluate_model_comprehensive(
        pd.Series(y_val),
        pd.Series(y_pred_proba),
        pd.Series(y_pred),
        model_type=model_type
    )
    
    base_models["meta_learner"] = {
        "model": meta_model,
        "metrics": ensemble_metrics
    }
    
    return base_models, ensemble_metrics


def train_advanced_model(
    sport: str,
    market: str,
    config: Optional[Dict] = None,
    algorithms: Optional[List[str]] = None,
    use_ensemble: bool = True,
    use_optuna: bool = False,
    n_trials: int = 50,
    model_version: Optional[str] = None
) -> Dict:
    """
    Train advanced model with multiple algorithms and ensemble.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals)
        config: Training configuration dict
        algorithms: List of algorithms to use (default: ["xgb", "catboost", "lightgbm"])
        use_ensemble: Whether to use stacking ensemble
        use_optuna: Whether to use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials per algorithm
        model_version: Version tag for model
    
    Returns:
        Dict with training results
    """
    
    if config is None:
        from app.training.pipeline import load_config
        config = load_config()
    
    if algorithms is None:
        algorithms = []
        if XGBOOST_AVAILABLE:
            algorithms.append("xgb")
        if CATBOOST_AVAILABLE:
            algorithms.append("catboost")
        if LIGHTGBM_AVAILABLE:
            algorithms.append("lightgbm")
        if len(algorithms) == 0:
            algorithms = ["gradient_boosting"]  # Fallback
    
    # Load data
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
        return {"success": False, "error": "No games found for training"}
    
    df = pd.concat(all_games, ignore_index=True)
    
    if len(df) < 100:
        return {"success": False, "error": f"Not enough games: {len(df)}"}
    
    # Build features (with sport-specific engineers)
    rolling_window = config.get("features", {}).get("rolling_window_games", 10)
    include_rest_days = config.get("features", {}).get("include_rest_days", True)
    include_h2h = config.get("features", {}).get("include_head_to_head", True)
    use_sport_specific = config.get("features", {}).get("use_sport_specific", True)
    
    print(f"Building features for {sport} {market}...")
    df = build_features(
        df,
        sport=sport,
        market=market,
        rolling_window=rolling_window,
        include_rest_days=include_rest_days,
        include_h2h=include_h2h,
        use_sport_specific=use_sport_specific
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
    
    # Remove rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    df = df[mask]
    
    if len(X) == 0:
        return {"success": False, "error": "No valid samples after cleaning"}
    
    # Temporal train-test split (80/20 by date)
    print("Splitting data temporally...")
    train_df, val_df = split_temporal(df, test_size=0.2)
    
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
    
    # Determine model type
    model_type = "regression" if market == "score_projection" else "classification"
    
    # Train ensemble
    if use_ensemble and len(algorithms) > 1:
        print(f"Training ensemble with {len(algorithms)} algorithms...")
        base_models, ensemble_metrics = train_ensemble(
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            algorithms, model_type, use_optuna, n_trials
        )
        
        best_model = base_models["meta_learner"]["model"]
        best_metrics = ensemble_metrics
        model_name = "ensemble"
    else:
        # Train single best model
        print(f"Training single model: {algorithms[0]}...")
        if use_optuna:
            hyperparams = optimize_hyperparameters(
                X_train_scaled, y_train.values,
                X_val_scaled, y_val.values,
                algorithms[0], model_type, n_trials=n_trials
            )
        else:
            hyperparams = None
        
        best_model, best_metrics = train_single_model(
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            algorithms[0], model_type, hyperparams
        )
        model_name = algorithms[0]
    
    # Calculate ROI if we have odds data
    roi_metrics = None
    if market == "moneyline" and "home_moneyline" in val_df.columns:
        # Get predictions
        if model_type == "classification":
            y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_pred_proba = best_model.predict(X_val_scaled)
        
        # Calculate edges
        home_odds = val_df["home_moneyline"].values
        away_odds = val_df["away_moneyline"].values
        
        home_implied = pd.Series([american_odds_to_implied_prob(odds) for odds in home_odds])
        away_implied = pd.Series([american_odds_to_implied_prob(odds) for odds in away_odds])
        
        # Use home win probability
        home_edges = pd.Series([calculate_edge(prob, imp) for prob, imp in zip(y_pred_proba, home_implied)])
        
        # Calculate ROI
        roi_metrics = calculate_roi(
            y_val,
            pd.Series(y_pred_proba),
            pd.Series(home_odds),
            home_edges,
            min_edge=0.05,
            kelly_fraction=0.25
        )
        best_metrics["roi"] = roi_metrics
    
    # Save model
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    
    model_name_file = f"{sport}_{market}_{model_name}_{model_version}.pkl"
    model_path = model_dir / model_name_file
    
    model_data = {
        "model": best_model,
        "scaler": scaler,
        "feature_columns": available_cols,
        "sport": sport,
        "market": market,
        "algorithm": model_name,
        "model_version": model_version,
        "training_date": datetime.now().isoformat(),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "eval_metrics": best_metrics,
        "algorithms_used": algorithms,
        "use_ensemble": use_ensemble,
    }
    
    joblib.dump(model_data, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save hyperparameters if available
    if use_optuna and "hyperparams" in locals():
        hyperparams_path = model_dir / f"{sport}_{market}_hyperparams_{model_version}.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=2)
        print(f"✓ Hyperparameters saved to {hyperparams_path}")
    
    return {
        "success": True,
        "model_path": str(model_path),
        "model_version": model_version,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "eval_metrics": best_metrics,
        "roi_metrics": roi_metrics,
    }
