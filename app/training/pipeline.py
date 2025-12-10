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


def validate_no_leakage(df: pd.DataFrame, features: list, target_col: str = None) -> bool:
    """
    Validate that features don't contain data leakage.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target_col: Name of target column (optional, will try to infer)
    
    Returns:
        True if no leakage detected, False otherwise
    """
    if target_col is None:
        # Try to find target column
        target_candidates = [col for col in df.columns if col in ['target', 'label', 'y', 'home_won', 'covered', 'over']]
        if target_candidates:
            target_col = target_candidates[0]
        else:
            print("⚠️  Could not find target column for leakage validation")
            return True
    
    if target_col not in df.columns:
        print("⚠️  Target column not found in dataframe")
        return True
    
    warnings = []
    errors = []
    
    # Check for suspicious correlations (>0.90)
    # Note: High correlation doesn't necessarily mean leakage - it could be a very predictive feature
    # But correlation >0.95 is suspicious and worth investigating
    for feat in features:
        if feat not in df.columns:
            continue
        try:
            corr = df[[feat, target_col]].corr().iloc[0, 1]
            if not pd.isna(corr) and abs(corr) > 0.95:  # Only warn for very high correlation (>0.95)
                # Check if it's a legitimate historical feature
                feat_lower = feat.lower()
                is_historical = any(p in feat_lower for p in ['win_rate', 'avg', 'last_', 'recent', 'ats_', 'h2h_'])
                if is_historical:
                    # Historical features with high correlation are usually fine (they're meant to be predictive)
                    # Only warn if correlation is extremely high (>0.98)
                    if abs(corr) > 0.98:
                        warnings.append(f"Very high correlation (investigate): {feat} = {corr:.4f}")
                else:
                    warnings.append(f"High correlation (investigate): {feat} = {corr:.4f}")
        except:
            pass
    
    # Check for outcome variable keywords in feature names
    # But exclude legitimate historical features (win_rate, wins_last, etc. from past games)
    outcome_keywords = ['final', 'result', 'actual', 'score_diff', 'winner', 'outcome']
    # Exclude legitimate patterns that indicate historical/past data
    legitimate_patterns = ['last_', 'recent', 'avg', 'rolling', 'history', 'win_rate', 'wins_last', 
                          'losses_last', 'streak', 'ats_', 'h2h_', 'points_for_avg', 'points_against_avg',
                          'efficiency', 'momentum', 'strength', 'opponent_strength']
    
    for feat in features:
        feat_lower = feat.lower()
        # Check if it contains outcome keywords
        has_outcome_kw = any(kw in feat_lower for kw in outcome_keywords)
        # Check if it's a legitimate historical feature
        is_legitimate = any(pattern in feat_lower for pattern in legitimate_patterns)
        
        if has_outcome_kw and not is_legitimate:
            errors.append(f"Outcome variable in features: {feat}")
    
    # Check for direct leakage features (features that encode the target directly)
    # These are features that use actual game outcomes or directly encode predictions
    leakage_features = ['spread_value', 'totals_value', 'home_cover_prob', 'over_prob', 
                       'spread_edge', 'totals_edge', 'actual_score', 'actual_margin',
                       'game_result', 'final_score', 'final_margin', 'corsi', 'time_of_possession']
    for feat in features:
        feat_lower = feat.lower()
        # Only flag if it's an exact match or clearly a leakage feature
        # Exclude legitimate features like "ats_win_rate" (past games only)
        if any(leak in feat_lower for leak in leakage_features):
            # But allow ATS features (they're historical, not current game)
            if 'ats_' not in feat_lower and 'h2h_' not in feat_lower:
                errors.append(f"Known leakage feature: {feat}")
    
    if warnings:
        print("⚠️  Leakage warnings:")
        for w in warnings[:10]:  # Show first 10
            print(f"   {w}")
    
    if errors:
        print("❌ Leakage errors detected:")
        for e in errors:
            print(f"   {e}")
        return False
    
    if not warnings and not errors:
        print("✓ Leakage validation passed")
    
    return len(errors) == 0
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
        # Need scores to calculate
        has_scores = df["home_score"].notna() & df["away_score"].notna()
        if not has_scores.any():
            print(f"  ⚠️  Warning: No games with scores, cannot train moneyline model")
            return pd.Series([np.nan] * len(df), index=df.index)
        
        result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        result[has_scores] = (df.loc[has_scores, "home_score"] > df.loc[has_scores, "away_score"]).astype(int)
        
        # Log distribution
        home_wins = (result == 1).sum()
        away_wins = (result == 0).sum()
        print(f"  Label distribution: {home_wins} home wins, {away_wins} away wins")
        
        if home_wins == 0 or away_wins == 0:
            print(f"  ⚠️  WARNING: All labels are the same! Home wins: {home_wins}, Away wins: {away_wins}")
            print(f"  This will cause training to fail. Check data quality.")
        
        return result
    
    elif market == "spread":
        # MLB doesn't use spreads (uses run lines instead) - skip training
        if sport == "MLB":
            print(f"  ⚠️  Warning: MLB does not use spreads (uses run lines), skipping spread training")
            return pd.Series([np.nan] * len(df), index=df.index)
        
        # 1 if home covers spread, 0 otherwise
        if "spread" not in df.columns:
            # No spread column at all - can't train
            print("  ⚠️  Warning: No spread column found, cannot train spread model")
            return pd.Series([np.nan] * len(df), index=df.index)
        
        spread_clean = df["spread"].copy()
        null_count = spread_clean.isna().sum()
        total_count = len(df)
        
        if null_count == total_count:
            # All spreads are null - can't train
            print(f"  ⚠️  Warning: All {total_count} games have null spreads, cannot train spread model")
            return pd.Series([np.nan] * total_count, index=df.index)
        
        # For games with null spreads, use spread = 0 as default (pick'em game)
        # This allows training while being conservative (assumes no spread advantage)
        if null_count > 0:
            print(f"  ⚠️  Warning: {null_count}/{total_count} games have null spreads, using spread=0 as default")
            spread_clean = spread_clean.fillna(0)
        
        # Calculate labels: 1 if home covers, 0 otherwise
        # Need scores to calculate
        has_scores = df["home_score"].notna() & df["away_score"].notna()
        if not has_scores.any():
            print(f"  ⚠️  Warning: No games with scores, cannot train spread model")
            return pd.Series([np.nan] * len(df), index=df.index)
        
        # Calculate labels only for games with scores
        result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        result[has_scores] = (df.loc[has_scores, "home_score"] - df.loc[has_scores, "away_score"] > spread_clean[has_scores]).astype(int)
        
        # Log distribution
        cover_count = (result == 1).sum()
        no_cover_count = (result == 0).sum()
        print(f"  Label distribution: {cover_count} covers, {no_cover_count} no-covers")
        
        if cover_count == 0 or no_cover_count == 0:
            print(f"  ⚠️  WARNING: All labels are the same! Covers: {cover_count}, No-covers: {no_cover_count}")
            print(f"  This will cause training to fail. Check data quality.")
        
        return result
    
    elif market == "totals":
        # 1 if over, 0 if under
        # Prefer ou_result if available (more reliable), otherwise calculate from scores
        
        # Check if we have ou_result column (new approach)
        has_ou_result = "ou_result" in df.columns
        has_closing_total = "closing_total" in df.columns or "over_under" in df.columns
        
        if not has_closing_total:
            print("  ⚠️  Warning: No over_under/closing_total column found, cannot train totals model")
            return pd.Series([np.nan] * len(df), index=df.index)
        
        # Use closing_total if available, fallback to over_under
        if has_closing_total and "closing_total" in df.columns:
            closing_total_col = df["closing_total"]
        else:
            closing_total_col = df["over_under"]
        
        null_count = closing_total_col.isna().sum()
        total_count = len(df)
        
        if null_count == total_count:
            print(f"  ⚠️  Warning: All {total_count} games have null closing_total, cannot train totals model")
            return pd.Series([np.nan] * total_count, index=df.index)
        
        result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        
        if has_ou_result:
            # Use pre-calculated ou_result (most reliable)
            valid_mask = df["ou_result"].notna()
            over_mask = df["ou_result"] == "OVER"
            under_mask = df["ou_result"] == "UNDER"
            push_mask = df["ou_result"] == "PUSH"
            
            result[valid_mask & over_mask] = 1
            result[valid_mask & under_mask] = 0
            # Pushes remain NaN (excluded)
            
            over_count = over_mask.sum()
            under_count = under_mask.sum()
            push_count = push_mask.sum()
            
            print(f"  Using ou_result column: {over_count} over, {under_count} under, {push_count} pushes (excluded)")
        else:
            # Fallback: calculate from scores and closing_total
            has_scores = df["home_score"].notna() & df["away_score"].notna()
            valid_mask = closing_total_col.notna() & has_scores
            
            if valid_mask.sum() == 0:
                print(f"  ⚠️  Warning: No games with both closing_total and scores, cannot train totals model")
                return result
            
            total_scores = df.loc[valid_mask, "home_score"] + df.loc[valid_mask, "away_score"]
            closing_values = closing_total_col[valid_mask]
            
            over_mask = total_scores > closing_values
            under_mask = total_scores < closing_values
            push_mask = total_scores == closing_values
            
            result[valid_mask & over_mask] = 1
            result[valid_mask & under_mask] = 0
            # Pushes remain NaN (excluded)
            
            over_count = over_mask.sum()
            under_count = under_mask.sum()
            push_count = push_mask.sum()
            
            print(f"  Calculated from scores: {over_count} over, {under_count} under, {push_count} pushes (excluded)")
        
        if null_count > 0:
            print(f"  ⚠️  Warning: {null_count}/{total_count} games have null closing_total (will be excluded)")
        
        if over_count == 0 or under_count == 0:
            print(f"  ⚠️  WARNING: All labels are the same! Over: {over_count}, Under: {under_count}")
            print(f"  This will cause training to fail. Check data quality.")
        
        return result
    
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
    
    # Debug: Log features being used (especially for spread/moneyline to check for data leakage)
    if market in ["spread", "moneyline"]:
        print(f"DEBUG: Using {len(available_cols)} features for {sport} {market}")
        # Check for known leakage features (not historical win_rate features - those are legitimate)
        known_leakage = [f for f in available_cols if any(x in f.lower() for x in [
            'spread_value', 'totals_value', 'cover_prob', 'over_prob', 
            'spread_edge', 'totals_edge', 'actual_score', 'final_margin'
        ])]
        if known_leakage:
            print(f"❌ ERROR: Known leakage features found: {known_leakage}")
        # Show first 20 features being used
        print(f"DEBUG: First 20 features: {available_cols[:20]}")
    
    X = df[available_cols].copy()
    
    # Validate no data leakage before training
    print("Validating features for data leakage...")
    temp_df = df.copy()
    temp_df['target'] = y
    validate_no_leakage(temp_df, available_cols, target_col='target')
    
    # Remove rows with NaN in features or labels
    mask = ~(X.isna().any(axis=1) | y.isna())
    
    # Debug: Log why samples are being filtered
    total_samples = len(X)
    nan_labels = y.isna().sum()
    nan_features = X.isna().any(axis=1).sum()
    valid_samples = mask.sum()
    
    if nan_labels > 0:
        print(f"  ⚠️  {nan_labels}/{total_samples} samples have NaN labels (will be excluded)")
    if nan_features > 0:
        print(f"  ⚠️  {nan_features}/{total_samples} samples have NaN features (will be excluded)")
    if valid_samples < total_samples:
        print(f"  ⚠️  Filtering: {total_samples} -> {valid_samples} valid samples")
    
    # Filter data and reset index to avoid index mismatches
    df_filtered = df[mask].copy().reset_index(drop=True)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    if len(X) == 0:
        error_msg = f"No valid samples after cleaning (total: {total_samples}, NaN labels: {nan_labels}, NaN features: {nan_features})"
        return {"success": False, "error": error_msg}
    
    # Train/validation split (use filtered dataframe with reset index)
    split_strategy = config.get("split_strategy", "season")
    validation_seasons = config.get("validation_seasons", 1)
    
    if split_strategy == "season":
        train_df, val_df = split_by_season(df_filtered, validation_seasons)
    elif split_strategy == "week":
        train_df, val_df = split_by_week(df_filtered, validation_seasons)
    else:
        train_df, val_df = split_random(df_filtered, test_size=0.2)
    
    # Verify temporal split (no data leakage)
    if "date" in df_filtered.columns:
        train_dates = train_df["date"]
        val_dates = val_df["date"]
        train_max_date = pd.to_datetime(train_dates).max()
        val_min_date = pd.to_datetime(val_dates).min()
        
        print(f"Train period: {train_dates.min()} to {train_dates.max()}")
        print(f"Val period: {val_dates.min()} to {val_dates.max()}")
        
        if train_max_date >= val_min_date:
            print(f"⚠️  WARNING: Temporal leakage detected! Train max ({train_max_date}) >= Val min ({val_min_date})")
        else:
            print(f"✓ Temporal split verified: Train ends before Val starts")
    
    # Create boolean masks by matching on a unique identifier
    # Use a combination of date and teams to uniquely identify rows
    # This avoids index mismatches when split functions reset indices
    if "date" in df_filtered.columns and "home_team" in df_filtered.columns and "away_team" in df_filtered.columns:
        # Create unique identifiers for matching
        df_filtered["_match_key"] = (
            df_filtered["date"].astype(str) + "_" + 
            df_filtered["home_team"].astype(str) + "_" + 
            df_filtered["away_team"].astype(str)
        )
        train_df["_match_key"] = (
            train_df["date"].astype(str) + "_" + 
            train_df["home_team"].astype(str) + "_" + 
            train_df["away_team"].astype(str)
        )
        val_df["_match_key"] = (
            val_df["date"].astype(str) + "_" + 
            val_df["home_team"].astype(str) + "_" + 
            val_df["away_team"].astype(str)
        )
        
        train_mask = df_filtered["_match_key"].isin(train_df["_match_key"])
        val_mask = df_filtered["_match_key"].isin(val_df["_match_key"])
        
        # Clean up temporary column
        df_filtered = df_filtered.drop(columns=["_match_key"])
    else:
        # Fallback: use indices (should work if split functions preserve them)
        train_mask = df_filtered.index.isin(train_df.index)
        val_mask = df_filtered.index.isin(val_df.index)
    
    # Use boolean masks to index X and y (all have same sequential indices)
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
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
    
    # Initialize feature importance dict (will be populated after training)
    feature_importance = {}
    
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
        
        # Extract feature importance for regression models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(available_cols, importances))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 Top 20 Most Important Features:")
            print("=" * 80)
            for i, (feat_name, importance) in enumerate(sorted_features[:20], 1):
                print(f"  {i:2d}. {feat_name:50s} {importance:8.6f}")
            print("=" * 80)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_proba = None  # Not applicable for regression
    else:
        # Classification model with regularization to prevent overfitting
        ModelClass = get_model_class(algorithm, model_type="classification")
        
        # Add regularization parameters for XGBoost/LightGBM to prevent overfitting
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": 42
        }
        
        # Add regularization if using XGBoost or LightGBM
        if algorithm.lower() in ["xgboost", "xgb"]:
            model_params.update({
                "max_depth": min(max_depth, 3),  # Prevent deep memorization
                "min_child_weight": 5,  # Require 5+ samples per leaf
                "subsample": 0.8,  # Use 80% of rows per tree
                "colsample_bytree": 0.8,  # Use 80% of features per tree
                "reg_alpha": 1.0,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
            })
        elif algorithm.lower() in ["lightgbm", "lgb"]:
            model_params.update({
                "max_depth": min(max_depth, 3),
                "min_child_samples": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
            })
        
        # Check for class imbalance and validate labels before training
        unique_classes = np.unique(y_train[~y_train.isna()])
        
        # Validate that we have at least 2 classes (for classification)
        if len(unique_classes) < 2:
            class_dist = {cls: (y_train == cls).sum() for cls in unique_classes}
            error_msg = f"Invalid classes in training data. Expected at least 2 classes, got: {unique_classes}. This usually means all labels are the same (all {unique_classes[0] if len(unique_classes) > 0 else 'NaN'}). Check label preparation and data filtering. Training samples: {len(y_train)}, Class distribution: {class_dist}"
            print(f"✗ {sport} {market} failed: {error_msg}")
            return {"success": False, "error": error_msg}
        if len(unique_classes) < 2:
            error_msg = f"Invalid classes in training data. Expected at least 2 classes, got: {unique_classes}. "
            error_msg += f"This usually means all labels are the same (all {unique_classes[0]}). "
            error_msg += f"Check label preparation and data filtering. "
            error_msg += f"Training samples: {len(y_train)}, Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
            return {"success": False, "error": error_msg}
        
        model = ModelClass(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Note: XGBoost probabilities are generally well-calibrated for binary classification
        # Calibration can be added later if needed, but it's causing compatibility issues
        # For now, we'll use XGBoost's native probabilities
        print("✓ Model trained (using native XGBoost probabilities)")
        
        # Extract and log feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Create dictionary of feature names and their importance scores
            feature_importance = dict(zip(available_cols, importances))
            # Sort by importance (descending)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 Top 20 Most Important Features:")
            print("=" * 80)
            for i, (feat_name, importance) in enumerate(sorted_features[:20], 1):
                print(f"  {i:2d}. {feat_name:50s} {importance:8.6f}")
            print("=" * 80)
            
            # Check for suspicious features (potential leakage)
            suspicious_keywords = ['actual', 'final', 'result', 'outcome', 'score', 'margin', 
                                 'edge', 'prob', 'value', 'cover', 'over_under', 'spread_value',
                                 'totals_value', 'home_cover_prob', 'over_prob']
            suspicious_features = []
            for feat_name, importance in sorted_features:
                feat_lower = feat_name.lower()
                if any(keyword in feat_lower for keyword in suspicious_keywords):
                    # But exclude legitimate historical features
                    if not any(prefix in feat_lower for prefix in ['avg', 'rate', 'h2h', 'ats_', 'last_']):
                        suspicious_features.append((feat_name, importance))
            
            if suspicious_features:
                print(f"\n⚠️  WARNING: Found {len(suspicious_features)} potentially suspicious features:")
                for feat_name, importance in suspicious_features[:10]:
                    print(f"  - {feat_name}: {importance:.6f}")
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Log validation accuracy and check for suspiciously high accuracy (possible leakage)
        val_acc = (y_val_pred == y_val).mean()
        print(f"\nValidation Accuracy: {val_acc:.3f}")
        
        # Expected ranges:
        # Moneyline: 0.50-0.58
        # Spread: 0.48-0.53 (should be near coin flip)
        # Total: 0.48-0.53 (should be near coin flip)
        if market == "spread" and val_acc > 0.60:
            print(f"⚠️  WARNING: Spread model accuracy {val_acc:.3f} > 60% - possible data leakage!")
        elif market == "totals" and val_acc > 0.60:
            print(f"⚠️  WARNING: Total model accuracy {val_acc:.3f} > 60% - possible data leakage!")
            print(f"   Review feature importance above to identify potential leakage sources.")
        elif market == "moneyline" and val_acc > 0.65:
            print(f"⚠️  WARNING: Moneyline model accuracy {val_acc:.3f} > 65% - investigate for leakage!")
    
    # Evaluate
    print("Evaluating model...")
    model_type = "regression" if market == "score_projection" else "classification"
    eval_results = evaluate_model_comprehensive(
        y_val,
        y_val_pred_proba if y_val_pred_proba is not None else y_val_pred,
        y_val_pred,
        model_type=model_type
    )
    
    # Print calibration summary
    if model_type == "classification" and "ece" in eval_results:
        ece = eval_results.get("ece", np.nan)
        brier = eval_results.get("brier_score", np.nan)
        print(f"\n📊 Calibration Metrics:")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        
        if not np.isnan(ece):
            if ece < 0.05:
                print(f"  ✓ Model is well-calibrated (ECE < 0.05)")
            elif ece < 0.10:
                print(f"  ⚠️  Model calibration is acceptable (ECE < 0.10)")
                print(f"     Consider adding probability calibration for better accuracy.")
            else:
                print(f"  ✗ Model calibration is poor (ECE > 0.10)")
                print(f"     Model probabilities may not be reliable. Review model training.")
        
        if not np.isnan(brier):
            # Brier score: lower is better (0 = perfect, 0.25 = random)
            if brier < 0.15:
                print(f"  ✓ Good Brier score (predictions are confident and accurate)")
            elif brier < 0.20:
                print(f"  ⚠️  Acceptable Brier score")
            else:
                print(f"  ⚠️  High Brier score - model may be underconfident or inaccurate")
    
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
        "feature_importance": feature_importance,
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
        "feature_importance": feature_importance,
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
