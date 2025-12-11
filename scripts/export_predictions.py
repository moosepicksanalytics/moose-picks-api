"""
Script to generate predictions for upcoming games and export to Lovable format.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import joblib
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Game
from sqlalchemy import func
from app.training.features import build_features, get_feature_columns
from app.utils.odds import (
    calculate_moneyline_edge,
    calculate_spread_edge,
    calculate_totals_edge,
    american_odds_to_implied_prob,
)
import logging

logger = logging.getLogger(__name__)
from app.utils.export import export_predictions
from app.prediction.storage import store_predictions_for_game, get_model_version
from app.config import settings


def load_latest_model(sport: str, market: str) -> dict:
    """
    Load the latest trained model for a sport/market.
    
    Args:
        sport: Sport code
        market: Market type
    
    Returns:
        Model data dict
    """
    model_dir = Path(settings.MODEL_DIR)
    
    # Find latest model
    pattern = f"{sport}_{market}_*.pkl"
    models = list(model_dir.glob(pattern))
    
    if not models:
        raise FileNotFoundError(f"No model found for {sport} {market}")
    
    # Get latest by modification time
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    
    return joblib.load(latest_model)


def enforce_probability_constraints(ml_prob, spread_prob, total_prob, spread_line):
    """
    Ensure probabilities are logically consistent.
    
    Args:
        ml_prob: Moneyline probability (team wins)
        spread_prob: Spread probability (team covers spread)
        total_prob: Total probability (over/under)
        spread_line: The spread value (negative = favorite)
    
    Returns:
        Adjusted probabilities that make logical sense
    """
    # Rule 1: Spread probability MUST be less than moneyline (for favorites)
    if spread_line is not None and spread_line < 0:  # Team is favorite
        # P(Win by spread) ≤ P(Win)
        if spread_prob > ml_prob:
            logger.warning(f"Adjusting spread_prob {spread_prob:.3f} to be ≤ ml_prob {ml_prob:.3f}")
            spread_prob = min(spread_prob, ml_prob * 0.85)
    elif spread_line is not None and spread_line > 0:  # Team is underdog
        # Underdog covering spread includes winning outright, so spread_prob >= ml_prob
        if spread_prob < ml_prob:
            spread_prob = max(spread_prob, ml_prob)
    
    # Rule 2: Total probabilities rarely exceed 70-75%
    if total_prob > 0.75:
        logger.warning(f"Capping total_prob {total_prob:.3f} at 0.75")
        total_prob = min(total_prob, 0.75)
    
    # Rule 3: Probabilities between 0 and 1
    ml_prob = max(0.01, min(0.99, ml_prob))
    spread_prob = max(0.01, min(0.99, spread_prob))
    total_prob = max(0.01, min(0.99, total_prob))
    
    return ml_prob, spread_prob, total_prob


def should_display_pick(prob, edge):
    """
    Filter out obviously miscalibrated or low-edge predictions.
    
    Args:
        prob: Model probability
        edge: Edge vs. market
    
    Returns:
        Boolean - whether to show this pick to users
    """
    # Don't show if probability is unrealistically high
    if prob > 0.70:
        return False  # Too confident, likely miscalibrated
    
    # Don't show if edge claim is unrealistic
    if abs(edge) > 0.10:
        logger.warning(f"Filtering pick with unrealistic edge {edge:.1%} (prob: {prob:.3f})")
        return False  # 10%+ edge is basically impossible
    
    # Don't show if edge is too small
    if abs(edge) < 0.02:
        return False  # <2% edge not worth showing
    
    return True


def load_score_models(sport: str) -> dict:
    """Load home and away score projection models."""
    model_dir = Path(settings.MODEL_DIR)
    
    home_pattern = f"{sport}_score_home_*.pkl"
    away_pattern = f"{sport}_score_away_*.pkl"
    
    home_models = list(model_dir.glob(home_pattern))
    away_models = list(model_dir.glob(away_pattern))
    
    if not home_models or not away_models:
        raise FileNotFoundError(f"No score models found for {sport}")
    
    latest_home = max(home_models, key=lambda p: p.stat().st_mtime)
    latest_away = max(away_models, key=lambda p: p.stat().st_mtime)
    
    return {
        "home": joblib.load(latest_home),
        "away": joblib.load(latest_away),
    }


def predict_for_game(
    game: Game,
    sport: str,
    config: dict
) -> dict:
    """
    Generate predictions for a single game.
    
    Args:
        game: Game database record
        sport: Sport code
        config: Config dict
    
    Returns:
        Prediction dict
    """
    # Convert game to DataFrame
    df = pd.DataFrame([{
        "game_id": game.id,
        "sport": game.sport,
        "league": game.league,
        "date": game.date,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "home_moneyline": game.home_moneyline,
        "away_moneyline": game.away_moneyline,
        "spread": game.spread,
        "over_under": game.over_under,
        "espn_data": game.espn_data,
    }])
    
    # Build features
    rolling_window = config.get("features", {}).get("rolling_window_games", 10)
    include_rest_days = config.get("features", {}).get("include_rest_days", True)
    include_h2h = config.get("features", {}).get("include_head_to_head", True)
    
    # Need historical data for feature engineering
    from app.data_loader import load_games_for_sport
    
    # Load recent games for context
    hist_df = load_games_for_sport(
        sport,
        min_games_per_team=0
    ).tail(100)  # Last 100 games for context
    
    if not hist_df.empty:
        # Combine with current game
        combined_df = pd.concat([hist_df, df], ignore_index=True)
        combined_df = build_features(
            combined_df,
            sport=sport,
            market="moneyline",
            rolling_window=rolling_window,
            include_rest_days=include_rest_days,
            include_h2h=include_h2h
        )
        # Get last row (our game)
        df = combined_df.iloc[[-1]].copy()
    else:
        # Fallback: minimal features
        df = build_features(
            df,
            sport=sport,
            market="moneyline",
            rolling_window=rolling_window,
            include_rest_days=include_rest_days,
            include_h2h=include_h2h
        )
    
    # Get feature columns
    feature_cols = get_feature_columns(sport, "moneyline")
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        return {"error": "No features available"}
    
    predictions = {
        "game_id": game.id,
        "league": game.league or sport,
        "season": str(game.date.year) if game.date else "",
        "date": game.date.strftime("%Y-%m-%d") if game.date else "",
        "home_team": game.home_team,
        "away_team": game.away_team,
    }
    
    # Moneyline predictions
    try:
        ml_model_data = load_latest_model(sport, "moneyline")
        ml_model = ml_model_data["model"]
        ml_scaler = ml_model_data["scaler"]
        
        # Get expected features from saved model data (stored during training)
        expected_features = ml_model_data.get("feature_columns", feature_cols)
        
        # Create feature vector with expected features in correct order
        # Build as dict first to avoid DataFrame fragmentation warnings
        feature_dict = {}
        for feat in expected_features:
            if feat in df.columns and len(df) > 0:
                feature_dict[feat] = df[feat].iloc[0]
            else:
                # Fill missing features with 0 (model was trained without this feature)
                feature_dict[feat] = 0
        X_aligned = pd.DataFrame([feature_dict])
        
        # Ensure columns are in the same order as training
        X_aligned = X_aligned[expected_features].fillna(0)
        X_scaled = ml_scaler.transform(X_aligned)
        
        # Use calibrated probabilities if calibrator exists (improves edge accuracy)
        ml_calibrator = ml_model_data.get("calibrator")
        
        # #region agent log
        try:
            import json
            debug_log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
            with open(debug_log_path, 'a') as f:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "M",
                    "location": "export_predictions.py:254",
                    "message": "Checking for moneyline calibrator",
                    "data": {
                        "has_calibrator": ml_calibrator is not None,
                        "calibrator_type": str(type(ml_calibrator)) if ml_calibrator is not None else None,
                        "model_keys": list(ml_model_data.keys())
                    },
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            pass
        # #endregion
        
        if ml_calibrator is not None:
            home_win_prob_raw = ml_calibrator.predict_proba(X_scaled)[0, 1]
            
            # #region agent log
            try:
                with open(debug_log_path, 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "M",
                        "location": "export_predictions.py:275",
                        "message": "Using calibrated probability",
                        "data": {
                            "calibrated_prob": float(home_win_prob_raw)
                        },
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                pass
            # #endregion
        else:
            home_win_prob_raw = ml_model.predict_proba(X_scaled)[0, 1]
            
            # #region agent log
            try:
                with open(debug_log_path, 'a') as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "M",
                        "location": "export_predictions.py:290",
                        "message": "Using raw (uncalibrated) probability",
                        "data": {
                            "raw_prob": float(home_win_prob_raw)
                        },
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                pass
            # #endregion
        away_win_prob_raw = 1 - home_win_prob_raw
        
        # Store raw probabilities (will be adjusted after all predictions)
        predictions["_raw_ml_prob"] = home_win_prob_raw
    except Exception as e:
        print(f"Error predicting moneyline: {e}")
        predictions["moneyline"] = {}
    
    # Spread predictions
    try:
        spread_model_data = load_latest_model(sport, "spread")
        spread_model = spread_model_data["model"]
        spread_scaler = spread_model_data["scaler"]
        
        # Rebuild features for spread (use combined_df if available)
        if not hist_df.empty:
            combined_df_spread = pd.concat([hist_df, df], ignore_index=True)
            df_spread = build_features(
                combined_df_spread,
                sport=sport,
                market="spread",
                rolling_window=rolling_window,
                include_rest_days=include_rest_days,
                include_h2h=include_h2h
            )
            df_spread = df_spread.iloc[[-1]].copy()
        else:
            df_spread = build_features(
                df,
                sport=sport,
                market="spread",
                rolling_window=rolling_window,
                include_rest_days=include_rest_days,
                include_h2h=include_h2h
            )
        feature_cols_spread = get_feature_columns(sport, "spread")
        
        # Get expected features from saved model data
        expected_features_spread = spread_model_data.get("feature_columns", feature_cols_spread)
        
        # Create aligned feature vector in correct order
        # Build as dict first to avoid DataFrame fragmentation warnings
        spread_feature_dict = {}
        for feat in expected_features_spread:
            if feat in df_spread.columns and len(df_spread) > 0:
                spread_feature_dict[feat] = df_spread[feat].iloc[0]
            else:
                spread_feature_dict[feat] = 0
        X_spread_aligned = pd.DataFrame([spread_feature_dict])
        
        # Ensure columns are in the same order as training
        X_spread_aligned = X_spread_aligned[expected_features_spread].fillna(0)
        X_spread_scaled = spread_scaler.transform(X_spread_aligned)
        
        # Use calibrated probabilities if calibrator exists
        spread_calibrator = spread_model_data.get("calibrator")
        if spread_calibrator is not None:
            cover_prob_raw = spread_calibrator.predict_proba(X_spread_scaled)[0, 1]
        else:
            cover_prob_raw = spread_model.predict_proba(X_spread_scaled)[0, 1]
        
        # Store raw probabilities (will be adjusted after all predictions)
        predictions["_raw_spread_prob"] = cover_prob_raw
    except Exception as e:
        print(f"Error predicting spread: {e}")
        predictions["spread"] = {}
    
    # Totals predictions
    try:
        totals_model_data = load_latest_model(sport, "totals")
        totals_model = totals_model_data["model"]
        totals_scaler = totals_model_data["scaler"]
        
        # Rebuild features for totals (use combined_df if available)
        if not hist_df.empty:
            combined_df_totals = pd.concat([hist_df, df], ignore_index=True)
            df_totals = build_features(
                combined_df_totals,
                sport=sport,
                market="totals",
                rolling_window=rolling_window,
                include_rest_days=include_rest_days,
                include_h2h=include_h2h
            )
            df_totals = df_totals.iloc[[-1]].copy()
        else:
            df_totals = build_features(
                df,
                sport=sport,
                market="totals",
                rolling_window=rolling_window,
                include_rest_days=include_rest_days,
                include_h2h=include_h2h
            )
        feature_cols_totals = get_feature_columns(sport, "totals")
        
        # Get expected features from saved model data
        expected_features_totals = totals_model_data.get("feature_columns", feature_cols_totals)
        
        # Create aligned feature vector in correct order
        X_totals_aligned = pd.DataFrame(index=[0])
        for feat in expected_features_totals:
            if feat in df_totals.columns:
                X_totals_aligned[feat] = df_totals[feat].iloc[0] if len(df_totals) > 0 else 0
            else:
                X_totals_aligned[feat] = 0
        
        # Ensure columns are in the same order as training
        X_totals_aligned = X_totals_aligned[expected_features_totals].fillna(0)
        X_totals_scaled = totals_scaler.transform(X_totals_aligned)
        
        # Use calibrated probabilities if calibrator exists
        totals_calibrator = totals_model_data.get("calibrator")
        if totals_calibrator is not None:
            over_prob_raw = totals_calibrator.predict_proba(X_totals_scaled)[0, 1]
        else:
            over_prob_raw = totals_model.predict_proba(X_totals_scaled)[0, 1]
        under_prob_raw = 1 - over_prob_raw
        
        # Store raw probabilities (will be adjusted after all predictions)
        predictions["_raw_total_prob"] = over_prob_raw
    except Exception as e:
        print(f"Error predicting totals: {e}")
        predictions["totals"] = {}
    
    # Score projections
    try:
        score_models = load_score_models(sport)
        
        # Use moneyline features for score projection
        feature_cols_score = get_feature_columns(sport, "score_projection")
        
        home_model_data = score_models["home"]
        home_model = home_model_data["model"]
        home_scaler = home_model_data["scaler"]
        away_model_data = score_models["away"]
        away_model = away_model_data["model"]
        away_scaler = away_model_data["scaler"]
        
        # Get expected features from saved model data
        expected_features_home = home_model_data.get("feature_columns", feature_cols_score)
        expected_features_away = away_model_data.get("feature_columns", feature_cols_score)
        
        # Create aligned feature vectors in correct order
        # Build as dicts first to avoid DataFrame fragmentation warnings
        home_feature_dict = {}
        for feat in expected_features_home:
            if feat in df.columns and len(df) > 0:
                home_feature_dict[feat] = df[feat].iloc[0]
            else:
                home_feature_dict[feat] = 0
        X_home_aligned = pd.DataFrame([home_feature_dict])
        
        away_feature_dict = {}
        for feat in expected_features_away:
            if feat in df.columns and len(df) > 0:
                away_feature_dict[feat] = df[feat].iloc[0]
            else:
                away_feature_dict[feat] = 0
        X_away_aligned = pd.DataFrame([away_feature_dict])
        
        # Ensure columns are in the same order as training
        X_home_aligned = X_home_aligned[expected_features_home].fillna(0)
        X_away_aligned = X_away_aligned[expected_features_away].fillna(0)
        
        X_home_scaled = home_scaler.transform(X_home_aligned)
        X_away_scaled = away_scaler.transform(X_away_aligned)
        
        proj_home_score = home_model.predict(X_home_scaled)[0]
        proj_away_score = away_model.predict(X_away_scaled)[0]
        
        predictions["proj_home_score"] = float(proj_home_score)
        predictions["proj_away_score"] = float(proj_away_score)
    except Exception as e:
        print(f"Error predicting scores: {e}")
        predictions["proj_home_score"] = None
        predictions["proj_away_score"] = None
    
    # Apply probability constraints after all predictions are generated
    # Collect raw probabilities
    ml_prob = predictions.get("_raw_ml_prob")
    spread_prob = predictions.get("_raw_spread_prob")
    total_prob = predictions.get("_raw_total_prob")
    
    # Apply constraints if we have all three probabilities
    if ml_prob is not None and spread_prob is not None and total_prob is not None:
        spread_line = game.spread if hasattr(game, 'spread') else None
        
        # Apply constraints
        ml_prob_adj, spread_prob_adj, total_prob_adj = enforce_probability_constraints(
            ml_prob, spread_prob, total_prob, spread_line
        )
        
        # Recalculate edges with adjusted probabilities
        # Moneyline
        away_win_prob_adj = 1 - ml_prob_adj
        ml_edges = calculate_moneyline_edge(
            ml_prob_adj,
            away_win_prob_adj,
            game.home_moneyline,
            game.away_moneyline
        )
        
        # Sanity check edges
        if abs(ml_edges["best_edge"]) > 0.10:
            logger.warning(f"Unrealistic moneyline edge: {ml_edges['best_edge']:.1%}")
        
        # CRITICAL: Don't recommend sides with negative edges
        # If best_edge is negative, set best_side to None (no value bet)
        if ml_edges["best_edge"] < 0:
            logger.warning(f"Negative edge detected ({ml_edges['best_edge']:.1%}) - not recommending any side")
            ml_edges["best_side"] = None
        
        # Validate probabilities before storing
        from app.prediction.validation import validate_prediction_before_storage
        if ml_edges["best_side"]:
            best_prob = ml_prob_adj if ml_edges["best_side"] == "home" else away_win_prob_adj
            validation = validate_prediction_before_storage(
                game.id,
                sport,
                "moneyline",
                best_prob,
                game.home_moneyline,
                game.away_moneyline
            )
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    logger.warning(f"  {warning}")
        
        predictions["moneyline"] = {
            "home_win_prob": ml_prob_adj,
            "away_win_prob": away_win_prob_adj,
            "home_odds": game.home_moneyline,
            "away_odds": game.away_moneyline,
            "home_implied_prob": american_odds_to_implied_prob(game.home_moneyline) if game.home_moneyline else 0.5,
            "away_implied_prob": american_odds_to_implied_prob(game.away_moneyline) if game.away_moneyline else 0.5,
            "home_edge": ml_edges["home_edge"],
            "away_edge": ml_edges["away_edge"],
            "best_side": ml_edges["best_side"],  # Will be None if negative edge
            "best_edge": ml_edges["best_edge"],
        }
        
        # Spread
        spread_odds = -110  # Default
        spread_edge = calculate_spread_edge(spread_prob_adj, spread_odds)
        
        if abs(spread_edge) > 0.10:
            logger.warning(f"Unrealistic spread edge: {spread_edge:.1%}")
        
        predictions["spread"] = {
            "cover_prob": spread_prob_adj,
            "line": game.spread,
            "price": spread_odds,
            "implied_prob": american_odds_to_implied_prob(spread_odds),
            "edge": spread_edge,
            "side": "favorite" if spread_prob_adj > 0.5 else "underdog",
        }
        
        # Totals
        under_prob_adj = 1 - total_prob_adj
        over_odds = -110
        under_odds = -110
        totals_edges = calculate_totals_edge(total_prob_adj, under_prob_adj, over_odds, under_odds)
        
        if abs(totals_edges["best_edge"]) > 0.10:
            logger.warning(f"Unrealistic totals edge: {totals_edges['best_edge']:.1%}")
        
        # CRITICAL: Don't recommend sides with negative edges
        if totals_edges["best_edge"] < 0:
            logger.warning(f"Negative totals edge detected ({totals_edges['best_edge']:.1%}) - not recommending any side")
            totals_edges["best_side"] = None
        
        predictions["totals"] = {
            "over_prob": total_prob_adj,
            "under_prob": under_prob_adj,
            "line": game.over_under,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "over_implied_prob": american_odds_to_implied_prob(over_odds),
            "under_implied_prob": american_odds_to_implied_prob(under_odds),
            "over_edge": totals_edges["over_edge"],
            "under_edge": totals_edges["under_edge"],
            "best_side": totals_edges["best_side"],
            "best_edge": totals_edges["best_edge"],
        }
        
        # Remove raw probability keys
        predictions.pop("_raw_ml_prob", None)
        predictions.pop("_raw_spread_prob", None)
        predictions.pop("_raw_total_prob", None)
    elif ml_prob is not None:
        # If only moneyline is available, still apply basic constraints
        ml_prob_adj = max(0.01, min(0.99, ml_prob))
        away_win_prob_adj = 1 - ml_prob_adj
        ml_edges = calculate_moneyline_edge(
            ml_prob_adj,
            away_win_prob_adj,
            game.home_moneyline,
            game.away_moneyline
        )
        if abs(ml_edges["best_edge"]) > 0.10:
            logger.warning(f"Unrealistic moneyline edge: {ml_edges['best_edge']:.1%}")
        
        # CRITICAL: Don't recommend sides with negative edges
        if ml_edges["best_edge"] < 0:
            logger.warning(f"Negative edge detected ({ml_edges['best_edge']:.1%}) - not recommending any side")
            ml_edges["best_side"] = None
        
        predictions["moneyline"] = {
            "home_win_prob": ml_prob_adj,
            "away_win_prob": away_win_prob_adj,
            "home_odds": game.home_moneyline,
            "away_odds": game.away_moneyline,
            "home_implied_prob": american_odds_to_implied_prob(game.home_moneyline) if game.home_moneyline else 0.5,
            "away_implied_prob": american_odds_to_implied_prob(game.away_moneyline) if game.away_moneyline else 0.5,
            "home_edge": ml_edges["home_edge"],
            "away_edge": ml_edges["away_edge"],
            "best_side": ml_edges["best_side"],  # Will be None if negative edge
            "best_edge": ml_edges["best_edge"],
        }
        predictions.pop("_raw_ml_prob", None)
    
    return predictions


def export_predictions_for_date(
    sport: str,
    date_str: str,
    config_path: str = "config.yaml",
    output_dir: str = "exports",
    min_edge: float = 0.05
):
    """
    Generate and export predictions for all games on a date.
    
    Args:
        sport: Sport code
        date_str: Date in YYYY-MM-DD format
        config_path: Path to config.yaml
        output_dir: Output directory
        min_edge: Minimum edge threshold
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load games from database
    db = SessionLocal()
    try:
        from datetime import timezone
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Games are stored in UTC from ESPN, so we need to query using UTC timezone
        # Query for games on this date in UTC
        start_datetime_utc = datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc)
        end_datetime_utc = datetime.combine(date_obj + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        
        # Try date-only comparison first (works with func.date)
        all_games = db.query(Game).filter(
            Game.sport == sport,
            func.date(Game.date) == date_obj
        ).all()
        
        # If no games, try UTC datetime range (for timezone-aware dates)
        if not all_games:
            all_games = db.query(Game).filter(
                Game.sport == sport,
                Game.date >= start_datetime_utc,
                Game.date < end_datetime_utc,
            ).all()
        
        # Final fallback: naive datetime range (for SQLite or if dates are naive)
        if not all_games:
            start_datetime_naive = datetime.combine(date_obj, datetime.min.time())
            end_datetime_naive = datetime.combine(date_obj + timedelta(days=1), datetime.min.time())
            all_games = db.query(Game).filter(
                Game.sport == sport,
                Game.date >= start_datetime_naive,
                Game.date < end_datetime_naive,
            ).all()
        
        if not all_games:
            print(f"No games found for {sport} on {date_str}")
            print(f"  (Checked date: {date_obj})")
            
            # Debug: Check what games exist for this sport (any date)
            all_sport_games = db.query(Game).filter(Game.sport == sport).order_by(Game.date.desc()).limit(10).all()
            if all_sport_games:
                print(f"  Debug: Found {len(all_sport_games)} recent {sport} games in database:")
                for g in all_sport_games[:5]:
                    print(f"    - {g.date} ({g.status}): {g.away_team} @ {g.home_team}")
            else:
                print(f"  Debug: No {sport} games found in database at all")
            
            # Try to fetch games from ESPN if none in DB
            print(f"  Attempting to fetch games from ESPN...")
            try:
                from app.espn_client.fetcher import fetch_games_for_date
                from app.espn_client.parser import parse_and_store_games
                fetched_games = fetch_games_for_date(sport, date_str)
                if fetched_games:
                    stored = parse_and_store_games(sport, fetched_games, only_final=False)
                    print(f"  ✓ Fetched and stored {stored} games from ESPN")
                    # Retry query with timezone-aware
                    all_games = db.query(Game).filter(
                        Game.sport == sport,
                        Game.date >= start_datetime,
                        Game.date < end_datetime,
                    ).all()
                    # If still none, try naive datetime
                    if not all_games:
                        start_datetime_naive = datetime.combine(date_obj, datetime.min.time())
                        end_datetime_naive = datetime.combine(date_obj + timedelta(days=1), datetime.min.time())
                        all_games = db.query(Game).filter(
                            Game.sport == sport,
                            Game.date >= start_datetime_naive,
                            Game.date < end_datetime_naive,
                        ).all()
                else:
                    print(f"  No games available from ESPN for {sport} on {date_str}")
                    return
            except Exception as e:
                print(f"  Error fetching from ESPN: {e}")
                return
        
        # Debug: Show what games were found
        print(f"Found {len(all_games)} total games for {sport} on {date_str}")
        if all_games:
            print(f"  Sample game dates: {[g.date for g in all_games[:3]]}")
            print(f"  Sample game statuses: {[g.status for g in all_games[:3]]}")
            print(f"  Sample game teams: {[f'{g.away_team} @ {g.home_team}' for g in all_games[:3]]}")
        
        # Filter for games that are not final (scheduled or in progress)
        # Also check if games have scores - if no scores, they can't be final
        # Also check game time - if game is in the future, it can't be final
        from datetime import timezone
        now = datetime.now(timezone.utc)
        
        games = []
        for g in all_games:
            # Game is final only if:
            # 1. Status is "final" AND
            # 2. Has scores AND
            # 3. Game time is in the past
            game_time = g.date
            if game_time.tzinfo is None:
                # If naive datetime, assume UTC
                game_time = game_time.replace(tzinfo=timezone.utc)
            
            is_final = (
                g.status and 
                g.status.lower() in ["final", "completed", "finished"] and
                g.home_score is not None and 
                g.away_score is not None and
                game_time < now  # Game has started/finished
            )
            
            # Also check: if game is in the future, it can't be final
            if game_time > now:
                # Game hasn't started yet - definitely not final
                is_final = False
            
            if not is_final:
                games.append(g)
        
        if not games:
            print(f"No scheduled/in-progress games found for {sport} on {date_str}")
            print(f"  Found {len(all_games)} total games, but all are final/completed")
            if all_games:
                print(f"  Game statuses: {[g.status for g in all_games[:5]]}")
                print(f"  Game scores: {[f'{g.away_score}-{g.home_score}' if g.home_score is not None else 'no scores' for g in all_games[:5]]}")
                print(f"  All game IDs: {[g.id for g in all_games[:5]]}")
                
                # Debug: Check if games might be misclassified
                misclassified = [g for g in all_games if g.status == "final" and (g.home_score is None or g.away_score is None)]
                if misclassified:
                    print(f"  ⚠️  Found {len(misclassified)} games marked 'final' but have no scores (likely misclassified)")
            return
        
        print(f"Found {len(games)} games for {sport} on {date_str}")
        
        # Get model version for storing predictions
        model_version = get_model_version()
        
        # Generate predictions
        predictions = []
        stored_count = 0
        for game in games:
            try:
                pred = predict_for_game(game, sport, config)
                if "error" not in pred:
                    predictions.append(pred)
                    
                    # Store predictions in database
                    try:
                        stored = store_predictions_for_game(
                            game_id=game.id,
                            sport=sport,
                            predictions_dict=pred,
                            model_version=model_version
                        )
                        stored_count += len(stored)
                    except Exception as e:
                        print(f"Warning: Could not store predictions for game {game.id}: {e}")
            except Exception as e:
                print(f"Error predicting for game {game.id}: {e}")
                continue
        
        if not predictions:
            print("No valid predictions generated")
            return
        
        print(f"✓ Stored {stored_count} predictions in database")
        
        # Export
        filename_prefix = f"{sport}_{date_str.replace('-', '')}"
        output_paths = export_predictions(
            predictions,
            output_dir=output_dir,
            formats=["csv", "json"],
            min_edge=min_edge,
            filename_prefix=filename_prefix
        )
        
        print(f"\n✓ Exported predictions to:")
        for fmt, path in output_paths.items():
            print(f"  {fmt}: {path}")
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export predictions for upcoming games")
    parser.add_argument("--sport", type=str, required=True, help="Sport code (NFL, NHL, NBA, MLB)")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge threshold")
    
    args = parser.parse_args()
    export_predictions_for_date(
        args.sport,
        args.date,
        args.config,
        args.output_dir,
        args.min_edge
    )
