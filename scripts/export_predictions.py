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
    
    X = df[available_cols].fillna(0)
    
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
        
        X_scaled = ml_scaler.transform(X)
        home_win_prob = ml_model.predict_proba(X_scaled)[0, 1]
        away_win_prob = 1 - home_win_prob
        
        ml_edges = calculate_moneyline_edge(
            home_win_prob,
            away_win_prob,
            game.home_moneyline,
            game.away_moneyline
        )
        
        predictions["moneyline"] = {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_odds": game.home_moneyline,
            "away_odds": game.away_moneyline,
            "home_implied_prob": american_odds_to_implied_prob(game.home_moneyline) if game.home_moneyline else 0.5,
            "away_implied_prob": american_odds_to_implied_prob(game.away_moneyline) if game.away_moneyline else 0.5,
            "home_edge": ml_edges["home_edge"],
            "away_edge": ml_edges["away_edge"],
            "best_side": ml_edges["best_side"],
            "best_edge": ml_edges["best_edge"],
        }
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
        available_cols_spread = [col for col in feature_cols_spread if col in df_spread.columns]
        X_spread = df_spread[available_cols_spread].fillna(0)
        
        X_spread_scaled = spread_scaler.transform(X_spread)
        cover_prob = spread_model.predict_proba(X_spread_scaled)[0, 1]
        
        # Assume -110 odds for spread
        spread_odds = -110
        edge = calculate_spread_edge(cover_prob, spread_odds)
        
        predictions["spread"] = {
            "cover_prob": cover_prob,
            "line": game.spread,
            "price": spread_odds,
            "implied_prob": american_odds_to_implied_prob(spread_odds),
            "edge": edge,
            "side": "favorite" if cover_prob > 0.5 else "underdog",
        }
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
        available_cols_totals = [col for col in feature_cols_totals if col in df_totals.columns]
        X_totals = df_totals[available_cols_totals].fillna(0)
        
        X_totals_scaled = totals_scaler.transform(X_totals)
        over_prob = totals_model.predict_proba(X_totals_scaled)[0, 1]
        under_prob = 1 - over_prob
        
        # Assume -110 odds for both sides
        over_odds = -110
        under_odds = -110
        totals_edges = calculate_totals_edge(over_prob, under_prob, over_odds, under_odds)
        
        predictions["totals"] = {
            "over_prob": over_prob,
            "under_prob": under_prob,
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
    except Exception as e:
        print(f"Error predicting totals: {e}")
        predictions["totals"] = {}
    
    # Score projections
    try:
        score_models = load_score_models(sport)
        
        # Use moneyline features for score projection
        feature_cols_score = get_feature_columns(sport, "score_projection")
        available_cols_score = [col for col in feature_cols_score if col in df.columns]
        X_score = df[available_cols_score].fillna(0)
        
        home_model = score_models["home"]["model"]
        home_scaler = score_models["home"]["scaler"]
        away_model = score_models["away"]["model"]
        away_scaler = score_models["away"]["scaler"]
        
        X_home_scaled = home_scaler.transform(X_score)
        X_away_scaled = away_scaler.transform(X_score)
        
        proj_home_score = home_model.predict(X_home_scaled)[0]
        proj_away_score = away_model.predict(X_away_scaled)[0]
        
        predictions["proj_home_score"] = float(proj_home_score)
        predictions["proj_away_score"] = float(proj_away_score)
    except Exception as e:
        print(f"Error predicting scores: {e}")
        predictions["proj_home_score"] = None
        predictions["proj_away_score"] = None
    
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
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Use date-only comparison to avoid timezone issues
        # This works regardless of whether dates are timezone-aware or naive
        all_games = db.query(Game).filter(
            Game.sport == sport,
            func.date(Game.date) == date_obj
        ).all()
        
        # If using PostgreSQL and func.date doesn't work, try datetime range
        if not all_games:
            from datetime import timezone
            start_datetime = datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc)
            end_datetime = datetime.combine(date_obj + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            all_games = db.query(Game).filter(
                Game.sport == sport,
                Game.date >= start_datetime,
                Game.date < end_datetime,
            ).all()
        
        # Final fallback: naive datetime (for SQLite)
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
        games = [
            g for g in all_games 
            if g.status and g.status.lower() not in ["final", "completed", "finished"]
        ]
        
        if not games:
            print(f"No scheduled/in-progress games found for {sport} on {date_str}")
            print(f"  Found {len(all_games)} total games, but all are final/completed")
            if all_games:
                print(f"  Game statuses: {[g.status for g in all_games[:5]]}")
                print(f"  All game IDs: {[g.id for g in all_games[:5]]}")
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
