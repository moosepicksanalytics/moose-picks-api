"""
API endpoints for Lovable integration and scheduled tasks.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
from typing import Optional
import asyncio

router = APIRouter()


@router.post("/backfill")
def trigger_backfill(
    background_tasks: BackgroundTasks,
    sports: Optional[str] = None,
    seasons: Optional[str] = None,
    delay: float = 0.1
):
    """
    Trigger historical data backfill from API.
    This endpoint can be called via HTTP POST to backfill historical game data.
    
    Args:
        sports: Comma-separated list of sports (e.g., "NFL,NHL") - default: all 4 sports
        seasons: Comma-separated list of season years (e.g., "2023,2024") - default: past 5 seasons
        delay: Delay between API calls in seconds (default: 0.1)
    
    Returns:
        Status message
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.backfill_historical_data import backfill_sport, backfill_all_sports
    from datetime import datetime
    
    # Parse sports
    if sports:
        sports_list = [s.strip().upper() for s in sports.split(",")]
    else:
        sports_list = ["NFL", "NHL", "NBA", "MLB"]
    
    # Parse seasons
    if seasons:
        seasons_list = [int(s.strip()) for s in seasons.split(",")]
    else:
        # Default: past 5 seasons
        current_year = datetime.now().year
        seasons_list = list(range(current_year - 4, current_year + 1))
    
    def run_backfill():
        try:
            if len(sports_list) == 4 and sports_list == ["NFL", "NHL", "NBA", "MLB"]:
                # All sports
                backfill_all_sports(seasons=seasons_list, delay=delay)
            else:
                # Specific sports
                total = 0
                for sport in sports_list:
                    games = backfill_sport(sport, seasons_list, delay=delay)
                    total += games
                print(f"\n✓ Total across all sports: {total} games")
        except Exception as e:
            print(f"Error in backfill: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_backfill)
    
    return {
        "status": "started",
        "message": "Backfill started in background",
        "sports": sports_list,
        "seasons": seasons_list,
        "delay": delay,
        "note": "Check Railway logs to monitor progress"
    }


@router.post("/backfill-odds")
def trigger_backfill_odds(
    background_tasks: BackgroundTasks,
    sport: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
    dry_run: bool = False
):
    """
    Backfill historical odds data for games in the database.
    
    ⚠️  COST WARNING: Historical odds cost 30 credits per date (10x more than current odds).
    With 20k credits/month, you can backfill ~666 dates.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB) - if not provided, backfills all sports
        start_date: Start date in YYYY-MM-DD format (required)
        end_date: End date in YYYY-MM-DD format (required)
        dry_run: If True, only report what would be done (no API calls)
    
    Returns:
        Status message with cost estimate
    
    Example:
        POST /api/backfill-odds?sport=NFL&start_date=2024-10-01&end_date=2024-10-31
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.backfill_odds import backfill_odds_for_date_range, backfill_all_sports
    
    if not start_date or not end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date and end_date are required (YYYY-MM-DD format)"
        )
    
    def run_backfill():
        try:
            if sport:
                result = backfill_odds_for_date_range(
                    sport=sport.upper(),
                    start_date=start_date,
                    end_date=end_date,
                    dry_run=dry_run,
                    use_historical=True
                )
            else:
                result = backfill_all_sports(
                    start_date=start_date,
                    end_date=end_date,
                    dry_run=dry_run,
                    use_historical=True
                )
            print(f"\n✓ Backfill completed: {result}")
        except Exception as e:
            print(f"Error in odds backfill: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_backfill)
    
    # Estimate cost
    from datetime import datetime
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        days = (end - start).days + 1
        cost_per_date = 30  # 3 markets × 1 region × 10 credits
        estimated_cost = days * cost_per_date
        if sport:
            total_cost = estimated_cost
        else:
            total_cost = estimated_cost * 4  # All 4 sports
    except:
        estimated_cost = None
        total_cost = None
    
    return {
        "status": "started",
        "message": "Odds backfill started in background",
        "sport": sport or "all",
        "start_date": start_date,
        "end_date": end_date,
        "dry_run": dry_run,
        "estimated_cost_credits": total_cost if total_cost else "unknown",
        "note": "Check Railway logs to monitor progress. Historical odds cost 30 credits per date."
    }


@router.post("/trigger-daily-workflow")
def trigger_daily_workflow(
    background_tasks: BackgroundTasks,
    train: bool = True,
    predict: bool = True,
    sports: Optional[str] = None,
    min_edge: float = 0.05
):
    """
    Trigger daily workflow from Lovable or external scheduler.
    This endpoint can be called by Lovable's scheduled tasks or Railway cron.
    
    Args:
        train: Whether to retrain models
        predict: Whether to generate predictions
        sports: Comma-separated list of sports (e.g., "NFL,NHL")
        min_edge: Minimum edge threshold
    
    Returns:
        Status message
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.daily_automation import daily_workflow
    
    sports_list = [s.strip() for s in sports.split(",")] if sports else ["NFL", "NHL", "NBA", "MLB"]
    
    def run_workflow():
        try:
            daily_workflow(
                sports=sports_list,
                train=train,
                predict=predict,
                min_edge=min_edge
            )
        except Exception as e:
            print(f"Error in daily workflow: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_workflow)
    
    return {
        "status": "started",
        "message": "Daily workflow started in background",
        "sports": sports_list,
        "train": train,
        "predict": predict
    }


@router.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    from app.database import SessionLocal
    from app.models.db_models import Game
    from pathlib import Path
    
    db = SessionLocal()
    try:
        game_count = db.query(Game).count()
        model_dir = Path("models")
        model_count = len(list(model_dir.glob("*.pkl"))) if model_dir.exists() else 0
        
        return {
            "status": "healthy",
            "database": "connected",
            "games_in_db": game_count,
            "models_trained": model_count,
            "timestamp": datetime.now().isoformat()
        }
    finally:
        db.close()


@router.get("/predictions/latest")
def get_latest_predictions(sport: str = "NFL", limit: int = 10):
    """
    Get latest predictions from database (unsettled predictions).
    Falls back to CSV exports if database has no predictions.
    Useful for Lovable to fetch recent predictions.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        limit: Maximum number of predictions to return
    """
    from app.database import SessionLocal
    from app.models.db_models import Prediction, Game
    from datetime import datetime, timedelta
    import logging
    
    logger = logging.getLogger(__name__)
    
    db = SessionLocal()
    try:
        # Get unsettled predictions for this sport, ordered by most recent
        try:
            predictions = db.query(Prediction).filter(
                Prediction.sport == sport,
                Prediction.settled == False
            ).order_by(Prediction.predicted_at.desc()).limit(limit * 2).all()  # Get more to filter by edge
        except Exception as db_error:
            logger.warning(f"Database query failed: {db_error}. Trying CSV fallback.")
            db.close()
            return _get_latest_predictions_from_csv(sport, limit)
        
        if not predictions:
            # No predictions in database, try CSV fallback
            logger.info(f"No predictions found in database for {sport}. Trying CSV fallback.")
            db.close()
            return _get_latest_predictions_from_csv(sport, limit)
        
        # Get associated games
        game_ids = [p.game_id for p in predictions]
        try:
            games = {g.id: g for g in db.query(Game).filter(Game.id.in_(game_ids)).all()}
        except Exception as game_error:
            logger.warning(f"Error fetching games: {game_error}")
            games = {}
        
        # Helper function to safely convert float values (handle nan)
        import math
        def safe_float(value):
            """Convert float to JSON-safe value (None if nan or None)."""
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return float(value) if value is not None else None
        
        # Convert to dict format
        picks = []
        for pred in predictions:
            game = games.get(pred.game_id)
            if not game:
                continue
            
            # Build prediction dict based on market
            pick = {
                "game_id": pred.game_id,
                "sport": pred.sport,
                "market": pred.market,
                "date": game.date.isoformat() if game.date else None,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "model_version": pred.model_version,
            }
            
            # Add market-specific data with safe float conversion
            if pred.market == "moneyline":
                home_prob = safe_float(pred.home_win_prob)
                pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                pick["home_win_prob"] = home_prob
                pick["edge"] = abs(home_prob - 0.5) if home_prob is not None else 0.0
            elif pred.market == "spread":
                spread_prob = safe_float(pred.spread_cover_prob)
                pick["side"] = "home" if spread_prob and spread_prob > 0.5 else "away"
                pick["spread_cover_prob"] = spread_prob
                pick["spread"] = safe_float(game.spread)
                pick["edge"] = abs(spread_prob - 0.5) if spread_prob is not None else 0.0
            elif pred.market == "totals" or pred.market == "over_under":
                over_prob = safe_float(pred.over_prob)
                pick["side"] = "over" if over_prob and over_prob > 0.5 else "under"
                pick["over_prob"] = over_prob
                pick["over_under"] = safe_float(game.over_under)
                pick["edge"] = abs(over_prob - 0.5) if over_prob is not None else 0.0
            
            pick["recommended_kelly"] = safe_float(pred.recommended_kelly)
            pick["recommended_unit_size"] = safe_float(pred.recommended_unit_size) if pred.recommended_unit_size is not None else 1.0
            
            picks.append(pick)
        
        # Sort by edge and limit
        picks_sorted = sorted(picks, key=lambda x: x.get("edge", 0), reverse=True)[:limit]
        
        return {
            "sport": sport,
            "source": "database",
            "total_predictions": len(picks),
            "top_picks": picks_sorted
        }
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Unexpected error in get_latest_predictions: {e}", exc_info=True)
        # If database query fails, try CSV fallback
        try:
            db.close()
            return _get_latest_predictions_from_csv(sport, limit)
        except Exception as csv_error:
            # If both fail, return empty result instead of 500 error
            logger.error(f"Both database and CSV fallback failed: {csv_error}")
            return {
                "sport": sport,
                "source": "none",
                "total_predictions": 0,
                "top_picks": [],
                "message": f"No predictions available. Error: {str(e)}"
            }
    finally:
        try:
            db.close()
        except:
            pass


def _get_latest_predictions_from_csv(sport: str, limit: int):
    """Fallback: Get predictions from CSV exports directory."""
    from pathlib import Path
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    exports_dir = Path("exports")
    if not exports_dir.exists():
        logger.info(f"Exports directory not found for {sport}")
        return {
            "sport": sport,
            "source": "none",
            "total_predictions": 0,
            "top_picks": [],
            "message": "No exports directory found and no database predictions available"
        }
    
    # Find latest CSV for the sport
    csv_files = sorted(exports_dir.glob(f"{sport}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not csv_files:
        logger.info(f"No CSV files found for {sport}")
        return {
            "sport": sport,
            "source": "none",
            "total_predictions": 0,
            "top_picks": [],
            "message": f"No predictions found for {sport} (no CSV files and no database predictions)"
        }
    
    latest_csv = csv_files[0]
    
    try:
        df = pd.read_csv(latest_csv)
        
        if df.empty:
            logger.info(f"CSV file {latest_csv.name} is empty")
            return {
                "sport": sport,
                "source": "csv",
                "file": latest_csv.name,
                "total_predictions": 0,
                "top_picks": [],
                "message": f"CSV file {latest_csv.name} is empty"
            }
        
        # Check if 'edge' column exists
        if "edge" not in df.columns:
            logger.warning(f"CSV file {latest_csv.name} missing 'edge' column")
            return {
                "sport": sport,
                "source": "csv",
                "file": latest_csv.name,
                "total_predictions": 0,
                "top_picks": [],
                "message": f"CSV file {latest_csv.name} missing 'edge' column"
            }
        
        # Return top picks by edge
        df_sorted = df.sort_values("edge", ascending=False).head(limit)
        
        return {
            "sport": sport,
            "source": "csv",
            "file": latest_csv.name,
            "total_predictions": len(df),
            "top_picks": df_sorted.to_dict("records")
        }
    except Exception as e:
        logger.error(f"Error reading CSV file {latest_csv.name}: {e}")
        return {
            "sport": sport,
            "source": "none",
            "total_predictions": 0,
            "top_picks": [],
            "message": f"Error reading CSV file {latest_csv.name}: {str(e)}"
        }


@router.get("/predictions/date-range")
def get_predictions_by_date_range(
    sport: str,
    start_date: str,
    end_date: Optional[str] = None,
    limit: int = 50
):
    """
    Get predictions for a date range.
    
    Examples:
    - NHL next 3 days: /api/predictions/date-range?sport=NHL&start_date=2024-12-08&days=3
    - NFL this week: /api/predictions/date-range?sport=NFL&start_date=2024-12-08&end_date=2024-12-14
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to start_date)
        limit: Maximum number of predictions to return
    """
    from pathlib import Path
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end = start
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    if end < start:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")
    
    # Query database for predictions in date range
    from app.database import SessionLocal
    from app.models.db_models import Prediction, Game
    
    db = SessionLocal()
    try:
        # Get predictions for games in the date range
        start_datetime = datetime.combine(start, datetime.min.time())
        end_datetime = datetime.combine(end + timedelta(days=1), datetime.min.time())
        
        # Get games in date range
        games = db.query(Game).filter(
            Game.sport == sport,
            Game.date >= start_datetime,
            Game.date < end_datetime
        ).all()
        
        game_ids = [g.id for g in games]
        
        if not game_ids:
            return {
                "sport": sport,
                "start_date": start_date,
                "end_date": end_date or start_date,
                "total_predictions": 0,
                "top_picks": []
            }
        
        # Get predictions for these games
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport,
            Prediction.game_id.in_(game_ids),
            Prediction.settled == False  # Only unsettled predictions
        ).all()
        
        # Helper function to safely convert float values (handle nan)
        import math
        def safe_float(value):
            """Convert float to JSON-safe value (None if nan or None)."""
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return float(value) if value is not None else None
        
        # Convert to dict format
        picks = []
        for pred in predictions:
            game = next((g for g in games if g.id == pred.game_id), None)
            if not game:
                continue
            
            # Build prediction dict based on market
            pick = {
                "game_id": pred.game_id,
                "sport": pred.sport,
                "market": pred.market,
                "date": game.date.isoformat() if game.date else None,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "model_version": pred.model_version,
            }
            
            # Add market-specific data with safe float conversion
            if pred.market == "moneyline":
                home_prob = safe_float(pred.home_win_prob)
                pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                pick["home_win_prob"] = home_prob
                pick["edge"] = abs(home_prob - 0.5) if home_prob is not None else 0.0
            elif pred.market == "spread":
                spread_prob = safe_float(pred.spread_cover_prob)
                pick["side"] = "home" if spread_prob and spread_prob > 0.5 else "away"
                pick["spread_cover_prob"] = spread_prob
                pick["spread"] = safe_float(game.spread)
                pick["edge"] = abs(spread_prob - 0.5) if spread_prob is not None else 0.0
            elif pred.market == "totals" or pred.market == "over_under":
                over_prob = safe_float(pred.over_prob)
                pick["side"] = "over" if over_prob and over_prob > 0.5 else "under"
                pick["over_prob"] = over_prob
                pick["over_under"] = safe_float(game.over_under)
                pick["edge"] = abs(over_prob - 0.5) if over_prob is not None else 0.0
            
            pick["recommended_kelly"] = safe_float(pred.recommended_kelly)
            pick["recommended_unit_size"] = safe_float(pred.recommended_unit_size) if pred.recommended_unit_size is not None else 1.0
            
            picks.append(pick)
        
        # Sort by edge and limit
        picks_sorted = sorted(picks, key=lambda x: x.get("edge", 0), reverse=True)[:limit]
        
        return {
            "sport": sport,
            "start_date": start_date,
            "end_date": end_date or start_date,
            "total_predictions": len(picks),
            "top_picks": picks_sorted
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching predictions: {str(e)}"
        )
    finally:
        db.close()


@router.get("/predictions/next-days")
def get_predictions_next_days(
    sport: str,
    days: int = 3,
    limit: int = 50
):
    """
    Get predictions for the next N days (convenience endpoint).
    
    Examples:
    - NHL next 3 days: /api/predictions/next-days?sport=NHL&days=3
    - NFL next 7 days: /api/predictions/next-days?sport=NFL&days=7
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        days: Number of days ahead (default: 3)
        limit: Maximum number of predictions to return
    """
    today = datetime.now().date()
    end_date = today + timedelta(days=days - 1)
    
    return get_predictions_by_date_range(
        sport=sport,
        start_date=today.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        limit=limit
    )


@router.get("/predictions/week")
def get_predictions_this_week(
    sport: str,
    limit: int = 50
):
    """
    Get predictions for the current week (Monday to Sunday).
    
    Example: /api/predictions/week?sport=NFL
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        limit: Maximum number of predictions to return
    """
    today = datetime.now().date()
    # Get Monday of current week
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    sunday = monday + timedelta(days=6)
    
    return get_predictions_by_date_range(
        sport=sport,
        start_date=monday.strftime("%Y-%m-%d"),
        end_date=sunday.strftime("%Y-%m-%d"),
        limit=limit
    )
