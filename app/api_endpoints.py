"""
API endpoints for Lovable integration and scheduled tasks.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends, Query
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from app.security import require_api_key

router = APIRouter()


@router.post("/backfill")
def trigger_backfill(
    background_tasks: BackgroundTasks,
    request: Request,
    sports: Optional[str] = None,
    seasons: Optional[str] = None,
    delay: float = 0.1,
    api_key: str = Depends(require_api_key)
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
    request: Request,
    sport: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
    dry_run: bool = False,
    api_key: str = Depends(require_api_key)
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


@router.post("/migrate-ou-columns")
def trigger_migrate_ou_columns(
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Depends(require_api_key)
):
    """
    Run database migration to add O/U columns (closing_total, actual_total, ou_result).
    
    Returns:
        Status message
        
    Example:
        POST /api/migrate-ou-columns
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.migrate_add_ou_columns import migrate_add_ou_columns
    
    def run_migration():
        try:
            migrate_add_ou_columns()
            print("✓ Migration completed successfully")
        except Exception as e:
            print(f"✗ Migration failed: {e}")
            import traceback
            traceback.print_exc()
    
    background_tasks.add_task(run_migration)
    
    return {
        "status": "started",
        "message": "Database migration started in background",
        "note": "Check Railway logs to monitor progress. This adds closing_total, actual_total, and ou_result columns."
    }


@router.post("/backfill-ou-data")
def trigger_backfill_ou_data(
    background_tasks: BackgroundTasks,
    request: Request,
    sport: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: str = Depends(require_api_key)
):
    """
    Backfill Over/Under (O/U) data for historical games.
    
    Calculates actual_total and ou_result from existing scores and over_under values.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB, or ALL) - default: ALL
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        Status message
        
    Example:
        POST /api/backfill-ou-data?sport=ALL
        POST /api/backfill-ou-data?sport=NFL&start_date=2024-01-01&end_date=2024-12-31
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.backfill_ou_data import backfill_ou_for_sport, backfill_all_sports
    from datetime import datetime
    
    def run_backfill():
        try:
            if sport and sport.upper() == "ALL":
                backfill_all_sports()
            elif sport:
                start = datetime.fromisoformat(start_date) if start_date else None
                end = datetime.fromisoformat(end_date) if end_date else None
                backfill_ou_for_sport(sport.upper(), start, end)
            else:
                backfill_all_sports()
            print("✓ O/U backfill completed")
        except Exception as e:
            print(f"✗ O/U backfill failed: {e}")
            import traceback
            traceback.print_exc()
    
    background_tasks.add_task(run_backfill)
    
    return {
        "status": "started",
        "message": "O/U data backfill started in background",
        "sport": sport or "ALL",
        "start_date": start_date,
        "end_date": end_date,
        "note": "Check Railway logs to monitor progress"
    }


@router.post("/train")
def train_model(
    sport: str,
    market: str,
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Depends(require_api_key)
):
    """
    Train a model for a specific sport and market.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals)
    
    Returns:
        Status message indicating training has started
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.training.pipeline import train_model_for_market
    
    def run_training():
        try:
            result = train_model_for_market(sport.upper(), market.lower())
            print(f"Training completed for {sport} {market}: {result}")
        except Exception as e:
            print(f"Error training {sport} {market}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_training)
    
    return {
        "status": "started",
        "message": f"Training started for {sport.upper()} {market.lower()}",
        "sport": sport.upper(),
        "market": market.lower()
    }


@router.get("/debug-ou-results")
def debug_ou_results(sport: str):
    """
    Debug endpoint to see what ou_result values actually exist in database.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
    
    Returns:
        Sample of ou_result values and counts
    """
    from app.database import SessionLocal
    from app.models.db_models import Game
    from sqlalchemy import func
    
    db = SessionLocal()
    try:
        # Get sample of actual ou_result values
        samples = db.query(Game.ou_result).filter(
            Game.sport == sport.upper(),
            Game.status == "final",
            Game.ou_result.isnot(None)
        ).limit(20).all()
        
        # Get counts by exact value
        exact_counts = db.query(
            Game.ou_result,
            func.count(Game.id).label('count')
        ).filter(
            Game.sport == sport.upper(),
            Game.status == "final",
            Game.ou_result.isnot(None)
        ).group_by(Game.ou_result).limit(10).all()
        
        return {
            "sport": sport.upper(),
            "sample_values": [repr(r[0]) for r in samples],
            "exact_counts": {str(r[0]): int(r[1]) for r in exact_counts if r[0]},
            "total_with_ou_result": len(samples)
        }
    finally:
        db.close()


@router.get("/validate-ou-coverage")
def get_ou_coverage(sport: Optional[str] = None):
    """
    Validate Over/Under data coverage for training.
    
    Returns coverage statistics and distribution for each sport.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB) - if not provided, returns all sports
    
    Returns:
        Coverage statistics including total games, games with O/U data, distribution, and training readiness
        
    Example:
        GET /api/validate-ou-coverage
        GET /api/validate-ou-coverage?sport=NFL
    """
    import sys
    from pathlib import Path
    import logging
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.backfill_ou_data import validate_ou_coverage
    
    # Suppress logging output for API response
    log_capture = io.StringIO()
    
    try:
        if sport:
            # Redirect stdout/stderr to capture and suppress logs
            with redirect_stdout(log_capture), redirect_stderr(log_capture):
                coverage = validate_ou_coverage(sport.upper())
            
            # Ensure distribution is a proper dict (not None)
            distribution = coverage.get("distribution", {}) or {}
            
            return {
                "sport": sport.upper(),
                "coverage": {
                    "sport": coverage.get("sport", sport.upper()),
                    "total_completed": int(coverage.get("total_completed", 0)),
                    "with_ou_data": int(coverage.get("with_ou_data", 0)),
                    "coverage_pct": float(coverage.get("coverage_pct", 0)),
                    "distribution": distribution,
                    "can_train": bool(coverage.get("can_train", False))
                }
            }
        else:
            results = {}
            for s in ["NFL", "NHL", "NBA", "MLB"]:
                with redirect_stdout(log_capture), redirect_stderr(log_capture):
                    coverage = validate_ou_coverage(s)
                
                # Ensure distribution is a proper dict (not None)
                distribution = coverage.get("distribution", {}) or {}
                
                results[s] = {
                    "sport": coverage.get("sport", s),
                    "total_completed": int(coverage.get("total_completed", 0)),
                    "with_ou_data": int(coverage.get("with_ou_data", 0)),
                    "coverage_pct": float(coverage.get("coverage_pct", 0)),
                    "distribution": distribution,
                    "can_train": bool(coverage.get("can_train", False))
                }
            return {
                "all_sports": results
            }
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500, 
            detail=f"Error validating O/U coverage: {str(e)}\n{traceback.format_exc()}"
        )


@router.post("/trigger-daily-workflow")
def trigger_daily_workflow(
    background_tasks: BackgroundTasks,
    request: Request,
    train: bool = True,
    predict: bool = True,
    sports: Optional[str] = None,
    min_edge: float = 0.05,
    api_key: str = Depends(require_api_key)
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


@router.get("/debug-logs")
def get_debug_logs(limit: Optional[int] = 100):
    """
    Retrieve debug logs from .cursor/debug.log file.
    
    Args:
        limit: Maximum number of log entries to return (default: 100, use 0 for all)
    
    Returns:
        JSON object with log entries and metadata
    """
    from pathlib import Path
    import json
    
    # Path to debug log file
    debug_log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
    
    if not debug_log_path.exists():
        return {
            "status": "not_found",
            "message": "Debug log file does not exist yet. Run predictions first.",
            "path": str(debug_log_path),
            "entries": [],
            "count": 0
        }
    
    try:
        # Read all lines from the log file
        with open(debug_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse each line as JSON (NDJSON format)
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
        
        # Reverse to get most recent first, then apply limit
        entries.reverse()
        if limit and limit > 0:
            entries = entries[:limit]
        
        return {
            "status": "success",
            "path": str(debug_log_path),
            "total_entries": len(entries),
            "file_size_bytes": debug_log_path.stat().st_size,
            "entries": entries
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading debug log: {str(e)}",
            "path": str(debug_log_path),
            "entries": [],
            "count": 0
        }


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
        
        # Import JSON sanitization and edge calculation utilities
        from app.utils.json_sanitize import safe_float, sanitize_dict
        from app.utils.odds import (
            calculate_moneyline_edge,
            calculate_spread_edge,
            calculate_totals_edge,
            american_odds_to_implied_prob
        )
        
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
                away_prob = 1 - home_prob if home_prob is not None else None
                pick["home_win_prob"] = home_prob
                
                # Calculate actual edge using odds
                if home_prob is not None and game.home_moneyline is not None and game.away_moneyline is not None:
                    ml_edges = calculate_moneyline_edge(
                        home_prob, away_prob,
                        safe_float(game.home_moneyline),
                        safe_float(game.away_moneyline)
                    )
                    # Sanitize edge calculation results to remove NaN/Inf
                    ml_edges = sanitize_dict(ml_edges)
                    
                    # Use best_side from edge calculation (which side has positive edge)
                    # This ensures we recommend the side with value, not just the favorite
                    best_edge = safe_float(ml_edges.get("best_edge", 0))
                    if ml_edges.get("best_side") and best_edge is not None and best_edge > 0:
                        pick["side"] = ml_edges["best_side"]
                        pick["edge"] = best_edge
                    else:
                        # No positive edge, use probability-based side
                        pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                        if pick["side"] == "home":
                            pick["edge"] = safe_float(ml_edges.get("home_edge")) or 0.0
                        else:
                            pick["edge"] = safe_float(ml_edges.get("away_edge")) or 0.0
                else:
                    # No odds available, use probability-based side
                    pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                    pick["edge"] = 0.0
            elif pred.market == "spread":
                spread_prob = safe_float(pred.spread_cover_prob)
                pick["side"] = "home" if spread_prob and spread_prob > 0.5 else "away"
                pick["spread_cover_prob"] = spread_prob
                pick["spread"] = safe_float(game.spread)
                
                # Calculate actual edge using odds (default to -110 if no odds)
                spread_odds = safe_float(game.spread_odds) if hasattr(game, 'spread_odds') and game.spread_odds else -110.0
                if spread_prob is not None and spread_odds is not None:
                    edge = calculate_spread_edge(spread_prob, spread_odds)
                    pick["edge"] = safe_float(edge) or 0.0
                else:
                    pick["edge"] = 0.0
            elif pred.market == "totals" or pred.market == "over_under":
                over_prob = safe_float(pred.over_prob)
                under_prob = 1 - over_prob if over_prob is not None else None
                pick["side"] = "over" if over_prob and over_prob > 0.5 else "under"
                pick["over_prob"] = over_prob
                pick["over_under"] = safe_float(game.over_under)
                
                # Calculate actual edge using odds (default to -110 if no odds)
                over_odds = safe_float(game.over_odds) if hasattr(game, 'over_odds') and game.over_odds else -110.0
                under_odds = safe_float(game.under_odds) if hasattr(game, 'under_odds') and game.under_odds else -110.0
                if over_prob is not None and over_odds is not None and under_odds is not None:
                    totals_edges = calculate_totals_edge(over_prob, under_prob, over_odds, under_odds)
                    # Sanitize edge calculation results to remove NaN/Inf
                    totals_edges = sanitize_dict(totals_edges)
                    
                    if pick["side"] == "over":
                        pick["edge"] = safe_float(totals_edges.get("over_edge")) or 0.0
                    else:
                        pick["edge"] = safe_float(totals_edges.get("under_edge")) or 0.0
                else:
                    pick["edge"] = 0.0
            
            pick["recommended_kelly"] = safe_float(pred.recommended_kelly)
            pick["recommended_unit_size"] = safe_float(pred.recommended_unit_size) if pred.recommended_unit_size is not None else 1.0
            
            picks.append(pick)
        
        # Sort by edge and limit
        picks_sorted = sorted(picks, key=lambda x: x.get("edge", 0) or 0, reverse=True)[:limit]
        
        # Final sanitization pass to ensure all values are JSON-safe
        picks_sorted = [sanitize_dict(pick) for pick in picks_sorted]
        
        return {
            "sport": sport,
            "source": "database",
            "total_predictions": len(picks),
            "top_picks": picks_sorted
        }
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_latest_predictions: {e}\n{error_trace}")
        
        # If database query fails, try CSV fallback
        try:
            db.close()
            return _get_latest_predictions_from_csv(sport, limit)
        except Exception as csv_error:
            # If both fail, return empty result instead of 500 error
            csv_trace = traceback.format_exc()
            logger.error(f"Both database and CSV fallback failed: {csv_error}\n{csv_trace}")
            
            # Return error details in development, generic message in production
            import os
            if os.getenv("DEBUG", "false").lower() == "true":
                return {
                    "sport": sport,
                    "source": "error",
                    "total_predictions": 0,
                    "top_picks": [],
                    "error": str(e),
                    "traceback": error_trace
                }
            else:
                return {
                    "sport": sport,
                    "source": "none",
                    "total_predictions": 0,
                    "top_picks": [],
                    "message": "No predictions available"
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
        
        # Return top picks by edge (handle NaN values in edge column)
        df_sorted = df.sort_values("edge", ascending=False, na_position='last').head(limit)
        
        # Sanitize DataFrame records to remove NaN values before converting to dict
        from app.utils.json_sanitize import sanitize_dict
        records = df_sorted.to_dict("records")
        sanitized_records = [sanitize_dict(record) for record in records]
        
        return {
            "sport": sport,
            "source": "csv",
            "file": latest_csv.name,
            "total_predictions": len(df),
            "top_picks": sanitized_records
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
        
        # Import JSON sanitization and edge calculation utilities
        from app.utils.json_sanitize import safe_float, sanitize_dict
        from app.utils.odds import (
            calculate_moneyline_edge,
            calculate_spread_edge,
            calculate_totals_edge,
            american_odds_to_implied_prob
        )
        
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
                away_prob = 1 - home_prob if home_prob is not None else None
                pick["home_win_prob"] = home_prob
                
                # Calculate actual edge using odds
                if home_prob is not None and game.home_moneyline is not None and game.away_moneyline is not None:
                    ml_edges = calculate_moneyline_edge(
                        home_prob, away_prob,
                        safe_float(game.home_moneyline),
                        safe_float(game.away_moneyline)
                    )
                    # Sanitize edge calculation results to remove NaN/Inf
                    ml_edges = sanitize_dict(ml_edges)
                    
                    # Use best_side from edge calculation (which side has positive edge)
                    # This ensures we recommend the side with value, not just the favorite
                    best_edge = safe_float(ml_edges.get("best_edge", 0))
                    if ml_edges.get("best_side") and best_edge is not None and best_edge > 0:
                        pick["side"] = ml_edges["best_side"]
                        pick["edge"] = best_edge
                    else:
                        # No positive edge, use probability-based side
                        pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                        if pick["side"] == "home":
                            pick["edge"] = safe_float(ml_edges.get("home_edge")) or 0.0
                        else:
                            pick["edge"] = safe_float(ml_edges.get("away_edge")) or 0.0
                else:
                    # No odds available, use probability-based side
                    pick["side"] = "home" if home_prob and home_prob > 0.5 else "away"
                    pick["edge"] = 0.0
            elif pred.market == "spread":
                spread_prob = safe_float(pred.spread_cover_prob)
                pick["side"] = "home" if spread_prob and spread_prob > 0.5 else "away"
                pick["spread_cover_prob"] = spread_prob
                pick["spread"] = safe_float(game.spread)
                
                # Calculate actual edge using odds (default to -110 if no odds)
                spread_odds = safe_float(game.spread_odds) if hasattr(game, 'spread_odds') and game.spread_odds else -110.0
                if spread_prob is not None and spread_odds is not None:
                    edge = calculate_spread_edge(spread_prob, spread_odds)
                    pick["edge"] = safe_float(edge) or 0.0
                else:
                    pick["edge"] = 0.0
            elif pred.market == "totals" or pred.market == "over_under":
                over_prob = safe_float(pred.over_prob)
                under_prob = 1 - over_prob if over_prob is not None else None
                pick["side"] = "over" if over_prob and over_prob > 0.5 else "under"
                pick["over_prob"] = over_prob
                pick["over_under"] = safe_float(game.over_under)
                
                # Calculate actual edge using odds (default to -110 if no odds)
                over_odds = safe_float(game.over_odds) if hasattr(game, 'over_odds') and game.over_odds else -110.0
                under_odds = safe_float(game.under_odds) if hasattr(game, 'under_odds') and game.under_odds else -110.0
                if over_prob is not None and over_odds is not None and under_odds is not None:
                    totals_edges = calculate_totals_edge(over_prob, under_prob, over_odds, under_odds)
                    # Sanitize edge calculation results to remove NaN/Inf
                    totals_edges = sanitize_dict(totals_edges)
                    
                    if pick["side"] == "over":
                        pick["edge"] = safe_float(totals_edges.get("over_edge")) or 0.0
                    else:
                        pick["edge"] = safe_float(totals_edges.get("under_edge")) or 0.0
                else:
                    pick["edge"] = 0.0
            
            pick["recommended_kelly"] = safe_float(pred.recommended_kelly)
            pick["recommended_unit_size"] = safe_float(pred.recommended_unit_size) if pred.recommended_unit_size is not None else 1.0
            
            picks.append(pick)
        
        # Sort by edge and limit
        picks_sorted = sorted(picks, key=lambda x: x.get("edge", 0) or 0, reverse=True)[:limit]
        
        # Final sanitization pass to ensure all values are JSON-safe
        picks_sorted = [sanitize_dict(pick) for pick in picks_sorted]
        
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


@router.get("/validate-probabilities")
def validate_model_probabilities(
    sport: str,
    market: str
):
    """
    Validate model probabilities and analyze edge distributions.
    
    Checks calibration, edge distribution, and historical performance.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals)
    
    Returns:
        Validation results including calibration metrics and edge analysis
        
    Example:
        GET /api/validate-probabilities?sport=NHL&market=moneyline
    """
    import sys
    from pathlib import Path
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.validate_model_probabilities import validate_model
    
    # Capture output
    output = io.StringIO()
    error_output = io.StringIO()
    
    try:
        with redirect_stdout(output), redirect_stderr(error_output):
            validate_model(sport.upper(), market.lower())
        
        result_text = output.getvalue()
        error_text = error_output.getvalue()
        
        return {
            "status": "success",
            "sport": sport.upper(),
            "market": market.lower(),
            "output": result_text,
            "errors": error_text if error_text else None
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "sport": sport.upper(),
            "market": market.lower(),
            "error": str(e),
            "traceback": error_trace
        }


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


@router.post("/settle-predictions")
def settle_predictions_endpoint(sport: str = Query(...)):
    """
    Settle all unsettled predictions for a sport by matching against game results.
    
    Process:
    1. Get all games with final status for this sport
    2. Get all unsettled predictions for those games
    3. Compare prediction vs actual result
    4. Mark as won/lost
    5. Calculate accuracy metrics
    
    Returns: {"settled": N, "accurate": M, "win_rate": X%}
    """
    from app.prediction.settling import (
        settle_moneyline_prediction,
        settle_spread_prediction,
        settle_totals_prediction
    )
    from app.database import SessionLocal
    from app.models.db_models import Prediction, Game
    
    db = SessionLocal()
    try:
        # Get all games with final status for this sport
        games = db.query(Game).filter(
            Game.sport == sport.upper(),
            Game.status == 'final',
            Game.home_score.isnot(None),
            Game.away_score.isnot(None)
        ).all()
        
        if not games:
            return {
                "sport": sport.upper(),
                "settled": 0,
                "accurate": 0,
                "win_rate": 0.0,
                "message": "No completed games found"
            }
        
        game_ids = [g.id for g in games]
        games_dict = {g.id: g for g in games}
        
        # Get all unsettled predictions for those games
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport.upper(),
            Prediction.game_id.in_(game_ids),
            Prediction.settled == False
        ).all()
        
        if not predictions:
            return {
                "sport": sport.upper(),
                "settled": 0,
                "accurate": 0,
                "win_rate": 0.0,
                "message": "No unsettled predictions found"
            }
        
        settled_count = 0
        accurate_count = 0
        win_count = 0
        loss_count = 0
        push_count = 0
        
        for pred in predictions:
            game = games_dict.get(pred.game_id)
            if not game:
                continue
            
            # Determine result based on market type
            if pred.market == "moneyline":
                result, pnl = settle_moneyline_prediction(pred, game)
            elif pred.market == "spread":
                result, pnl = settle_spread_prediction(pred, game)
            elif pred.market in ["totals", "over_under"]:
                result, pnl = settle_totals_prediction(pred, game)
            else:
                continue
            
            # Update prediction
            pred.settled = True
            pred.result = result  # Keep for backward compatibility
            pred.settled_result = result  # New column
            pred.pnl = pnl
            pred.settled_at = datetime.now()
            
            settled_count += 1
            if result == "win":
                accurate_count += 1
                win_count += 1
            elif result == "loss":
                loss_count += 1
            else:
                push_count += 1
        
        db.commit()
        
        win_rate = accurate_count / settled_count if settled_count > 0 else 0.0
        
        return {
            "sport": sport.upper(),
            "settled": settled_count,
            "accurate": accurate_count,
            "wins": win_count,
            "losses": loss_count,
            "pushes": push_count,
            "win_rate": round(win_rate, 3)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error settling predictions: {str(e)}"
        )
    finally:
        db.close()


@router.get("/metrics/accuracy/{sport}")
def get_accuracy_metrics(sport: str, days: int = Query(30, ge=1, le=365)):
    """
    Get accuracy metrics for a sport (last N days of settled predictions).
    
    Returns:
    {
      "sport": "NFL",
      "total_predictions": 234,
      "wins": 122,
      "losses": 112,
      "win_rate": 0.521,
      "vs_random": 0.021,  # How much better than 50%
      "sample_size_adequate": true
    }
    """
    from app.database import SessionLocal
    from app.models.db_models import Prediction
    from datetime import datetime, timedelta
    
    db = SessionLocal()
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get settled predictions from the last N days
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport.upper(),
            Prediction.settled == True,
            Prediction.settled_at >= cutoff_date
        ).all()
        
        if not predictions:
            return {
                "sport": sport.upper(),
                "total_predictions": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "win_rate": 0.0,
                "vs_random": 0.0,
                "sample_size_adequate": False,
                "message": f"No settled predictions found in last {days} days"
            }
        
        wins = sum(1 for p in predictions if p.settled_result == "win")
        losses = sum(1 for p in predictions if p.settled_result == "loss")
        pushes = sum(1 for p in predictions if p.settled_result == "push")
        
        # Calculate win rate (excluding pushes)
        total_non_push = wins + losses
        win_rate = wins / total_non_push if total_non_push > 0 else 0.0
        
        return {
            "sport": sport.upper(),
            "total_predictions": len(predictions),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(win_rate, 3),
            "vs_random": round(win_rate - 0.5, 3),  # How much better than 50%
            "sample_size_adequate": len(predictions) >= 30
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching accuracy metrics: {str(e)}"
        )
    finally:
        db.close()


@router.post("/jobs/settle-daily")
def settle_daily_predictions(api_key: str = Depends(require_api_key)):
    """
    Scheduled job (run daily via Railway cron).
    Settles all predictions for finished games.
    
    This endpoint should be called daily at 11 PM UTC to settle all predictions
    for games that have finished.
    """
    from app.prediction.settling import settle_predictions
    
    results = {}
    for sport in ["NFL", "NBA", "NHL", "MLB"]:
        try:
            # Use the settle_predictions function from settling.py
            # It needs a date, so we'll settle for today and yesterday
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # Settle yesterday's predictions
            settle_predictions(yesterday.strftime("%Y-%m-%d"), sport=sport)
            
            # Also try today in case games finished early
            settle_predictions(today.strftime("%Y-%m-%d"), sport=sport)
            
            results[sport] = "completed"
        except Exception as e:
            results[sport] = f"error: {str(e)}"
    
    return {
        "status": "daily settling complete",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/diagnose-accuracy")
def diagnose_model_accuracy(
    background_tasks: BackgroundTasks,
    sport: str = Query(...),
    source: str = Query("csv", regex="^(csv|db)$"),
    api_key: str = Depends(require_api_key)
):
    """
    Run model accuracy diagnostic script via API.
    
    This endpoint runs the diagnostic script to analyze predictions vs actual results.
    Results are saved to the exports directory and can be retrieved via logs.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        source: Data source - "csv" (from exports) or "db" (from database)
    
    Returns:
        Status message with output file paths
    
    Example:
        POST /api/diagnose-accuracy?sport=NFL&source=csv
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.diagnose_model_accuracy import (
        load_predictions_from_csv,
        load_predictions_from_db,
        get_game_results,
        determine_prediction_result,
        calculate_metrics,
    )
    import pandas as pd
    from datetime import datetime
    
    def run_diagnostic():
        try:
            # Load predictions
            if source == 'csv':
                predictions_df = load_predictions_from_csv(sport.upper())
            else:
                predictions_df = load_predictions_from_db(sport.upper())
            
            if len(predictions_df) == 0:
                print(f"No predictions found for {sport}")
                return
            
            # Get game results
            game_ids = predictions_df['game_id'].unique().tolist()
            games_df = get_game_results(sport.upper(), game_ids)
            
            if len(games_df) == 0:
                print(f"No completed games found for {sport}")
                return
            
            # Merge and calculate
            merged_df = predictions_df.merge(games_df, on='game_id', how='inner')
            results = merged_df.apply(determine_prediction_result, axis=1)
            merged_df['is_correct'] = [r[0] for r in results]
            merged_df['actual_outcome'] = [r[1] for r in results]
            
            # Calculate metrics
            metrics = calculate_metrics(merged_df)
            
            # Save results
            output_dir = Path("exports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            csv_path = output_dir / f"{sport}_accuracy_diagnostic_{timestamp}.csv"
            merged_df.to_csv(csv_path, index=False)
            
            json_path = output_dir / f"{sport}_accuracy_summary_{timestamp}.json"
            import json
            with open(json_path, 'w') as f:
                json.dump({
                    'sport': sport.upper(),
                    'timestamp': timestamp,
                    'metrics': metrics,
                }, f, indent=2)
            
            print(f"✓ Diagnostic complete: {csv_path.name}, {json_path.name}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"  ECE: {metrics.get('confidence', {}).get('ece', 'N/A')}")
            
        except Exception as e:
            print(f"Error in diagnostic: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_diagnostic)
    
    return {
        "status": "started",
        "message": f"Diagnostic started for {sport.upper()} (source: {source})",
        "sport": sport.upper(),
        "source": source,
        "note": "Check Railway logs to see results. Output files saved to exports/ directory."
    }


@router.post("/recalibrate-model")
def recalibrate_model_endpoint(
    background_tasks: BackgroundTasks,
    sport: str = Query(...),
    model: str = Query(...),
    test_weeks: int = Query(2, ge=1, le=4),
    api_key: str = Depends(require_api_key)
):
    """
    Recalibrate a model using Platt scaling via API.
    
    This endpoint loads a trained model, applies Platt scaling calibration,
    and saves a calibrated version.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        model: Model filename (e.g., "NFL_spread_20251207_191802.pkl") or full path
        test_weeks: Number of weeks to use for test set (default: 2)
    
    Returns:
        Status message
    
    Example:
        POST /api/recalibrate-model?sport=NFL&model=NFL_spread_20251207_191802.pkl
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.recalibrate_model import (
        load_model,
        prepare_calibration_data,
        apply_platt_scaling,
        evaluate_calibration,
        create_calibrated_wrapper,
    )
    import joblib
    from app.config import settings
    
    def run_recalibration():
        try:
            # Resolve model path
            model_path = Path(model)
            if not model_path.is_absolute():
                # Assume it's in models directory
                model_path = Path(settings.MODEL_DIR) / model
            
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                return
            
            # Load model
            model_data = load_model(str(model_path))
            base_model = model_data['model']
            market = model_data.get('market', 'unknown')
            
            print(f"Recalibrating {sport.upper()} {market} model...")
            
            # Prepare data
            X_train, y_train, X_cal, y_cal, X_test, y_test, feature_cols = prepare_calibration_data(
                sport.upper(), market, model_data, test_weeks=test_weeks
            )
            
            # Evaluate base model
            base_metrics = evaluate_calibration(base_model, X_test, y_test, "Base Model")
            
            # Apply Platt scaling
            calibrator = apply_platt_scaling(base_model, X_cal, y_cal)
            calibrated_model = create_calibrated_wrapper(base_model, calibrator)
            
            # Evaluate calibrated model
            calibrated_metrics = evaluate_calibration(calibrated_model, X_test, y_test, "Calibrated Model")
            
            # Save calibrated model
            model_dir = Path(settings.MODEL_DIR)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibrated_model_path = model_dir / f"{sport.upper()}_{market}_CALIBRATED_{timestamp}.pkl"
            
            calibrated_model_data = {
                **model_data,
                'model': calibrated_model,
                'calibrator': calibrator,
                'calibration_date': datetime.now().isoformat(),
                'calibration_method': 'platt_scaling',
                'base_ece': float(base_metrics['ece']),
                'calibrated_ece': float(calibrated_metrics['ece']),
                'ece_improvement': float(base_metrics['ece'] - calibrated_metrics['ece']),
            }
            
            joblib.dump(calibrated_model_data, calibrated_model_path)
            
            print(f"✓ Calibrated model saved: {calibrated_model_path.name}")
            print(f"  ECE Before: {base_metrics['ece']:.4f}")
            print(f"  ECE After: {calibrated_metrics['ece']:.4f}")
            print(f"  Improvement: {base_metrics['ece'] - calibrated_metrics['ece']:+.4f}")
            
        except Exception as e:
            print(f"Error in recalibration: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background
    background_tasks.add_task(run_recalibration)
    
    return {
        "status": "started",
        "message": f"Recalibration started for {sport.upper()} model: {model}",
        "sport": sport.upper(),
        "model": model,
        "test_weeks": test_weeks,
        "note": "Check Railway logs to see results. Calibrated model saved to models/ directory."
    }