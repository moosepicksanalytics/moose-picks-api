"""
API endpoints for Lovable integration and scheduled tasks.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
from typing import Optional
import asyncio

router = APIRouter()


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
    
    sports_list = [s.strip() for s in sports.split(",")] if sports else ["NFL", "NHL"]
    
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
    Get latest predictions from exports directory.
    Useful for Lovable to fetch recent predictions.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        limit: Maximum number of predictions to return
    """
    import json
    from pathlib import Path
    import pandas as pd
    
    exports_dir = Path("exports")
    if not exports_dir.exists():
        raise HTTPException(status_code=404, detail="No exports found")
    
    # Find latest CSV for the sport
    csv_files = sorted(exports_dir.glob(f"{sport}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No predictions found for {sport}")
    
    latest_csv = csv_files[0]
    df = pd.read_csv(latest_csv)
    
    # Return top picks by edge
    df_sorted = df.sort_values("edge", ascending=False).head(limit)
    
    return {
        "sport": sport,
        "file": latest_csv.name,
        "total_predictions": len(df),
        "top_picks": df_sorted.to_dict("records")
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
            
            # Add market-specific data
            if pred.market == "moneyline":
                pick["side"] = "home" if pred.home_win_prob and pred.home_win_prob > 0.5 else "away"
                pick["home_win_prob"] = pred.home_win_prob
                pick["edge"] = abs(pred.home_win_prob - 0.5) if pred.home_win_prob else 0
            elif pred.market == "spread":
                pick["side"] = "home" if pred.spread_cover_prob and pred.spread_cover_prob > 0.5 else "away"
                pick["spread_cover_prob"] = pred.spread_cover_prob
                pick["spread"] = game.spread
                pick["edge"] = abs(pred.spread_cover_prob - 0.5) if pred.spread_cover_prob else 0
            elif pred.market == "totals" or pred.market == "over_under":
                pick["side"] = "over" if pred.over_prob and pred.over_prob > 0.5 else "under"
                pick["over_prob"] = pred.over_prob
                pick["over_under"] = game.over_under
                pick["edge"] = abs(pred.over_prob - 0.5) if pred.over_prob else 0
            
            pick["recommended_kelly"] = pred.recommended_kelly
            pick["recommended_unit_size"] = pred.recommended_unit_size
            
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
