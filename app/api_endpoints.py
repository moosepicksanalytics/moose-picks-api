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
