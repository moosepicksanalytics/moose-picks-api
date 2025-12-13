"""
Functions for storing predictions in the database.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional
from app.database import SessionLocal
from app.models.db_models import Prediction, Game
from app.config import settings


def get_model_version() -> str:
    """Get the latest model version from model directory."""
    from pathlib import Path
    model_dir = Path(settings.MODEL_DIR)
    
    if not model_dir.exists():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find any model file to get version
    pkl_files = list(model_dir.glob("*.pkl"))
    if not pkl_files:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get latest model's timestamp
    latest = max(pkl_files, key=lambda p: p.stat().st_mtime)
    # Extract version from filename (format: sport_market_version.pkl)
    parts = latest.stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])  # Everything after sport_market
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def store_prediction(
    game_id: str,
    sport: str,
    market: str,
    home_win_prob: Optional[float] = None,
    spread_cover_prob: Optional[float] = None,
    over_prob: Optional[float] = None,
    model_version: Optional[str] = None
) -> Prediction:
    """
    Store a prediction in the database.
    
    Args:
        game_id: Game ID
        sport: Sport code
        market: Market type (moneyline, spread, totals)
        home_win_prob: Home win probability (for moneyline)
        spread_cover_prob: Spread cover probability (for spread)
        over_prob: Over probability (for totals)
        model_version: Model version string (if None, auto-detected)
    
    Returns:
        Prediction object
    """
    db = SessionLocal()
    
    try:
        if model_version is None:
            model_version = get_model_version()
        
        # Check if prediction already exists
        existing = db.query(Prediction).filter(
            Prediction.game_id == game_id,
            Prediction.market == market
        ).first()
        
        if existing:
            # Update existing prediction
            if home_win_prob is not None:
                existing.home_win_prob = home_win_prob
            if spread_cover_prob is not None:
                existing.spread_cover_prob = spread_cover_prob
            if over_prob is not None:
                existing.over_prob = over_prob
            existing.model_version = model_version
            prediction = existing
        else:
            # Create new prediction
            prediction = Prediction(
                id=str(uuid.uuid4()),
                game_id=game_id,
                sport=sport,
                market=market,
                model_version=model_version,
                home_win_prob=home_win_prob,
                spread_cover_prob=spread_cover_prob,
                over_prob=over_prob,
                settled=False
            )
            db.add(prediction)
        
        db.commit()
        db.refresh(prediction)
        return prediction
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def store_predictions_for_game(
    game_id: str,
    sport: str,
    predictions_dict: Dict,
    model_version: Optional[str] = None
) -> list:
    """
    Store all predictions for a game (moneyline, spread, totals).
    Uses a single database transaction for atomicity.
    
    Args:
        game_id: Game ID
        sport: Sport code
        predictions_dict: Prediction dict from predict_for_game()
        model_version: Model version string (if None, auto-detected)
    
    Returns:
        List of stored Prediction objects
    """
    db = SessionLocal()
    stored = []
    
    try:
        if model_version is None:
            model_version = get_model_version()
        
        # Store moneyline prediction
        if "moneyline" in predictions_dict and predictions_dict["moneyline"]:
            ml = predictions_dict["moneyline"]
            if "home_win_prob" in ml:
                best_side = ml.get("best_side")
                best_edge = ml.get("best_edge", 0)
                
                if best_edge < 0:
                    print(f"  ⚠️  Storing prediction with negative edge ({best_edge:.1%}) - no value bet recommended")
                
                # Check if prediction already exists
                existing = db.query(Prediction).filter(
                    Prediction.game_id == game_id,
                    Prediction.market == "moneyline"
                ).first()
                
                if existing:
                    existing.home_win_prob = ml["home_win_prob"]
                    existing.model_version = model_version
                    stored.append(existing)
                else:
                    pred = Prediction(
                        id=str(uuid.uuid4()),
                        game_id=game_id,
                        sport=sport,
                        market="moneyline",
                        model_version=model_version,
                        home_win_prob=ml["home_win_prob"],
                        settled=False
                    )
                    db.add(pred)
                    stored.append(pred)
        
        # Store spread prediction
        if "spread" in predictions_dict and predictions_dict["spread"]:
            spread = predictions_dict["spread"]
            if "cover_prob" in spread:
                existing = db.query(Prediction).filter(
                    Prediction.game_id == game_id,
                    Prediction.market == "spread"
                ).first()
                
                if existing:
                    existing.spread_cover_prob = spread["cover_prob"]
                    existing.model_version = model_version
                    stored.append(existing)
                else:
                    pred = Prediction(
                        id=str(uuid.uuid4()),
                        game_id=game_id,
                        sport=sport,
                        market="spread",
                        model_version=model_version,
                        spread_cover_prob=spread["cover_prob"],
                        settled=False
                    )
                    db.add(pred)
                    stored.append(pred)
        
        # Store totals prediction
        if "totals" in predictions_dict and predictions_dict["totals"]:
            totals = predictions_dict["totals"]
            if "over_prob" in totals:
                existing = db.query(Prediction).filter(
                    Prediction.game_id == game_id,
                    Prediction.market == "totals"
                ).first()
                
                if existing:
                    existing.over_prob = totals["over_prob"]
                    existing.model_version = model_version
                    stored.append(existing)
                else:
                    pred = Prediction(
                        id=str(uuid.uuid4()),
                        game_id=game_id,
                        sport=sport,
                        market="totals",
                        model_version=model_version,
                        over_prob=totals["over_prob"],
                        settled=False
                    )
                    db.add(pred)
                    stored.append(pred)
        
        # Commit all predictions in a single transaction
        db.commit()
        
        # Refresh all stored predictions
        for pred in stored:
            db.refresh(pred)
        
        return stored
        
    except Exception as e:
        db.rollback()
        print(f"Error storing predictions for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        db.close()
