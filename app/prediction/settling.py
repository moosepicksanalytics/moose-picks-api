from app.database import SessionLocal
from app.models.db_models import Prediction, Game


def settle_predictions(date_str: str):
    """
    Compare predictions from date_str against final scores.
    """
    db = SessionLocal()
    
    try:
        from datetime import datetime
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        predictions = db.query(Prediction).filter(
            Prediction.settled == False,
            Prediction.predicted_at >= target_date,
        ).all()
        
        for pred in predictions:
            game = db.query(Game).filter(Game.id == pred.game_id).first()
            if not game or not game.home_score:
                continue
            
            # Determine result
            if pred.market == "moneyline":
                home_win = game.home_score > game.away_score
                result = "win" if (home_win and pred.home_win_prob > 0.5) or (not home_win and pred.home_win_prob < 0.5) else "loss"
            else:
                result = "push"  # simplified
            
            pred.settled = True
            pred.result = result
            pred.pnl = 1.0 if result == "win" else -1.0 if result == "loss" else 0
        
        db.commit()
        print(f"✓ Settled {len(predictions)} predictions")
    except Exception as e:
        print(f"Error settling: {e}")
        db.rollback()
    finally:
        db.close()
