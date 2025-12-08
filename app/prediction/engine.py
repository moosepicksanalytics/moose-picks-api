"""
Prediction engine for generating predictions using trained models.
"""
from app.database import SessionLocal
from app.models.db_models import Game


def predict_for_game(sport: str, market: str, game_id: str):
    """
    Generate prediction for a single game.
    
    Args:
        sport: Sport code
        market: Market type
        game_id: Game ID
    
    Returns:
        Prediction dict
    """
    db = SessionLocal()
    try:
        game = db.query(Game).filter(Game.id == game_id).first()
        if not game:
            return {"error": "Game not found"}
        
        # Import here to avoid circular imports
        from scripts.export_predictions import predict_for_game as predict_func
        from app.training.pipeline import load_config
        
        config = load_config()
        return predict_func(game, sport, config)
    finally:
        db.close()
