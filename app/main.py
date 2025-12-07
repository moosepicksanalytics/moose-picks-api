from fastapi import FastAPI, HTTPException, BackgroundTasks
from datetime import datetime
from app.database import engine, SessionLocal
from app.models.db_models import Base
from app.config import settings
from app.prediction.engine import predict_for_game
from app.prediction.settling import settle_predictions
from app.training.pipeline import train_model_for_market
from app.espn_client.fetcher import fetch_games_for_date
from app.espn_client.parser import parse_and_store_games

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Moose Picks API", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1"}


@app.get("/predict")
def predict(sport: str, market: str, game_id: str):
    """
    Predict for a single game.
    Example: /predict?sport=NFL&market=moneyline&game_id=401547123
    """
    result = predict_for_game(sport, market, game_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/daily-picks")
def daily_picks(date_str: str = None):
    """
    Get all picks for a date (default today).
    date format: YYYY-MM-DD
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    picks = {}
    for sport in settings.SPORTS_CONFIG.keys():
        games = fetch_games_for_date(sport, date_str)
        parse_and_store_games(sport, games)
        
        picks[sport] = []
        
        for game in games:
            game_id = game["id"]
            for market in settings.SPORTS_CONFIG[sport]["markets"]:
                pred = predict_for_game(sport, market, game_id)
                if "error" not in pred:
                    picks[sport].append(pred)
    
    return picks


@app.post("/train")
def train(sport: str = None, market: str = None, background_tasks: BackgroundTasks = None):
    """
    Trigger retraining.
    If sport/market specified, train just those.
    Otherwise, train all configured markets.
    """
    def retrain():
        if sport and market:
            train_model_for_market(sport, market)
        else:
            for s in settings.SPORTS_CONFIG.keys():
                for m in settings.SPORTS_CONFIG[s]["markets"]:
                    train_model_for_market(s, m)
    
    if background_tasks:
        background_tasks.add_task(retrain)
    else:
        retrain()
    
    return {"status": "training started"}


@app.post("/settle")
def settle(date_str: str = None):
    """
    Settle predictions from a date against final scores.
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    settle_predictions(date_str)
    return {"status": f"settled predictions for {date_str}"}


@app.get("/models")
def list_models():
    """List all trained models."""
    from pathlib import Path
    model_dir = Path(settings.MODEL_DIR)
    if not model_dir.exists():
        return {"models": []}
    
    models = sorted([f.stem for f in model_dir.glob("*.pkl")], reverse=True)
    return {"models": models}
