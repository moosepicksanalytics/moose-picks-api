from fastapi import FastAPI
from datetime import datetime
from app.database import engine, SessionLocal
from app.models.db_models import Base
from app.config import settings
from app.prediction.engine import predict_for_game
from app.prediction.settling import settle_predictions
from app.training.pipeline import train_model_for_market
from app.espn_client.fetcher import fetch_games_for_date
from app.espn_client.parser import parse_and_store_games
from app.api_endpoints import router as api_router

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Moose Picks API", version="0.1")

# Include API router for Lovable/scheduled tasks
app.include_router(api_router, prefix="/api", tags=["automation"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1"}


@app.get("/predict")
def predict(sport: str, market: str, game_id: str):
    """
    Get prediction for a single game.
    """
    return predict_for_game(sport, market, game_id)


@app.post("/train")
def train(sport: str, market: str):
    """
    Manually trigger model training.
    """
    result = train_model_for_market(sport, market)
    return result


@app.get("/models")
def list_models():
    """List all trained models."""
    from pathlib import Path
    model_dir = Path(settings.MODEL_DIR)
    if not model_dir.exists():
        return {"models": []}
    
    models = sorted([f.stem for f in model_dir.glob("*.pkl")], reverse=True)
    return {"models": models}
