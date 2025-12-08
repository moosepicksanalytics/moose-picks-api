from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.database import engine, SessionLocal
    from app.models.db_models import Base
    from app.config import settings
    from app.prediction.engine import predict_for_game
    from app.prediction.settling import settle_predictions
    from app.training.pipeline import train_model_for_market
    from app.espn_client.fetcher import fetch_games_for_date
    from app.espn_client.parser import parse_and_store_games
    from app.api_endpoints import router as api_router
    
    # Create tables with error handling
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        # Continue anyway - tables might already exist
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    raise

app = FastAPI(title="Moose Picks API", version="0.1")

# Add CORS middleware to allow requests from Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted to specific Lovable domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include API router for Lovable/scheduled tasks
app.include_router(api_router, prefix="/api", tags=["automation"])


@app.on_event("startup")
async def startup_event():
    """Initialize database and verify setup on startup."""
    try:
        logger.info("Starting Moose Picks API...")
        logger.info(f"Database URL: {settings.DATABASE_URL[:20]}...")  # Log partial URL for security
        logger.info(f"Model directory: {settings.MODEL_DIR}")
        
        # Verify database connection
        from sqlalchemy import text
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db.commit()
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
        finally:
            db.close()
            
        logger.info("Moose Picks API started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't raise - allow app to start even if some checks fail


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
