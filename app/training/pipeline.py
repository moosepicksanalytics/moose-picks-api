from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from datetime import datetime
from app.database import SessionLocal
from app.models.db_models import Game
from app.config import settings
from app.training.features import (
    extract_features_nfl_moneyline,
    extract_features_nba_spread,
    extract_features_nhl_totals,
    extract_features_mlb_moneyline,
)
import pandas as pd


FEATURE_FUNCS = {
    ("NFL", "moneyline"): extract_features_nfl_moneyline,
    ("NBA", "spread"): extract_features_nba_spread,
    ("NHL", "over_under"): extract_features_nhl_totals,
    ("MLB", "moneyline"): extract_features_mlb_moneyline,
}


def train_model_for_market(sport: str, market: str):
    """
    Load historical games, extract features, train model, save.
    """
    db = SessionLocal()
    
    try:
        # Load games
        games = db.query(Game).filter(
            Game.sport == sport,
            Game.status == "final",
            Game.home_score.isnot(None),
        ).all()
        
        if len(games) < settings.SPORTS_CONFIG[sport]["min_training_games"]:
            print(f"Not enough games for {sport} {market}: {len(games)}")
            return False
        
        feature_func = FEATURE_FUNCS.get((sport, market))
        if not feature_func:
            print(f"No feature function for {sport} {market}")
            return False
        
        # Extract features
        feature_dicts = []
        labels = []
        
        for game in games:
            try:
                features = feature_func(game.espn_data)
                if not features:
                    continue
                
                feature_dicts.append(features)
                
                # Label
                if market == "moneyline":
                    labels.append(1 if game.home_score > game.away_score else 0)
                elif market == "spread":
                    labels.append(1 if (game.home_score - game.away_score) > game.spread else 0)
                elif market == "over_under":
                    labels.append(1 if (game.home_score + game.away_score) > game.over_under else 0)
            except Exception as e:
                print(f"Error on game {game.id}: {e}")
                continue
        
        if not feature_dicts:
            print(f"No valid features for {sport} {market}")
            return False
        
        # Build DataFrame
        df = pd.DataFrame(feature_dicts)
        X = df.select_dtypes(include=["number"])
        y = pd.Series(labels)
        
        # Train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        
        # Save
        Path(settings.MODEL_DIR).mkdir(exist_ok=True)
        model_name = f"{sport}_{market}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = f"{settings.MODEL_DIR}{model_name}"
        joblib.dump({"model": model, "scaler": scaler}, model_path)
        
        print(f"✓ Trained and saved {model_path}")
        return True
    except Exception as e:
        print(f"Error training {sport} {market}: {e}")
        return False
    finally:
        db.close()
