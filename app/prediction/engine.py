import joblib
from pathlib import Path
from app.config import settings
from app.espn_client.fetcher import fetch_game_detail
from app.training.features import (
    extract_features_nfl_moneyline,
    extract_features_nba_spread,
    extract_features_nhl_totals,
    extract_features_mlb_moneyline,
)

FEATURE_FUNCS = {
    ("NFL", "moneyline"): extract_features_nfl_moneyline,
    ("NBA", "spread"): extract_features_nba_spread,
    ("NHL", "over_under"): extract_features_nhl_totals,
    ("MLB", "moneyline"): extract_features_mlb_moneyline,
}


def predict_for_game(sport: str, market: str, game_id: str) -> dict:
    """
    Fetch game, extract features, run model, return probability + Kelly.
    """
    try:
        # Load latest model
        model_dir = Path(settings.MODEL_DIR)
        model_files = sorted(model_dir.glob(f"{sport}_{market}_*.pkl"), reverse=True)
        
        if not model_files:
            return {"error": f"No model found for {sport} {market}"}
        
        model_artifact = joblib.load(model_files[0])
        model = model_artifact["model"]
        scaler = model_artifact["scaler"]
        
        # Fetch game
        game_detail = fetch_game_detail(sport, game_id)
        if not game_detail:
            return {"error": f"Could not fetch {game_id}"}
        
        feature_func = FEATURE_FUNCS.get((sport, market))
        if not feature_func:
            return {"error": f"No feature func for {sport} {market}"}
        
        features = feature_func(game_detail)
        if not features:
            return {"error": "Could not extract features"}
        
        # Predict
        import pandas as pd
        df = pd.DataFrame([features])
        X = df.select_dtypes(include=["number"])
        X_scaled = scaler.transform(X)
        
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # Kelly sizing (simplified)
        kelly = max(0, min((prob - 0.5) * 0.2, 0.25))  # cap at 25%
        
        return {
            "game_id": game_id,
            "sport": sport,
            "market": market,
            "probability": float(prob),
            "kelly_fraction": float(kelly),
            "model_version": model_files[0].stem,
        }
    except Exception as e:
        print(f"Error predicting: {e}")
        return {"error": str(e)}
