"""
Probability validation utilities for model predictions.
Validates calibration, flags unrealistic probabilities, and checks edge calculations.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from app.database import SessionLocal
from app.models.db_models import Game, Prediction
from app.utils.odds import american_odds_to_implied_prob


def validate_probability(
    model_prob: float,
    implied_prob: float,
    team_name: str,
    is_underdog: bool,
    min_edge_threshold: float = 0.05
) -> Dict[str, any]:
    """
    Validate a model probability prediction against implied odds.
    
    Args:
        model_prob: Model's predicted probability (0-1)
        implied_prob: Implied probability from betting odds (0-1)
        team_name: Team name (for logging)
        is_underdog: Whether this team is the underdog
        min_edge_threshold: Minimum edge to flag as suspicious
        
    Returns:
        Dict with validation results:
        - is_valid: bool
        - warnings: List of warning messages
        - edge: Calculated edge
        - calibration_check: Expected vs actual check
    """
    warnings = []
    edge = model_prob - implied_prob
    
    # Check 1: Unrealistic probability for underdog
    if is_underdog and model_prob > 0.60:
        warnings.append(
            f"⚠️  High probability ({model_prob:.1%}) for underdog {team_name}. "
            f"Verify model calibration - underdogs rarely have >60% win probability."
        )
    
    # Check 2: Unrealistic probability for favorite
    if not is_underdog and model_prob < 0.40:
        warnings.append(
            f"⚠️  Low probability ({model_prob:.1%}) for favorite {team_name}. "
            f"Verify model calibration - favorites rarely have <40% win probability."
        )
    
    # Check 3: Extremely high edge (rare in efficient markets)
    if abs(edge) > 0.25:
        warnings.append(
            f"⚠️  Very high edge ({edge:.1%}) detected. "
            f"Edges >25% are extremely rare in efficient markets. "
            f"Double-check: model_prob={model_prob:.1%}, implied_prob={implied_prob:.1%}"
        )
    
    # Check 4: Edge direction mismatch
    if is_underdog and edge < 0:
        # Underdog with negative edge is normal
        pass
    elif not is_underdog and edge > 0.20:
        warnings.append(
            f"⚠️  Large positive edge ({edge:.1%}) on favorite. "
            f"While possible, verify odds are correct and model is well-calibrated."
        )
    
    # Check 5: Probability bounds
    if model_prob < 0.05 or model_prob > 0.95:
        warnings.append(
            f"⚠️  Extreme probability ({model_prob:.1%}) - model may be overconfident. "
            f"Consider capping probabilities between 5% and 95%."
        )
    
    is_valid = len(warnings) == 0 or all("verify" not in w.lower() for w in warnings)
    
    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "edge": edge,
        "model_prob": model_prob,
        "implied_prob": implied_prob,
    }


def check_calibration_by_bucket(
    sport: str,
    market: str,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Check model calibration by comparing predicted probabilities to actual outcomes.
    
    Uses historical predictions that have been settled to validate calibration.
    
    Args:
        sport: Sport code
        market: Market type (moneyline, spread, totals)
        n_bins: Number of probability bins
        
    Returns:
        DataFrame with calibration statistics by probability bucket
    """
    db = SessionLocal()
    try:
        # Get settled predictions with outcomes
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport,
            Prediction.market == market,
            Prediction.settled == True
        ).all()
        
        if len(predictions) < 50:
            return pd.DataFrame()  # Not enough data
        
        results = []
        for pred in predictions:
            game = db.query(Game).filter(Game.id == pred.game_id).first()
            if not game or game.home_score is None or game.away_score is None:
                continue
            
            # Get model probability based on market
            if market == "moneyline":
                model_prob = pred.home_win_prob
                if model_prob is None:
                    continue
                # Determine if home won
                actual_outcome = 1 if game.home_score > game.away_score else 0
            elif market == "spread":
                model_prob = pred.spread_cover_prob
                if model_prob is None or game.spread is None:
                    continue
                # Determine if home covered
                actual_outcome = 1 if (game.home_score - game.away_score) > game.spread else 0
            elif market == "totals":
                model_prob = pred.over_prob
                if model_prob is None or game.closing_total is None:
                    continue
                total = game.home_score + game.away_score
                actual_outcome = 1 if total > game.closing_total else 0
            else:
                continue
            
            results.append({
                "model_prob": model_prob,
                "actual_outcome": actual_outcome
            })
        
        if len(results) < 50:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Bin probabilities
        df["prob_bin"] = pd.cut(df["model_prob"], bins=n_bins, labels=False)
        
        # Calculate calibration by bin
        calibration = []
        for bin_idx in range(n_bins):
            bin_data = df[df["prob_bin"] == bin_idx]
            if len(bin_data) == 0:
                continue
            
            avg_predicted = bin_data["model_prob"].mean()
            avg_actual = bin_data["actual_outcome"].mean()
            n_samples = len(bin_data)
            calibration_error = abs(avg_predicted - avg_actual)
            
            calibration.append({
                "prob_bin": bin_idx,
                "prob_range": f"{bin_data['model_prob'].min():.2f}-{bin_data['model_prob'].max():.2f}",
                "avg_predicted": avg_predicted,
                "avg_actual": avg_actual,
                "calibration_error": calibration_error,
                "n_samples": n_samples
            })
        
        return pd.DataFrame(calibration)
        
    finally:
        db.close()


def validate_prediction_before_storage(
    game_id: str,
    sport: str,
    market: str,
    model_prob: float,
    home_odds: Optional[float] = None,
    away_odds: Optional[float] = None
) -> Dict[str, any]:
    """
    Validate a prediction before storing it in the database.
    
    Args:
        game_id: Game ID
        sport: Sport code
        market: Market type
        model_prob: Model's predicted probability
        home_odds: Home team American odds
        away_odds: Away team American odds
        
    Returns:
        Dict with validation results and warnings
    """
    db = SessionLocal()
    try:
        game = db.query(Game).filter(Game.id == game_id).first()
        if not game:
            return {"is_valid": False, "warnings": ["Game not found"]}
        
        if market == "moneyline":
            # Determine which team's probability we're validating
            if model_prob > 0.5:
                # Home team is favorite
                team_name = game.home_team
                is_underdog = False
                implied_prob = american_odds_to_implied_prob(home_odds) if home_odds else None
            else:
                # Away team is favorite (or model thinks away wins)
                team_name = game.away_team
                is_underdog = True
                implied_prob = american_odds_to_implied_prob(away_odds) if away_odds else None
            
            if implied_prob is None:
                return {"is_valid": True, "warnings": ["No odds available for validation"]}
            
            # Use the appropriate probability (home_win_prob or 1 - home_win_prob)
            if model_prob > 0.5:
                team_prob = model_prob
            else:
                team_prob = 1 - model_prob
            
            return validate_probability(
                team_prob,
                implied_prob,
                team_name,
                is_underdog
            )
        else:
            # For spread/totals, just check probability bounds
            warnings = []
            if model_prob < 0.05 or model_prob > 0.95:
                warnings.append(
                    f"⚠️  Extreme probability ({model_prob:.1%}) for {market}. "
                    f"Consider capping between 5% and 95%."
                )
            
            return {
                "is_valid": len(warnings) == 0,
                "warnings": warnings,
                "model_prob": model_prob
            }
            
    finally:
        db.close()

