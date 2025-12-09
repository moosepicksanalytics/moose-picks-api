"""
Export pipeline for generating predictions in a standardized format for Lovable.
"""
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import json


def find_top_pick(pred: Dict, min_edge: float = 0.05) -> Optional[Dict]:
    """Find the top pick (highest edge) from a prediction."""
    best_pick = None
    best_edge = 0.0
    
    # Check moneyline
    if "moneyline" in pred and pred["moneyline"]:
        ml = pred["moneyline"]
        if ml.get("best_edge", 0) >= min_edge and ml.get("best_edge", 0) > best_edge:
            best_edge = ml["best_edge"]
            best_pick = {
                "market": "moneyline",
                "side": ml["best_side"],
                "edge": ml["best_edge"],
                "prob": ml.get(f"{ml['best_side']}_win_prob", 0),
            }
    
    # Check spread
    if "spread" in pred and pred["spread"]:
        spread = pred["spread"]
        if spread.get("edge", 0) >= min_edge and spread.get("edge", 0) > best_edge:
            best_edge = spread["edge"]
            best_pick = {
                "market": "spread",
                "side": spread.get("side", "favorite"),
                "edge": spread["edge"],
                "prob": spread.get("cover_prob", 0),
            }
    
    # Check totals
    if "totals" in pred and pred["totals"]:
        totals = pred["totals"]
        if totals.get("best_edge", 0) >= min_edge and totals.get("best_edge", 0) > best_edge:
            best_edge = totals["best_edge"]
            best_pick = {
                "market": "totals",
                "side": totals["best_side"],
                "edge": totals["best_edge"],
                "prob": totals.get(f"{totals['best_side']}_prob", 0),
            }
    
    return best_pick


def export_predictions_to_csv(
    predictions: List[Dict],
    output_path: str,
    min_edge: float = 0.05
) -> None:
    """
    Export predictions to CSV in standardized format for Lovable.
    
    Args:
        predictions: List of prediction dicts
        output_path: Path to output CSV file
        min_edge: Minimum edge threshold for filtering
    """
    rows = []
    
    for pred in predictions:
        game_id = pred.get("game_id", "")
        league = pred.get("league", "")
        season = pred.get("season", "")
        date = pred.get("date", "")
        home_team = pred.get("home_team", "")
        away_team = pred.get("away_team", "")
        
        # Moneyline
        if "moneyline" in pred:
            ml = pred["moneyline"]
            for side in ["home", "away"]:
                if f"{side}_edge" in ml and ml[f"{side}_edge"] >= min_edge:
                    prob = ml.get(f"{side}_win_prob", 0)
                    edge = ml.get(f"{side}_edge", 0)
                    # Apply display filter: don't show unrealistic predictions
                    if prob > 0.70 or abs(edge) > 0.10:
                        continue  # Skip unrealistic predictions
                    rows.append({
                        "game_id": game_id,
                        "league": league,
                        "season": season,
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": "moneyline",
                        "side": side,
                        "line": None,
                        "price": ml.get(f"{side}_odds"),
                        "model_prob": ml.get(f"{side}_win_prob"),
                        "implied_prob": ml.get(f"{side}_implied_prob"),
                        "edge": ml.get(f"{side}_edge"),
                        "proj_home_score": pred.get("proj_home_score"),
                        "proj_away_score": pred.get("proj_away_score"),
                    })
        
        # Spread
        if "spread" in pred:
            spread = pred["spread"]
            if spread.get("edge", 0) >= min_edge:
                prob = spread.get("cover_prob", 0)
                edge = spread.get("edge", 0)
                # Apply display filter: don't show unrealistic predictions
                if prob > 0.70 or abs(edge) > 0.10:
                    continue  # Skip unrealistic predictions
                rows.append({
                    "game_id": game_id,
                    "league": league,
                    "season": season,
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": "spread",
                    "side": spread.get("side", "favorite"),
                    "line": spread.get("line"),
                    "price": spread.get("price"),
                    "model_prob": spread.get("cover_prob"),
                    "implied_prob": spread.get("implied_prob"),
                    "edge": spread.get("edge"),
                    "proj_home_score": pred.get("proj_home_score"),
                    "proj_away_score": pred.get("proj_away_score"),
                })
        
        # Totals
        if "totals" in pred:
            totals = pred["totals"]
            for side in ["over", "under"]:
                if f"{side}_edge" in totals and totals[f"{side}_edge"] >= min_edge:
                    prob = totals.get(f"{side}_prob", 0)
                    edge = totals.get(f"{side}_edge", 0)
                    # Apply display filter: don't show unrealistic predictions
                    if prob > 0.70 or abs(edge) > 0.10:
                        continue  # Skip unrealistic predictions
                    rows.append({
                        "game_id": game_id,
                        "league": league,
                        "season": season,
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": "totals",
                        "side": side,
                        "line": totals.get("line"),
                        "price": totals.get(f"{side}_odds"),
                        "model_prob": totals.get(f"{side}_prob"),
                        "implied_prob": totals.get(f"{side}_implied_prob"),
                        "edge": totals.get(f"{side}_edge"),
                        "proj_home_score": pred.get("proj_home_score"),
                        "proj_away_score": pred.get("proj_away_score"),
                    })
    
    if not rows:
        print("No predictions to export (all below edge threshold)")
        return
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} predictions to {output_path}")


def export_predictions_to_json(
    predictions: List[Dict],
    output_path: str,
    min_edge: float = 0.05
) -> None:
    """
    Export predictions to JSON in standardized format for Lovable.
    
    Args:
        predictions: List of prediction dicts
        output_path: Path to output JSON file
        min_edge: Minimum edge threshold for filtering
    """
    # Filter by edge threshold
    filtered = []
    for pred in predictions:
        # Check if any market has edge above threshold
        has_edge = False
        
        if "moneyline" in pred:
            ml = pred["moneyline"]
            if ml.get("best_edge", 0) >= min_edge:
                has_edge = True
        
        if "spread" in pred:
            if pred["spread"].get("edge", 0) >= min_edge:
                has_edge = True
        
        if "totals" in pred:
            totals = pred["totals"]
            if totals.get("best_edge", 0) >= min_edge:
                has_edge = True
        
        if has_edge or min_edge == 0:
            # Add top pick
            top_pick = find_top_pick(pred, min_edge)
            pred["top_pick"] = top_pick
            filtered.append(pred)
    
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2, default=str)
    
    print(f"✓ Exported {len(filtered)} games to {output_path}")


def format_prediction_row(
    game_id: str,
    league: str,
    season: str,
    date: str,
    home_team: str,
    away_team: str,
    market_type: str,
    side: str,
    line: Optional[float],
    price: Optional[float],
    model_prob: float,
    implied_prob: float,
    edge: float,
    proj_home_score: Optional[float],
    proj_away_score: Optional[float],
) -> Dict:
    """
    Format a single prediction row in the standardized schema.
    
    Returns:
        Dict with all required fields
    """
    return {
        "game_id": game_id,
        "league": league,
        "season": season,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "market_type": market_type,
        "side": side,
        "line": line,
        "price": price,
        "model_prob": model_prob,
        "implied_prob": implied_prob,
        "edge": edge,
        "proj_home_score": proj_home_score,
        "proj_away_score": proj_away_score,
    }


def export_predictions(
    predictions: List[Dict],
    output_dir: str = "exports",
    formats: List[str] = ["csv", "json"],
    min_edge: float = 0.05,
    filename_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    Export predictions in multiple formats.
    
    Args:
        predictions: List of prediction dicts
        output_dir: Output directory
        formats: List of formats to export (csv, json)
        min_edge: Minimum edge threshold
        filename_prefix: Optional prefix for filenames
    
    Returns:
        Dict mapping format to output path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if filename_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"predictions_{timestamp}"
    
    output_paths = {}
    
    if "csv" in formats:
        csv_path = Path(output_dir) / f"{filename_prefix}.csv"
        export_predictions_to_csv(predictions, str(csv_path), min_edge)
        output_paths["csv"] = str(csv_path)
    
    if "json" in formats:
        json_path = Path(output_dir) / f"{filename_prefix}.json"
        export_predictions_to_json(predictions, str(json_path), min_edge)
        output_paths["json"] = str(json_path)
    
    return output_paths
