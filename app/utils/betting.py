"""
Betting utilities: edge calculation, Kelly Criterion, and ROI simulation.
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np


def american_odds_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -110, +150)
    
    Returns:
        Implied probability (0-1)
    """
    if pd.isna(odds) or odds == 0:
        return 0.5
    
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_edge(
    model_prob: float,
    implied_prob: float
) -> float:
    """
    Calculate betting edge (model probability - implied probability).
    
    Args:
        model_prob: Model's predicted probability
        implied_prob: Implied probability from odds
    
    Returns:
        Edge (positive = value bet)
    """
    return model_prob - implied_prob


def calculate_kelly_fraction(
    model_prob: float,
    odds: float,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion bet size (fractional Kelly).
    
    Kelly Formula: f = (bp - q) / b
    where:
        f = fraction of bankroll to bet
        b = odds received (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    
    We use fractional Kelly (default 1/4 Kelly) to reduce variance.
    
    Args:
        model_prob: Model's predicted probability of winning
        odds: American odds
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = 1/4 Kelly)
    
    Returns:
        Bet size as fraction of bankroll (0-1)
    """
    if pd.isna(model_prob) or pd.isna(odds) or odds == 0:
        return 0.0
    
    # Convert American odds to decimal odds
    if odds > 0:
        decimal_odds = (odds + 100) / 100
    else:
        decimal_odds = (abs(odds) + 100) / abs(odds)
    
    # Kelly formula
    b = decimal_odds - 1  # Net odds
    p = model_prob
    q = 1 - p
    
    # Full Kelly
    if b == 0:
        return 0.0
    
    full_kelly = (b * p - q) / b
    
    # Apply fractional Kelly and ensure non-negative
    fractional_kelly = max(0.0, full_kelly * kelly_fraction)
    
    # Cap at reasonable maximum (e.g., 10% of bankroll)
    return min(fractional_kelly, 0.10)


def calculate_roi(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    odds: pd.Series,
    edges: pd.Series,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25
) -> Dict:
    """
    Calculate ROI from betting simulation using Kelly Criterion.
    
    Args:
        y_true: True outcomes (1 = win, 0 = loss)
        y_pred_proba: Model predicted probabilities
        odds: American odds
        edges: Calculated edges
        min_edge: Minimum edge to place bet (default 5%)
        kelly_fraction: Fraction of full Kelly to use
    
    Returns:
        Dict with ROI metrics
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "odds": odds,
        "edge": edges,
    })
    
    # Filter for value bets (edge >= min_edge)
    value_bets = df[df["edge"] >= min_edge].copy()
    
    if len(value_bets) == 0:
        return {
            "total_bets": 0,
            "value_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_wagered": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "kelly_roi": 0.0,
        }
    
    # Calculate Kelly bet sizes
    value_bets["kelly_bet"] = value_bets.apply(
        lambda row: calculate_kelly_fraction(
            row["y_pred_proba"],
            row["odds"],
            kelly_fraction
        ),
        axis=1
    )
    
    # Simulate betting with Kelly sizing
    # Assume starting bankroll of 100 units
    bankroll = 100.0
    total_wagered = 0.0
    total_profit = 0.0
    
    for _, bet in value_bets.iterrows():
        # Bet size in units
        bet_size = bankroll * bet["kelly_bet"]
        total_wagered += bet_size
        
        if bet["y_true"] == 1:  # Win
            # Calculate payout from American odds
            if bet["odds"] > 0:
                payout = bet_size * (bet["odds"] / 100)
            else:
                payout = bet_size * (100 / abs(bet["odds"]))
            
            profit = payout
            total_profit += profit
            bankroll += profit
        else:  # Loss
            total_profit -= bet_size
            bankroll -= bet_size
    
    # Calculate ROI
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    kelly_roi = ((bankroll - 100) / 100 * 100) if bankroll > 0 else 0.0
    
    wins = value_bets["y_true"].sum()
    losses = len(value_bets) - wins
    
    return {
        "total_bets": len(df),
        "value_bets": len(value_bets),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": wins / len(value_bets) if len(value_bets) > 0 else 0.0,
        "total_wagered": total_wagered,
        "total_profit": total_profit,
        "roi": roi,
        "kelly_roi": kelly_roi,
        "final_bankroll": bankroll,
    }


def filter_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = 0.05
) -> pd.DataFrame:
    """
    Filter predictions to only include value bets (edge >= min_edge).
    
    Args:
        predictions: DataFrame with predictions and edges
        min_edge: Minimum edge threshold (default 5%)
    
    Returns:
        Filtered DataFrame with only value bets
    """
    return predictions[predictions["edge"] >= min_edge].copy()
