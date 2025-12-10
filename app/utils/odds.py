"""
Odds conversion and edge calculation utilities.
"""
from typing import Dict, Optional
import numpy as np


def american_odds_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -110, +150)
    
    Returns:
        Implied probability (0.0 to 1.0)
    """
    if odds is None or np.isnan(odds):
        return 0.5
    
    if odds > 0:
        # Positive odds: probability = 100 / (odds + 100)
        return 100 / (odds + 100)
    else:
        # Negative odds: probability = |odds| / (|odds| + 100)
        return abs(odds) / (abs(odds) + 100)


def implied_prob_to_american_odds(prob: float) -> float:
    """
    Convert implied probability to American odds.
    
    Args:
        prob: Implied probability (0.0 to 1.0)
    
    Returns:
        American odds
    """
    if prob <= 0 or prob >= 1:
        return 0
    
    if prob < 0.5:
        # Underdog: positive odds
        return (100 / prob) - 100
    else:
        # Favorite: negative odds
        return -((100 * prob) / (1 - prob))


def calculate_moneyline_edge(
    home_win_prob: float,
    away_win_prob: float,
    home_odds: Optional[float],
    away_odds: Optional[float]
) -> Dict[str, float]:
    """
    Calculate betting edge for moneyline.
    
    Args:
        home_win_prob: Model's probability of home win
        away_win_prob: Model's probability of away win
        home_odds: Home team American odds
        away_odds: Away team American odds
    
    Returns:
        Dict with edges and best side
    """
    if home_odds is None or away_odds is None:
        return {
            "home_edge": 0.0,
            "away_edge": 0.0,
            "best_side": None,
            "best_edge": 0.0,
        }
    
    home_implied = american_odds_to_implied_prob(home_odds)
    away_implied = american_odds_to_implied_prob(away_odds)
    
    home_edge = home_win_prob - home_implied
    away_edge = away_win_prob - away_implied
    
    # Only recommend a side if it has a positive edge
    # If both edges are negative, don't recommend either side
    if home_edge > away_edge and home_edge > 0:
        best_side = "home"
        best_edge = home_edge
    elif away_edge > 0:
        best_side = "away"
        best_edge = away_edge
    else:
        # Both edges are negative or zero - no value bet
        best_side = None
        best_edge = max(home_edge, away_edge)  # Most negative (least bad)
    
    return {
        "home_edge": home_edge,
        "away_edge": away_edge,
        "best_side": best_side,
        "best_edge": best_edge,
    }


def calculate_spread_edge(cover_prob: float, odds: float) -> float:
    """
    Calculate betting edge for spread.
    
    Args:
        cover_prob: Model's probability of covering
        odds: American odds (typically -110)
    
    Returns:
        Edge (model_prob - implied_prob)
    """
    implied_prob = american_odds_to_implied_prob(odds)
    return cover_prob - implied_prob


def calculate_totals_edge(
    over_prob: float,
    under_prob: float,
    over_odds: float,
    under_odds: float
) -> Dict[str, float]:
    """
    Calculate betting edge for totals (over/under).
    
    Args:
        over_prob: Model's probability of over
        under_prob: Model's probability of under
        over_odds: Over American odds
        under_odds: Under American odds
    
    Returns:
        Dict with edges and best side
    """
    over_implied = american_odds_to_implied_prob(over_odds)
    under_implied = american_odds_to_implied_prob(under_odds)
    
    over_edge = over_prob - over_implied
    under_edge = under_prob - under_implied
    
    if over_edge > under_edge and over_edge > 0:
        best_side = "over"
        best_edge = over_edge
    elif under_edge > 0:
        best_side = "under"
        best_edge = under_edge
    else:
        best_side = None
        best_edge = max(over_edge, under_edge)
    
    return {
        "over_edge": over_edge,
        "under_edge": under_edge,
        "best_side": best_side,
        "best_edge": best_edge,
    }
