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


def calculate_vig(home_implied: float, away_implied: float) -> float:
    """
    Calculate sportsbook vig (overround).
    
    Args:
        home_implied: Implied probability of home win
        away_implied: Implied probability of away win
    
    Returns:
        Vig (overround) percentage. Typically 2-5% for efficient markets.
    """
    return (home_implied + away_implied) - 1.0


def adjust_for_vig(implied_prob: float, total_implied: float) -> float:
    """
    Adjust implied probability to remove vig (calculate no-vig line).
    
    This removes the sportsbook's margin to get the "true" implied probability.
    
    Args:
        implied_prob: Original implied probability (includes vig)
        total_implied: Sum of all implied probabilities (1.0 + vig)
    
    Returns:
        No-vig implied probability
    """
    if total_implied <= 0:
        return implied_prob
    
    return implied_prob / total_implied


def calculate_moneyline_edge(
    home_win_prob: float,
    away_win_prob: float,
    home_odds: Optional[float],
    away_odds: Optional[float],
    use_no_vig: bool = True
) -> Dict[str, float]:
    """
    Calculate betting edge for moneyline with optional vig adjustment.
    
    Args:
        home_win_prob: Model's probability of home win
        away_win_prob: Model's probability of away win
        home_odds: Home team American odds
        away_odds: Away team American odds
        use_no_vig: If True, adjust for vig (recommended for accurate edges)
    
    Returns:
        Dict with edges and best side
    """
    if home_odds is None or away_odds is None:
        return {
            "home_edge": 0.0,
            "away_edge": 0.0,
            "best_side": None,
            "best_edge": 0.0,
            "vig": 0.0,
        }
    
    home_implied = american_odds_to_implied_prob(home_odds)
    away_implied = american_odds_to_implied_prob(away_odds)
    
    # Calculate vig
    vig = calculate_vig(home_implied, away_implied)
    total_implied = home_implied + away_implied
    
    # Adjust for vig (remove sportsbook margin) if requested
    if use_no_vig and total_implied > 1.0:
        home_implied_adj = adjust_for_vig(home_implied, total_implied)
        away_implied_adj = adjust_for_vig(away_implied, total_implied)
    else:
        home_implied_adj = home_implied
        away_implied_adj = away_implied
    
    home_edge = home_win_prob - home_implied_adj
    away_edge = away_win_prob - away_implied_adj
    
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
        "vig": vig,
        "home_implied_no_vig": home_implied_adj,
        "away_implied_no_vig": away_implied_adj,
    }


def calculate_spread_edge(cover_prob: float, odds: float, use_no_vig: bool = True) -> float:
    """
    Calculate betting edge for spread with optional vig adjustment.
    
    Args:
        cover_prob: Model's probability of covering
        odds: American odds (typically -110)
        use_no_vig: If True, adjust for vig (recommended for accurate edges)
    
    Returns:
        Edge (model_prob - implied_prob)
    
    Note:
        For spread bets, both sides typically have -110 odds, which includes ~4.55% vig.
        With no-vig adjustment, this becomes ~52.38% implied probability per side.
    """
    implied_prob = american_odds_to_implied_prob(odds)
    
    # For spread, both sides typically have same odds (-110), so vig is built into each side
    # With -110 odds: implied_prob = 0.5238, so total = 1.0476, vig = 4.76%
    # No-vig probability = 0.5238 / 1.0476 = 0.5000
    if use_no_vig:
        # Assume standard -110 on both sides (typical for spreads)
        total_implied = implied_prob * 2
        if total_implied > 1.0:
            implied_prob_adj = adjust_for_vig(implied_prob, total_implied)
        else:
            implied_prob_adj = implied_prob
    else:
        implied_prob_adj = implied_prob
    
    return cover_prob - implied_prob_adj


def calculate_totals_edge(
    over_prob: float,
    under_prob: float,
    over_odds: float,
    under_odds: float,
    use_no_vig: bool = True
) -> Dict[str, float]:
    """
    Calculate betting edge for totals (over/under) with optional vig adjustment.
    
    Args:
        over_prob: Model's probability of over
        under_prob: Model's probability of under
        over_odds: Over American odds
        under_odds: Under American odds
        use_no_vig: If True, adjust for vig (recommended for accurate edges)
    
    Returns:
        Dict with edges and best side
    """
    over_implied = american_odds_to_implied_prob(over_odds)
    under_implied = american_odds_to_implied_prob(under_odds)
    
    # Calculate vig
    vig = calculate_vig(over_implied, under_implied)
    total_implied = over_implied + under_implied
    
    # Adjust for vig if requested
    if use_no_vig and total_implied > 1.0:
        over_implied_adj = adjust_for_vig(over_implied, total_implied)
        under_implied_adj = adjust_for_vig(under_implied, total_implied)
    else:
        over_implied_adj = over_implied
        under_implied_adj = under_implied
    
    over_edge = over_prob - over_implied_adj
    under_edge = under_prob - under_implied_adj
    
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
        "vig": vig,
        "over_implied_no_vig": over_implied_adj,
        "under_implied_no_vig": under_implied_adj,
    }
