"""
Feature engineering for NFL, NHL, NBA, and MLB game predictions.
Builds comprehensive features including rolling stats, rest days, home/away splits, 
head-to-head records, ATS records, betting market features, and advanced metrics.
Supports all four major sports with sport-specific optimizations.

Now uses sport-specific feature engineers for advanced features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import timedelta
from app.training.sport_feature_engineers import get_feature_engineer


def calculate_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rest days between games for each team.
    
    Args:
        df: DataFrame with game data (must be sorted by date)
    
    Returns:
        DataFrame with rest_days_home, rest_days_away, rest_advantage, and back_to_back columns
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize rest days columns
    df["rest_days_home"] = np.nan
    df["rest_days_away"] = np.nan
    df["rest_advantage"] = 0.0
    df["home_back_to_back"] = 0
    df["away_back_to_back"] = 0
    
    # Track last game date for each team
    last_game_home = {}
    last_game_away = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        game_date = pd.to_datetime(row["date"])
        
        # Calculate rest days for home team
        if home_team in last_game_home:
            last_date = last_game_home[home_team]
            rest_days = (game_date - last_date).days - 1
            df.at[idx, "rest_days_home"] = max(0, rest_days) if rest_days >= 0 else np.nan
            # Back-to-back indicator (0 or 1 day rest)
            if rest_days <= 1:
                df.at[idx, "home_back_to_back"] = 1
        last_game_home[home_team] = game_date
        
        # Calculate rest days for away team
        if away_team in last_game_away:
            last_date = last_game_away[away_team]
            rest_days = (game_date - last_date).days - 1
            df.at[idx, "rest_days_away"] = max(0, rest_days) if rest_days >= 0 else np.nan
            if rest_days <= 1:
                df.at[idx, "away_back_to_back"] = 1
        last_game_away[away_team] = game_date
        
        # Rest advantage (positive = home team has more rest)
        if pd.notna(df.at[idx, "rest_days_home"]) and pd.notna(df.at[idx, "rest_days_away"]):
            df.at[idx, "rest_advantage"] = df.at[idx, "rest_days_home"] - df.at[idx, "rest_days_away"]
    
    return df


def calculate_multi_window_rolling_stats(
    df: pd.DataFrame,
    sport: str,
    rolling_windows: List[int] = [3, 5, 10, 15]
) -> pd.DataFrame:
    """
    Calculate rolling statistics for multiple time windows.
    
    Args:
        df: DataFrame with game data (must be sorted by date)
        sport: Sport code (NFL, NHL, NBA, MLB)
        rolling_windows: List of window sizes to calculate
    
    Returns:
        DataFrame with rolling statistics for each window
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    if "home_score" not in df.columns or "away_score" not in df.columns:
        return df
    
    # Track game history for each team
    team_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        # Process both teams
        for team, is_home, team_score, opp_score in [
            (home_team, True, home_score, away_score),
            (away_team, False, away_score, home_score)
        ]:
            prefix = "home" if is_home else "away"
            
            if team not in team_history:
                team_history[team] = []
            
            # Get recent games (before this game, with valid scores)
            recent_games = [
                g for g in team_history[team] 
                if g["date"] < game_date and g.get("has_score", False)
            ]
            recent_games = sorted(recent_games, key=lambda x: x["date"], reverse=True)
            
            # Calculate stats for each window
            for window in rolling_windows:
                window_games = recent_games[:window]
                
                if window_games:
                    wins = sum(1 for g in window_games if g["won"])
                    points_for = [g["points_for"] for g in window_games]
                    points_against = [g["points_against"] for g in window_games]
                    
                    df.at[idx, f"{prefix}_win_rate_{window}"] = wins / len(window_games)
                    df.at[idx, f"{prefix}_points_for_avg_{window}"] = np.mean(points_for)
                    df.at[idx, f"{prefix}_points_against_avg_{window}"] = np.mean(points_against)
                    df.at[idx, f"{prefix}_point_diff_avg_{window}"] = np.mean([pf - pa for pf, pa in zip(points_for, points_against)])
                    
                    # Momentum: recent vs older performance
                    if len(recent_games) >= window * 2:
                        recent_half = window_games[:window//2] if window >= 4 else window_games
                        older_half = recent_games[window:window+window//2] if window >= 4 else []
                        if older_half:
                            recent_avg = np.mean([g["points_for"] - g["points_against"] for g in recent_half])
                            older_avg = np.mean([g["points_for"] - g["points_against"] for g in older_half])
                            df.at[idx, f"{prefix}_momentum_{window}"] = recent_avg - older_avg
                    
                    # Efficiency metrics
                    df.at[idx, f"{prefix}_offensive_efficiency_{window}"] = np.mean(points_for)
                    df.at[idx, f"{prefix}_defensive_efficiency_{window}"] = np.mean(points_against)
            
            # Add this game to history (only if it has scores)
            if has_scores:
                team_history[team].append({
                    "date": game_date,
                    "points_for": team_score,
                    "points_against": opp_score,
                    "won": team_score > opp_score,
                    "is_home": is_home,
                    "has_score": True
                })
    
    return df


def calculate_ats_records(
    df: pd.DataFrame,
    rolling_window: int = 10
) -> pd.DataFrame:
    """
    Calculate Against The Spread (ATS) records.
    
    Args:
        df: DataFrame with game data
        rolling_window: Number of recent games to include
    
    Returns:
        DataFrame with ATS statistics
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize ATS columns
    for prefix in ["home", "away"]:
        df[f"{prefix}_ats_win_rate"] = np.nan
        df[f"{prefix}_ats_wins"] = 0
        df[f"{prefix}_ats_games"] = 0
    
    team_ats_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        spread = row.get("spread", 0) if pd.notna(row.get("spread")) else 0
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        # Process home team ATS
        if home_team not in team_ats_history:
            team_ats_history[home_team] = []
        
        recent_ats = [
            g for g in team_ats_history[home_team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_ats = sorted(recent_ats, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_ats:
            ats_wins = sum(1 for g in recent_ats if g["covered"])
            df.at[idx, "home_ats_wins"] = ats_wins
            df.at[idx, "home_ats_games"] = len(recent_ats)
            df.at[idx, "home_ats_win_rate"] = ats_wins / len(recent_ats) if len(recent_ats) > 0 else 0
        
        # Process away team ATS
        if away_team not in team_ats_history:
            team_ats_history[away_team] = []
        
        recent_ats = [
            g for g in team_ats_history[away_team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_ats = sorted(recent_ats, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_ats:
            ats_wins = sum(1 for g in recent_ats if g["covered"])
            df.at[idx, "away_ats_wins"] = ats_wins
            df.at[idx, "away_ats_games"] = len(recent_ats)
            df.at[idx, "away_ats_win_rate"] = ats_wins / len(recent_ats) if len(recent_ats) > 0 else 0
        
        # Add this game to ATS history
        if has_scores and pd.notna(spread):
            # Home team covers if (home_score - away_score) > spread
            home_covered = (home_score - away_score) > spread
            team_ats_history[home_team].append({
                "date": game_date,
                "covered": home_covered,
                "spread": spread,
                "has_score": True
            })
            
            # Away team covers if (away_score - home_score) > -spread (i.e., home doesn't cover by spread)
            away_covered = (away_score - home_score) > -spread
            team_ats_history[away_team].append({
                "date": game_date,
                "covered": away_covered,
                "spread": -spread,
                "has_score": True
            })
    
    return df


def calculate_totals_records(
    df: pd.DataFrame,
    rolling_window: int = 10
) -> pd.DataFrame:
    """
    Calculate Over/Under records.
    
    Args:
        df: DataFrame with game data
        rolling_window: Number of recent games to include
    
    Returns:
        DataFrame with Over/Under statistics
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize totals columns
    for prefix in ["home", "away"]:
        df[f"{prefix}_over_rate"] = np.nan
        df[f"{prefix}_over_games"] = 0
    
    team_totals_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        over_under = row.get("over_under", 0) if pd.notna(row.get("over_under")) else 0
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        total_score = home_score + away_score if has_scores else 0
        
        # Process home team totals
        if home_team not in team_totals_history:
            team_totals_history[home_team] = []
        
        recent_totals = [
            g for g in team_totals_history[home_team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_totals = sorted(recent_totals, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_totals:
            overs = sum(1 for g in recent_totals if g["over"])
            df.at[idx, "home_over_games"] = len(recent_totals)
            df.at[idx, "home_over_rate"] = overs / len(recent_totals) if len(recent_totals) > 0 else 0
        
        # Process away team totals
        if away_team not in team_totals_history:
            team_totals_history[away_team] = []
        
        recent_totals = [
            g for g in team_totals_history[away_team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_totals = sorted(recent_totals, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_totals:
            overs = sum(1 for g in recent_totals if g["over"])
            df.at[idx, "away_over_games"] = len(recent_totals)
            df.at[idx, "away_over_rate"] = overs / len(recent_totals) if len(recent_totals) > 0 else 0
        
        # Add this game to totals history
        if has_scores and pd.notna(over_under) and over_under > 0:
            went_over = total_score > over_under
            team_totals_history[home_team].append({
                "date": game_date,
                "over": went_over,
                "total": total_score,
                "has_score": True
            })
            team_totals_history[away_team].append({
                "date": game_date,
                "over": went_over,
                "total": total_score,
                "has_score": True
            })
    
    return df


def calculate_recent_outcomes(
    df: pd.DataFrame,
    windows: List[int] = [3, 5]
) -> pd.DataFrame:
    """
    Calculate recent game outcomes (wins/losses in last N games).
    
    Args:
        df: DataFrame with game data
        windows: List of window sizes
    
    Returns:
        DataFrame with recent outcome features
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    team_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        for team, is_home, team_score, opp_score in [
            (home_team, True, home_score, away_score),
            (away_team, False, away_score, home_score)
        ]:
            prefix = "home" if is_home else "away"
            
            if team not in team_history:
                team_history[team] = []
            
            recent_games = [
                g for g in team_history[team]
                if g["date"] < game_date and g.get("has_score", False)
            ]
            recent_games = sorted(recent_games, key=lambda x: x["date"], reverse=True)
            
            for window in windows:
                window_games = recent_games[:window]
                if window_games:
                    wins = sum(1 for g in window_games if g["won"])
                    df.at[idx, f"{prefix}_wins_last_{window}"] = wins
                    df.at[idx, f"{prefix}_losses_last_{window}"] = len(window_games) - wins
                    df.at[idx, f"{prefix}_win_streak"] = 0
                    
                    # Calculate current streak
                    streak = 0
                    for g in window_games:
                        if g["won"]:
                            streak += 1
                        else:
                            break
                    df.at[idx, f"{prefix}_win_streak"] = streak
            
            if has_scores:
                team_history[team].append({
                    "date": game_date,
                    "won": team_score > opp_score,
                    "has_score": True
                })
    
    return df


def calculate_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate contextual features like blowouts, close games, etc.
    
    Args:
        df: DataFrame with game data
    
    Returns:
        DataFrame with contextual features
    """
    df = df.copy()
    
    # Blowout threshold (sport-specific)
    sport_str = str(df.get("sport", "").iloc[0] if not df.empty else "")
    if "NFL" in sport_str or "NBA" in sport_str:
        blowout_threshold = 14  # Football and basketball
    elif "MLB" in sport_str:
        blowout_threshold = 5   # Baseball
    else:
        blowout_threshold = 3   # Hockey
    
    # Close game threshold
    if "NFL" in sport_str or "NBA" in sport_str:
        close_threshold = 7
    elif "MLB" in sport_str:
        close_threshold = 2
    else:
        close_threshold = 1
    
    # Calculate for historical context (would need team history, simplified here)
    df["expected_margin"] = (
        df.get("home_point_diff_avg", pd.Series(0)) - 
        df.get("away_point_diff_avg", pd.Series(0))
    )
    
    return df


def calculate_opponent_adjusted_stats(
    df: pd.DataFrame,
    rolling_window: int = 10
) -> pd.DataFrame:
    """
    Calculate opponent-adjusted statistics (strength of schedule).
    
    Args:
        df: DataFrame with game data
        rolling_window: Number of recent games to include
    
    Returns:
        DataFrame with opponent-adjusted features
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Calculate average opponent strength for each team
    team_history = {}
    team_opponent_strength = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        # Get opponent strengths
        home_opp_strength = team_opponent_strength.get(away_team, 0)
        away_opp_strength = team_opponent_strength.get(home_team, 0)
        
        df.at[idx, "home_opponent_strength"] = home_opp_strength
        df.at[idx, "away_opponent_strength"] = away_opp_strength
        
        # Update opponent strength based on recent performance
        if has_scores:
            for team, team_score, opp_score in [
                (home_team, home_score, away_score),
                (away_team, away_score, home_score)
            ]:
                if team not in team_history:
                    team_history[team] = []
                
                recent_games = [
                    g for g in team_history[team]
                    if g["date"] < game_date and g.get("has_score", False)
                ]
                recent_games = sorted(recent_games, key=lambda x: x["date"], reverse=True)[:rolling_window]
                
                if recent_games:
                    avg_point_diff = np.mean([g["points_for"] - g["points_against"] for g in recent_games])
                    team_opponent_strength[team] = avg_point_diff
                
                team_history[team].append({
                    "date": game_date,
                    "points_for": team_score,
                    "points_against": opp_score,
                    "has_score": True
                })
    
    return df


def calculate_betting_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting market features (CLV, line value, etc.).
    
    Args:
        df: DataFrame with game data
    
    Returns:
        DataFrame with betting market features
    """
    df = df.copy()
    
    # Implied probabilities from moneylines
    if "home_moneyline" in df.columns and "away_moneyline" in df.columns:
        def ml_to_prob(ml):
            if pd.isna(ml) or ml == 0:
                return 0.5
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        
        df["home_implied_prob"] = df["home_moneyline"].apply(ml_to_prob)
        df["away_implied_prob"] = df["away_moneyline"].apply(ml_to_prob)
        df["market_total_prob"] = df["home_implied_prob"] + df["away_implied_prob"]
    
    # REMOVED: spread_value and totals_value calculation
    # These features cause data leakage because they directly encode the prediction target:
    # - spread_value > 0 means "cover", spread_value < 0 means "don't cover" (directly encodes label)
    # - totals_value > 0 means "over", totals_value < 0 means "under" (directly encodes label)
    # Even if calculated from rolling averages, these features allow the model to trivially predict outcomes
    # by learning: if spread_value > 0, predict 1, else predict 0
    # 
    # If we need market value features in the future, they should be calculated differently:
    # - Use pre-game projections from separate models, not rolling averages
    # - Or use betting market implied probabilities only (which we already have)
    pass
    
    return df


def calculate_home_away_splits(
    df: pd.DataFrame,
    rolling_window: int = 10
) -> pd.DataFrame:
    """
    Calculate home/away specific statistics.
    
    Args:
        df: DataFrame with game data
        rolling_window: Number of recent games to include
    
    Returns:
        DataFrame with home/away split statistics
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize columns
    df["home_win_rate_home"] = np.nan
    df["home_win_rate_away"] = np.nan
    df["away_win_rate_home"] = np.nan
    df["away_win_rate_away"] = np.nan
    df["home_field_advantage"] = 0.0
    
    # Track home/away game history for each team
    team_home_history = {}
    team_away_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        # Home team's home games
        if home_team not in team_home_history:
            team_home_history[home_team] = []
        
        recent_home = [
            g for g in team_home_history[home_team] 
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_home = sorted(recent_home, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_home:
            wins = sum(1 for g in recent_home if g["won"])
            df.at[idx, "home_win_rate_home"] = wins / len(recent_home) if len(recent_home) > 0 else 0
        
        if has_scores:
            team_home_history[home_team].append({
                "date": game_date,
                "won": home_score > away_score,
                "has_score": True
            })
        
        # Home team's away games
        if home_team not in team_away_history:
            team_away_history[home_team] = []
        
        recent_away = [
            g for g in team_away_history[home_team] 
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_away = sorted(recent_away, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_away:
            wins = sum(1 for g in recent_away if g["won"])
            df.at[idx, "home_win_rate_away"] = wins / len(recent_away) if len(recent_away) > 0 else 0
        
        # Away team's home games
        if away_team not in team_home_history:
            team_home_history[away_team] = []
        
        recent_home = [
            g for g in team_home_history[away_team] 
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_home = sorted(recent_home, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_home:
            wins = sum(1 for g in recent_home if g["won"])
            df.at[idx, "away_win_rate_home"] = wins / len(recent_home) if len(recent_home) > 0 else 0
        
        # Away team's away games
        if away_team not in team_away_history:
            team_away_history[away_team] = []
        
        recent_away = [
            g for g in team_away_history[away_team] 
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_away = sorted(recent_away, key=lambda x: x["date"], reverse=True)[:rolling_window]
        
        if recent_away:
            wins = sum(1 for g in recent_away if g["won"])
            df.at[idx, "away_win_rate_away"] = wins / len(recent_away) if len(recent_away) > 0 else 0
        
        if has_scores:
            team_away_history[away_team].append({
                "date": game_date,
                "won": away_score > home_score,
                "has_score": True
            })
        
        # Home field advantage (difference between home win rate at home vs away)
        if pd.notna(df.at[idx, "home_win_rate_home"]) and pd.notna(df.at[idx, "home_win_rate_away"]):
            df.at[idx, "home_field_advantage"] = (
                df.at[idx, "home_win_rate_home"] - df.at[idx, "home_win_rate_away"]
            )
    
    return df


def calculate_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head statistics between teams.
    
    Args:
        df: DataFrame with game data (must be sorted by date)
    
    Returns:
        DataFrame with head-to-head statistics
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Initialize columns
    df["h2h_home_wins"] = 0
    df["h2h_total_games"] = 0
    df["h2h_home_win_rate"] = np.nan
    df["h2h_avg_margin"] = 0.0
    
    # Track head-to-head history
    h2h_history = {}
    
    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        game_date = pd.to_datetime(row["date"])
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        
        # Create match key (sorted team names for consistency)
        match_key = tuple(sorted([home_team, away_team]))
        
        if match_key not in h2h_history:
            h2h_history[match_key] = []
        
        # Get previous head-to-head games (before this game, with valid scores)
        prev_games = [
            g for g in h2h_history[match_key] 
            if g["date"] < game_date and g.get("has_score", False)
        ]
        
        if prev_games:
            # Count wins for home team in previous matchups
            home_wins = 0
            margins = []
            for g in prev_games:
                if g["home_team"] == home_team:
                    margin = g["home_score"] - g["away_score"]
                    if margin > 0:
                        home_wins += 1
                else:
                    margin = g["away_score"] - g["home_score"]
                    if margin > 0:
                        home_wins += 1
                margins.append(abs(margin))
            
            df.at[idx, "h2h_home_wins"] = home_wins
            df.at[idx, "h2h_total_games"] = len(prev_games)
            df.at[idx, "h2h_home_win_rate"] = home_wins / len(prev_games) if len(prev_games) > 0 else 0
            df.at[idx, "h2h_avg_margin"] = np.mean(margins) if margins else 0
        
        # Add this game to history (only if it has scores)
        if has_scores:
            h2h_history[match_key].append({
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "has_score": True
            })
    
    return df


def calculate_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team strength metrics (ELO-like ratings).
    
    Args:
        df: DataFrame with game data
    
    Returns:
        DataFrame with team strength features
    """
    df = df.copy()
    
    # Simple strength metric: point differential (use 10-game window if available)
    home_diff_col = "home_point_diff_avg_10" if "home_point_diff_avg_10" in df.columns else "home_point_diff_avg"
    away_diff_col = "away_point_diff_avg_10" if "away_point_diff_avg_10" in df.columns else "away_point_diff_avg"
    
    df["home_strength"] = df[home_diff_col].fillna(0) if home_diff_col in df.columns else 0
    df["away_strength"] = df[away_diff_col].fillna(0) if away_diff_col in df.columns else 0
    df["strength_diff"] = df["home_strength"] - df["away_strength"]
    
    # Win rate difference
    home_wr_col = "home_win_rate_10" if "home_win_rate_10" in df.columns else "home_win_rate"
    away_wr_col = "away_win_rate_10" if "away_win_rate_10" in df.columns else "away_win_rate"
    
    home_wr = df[home_wr_col].fillna(0.5) if home_wr_col in df.columns else pd.Series([0.5] * len(df))
    away_wr = df[away_wr_col].fillna(0.5) if away_wr_col in df.columns else pd.Series([0.5] * len(df))
    df["win_rate_diff"] = home_wr - away_wr
    
    return df


def calculate_market_specific_features(
    df: pd.DataFrame,
    sport: str,
    market: str
) -> pd.DataFrame:
    """
    Calculate market-specific features.
    
    Args:
        df: DataFrame with game data
        sport: Sport code
        market: Market type
    
    Returns:
        DataFrame with market-specific features
    """
    df = df.copy()
    
    if market == "spread":
        # Spread-specific features
        # REMOVED: spread_line - allows model to reconstruct edge with home_point_diff_avg
        # REMOVED: home_cover_prob - this was data leakage (directly encoding the label)
        # REMOVED: spread_edge - this was also data leakage (edge > 0 = cover, edge < 0 = don't cover)
        # The model can learn: if spread_edge > 0, predict 1, else predict 0
        pass  # No spread-specific features to avoid data leakage
    
    elif market == "totals":
        # Totals-specific features
        home_col = "home_points_for_avg_10" if "home_points_for_avg_10" in df.columns else "home_points_for_avg"
        away_col = "away_points_for_avg_10" if "away_points_for_avg_10" in df.columns else "away_points_for_avg"
        home_avg = df[home_col].fillna(0) if home_col in df.columns else pd.Series([0] * len(df))
        away_avg = df[away_col].fillna(0) if away_col in df.columns else pd.Series([0] * len(df))
        df["total_projection"] = home_avg + away_avg
        # REMOVED: over_under_line - allows model to reconstruct edge with total_projection
        # REMOVED: over_prob - this was data leakage (directly encoding the label)
        # REMOVED: totals_edge - this was also data leakage (edge > 0 = over, edge < 0 = under)
        # The model can learn: if totals_edge > 0, predict 1, else predict 0
    
    elif market == "score_projection":
        # Score projection features
        if "home_score" in df.columns:
            df["home_score"] = df["home_score"].fillna(0)
        if "away_score" in df.columns:
            df["away_score"] = df["away_score"].fillna(0)
    
    return df


def build_features(
    df: pd.DataFrame,
    sport: str,
    market: str,
    rolling_window: int = 10,
    include_rest_days: bool = True,
    include_h2h: bool = True,
    use_sport_specific: bool = True
) -> pd.DataFrame:
    """
    Build comprehensive features for game predictions.
    
    Now uses sport-specific feature engineers for advanced features.
    
    Args:
        df: DataFrame with game data
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals, score_projection)
        rolling_window: Number of recent games for rolling stats
        include_rest_days: Whether to include rest days feature
        include_h2h: Whether to include head-to-head features
        use_sport_specific: Whether to use sport-specific feature engineers
    
    Returns:
        DataFrame with features added
    """
    if df.empty:
        return df
    
    # Use sport-specific feature engineer if available
    if use_sport_specific:
        try:
            engineer = get_feature_engineer(sport)
            engineer.rolling_windows = [3, 5, 10, 15]
            engineer.include_rest_days = include_rest_days
            engineer.include_h2h = include_h2h
            
            # Build features using sport-specific engineer
            df = engineer.build_features(df, sport, market)
            
            # Add standard features that aren't in base class
            print("Calculating ATS records...")
            df = calculate_ats_records(df, rolling_window)
            
            print("Calculating Over/Under records...")
            df = calculate_totals_records(df, rolling_window)
            
            print("Calculating recent outcomes...")
            df = calculate_recent_outcomes(df, windows=[3, 5])
            
            print("Calculating opponent-adjusted statistics...")
            df = calculate_opponent_adjusted_stats(df, rolling_window)
            
            print("Calculating team strength metrics...")
            df = calculate_team_strength(df)
            
            print("Calculating betting market features...")
            df = calculate_betting_market_features(df)
            
            print(f"Calculating {market}-specific features...")
            df = calculate_market_specific_features(df, sport, market)
            
            # Fill remaining NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            return df
        except Exception as e:
            print(f"Warning: Sport-specific feature engineer failed ({e}), falling back to standard features")
            # Fall through to standard implementation
    
    # Standard feature building (fallback)
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date for proper feature calculation
    df = df.sort_values("date").reset_index(drop=True)
    
    # Add sport column if not present
    if "sport" not in df.columns:
        df["sport"] = sport
    
    # Calculate multi-window rolling statistics
    print(f"Calculating multi-window rolling statistics...")
    df = calculate_multi_window_rolling_stats(df, sport, rolling_windows=[3, 5, 10, 15])
    
    # Calculate rest days and rest advantage
    if include_rest_days:
        print("Calculating rest days and rest advantage...")
        df = calculate_rest_days(df)
        df["rest_days_home"] = df["rest_days_home"].fillna(df["rest_days_home"].median() if not df["rest_days_home"].isna().all() else 3)
        df["rest_days_away"] = df["rest_days_away"].fillna(df["rest_days_away"].median() if not df["rest_days_away"].isna().all() else 3)
    
    # Calculate ATS records
    print("Calculating ATS records...")
    df = calculate_ats_records(df, rolling_window)
    
    # Calculate Over/Under records
    print("Calculating Over/Under records...")
    df = calculate_totals_records(df, rolling_window)
    
    # Calculate recent outcomes
    print("Calculating recent outcomes...")
    df = calculate_recent_outcomes(df, windows=[3, 5])
    
    # Calculate home/away splits
    print("Calculating home/away splits...")
    df = calculate_home_away_splits(df, rolling_window)
    
    # Calculate head-to-head
    if include_h2h:
        print("Calculating head-to-head records...")
        df = calculate_head_to_head(df)
        df["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.5)
    
    # Calculate opponent-adjusted stats
    print("Calculating opponent-adjusted statistics...")
    df = calculate_opponent_adjusted_stats(df, rolling_window)
    
    # Calculate team strength (before betting market features)
    print("Calculating team strength metrics...")
    df = calculate_team_strength(df)
    
    # Calculate betting market features
    print("Calculating betting market features...")
    df = calculate_betting_market_features(df)
    
    # Market-specific features
    print(f"Calculating {market}-specific features...")
    df = calculate_market_specific_features(df, sport, market)
    
    # Fill remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def get_feature_columns(sport: str, market: str) -> List[str]:
    """
    Get list of feature column names for a sport and market.
    Includes sport-specific features.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        market: Market type (moneyline, spread, totals, score_projection)
    
    Returns:
        List of feature column names
    """
    base_features = []
    
    # Multi-window rolling statistics
    for window in [3, 5, 10, 15]:
        for prefix in ["home", "away"]:
            features_to_add = [
                f"{prefix}_win_rate_{window}",
                f"{prefix}_points_for_avg_{window}",
                f"{prefix}_points_against_avg_{window}",
                f"{prefix}_offensive_efficiency_{window}",
                f"{prefix}_defensive_efficiency_{window}",
            ]
            # REMOVED for spread market: point_diff_avg allows model to reconstruct edge
            # (if home_point_diff_avg > spread_threshold, predict cover)
            if market != "spread":
                features_to_add.append(f"{prefix}_point_diff_avg_{window}")
            base_features.extend(features_to_add)
            # REMOVED for spread market: momentum uses point differential (recent_avg - older_avg)
            # which can allow edge reconstruction
            if window >= 4 and market != "spread":
                base_features.append(f"{prefix}_momentum_{window}")
    
    # Rest days
    base_features.extend([
        "rest_days_home",
        "rest_days_away",
        "rest_advantage",
        "home_back_to_back",
        "away_back_to_back",
    ])
    
    # ATS records
    base_features.extend([
        "home_ats_win_rate",
        "away_ats_win_rate",
        "home_ats_wins",
        "away_ats_wins",
        "home_ats_games",
        "away_ats_games",
    ])
    
    # Over/Under records
    base_features.extend([
        "home_over_rate",
        "away_over_rate",
        "home_over_games",
        "away_over_games",
    ])
    
    # Recent outcomes
    for window in [3, 5]:
        for prefix in ["home", "away"]:
            base_features.extend([
                f"{prefix}_wins_last_{window}",
                f"{prefix}_losses_last_{window}",
            ])
    base_features.extend(["home_win_streak", "away_win_streak"])
    
    # Home/away splits
    base_features.extend([
        "home_win_rate_home",
        "home_win_rate_away",
        "away_win_rate_home",
        "away_win_rate_away",
        "home_field_advantage",
    ])
    
    # Head-to-head
    h2h_features = [
        "h2h_home_wins",
        "h2h_total_games",
        "h2h_home_win_rate",
    ]
    # REMOVED for spread market: h2h_avg_margin allows model to reconstruct edge
    if market != "spread":
        h2h_features.append("h2h_avg_margin")
    base_features.extend(h2h_features)
    
    # Opponent-adjusted
    # REMOVED for spread market: opponent_strength uses point_diff_avg
    # Model can reconstruct edge: away_opponent_strength - home_opponent_strength = strength_diff
    if market == "spread":
        # Skip opponent_strength for spread (uses point differential)
        pass
    else:
        base_features.extend([
            "home_opponent_strength",
            "away_opponent_strength",
        ])
    
    # Betting market features
    base_features.extend([
        "home_implied_prob",
        "away_implied_prob",
        "market_total_prob",
        # REMOVED: "spread_value" - data leakage for spread market (value > 0 = cover, value < 0 = don't cover)
        # REMOVED: "totals_value" - data leakage for totals market (value > 0 = over, value < 0 = under)
    ])
    
    # Team strength
    # REMOVED for spread market: home_strength, away_strength, strength_diff use point_diff_avg
    # which allows model to reconstruct edge (if strength_diff > threshold, predict cover)
    # This is especially problematic when spreads are missing/null (filled with 0),
    # making the label "did home win?" which strength_diff can perfectly predict
    if market == "spread":
        base_features.extend([
            "win_rate_diff",  # Only win rate diff is safe (doesn't use point differential)
        ])
    else:
        base_features.extend([
            "home_strength",
            "away_strength",
            "strength_diff",
            "win_rate_diff",
        ])
    
    # Sport-specific features
    sport_upper = sport.upper()
    if sport_upper == "NFL":
        base_features.extend([
            "division_matchup",
            "home_third_down_pct",
            "away_third_down_pct",
            "home_redzone_efficiency",
            "away_redzone_efficiency",
            "turnover_differential",
            # REMOVED: home_time_of_possession, away_time_of_possession - was causing data leakage
            # Time of possession was being calculated from current game outcome (home_score > away_score)
            # which directly encodes the target variable. Until we can calculate from historical data only,
            # we'll exclude it.
        ])
        for window in [3, 5, 10, 15]:
            for prefix in ["home", "away"]:
                base_features.extend([
                    f"{prefix}_third_down_pct_{window}",
                    f"{prefix}_redzone_efficiency_{window}",
                    f"{prefix}_turnover_differential_{window}",
                    # REMOVED: f"{prefix}_time_of_possession_{window}" - data leakage
                ])
    elif sport_upper == "NHL":
        base_features.extend([
            "home_goalie_save_pct",
            "away_goalie_save_pct",
            "home_goalie_gaa",
            "away_goalie_gaa",
            # REMOVED: home_corsi, away_corsi - was causing data leakage
            # Corsi was being calculated from current game outcome (home_score > away_score)
            # which directly encodes the target variable. Until we can calculate Corsi from
            # historical shot data only, we'll exclude it.
            "home_powerplay_pct",
            "away_powerplay_pct",
            "home_penaltykill_pct",
            "away_penaltykill_pct",
            "home_goalie_back_to_back",
            "away_goalie_back_to_back",
        ])
        for window in [3, 5, 10, 15]:
            for prefix in ["home", "away"]:
                base_features.extend([
                    f"{prefix}_goalie_save_pct_{window}",
                    f"{prefix}_goalie_gaa_{window}",
                    # REMOVED: f"{prefix}_corsi_{window}" - data leakage
                    f"{prefix}_powerplay_pct_{window}",
                    f"{prefix}_penaltykill_pct_{window}",
                ])
    elif sport_upper == "NBA":
        base_features.extend([
            "home_pace",
            "away_pace",
            "home_efg_pct",
            "away_efg_pct",
            "home_true_shooting_pct",
            "away_true_shooting_pct",
            "home_off_reb_pct",
            "away_off_reb_pct",
            "home_def_reb_pct",
            "away_def_reb_pct",
            "home_assists_per_game",
            "away_assists_per_game",
            "home_turnovers_per_game",
            "away_turnovers_per_game",
            "home_3p_pct",
            "away_3p_pct",
            "conference_matchup",
        ])
        for window in [3, 5, 10, 15]:
            for prefix in ["home", "away"]:
                base_features.extend([
                    f"{prefix}_pace_{window}",
                    f"{prefix}_efg_pct_{window}",
                    f"{prefix}_true_shooting_pct_{window}",
                    f"{prefix}_off_reb_pct_{window}",
                    f"{prefix}_def_reb_pct_{window}",
                    f"{prefix}_assists_per_game_{window}",
                    f"{prefix}_turnovers_per_game_{window}",
                    f"{prefix}_3p_pct_{window}",
                ])
    elif sport_upper == "MLB":
        base_features.extend([
            "home_pitcher_era",
            "away_pitcher_era",
            "home_pitcher_whip",
            "away_pitcher_whip",
            "home_pitcher_k9",
            "away_pitcher_k9",
            "home_pitcher_rest_days",
            "away_pitcher_rest_days",
            "day_game",
            "home_ops",
            "away_ops",
            "home_hr_rate",
            "away_hr_rate",
            "ballpark_factor",
        ])
        for window in [3, 5, 10, 15]:
            for prefix in ["home", "away"]:
                base_features.extend([
                    f"{prefix}_pitcher_era_{window}",
                    f"{prefix}_pitcher_whip_{window}",
                    f"{prefix}_pitcher_k9_{window}",
                    f"{prefix}_ops_{window}",
                    f"{prefix}_hr_rate_{window}",
                ])
    
    # Market-specific features
    if market == "spread":
        market_features = [
            # REMOVED: "spread_line" - allows model to reconstruct edge with home_point_diff_avg
            # REMOVED: "home_cover_prob" - data leakage (directly encodes label)
            # REMOVED: "spread_edge" - data leakage (edge > 0 = cover, edge < 0 = don't cover)
        ]
    elif market == "totals":
        market_features = [
            "total_projection",  # Keep this - it's the expected total, not the betting line
            # REMOVED: "over_under_line" - allows model to reconstruct edge with total_projection
            # REMOVED: "over_prob" - data leakage (directly encodes label)
            # REMOVED: "totals_edge" - data leakage (edge > 0 = over, edge < 0 = under)
        ]
    elif market == "score_projection":
        market_features = []
    else:  # moneyline
        market_features = []
    
    # Combine all features
    all_features = base_features + market_features
    
    # Filter out None values and return unique list
    return list(dict.fromkeys([f for f in all_features if f is not None]))
