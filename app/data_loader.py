"""
Data loading utilities for historical game data.
Supports loading from database with season/week splits for train/validation.
"""
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.database import SessionLocal
from app.models.db_models import Game


def load_games_for_sport(
    sport: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_games_per_team: int = 10,
    status: str = "final"
) -> pd.DataFrame:
    """
    Load historical games for a sport from the database.
    
    Args:
        sport: Sport code (NFL, NHL, etc.)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        min_games_per_team: Minimum games a team must have to be included
        status: Game status filter (default: "final")
    
    Returns:
        DataFrame with game data
    """
    db = SessionLocal()
    
    try:
        query = db.query(Game).filter(
            Game.sport == sport,
            Game.home_score.isnot(None),
            Game.away_score.isnot(None)
        )
        
        # Handle status filter (case-insensitive, handle ESPN format)
        if status:
            # Normalize status for comparison
            status_normalized = status.lower()
            if status_normalized == "final":
                # Match both "final" and "STATUS_FINAL" formats
                query = query.filter(
                    or_(
                        Game.status.ilike("%final%"),
                        Game.status == "final"
                    )
                )
            else:
                query = query.filter(Game.status.ilike(f"%{status_normalized}%"))
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(Game.date >= start_dt)
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            query = query.filter(Game.date <= end_dt)
        
        games = query.order_by(Game.date).all()
        
        if not games:
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for game in games:
            records.append({
                "game_id": game.id,
                "sport": game.sport,
                "league": game.league,
                "date": game.date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "home_moneyline": game.home_moneyline,
                "away_moneyline": game.away_moneyline,
                "spread": game.spread,
                "over_under": game.over_under,
                "espn_data": game.espn_data,
            })
        
        df = pd.DataFrame(records)
        
        # Filter teams with minimum games
        if min_games_per_team > 0:
            team_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
            valid_teams = set(team_counts[team_counts >= min_games_per_team].index)
            df = df[df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)]
        
        return df
    
    finally:
        db.close()


def split_by_season(
    df: pd.DataFrame,
    validation_seasons: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by season, holding out the last N seasons for validation.
    
    Args:
        df: DataFrame with game data (must have 'date' column)
        validation_seasons: Number of seasons to hold out
    
    Returns:
        Tuple of (train_df, val_df)
    """
    if df.empty:
        return df, df
    
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    
    # For NFL: season year is the year the season starts (e.g., 2023 season = 2023)
    # For NHL: season spans two years (e.g., 2023-24 season = 2023)
    # We'll use the year of the date as the season identifier
    years = sorted(df["year"].unique())
    
    if len(years) <= validation_seasons:
        # Not enough seasons, use 80/20 split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
    else:
        val_years = set(years[-validation_seasons:])
        train_df = df[~df["year"].isin(val_years)].copy()
        val_df = df[df["year"].isin(val_years)].copy()
    
    return train_df.drop(columns=["year"]), val_df.drop(columns=["year"])


def split_by_week(
    df: pd.DataFrame,
    validation_weeks: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by week, holding out the last N weeks for validation.
    
    Args:
        df: DataFrame with game data (must have 'date' column)
        validation_weeks: Number of weeks to hold out
    
    Returns:
        Tuple of (train_df, val_df)
    """
    if df.empty:
        return df, df
    
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")
    
    # Calculate weeks from start
    start_date = df["date_dt"].min()
    df["weeks_from_start"] = ((df["date_dt"] - start_date).dt.days // 7)
    
    max_week = df["weeks_from_start"].max()
    split_week = max_week - validation_weeks
    
    train_df = df[df["weeks_from_start"] <= split_week].copy()
    val_df = df[df["weeks_from_start"] > split_week].copy()
    
    return train_df.drop(columns=["weeks_from_start", "date_dt"]), val_df.drop(columns=["weeks_from_start", "date_dt"])


def split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split of data by date (no data leakage).
    Ensures training data ends before test data starts.
    
    Args:
        df: DataFrame with game data (must have 'date' column)
        test_size: Proportion of data for validation (default 0.2 = 20%)
    
    Returns:
        Tuple of (train_df, val_df) sorted by date
    """
    if df.empty:
        return df, df
    
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)
    
    # Calculate split index (80/20 by default)
    split_idx = int(len(df) * (1 - test_size))
    
    # Ensure we split at a date boundary (no overlap)
    split_date = df.iloc[split_idx]["date_dt"]
    
    train_df = df[df["date_dt"] < split_date].copy()
    val_df = df[df["date_dt"] >= split_date].copy()
    
    return train_df.drop(columns=["date_dt"]), val_df.drop(columns=["date_dt"])


def split_random(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random split of data.
    WARNING: This can cause data leakage. Use split_temporal() for production.
    
    Args:
        df: DataFrame with game data
        test_size: Proportion of data for validation
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, val_df)
    """
    if df.empty:
        return df, df
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df


def get_season_date_range(sport: str, season_year: int) -> Tuple[str, str]:
    """
    Get start and end dates for a season.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        season_year: Year of the season (year season starts)
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    if sport == "NFL":
        # NFL season: September to February (next year)
        start_date = f"{season_year}-09-01"
        end_date = f"{season_year + 1}-02-28"
    elif sport == "NHL":
        # NHL season: October to June (next year)
        start_date = f"{season_year}-10-01"
        end_date = f"{season_year + 1}-06-30"
    elif sport == "NBA":
        # NBA season: October to June (next year)
        start_date = f"{season_year}-10-01"
        end_date = f"{season_year + 1}-06-30"
    elif sport == "MLB":
        # MLB season: March to November (same year)
        start_date = f"{season_year}-03-01"
        end_date = f"{season_year}-11-30"
    else:
        # Default: calendar year
        start_date = f"{season_year}-01-01"
        end_date = f"{season_year}-12-31"
    
    return start_date, end_date
