"""
Base feature engineering class with shared utilities for all sports.
Provides rolling windows, recency weighting, and temporal validation.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import timedelta


class BaseFeatureEngineer(ABC):
    """
    Base class for sport-specific feature engineering.
    Provides shared utilities for rolling stats, recency weighting, and temporal validation.
    """
    
    def __init__(
        self,
        rolling_windows: List[int] = [3, 5, 10, 15],
        recency_decay: float = 0.9977,
        include_rest_days: bool = True,
        include_h2h: bool = True
    ):
        """
        Initialize base feature engineer.
        
        Args:
            rolling_windows: List of window sizes for rolling statistics
            recency_decay: Exponential decay factor per day (default 0.9977)
            include_rest_days: Whether to include rest days features
            include_h2h: Whether to include head-to-head features
        """
        self.rolling_windows = rolling_windows
        self.recency_decay = recency_decay
        self.include_rest_days = include_rest_days
        self.include_h2h = include_h2h
    
    def calculate_recency_weight(self, days_ago: int) -> float:
        """
        Calculate recency weight using exponential decay.
        
        Args:
            days_ago: Number of days ago the game occurred
        
        Returns:
            Weight (0-1) with more recent games weighted higher
        """
        return self.recency_decay ** days_ago
    
    def calculate_weighted_rolling_stats(
        self,
        df: pd.DataFrame,
        team_history: Dict,
        team: str,
        game_date: pd.Timestamp,
        prefix: str,
        window: int
    ) -> Dict[str, float]:
        """
        Calculate recency-weighted rolling statistics.
        
        Args:
            df: DataFrame with game data
            team_history: Dictionary tracking team game history
            team: Team name
            game_date: Current game date
            prefix: Feature prefix ("home" or "away")
            window: Rolling window size
        
        Returns:
            Dictionary of weighted statistics
        """
        if team not in team_history:
            return {}
        
        # Get recent games before this game
        recent_games = [
            g for g in team_history[team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_games = sorted(recent_games, key=lambda x: x["date"], reverse=True)[:window]
        
        if not recent_games:
            return {}
        
        # Calculate weights
        weights = []
        for g in recent_games:
            days_ago = (game_date - g["date"]).days
            weights.append(self.calculate_recency_weight(days_ago))
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted statistics
        points_for = np.array([g["points_for"] for g in recent_games])
        points_against = np.array([g["points_against"] for g in recent_games])
        wins = np.array([1 if g["won"] else 0 for g in recent_games])
        
        stats = {
            f"{prefix}_weighted_win_rate_{window}": np.dot(wins, weights),
            f"{prefix}_weighted_points_for_{window}": np.dot(points_for, weights),
            f"{prefix}_weighted_points_against_{window}": np.dot(points_against, weights),
            f"{prefix}_weighted_point_diff_{window}": np.dot(points_for - points_against, weights),
        }
        
        return stats
    
    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        team_history: Dict,
        team: str,
        game_date: pd.Timestamp,
        prefix: str,
        window: int
    ) -> Dict[str, float]:
        """
        Calculate standard (unweighted) rolling statistics.
        
        Args:
            df: DataFrame with game data
            team_history: Dictionary tracking team game history
            team: Team name
            game_date: Current game date
            prefix: Feature prefix ("home" or "away")
            window: Rolling window size
        
        Returns:
            Dictionary of rolling statistics
        """
        if team not in team_history:
            return {}
        
        recent_games = [
            g for g in team_history[team]
            if g["date"] < game_date and g.get("has_score", False)
        ]
        recent_games = sorted(recent_games, key=lambda x: x["date"], reverse=True)[:window]
        
        if not recent_games:
            return {}
        
        points_for = [g["points_for"] for g in recent_games]
        points_against = [g["points_against"] for g in recent_games]
        wins = sum(1 for g in recent_games if g["won"])
        
        stats = {
            f"{prefix}_win_rate_{window}": wins / len(recent_games),
            f"{prefix}_points_for_avg_{window}": np.mean(points_for),
            f"{prefix}_points_against_avg_{window}": np.mean(points_against),
            f"{prefix}_point_diff_avg_{window}": np.mean([pf - pa for pf, pa in zip(points_for, points_against)]),
        }
        
        # Momentum: recent vs older performance
        if len(recent_games) >= window * 2:
            recent_half = recent_games[:window//2] if window >= 4 else recent_games
            older_half = [
                g for g in team_history[team]
                if g["date"] < game_date and g.get("has_score", False)
            ][window:window+window//2] if window >= 4 else []
            
            if older_half:
                recent_avg = np.mean([g["points_for"] - g["points_against"] for g in recent_half])
                older_avg = np.mean([g["points_for"] - g["points_against"] for g in older_half])
                stats[f"{prefix}_momentum_{window}"] = recent_avg - older_avg
        
        return stats
    
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rest days between games for each team.
        
        Args:
            df: DataFrame with game data (must be sorted by date)
        
        Returns:
            DataFrame with rest_days_home, rest_days_away, rest_advantage, and back_to_back columns
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        df["rest_days_home"] = np.nan
        df["rest_days_away"] = np.nan
        df["rest_advantage"] = 0.0
        df["home_back_to_back"] = 0
        df["away_back_to_back"] = 0
        
        last_game_home = {}
        last_game_away = {}
        
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            game_date = pd.to_datetime(row["date"])
            
            # Home team rest days
            if home_team in last_game_home:
                last_date = last_game_home[home_team]
                rest_days = (game_date - last_date).days - 1
                df.at[idx, "rest_days_home"] = max(0, rest_days) if rest_days >= 0 else np.nan
                if rest_days <= 1:
                    df.at[idx, "home_back_to_back"] = 1
            last_game_home[home_team] = game_date
            
            # Away team rest days
            if away_team in last_game_away:
                last_date = last_game_away[away_team]
                rest_days = (game_date - last_date).days - 1
                df.at[idx, "rest_days_away"] = max(0, rest_days) if rest_days >= 0 else np.nan
                if rest_days <= 1:
                    df.at[idx, "away_back_to_back"] = 1
            last_game_away[away_team] = game_date
            
            # Rest advantage
            if pd.notna(df.at[idx, "rest_days_home"]) and pd.notna(df.at[idx, "rest_days_away"]):
                df.at[idx, "rest_advantage"] = df.at[idx, "rest_days_home"] - df.at[idx, "rest_days_away"]
        
        return df
    
    def calculate_home_away_splits(
        self,
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
        
        df["home_win_rate_home"] = np.nan
        df["home_win_rate_away"] = np.nan
        df["away_win_rate_home"] = np.nan
        df["away_win_rate_away"] = np.nan
        df["home_field_advantage"] = 0.0
        
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
                df.at[idx, "home_win_rate_home"] = wins / len(recent_home)
            
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
                df.at[idx, "home_win_rate_away"] = wins / len(recent_away)
            
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
                df.at[idx, "away_win_rate_home"] = wins / len(recent_home)
            
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
                df.at[idx, "away_win_rate_away"] = wins / len(recent_away)
            
            if has_scores:
                team_away_history[away_team].append({
                    "date": game_date,
                    "won": away_score > home_score,
                    "has_score": True
                })
            
            # Home field advantage
            if pd.notna(df.at[idx, "home_win_rate_home"]) and pd.notna(df.at[idx, "home_win_rate_away"]):
                df.at[idx, "home_field_advantage"] = (
                    df.at[idx, "home_win_rate_home"] - df.at[idx, "home_win_rate_away"]
                )
        
        return df
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head statistics between teams.
        
        Args:
            df: DataFrame with game data (must be sorted by date)
        
        Returns:
            DataFrame with head-to-head statistics
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        df["h2h_home_wins"] = 0
        df["h2h_total_games"] = 0
        df["h2h_home_win_rate"] = np.nan
        df["h2h_avg_margin"] = 0.0
        
        h2h_history = {}
        
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            game_date = pd.to_datetime(row["date"])
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            
            match_key = tuple(sorted([home_team, away_team]))
            
            if match_key not in h2h_history:
                h2h_history[match_key] = []
            
            prev_games = [
                g for g in h2h_history[match_key]
                if g["date"] < game_date and g.get("has_score", False)
            ]
            
            if prev_games:
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
    
    @abstractmethod
    def build_sport_specific_features(
        self,
        df: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Build sport-specific features. Must be implemented by subclasses.
        
        Args:
            df: DataFrame with game data
            market: Market type (moneyline, spread, totals, score_projection)
        
        Returns:
            DataFrame with sport-specific features added
        """
        pass
    
    def calculate_multi_window_rolling_stats(
        self,
        df: pd.DataFrame,
        sport: str
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for multiple time windows.
        Uses the base implementation from features.py.
        
        Args:
            df: DataFrame with game data
            sport: Sport code
        
        Returns:
            DataFrame with rolling statistics
        """
        # Import here to avoid circular dependency
        from app.training.features import calculate_multi_window_rolling_stats as calc_rolling
        
        return calc_rolling(df, sport, self.rolling_windows)
    
    def build_features(
        self,
        df: pd.DataFrame,
        sport: str,
        market: str
    ) -> pd.DataFrame:
        """
        Build all features (base + sport-specific).
        
        Args:
            df: DataFrame with game data
            sport: Sport code
            market: Market type
        
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            return df
        
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
        df = self.calculate_multi_window_rolling_stats(df, sport)
        
        # Calculate rest days
        if self.include_rest_days:
            print("Calculating rest days and rest advantage...")
            df = self.calculate_rest_days(df)
            df["rest_days_home"] = df["rest_days_home"].fillna(
                df["rest_days_home"].median() if not df["rest_days_home"].isna().all() else 3
            )
            df["rest_days_away"] = df["rest_days_away"].fillna(
                df["rest_days_away"].median() if not df["rest_days_away"].isna().all() else 3
            )
        
        # Calculate head-to-head
        if self.include_h2h:
            print("Calculating head-to-head records...")
            df = self.calculate_head_to_head(df)
            df["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.5)
        
        # Calculate home/away splits
        print("Calculating home/away splits...")
        df = self.calculate_home_away_splits(df, rolling_window=max(self.rolling_windows))
        
        # Build sport-specific features
        print(f"Calculating {sport}-specific features...")
        df = self.build_sport_specific_features(df, market)
        
        # Fill remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
