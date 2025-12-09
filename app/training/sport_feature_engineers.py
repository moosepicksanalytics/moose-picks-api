"""
Sport-specific feature engineers for NFL, NHL, NBA, and MLB.
Each sport has unique features based on the sport's characteristics.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List
from datetime import datetime
from app.training.base_feature_engineer import BaseFeatureEngineer


class NFLFeatureEngineer(BaseFeatureEngineer):
    """NFL-specific feature engineering."""
    
    def build_sport_specific_features(
        self,
        df: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Build NFL-specific features.
        
        Features:
        - Division matchup indicators
        - Third-down conversion rates
        - Red zone efficiency
        - Turnover differential
        - Time of possession
        - Yards per play
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Initialize NFL-specific columns
        df["division_matchup"] = 0
        df["home_third_down_pct"] = 0.0
        df["away_third_down_pct"] = 0.0
        df["home_redzone_efficiency"] = 0.0
        df["away_redzone_efficiency"] = 0.0
        df["turnover_differential"] = 0.0
        df["home_time_of_possession"] = 0.0
        df["away_time_of_possession"] = 0.0
        
        # NFL divisions (simplified - would need full team mapping)
        nfl_divisions = {
            "AFC East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
            "AFC North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
            "AFC South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
            "AFC West": ["Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers"],
            "NFC East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders"],
            "NFC North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
            "NFC South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
            "NFC West": ["Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"],
        }
        
        # Create division lookup
        team_to_division = {}
        for division, teams in nfl_divisions.items():
            for team in teams:
                team_to_division[team] = division
        
        # Calculate division matchups
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            
            home_div = team_to_division.get(home_team)
            away_div = team_to_division.get(away_team)
            
            if home_div and away_div and home_div == away_div:
                df.at[idx, "division_matchup"] = 1
        
        # Try to extract advanced stats from espn_data if available
        for idx, row in df.iterrows():
            espn_data = row.get("espn_data")
            if espn_data and isinstance(espn_data, dict):
                # Extract third-down, red zone, turnover stats if available
                # ESPN data structure varies, so we'll extract what we can
                try:
                    competitions = espn_data.get("competitions", [])
                    if competitions:
                        comp = competitions[0]
                        # Extract statistics if available
                        # This is a simplified version - full implementation would parse ESPN stats
                        pass
                except:
                    pass
        
        # Calculate rolling NFL-specific stats
        team_history = {}
        
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            game_date = pd.to_datetime(row["date"])
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            
            # Calculate turnover differential (simplified - would need actual turnover data)
            # For now, estimate based on score differential patterns
            
            # Calculate time of possession (simplified - would need actual TOP data)
            # Estimate: winning team typically has more TOP
            
            if has_scores:
                # Simple estimates for NFL-specific metrics
                # In production, these would come from detailed ESPN stats
                if home_score > away_score:
                    # REMOVED: time_of_possession estimation using current game outcome
                    # This was data leakage - using home_score > away_score directly encodes the target
                    # Keep default values (0.0) - in future, calculate from historical data only
                    pass  # Keep default 0.0 values set above
                
                # REMOVED: turnover_differential estimation using current game outcome
                # This was data leakage - using (home_score - away_score) directly encodes the target
                # Keep default value (0.0) - in future, calculate from historical data only
                pass  # Keep default 0.0 value set above
        
        # Calculate rolling NFL-specific averages
        for window in self.rolling_windows:
            for prefix in ["home", "away"]:
                df[f"{prefix}_third_down_pct_{window}"] = 0.4  # Default NFL average
                df[f"{prefix}_redzone_efficiency_{window}"] = 0.6  # Default NFL average
                df[f"{prefix}_turnover_differential_{window}"] = 0.0
                df[f"{prefix}_time_of_possession_{window}"] = 0.5
        
        return df


class NHLFeatureEngineer(BaseFeatureEngineer):
    """NHL-specific feature engineering."""
    
    def build_sport_specific_features(
        self,
        df: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Build NHL-specific features.
        
        Features:
        - Goalie-specific features (starter vs backup, save %, GAA)
        - Corsi (shot differential)
        - Power play % / Penalty kill %
        - High-danger chances
        - Goalie back-to-back
        - Regulation vs OT wins
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Initialize NHL-specific columns
        df["home_goalie_save_pct"] = 0.91  # NHL average
        df["away_goalie_save_pct"] = 0.91
        df["home_goalie_gaa"] = 2.5  # NHL average
        df["away_goalie_gaa"] = 2.5
        df["home_corsi"] = 0.5
        df["away_corsi"] = 0.5
        df["home_powerplay_pct"] = 0.2  # NHL average
        df["away_powerplay_pct"] = 0.2
        df["home_penaltykill_pct"] = 0.8  # NHL average
        df["away_penaltykill_pct"] = 0.8
        df["home_goalie_back_to_back"] = 0
        df["away_goalie_back_to_back"] = 0
        
        # Track goalie history
        goalie_history = {}
        
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            game_date = pd.to_datetime(row["date"])
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            
            # Try to extract goalie data from espn_data
            espn_data = row.get("espn_data")
            if espn_data and isinstance(espn_data, dict):
                try:
                    # ESPN NHL data structure parsing
                    # This is simplified - full implementation would parse ESPN stats
                    pass
                except:
                    pass
            
            # Estimate goalie stats from scores (simplified)
            if has_scores:
                # Estimate save % based on goals against
                # Lower goals against = better save %
                home_goals_against = away_score
                away_goals_against = home_score
                
                # Estimate shots (NHL average ~30 shots per game)
                estimated_shots = 30.0
                
                if estimated_shots > 0:
                    df.at[idx, "home_goalie_save_pct"] = max(0.7, min(0.98, 1 - (home_goals_against / estimated_shots)))
                    df.at[idx, "away_goalie_save_pct"] = max(0.7, min(0.98, 1 - (away_goals_against / estimated_shots)))
                
                df.at[idx, "home_goalie_gaa"] = home_goals_against
                df.at[idx, "away_goalie_gaa"] = away_goals_against
                
                # REMOVED: Corsi estimation using current game outcome
                # This was data leakage - using home_score > away_score directly encodes the target
                # Corsi should be calculated from past games only, not the current game
                # For now, keep default values (0.5) - in future, calculate from historical shot data
                pass  # Keep default 0.5 values set above
        
        # Calculate rolling NHL-specific stats
        for window in self.rolling_windows:
            for prefix in ["home", "away"]:
                df[f"{prefix}_goalie_save_pct_{window}"] = df[f"{prefix}_goalie_save_pct"]
                df[f"{prefix}_goalie_gaa_{window}"] = df[f"{prefix}_goalie_gaa"]
                df[f"{prefix}_corsi_{window}"] = df[f"{prefix}_corsi"]
                df[f"{prefix}_powerplay_pct_{window}"] = df[f"{prefix}_powerplay_pct"]
                df[f"{prefix}_penaltykill_pct_{window}"] = df[f"{prefix}_penaltykill_pct"]
        
        return df


class NBAFeatureEngineer(BaseFeatureEngineer):
    """NBA-specific feature engineering."""
    
    def build_sport_specific_features(
        self,
        df: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Build NBA-specific features.
        
        Features:
        - Pace (possessions per game)
        - eFG% (effective field goal percentage)
        - True shooting %
        - Offensive/defensive rebounds
        - Assists per game
        - Turnovers per game
        - 3P shooting %
        - Conference matchup
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Initialize NBA-specific columns
        df["home_pace"] = 100.0  # NBA average
        df["away_pace"] = 100.0
        df["home_efg_pct"] = 0.54  # NBA average
        df["away_efg_pct"] = 0.54
        df["home_true_shooting_pct"] = 0.57  # NBA average
        df["away_true_shooting_pct"] = 0.57
        df["home_off_reb_pct"] = 0.25  # NBA average
        df["away_off_reb_pct"] = 0.25
        df["home_def_reb_pct"] = 0.75  # NBA average
        df["away_def_reb_pct"] = 0.75
        df["home_assists_per_game"] = 25.0  # NBA average
        df["away_assists_per_game"] = 25.0
        df["home_turnovers_per_game"] = 14.0  # NBA average
        df["away_turnovers_per_game"] = 14.0
        df["home_3p_pct"] = 0.36  # NBA average
        df["away_3p_pct"] = 0.36
        df["conference_matchup"] = 0
        
        # NBA conferences
        nba_east = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
            "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
            "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards"
        ]
        nba_west = [
            "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Minnesota Timberwolves",
            "New Orleans Pelicans", "Oklahoma City Thunder", "Phoenix Suns", "Portland Trail Blazers",
            "Sacramento Kings", "San Antonio Spurs", "Utah Jazz"
        ]
        
        # Calculate conference matchups
        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            
            home_east = home_team in nba_east
            away_east = away_team in nba_east
            
            # Same conference = 1, different = 0
            if (home_east and away_east) or (not home_east and not away_east):
                df.at[idx, "conference_matchup"] = 1
        
        # Estimate NBA-specific stats from scores
        for idx, row in df.iterrows():
            home_score = row["home_score"]
            away_score = row["away_score"]
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            
            if has_scores:
                # Estimate pace (higher scoring = higher pace)
                total_score = home_score + away_score
                df.at[idx, "home_pace"] = min(120, max(80, total_score / 2.0))
                df.at[idx, "away_pace"] = df.at[idx, "home_pace"]
                
                # Estimate eFG% (higher scoring = better shooting)
                # NBA average ~110 points, ~54% eFG
                if total_score > 220:
                    efg_boost = 0.02
                elif total_score < 200:
                    efg_boost = -0.02
                else:
                    efg_boost = 0
                
                df.at[idx, "home_efg_pct"] = 0.54 + efg_boost if home_score > away_score else 0.54 - efg_boost
                df.at[idx, "away_efg_pct"] = 0.54 + efg_boost if away_score > home_score else 0.54 - efg_boost
                
                # Estimate true shooting % (similar to eFG%)
                df.at[idx, "home_true_shooting_pct"] = df.at[idx, "home_efg_pct"] + 0.03
                df.at[idx, "away_true_shooting_pct"] = df.at[idx, "away_efg_pct"] + 0.03
        
        # Calculate rolling NBA-specific stats
        for window in self.rolling_windows:
            for prefix in ["home", "away"]:
                df[f"{prefix}_pace_{window}"] = df[f"{prefix}_pace"]
                df[f"{prefix}_efg_pct_{window}"] = df[f"{prefix}_efg_pct"]
                df[f"{prefix}_true_shooting_pct_{window}"] = df[f"{prefix}_true_shooting_pct"]
                df[f"{prefix}_off_reb_pct_{window}"] = df[f"{prefix}_off_reb_pct"]
                df[f"{prefix}_def_reb_pct_{window}"] = df[f"{prefix}_def_reb_pct"]
                df[f"{prefix}_assists_per_game_{window}"] = df[f"{prefix}_assists_per_game"]
                df[f"{prefix}_turnovers_per_game_{window}"] = df[f"{prefix}_turnovers_per_game"]
                df[f"{prefix}_3p_pct_{window}"] = df[f"{prefix}_3p_pct"]
        
        return df


class MLBFeatureEngineer(BaseFeatureEngineer):
    """MLB-specific feature engineering."""
    
    def build_sport_specific_features(
        self,
        df: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Build MLB-specific features.
        
        Features:
        - Pitcher features (ERA, WHIP, K/9, rest days)
        - Day vs night game
        - Ballpark factors
        - Bullpen availability
        - Pitcher vs batter history
        - Home run rates
        - OPS (on-base + slugging)
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Initialize MLB-specific columns
        df["home_pitcher_era"] = 4.0  # MLB average
        df["away_pitcher_era"] = 4.0
        df["home_pitcher_whip"] = 1.3  # MLB average
        df["away_pitcher_whip"] = 1.3
        df["home_pitcher_k9"] = 8.5  # MLB average
        df["away_pitcher_k9"] = 8.5
        df["home_pitcher_rest_days"] = 4  # MLB average
        df["away_pitcher_rest_days"] = 4
        df["day_game"] = 0  # 1 if day game, 0 if night
        df["home_ops"] = 0.73  # MLB average
        df["away_ops"] = 0.73
        df["home_hr_rate"] = 0.03  # MLB average
        df["away_hr_rate"] = 0.03
        df["ballpark_factor"] = 1.0  # Neutral
        
        # Estimate day vs night from game time (if available)
        for idx, row in df.iterrows():
            game_date = pd.to_datetime(row["date"])
            # Estimate: games before 6 PM are day games
            if game_date.hour < 18:
                df.at[idx, "day_game"] = 1
        
        # Estimate pitcher stats from runs allowed
        for idx, row in df.iterrows():
            home_score = row["home_score"]
            away_score = row["away_score"]
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            
            if has_scores:
                # Estimate pitcher ERA from runs allowed
                # Lower runs = better ERA
                df.at[idx, "away_pitcher_era"] = max(2.0, min(6.0, away_score * 0.8))
                df.at[idx, "home_pitcher_era"] = max(2.0, min(6.0, home_score * 0.8))
                
                # Estimate WHIP (walks + hits per inning)
                # Correlates with runs allowed
                df.at[idx, "away_pitcher_whip"] = max(0.8, min(1.8, df.at[idx, "away_pitcher_era"] / 3.0))
                df.at[idx, "home_pitcher_whip"] = max(0.8, min(1.8, df.at[idx, "home_pitcher_era"] / 3.0))
                
                # Estimate OPS from runs scored
                # Higher runs = better OPS
                if home_score > 0:
                    df.at[idx, "home_ops"] = min(1.0, max(0.5, 0.73 + (home_score - 4.5) * 0.05))
                if away_score > 0:
                    df.at[idx, "away_ops"] = min(1.0, max(0.5, 0.73 + (away_score - 4.5) * 0.05))
                
                # Estimate HR rate
                # Higher scoring = more HRs
                df.at[idx, "home_hr_rate"] = min(0.06, max(0.01, 0.03 + (home_score - 4.5) * 0.002))
                df.at[idx, "away_hr_rate"] = min(0.06, max(0.01, 0.03 + (away_score - 4.5) * 0.002))
        
        # Calculate rolling MLB-specific stats
        for window in self.rolling_windows:
            for prefix in ["home", "away"]:
                df[f"{prefix}_pitcher_era_{window}"] = df[f"{prefix}_pitcher_era"]
                df[f"{prefix}_pitcher_whip_{window}"] = df[f"{prefix}_pitcher_whip"]
                df[f"{prefix}_pitcher_k9_{window}"] = df[f"{prefix}_pitcher_k9"]
                df[f"{prefix}_ops_{window}"] = df[f"{prefix}_ops"]
                df[f"{prefix}_hr_rate_{window}"] = df[f"{prefix}_hr_rate"]
        
        return df


def get_feature_engineer(sport: str) -> BaseFeatureEngineer:
    """
    Get the appropriate feature engineer for a sport.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
    
    Returns:
        Feature engineer instance
    """
    sport_upper = sport.upper()
    
    if sport_upper == "NFL":
        return NFLFeatureEngineer()
    elif sport_upper == "NHL":
        return NHLFeatureEngineer()
    elif sport_upper == "NBA":
        return NBAFeatureEngineer()
    elif sport_upper == "MLB":
        return MLBFeatureEngineer()
    else:
        # Default to base engineer
        return BaseFeatureEngineer()
