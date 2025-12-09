"""
Over/Under (O/U) data extraction and calculation utilities.
Extracts closing totals from ESPN, calculates actuals from final scores,
and determines OVER/UNDER/PUSH results for model training.
"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class OUCalculator:
    """Extract and calculate Over/Under data from ESPN responses"""
    
    @staticmethod
    def extract_closing_total(game_data: Dict) -> Optional[float]:
        """
        Extract closing O/U total from ESPN game response.
        
        Args:
            game_data: ESPN event dictionary
            
        Returns:
            Closing total (float) or None if not available
        """
        try:
            competitions = game_data.get('competitions', [])
            if not competitions:
                return None
            
            odds_list = competitions[0].get('odds', [])
            if not odds_list:
                return None
            
            closing_total = odds_list[0].get('overUnder')
            if closing_total is None:
                return None
            
            # Convert to float, handle string values
            try:
                return float(closing_total)
            except (ValueError, TypeError):
                return None
                
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.debug(f"Error extracting closing total: {e}")
            return None
    
    @staticmethod
    def calculate_actual_total(game_data: Dict) -> Optional[int]:
        """
        Calculate actual total points scored in game.
        
        Args:
            game_data: ESPN event dictionary
            
        Returns:
            Actual total (int) or None if scores not available
        """
        try:
            competitions = game_data.get('competitions', [])
            if not competitions:
                return None
            
            competitors = competitions[0].get('competitors', [])
            if len(competitors) < 2:
                return None
            
            scores = []
            for competitor in competitors[:2]:
                score = competitor.get('score')
                if score is None:
                    return None
                try:
                    scores.append(int(score))
                except (ValueError, TypeError):
                    return None
            
            return sum(scores)
            
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.debug(f"Error calculating actual total: {e}")
            return None
    
    @staticmethod
    def determine_ou_result(
        actual_total: Optional[int],
        closing_total: Optional[float]
    ) -> Optional[str]:
        """
        Determine if game went OVER, UNDER, or PUSH.
        
        Args:
            actual_total: Total points scored in game
            closing_total: Closing O/U line
            
        Returns:
            "OVER", "UNDER", "PUSH", or None if data incomplete
        """
        if actual_total is None or closing_total is None:
            return None
        
        if actual_total > closing_total:
            return "OVER"
        elif actual_total < closing_total:
            return "UNDER"
        else:
            return "PUSH"
    
    @staticmethod
    def process_game_ou_data(game_data: Dict) -> Dict:
        """
        Complete O/U processing for a single game.
        
        Args:
            game_data: ESPN event dictionary
            
        Returns:
            Dictionary with closing_total, actual_total, and ou_result
        """
        closing_total = OUCalculator.extract_closing_total(game_data)
        actual_total = OUCalculator.calculate_actual_total(game_data)
        ou_result = OUCalculator.determine_ou_result(actual_total, closing_total)
        
        return {
            'closing_total': closing_total,
            'actual_total': actual_total,
            'ou_result': ou_result
        }
    
    @staticmethod
    def process_game_from_scores(
        home_score: Optional[int],
        away_score: Optional[int],
        over_under: Optional[float]
    ) -> Dict:
        """
        Process O/U data from existing game scores and over_under value.
        Useful for backfilling existing games.
        
        Args:
            home_score: Home team score
            away_score: Away team score
            over_under: Over/under line value
            
        Returns:
            Dictionary with closing_total, actual_total, and ou_result
        """
        if home_score is None or away_score is None:
            actual_total = None
        else:
            actual_total = home_score + away_score
        
        closing_total = over_under
        ou_result = OUCalculator.determine_ou_result(actual_total, closing_total)
        
        return {
            'closing_total': closing_total,
            'actual_total': actual_total,
            'ou_result': ou_result
        }
