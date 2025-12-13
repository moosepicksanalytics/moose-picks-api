"""
Diagnostic script to analyze model accuracy by comparing predictions to actual game results.

This script:
1. Loads recent predictions from CSV exports or database
2. Queries game results from database
3. Matches predictions to actual results
4. Calculates win rates, calibration metrics, and edge statistics
5. Outputs console report, CSV, and JSON summary
"""
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Game, Prediction
from app.training.evaluate import calculate_ece
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions_from_csv(sport: str) -> pd.DataFrame:
    """Load predictions from CSV export files."""
    exports_dir = Path("exports")
    if not exports_dir.exists():
        raise FileNotFoundError(f"Exports directory not found")
    
    # Find latest CSV for the sport
    csv_files = sorted(
        exports_dir.glob(f"{sport}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {sport}")
    
    latest_csv = csv_files[0]
    logger.info(f"Loading predictions from {latest_csv.name}")
    
    df = pd.read_csv(latest_csv)
    
    # Standardize column names
    if 'market_type' in df.columns:
        df = df.rename(columns={'market_type': 'market'})
    
    # Convert game_id to string (database stores as VARCHAR)
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str)
    
    return df


def load_predictions_from_db(sport: str) -> pd.DataFrame:
    """Load predictions from database."""
    db = SessionLocal()
    try:
        # Get recent predictions (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport,
            Prediction.predicted_at >= cutoff_date
        ).all()
        
        if not predictions:
            logger.warning(f"No predictions found in database for {sport}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for pred in predictions:
            record = {
                'game_id': pred.game_id,
                'sport': pred.sport,
                'market': pred.market,
                'home_win_prob': pred.home_win_prob,
                'spread_cover_prob': pred.spread_cover_prob,
                'over_prob': pred.over_prob,
                'predicted_at': pred.predicted_at,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    finally:
        db.close()


def get_game_results(sport: str, game_ids: list) -> pd.DataFrame:
    """Query game results from database."""
    db = SessionLocal()
    try:
        # Convert game_ids to strings (database stores id as VARCHAR)
        game_ids_str = [str(gid) for gid in game_ids]
        
        games = db.query(Game).filter(
            Game.sport == sport,
            Game.id.in_(game_ids_str),
            Game.status == 'final',
            Game.home_score.isnot(None),
            Game.away_score.isnot(None)
        ).all()
        
        records = []
        for game in games:
            record = {
                'game_id': game.id,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'home_score': game.home_score,
                'away_score': game.away_score,
                'spread': game.spread,
                'over_under': game.over_under,
                'date': game.date,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    finally:
        db.close()


def determine_prediction_result(row: pd.Series) -> tuple:
    """
    Determine if a prediction was correct.
    
    Returns:
        (is_correct: bool, actual_outcome: str)
    """
    market = row['market']
    
    if market == 'moneyline':
        # Check if predicted team won
        if pd.isna(row.get('home_win_prob')):
            return None, None
        
        predicted_home_win = row['home_win_prob'] > 0.5
        actual_home_win = row['home_score'] > row['away_score']
        
        is_correct = (predicted_home_win == actual_home_win)
        outcome = 'home_win' if actual_home_win else 'away_win'
        
        return is_correct, outcome
    
    elif market == 'spread':
        if pd.isna(row.get('spread_cover_prob')) or pd.isna(row.get('spread')):
            return None, None
        
        # Calculate actual margin
        actual_margin = row['home_score'] - row['away_score']
        spread = row['spread']
        
        # Determine if home covered
        if spread < 0:
            # Home is favorite (spread is negative, e.g., -3.5)
            home_covers = actual_margin > abs(spread)
        else:
            # Home is underdog (spread is positive, e.g., +3.5)
            home_covers = actual_margin > -spread
        
        predicted_home_covers = row['spread_cover_prob'] > 0.5
        
        is_correct = (home_covers == predicted_home_covers)
        
        # Check for push
        if abs(actual_margin - spread) < 0.1:
            return None, 'push'
        
        outcome = 'home_cover' if home_covers else 'away_cover'
        return is_correct, outcome
    
    elif market in ['totals', 'over_under']:
        if pd.isna(row.get('over_prob')) or pd.isna(row.get('over_under')):
            return None, None
        
        total_score = row['home_score'] + row['away_score']
        predicted_over = row['over_prob'] > 0.5
        actual_over = total_score > row['over_under']
        
        # Check for push
        if abs(total_score - row['over_under']) < 0.1:
            return None, 'push'
        
        is_correct = (predicted_over == actual_over)
        outcome = 'over' if actual_over else 'under'
        
        return is_correct, outcome
    
    return None, None


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive accuracy metrics."""
    if len(df) == 0:
        return {}
    
    # Filter out pushes and invalid predictions
    valid_df = df[df['is_correct'].notna()].copy()
    
    if len(valid_df) == 0:
        return {'error': 'No valid predictions to analyze'}
    
    # Overall metrics
    total = len(valid_df)
    wins = valid_df['is_correct'].sum()
    losses = total - wins
    win_rate = wins / total if total > 0 else 0.0
    
    # By market
    market_metrics = {}
    for market in valid_df['market'].unique():
        market_df = valid_df[valid_df['market'] == market]
        market_total = len(market_df)
        market_wins = market_df['is_correct'].sum()
        market_win_rate = market_wins / market_total if market_total > 0 else 0.0
        
        market_metrics[market] = {
            'total': int(market_total),
            'wins': int(market_wins),
            'losses': int(market_total - market_wins),
            'win_rate': float(market_win_rate)
        }
    
    # Confidence metrics
    confidence_col = None
    if 'model_prob' in valid_df.columns:
        confidence_col = 'model_prob'
    elif 'home_win_prob' in valid_df.columns:
        confidence_col = 'home_win_prob'
    elif 'spread_cover_prob' in valid_df.columns:
        confidence_col = 'spread_cover_prob'
    elif 'over_prob' in valid_df.columns:
        confidence_col = 'over_prob'
    
    confidence_metrics = {}
    if confidence_col:
        avg_confidence = valid_df[confidence_col].mean()
        
        # Calculate ECE
        # For moneyline, use home_win_prob vs actual home win
        # For spread, use spread_cover_prob vs actual cover
        # For totals, use over_prob vs actual over
        y_true = []
        y_pred_proba = []
        
        for _, row in valid_df.iterrows():
            if row['market'] == 'moneyline':
                actual_home_win = row['home_score'] > row['away_score']
                prob = row.get('home_win_prob', 0.5)
                y_true.append(1 if actual_home_win else 0)
                y_pred_proba.append(prob)
            elif row['market'] == 'spread':
                actual_margin = row['home_score'] - row['away_score']
                spread = row.get('spread', 0)
                if spread < 0:
                    home_covers = actual_margin > abs(spread)
                else:
                    home_covers = actual_margin > -spread
                prob = row.get('spread_cover_prob', 0.5)
                y_true.append(1 if home_covers else 0)
                y_pred_proba.append(prob)
            elif row['market'] in ['totals', 'over_under']:
                total_score = row['home_score'] + row['away_score']
                actual_over = total_score > row.get('over_under', 0)
                prob = row.get('over_prob', 0.5)
                y_true.append(1 if actual_over else 0)
                y_pred_proba.append(prob)
        
        if len(y_true) > 0:
            ece = calculate_ece(np.array(y_true), np.array(y_pred_proba), n_bins=10)
            confidence_metrics = {
                'average_confidence': float(avg_confidence),
                'ece': float(ece) if not np.isnan(ece) else None,
            }
    
    # Edge statistics
    edge_metrics = {}
    if 'edge' in valid_df.columns:
        edge_series = valid_df['edge'].dropna()
        if len(edge_series) > 0:
            edge_metrics = {
                'mean': float(edge_series.mean()),
                'median': float(edge_series.median()),
                'min': float(edge_series.min()),
                'max': float(edge_series.max()),
                'std': float(edge_series.std()),
            }
    
    return {
        'total_predictions': int(total),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'vs_random': float(win_rate - 0.5),  # How much better than 50%
        'sample_size_adequate': total >= 30,  # Minimum for statistical significance
        'by_market': market_metrics,
        'confidence': confidence_metrics,
        'edge': edge_metrics,
    }


def print_console_report(sport: str, metrics: dict, df: pd.DataFrame):
    """Print formatted console report."""
    print("\n" + "=" * 80)
    print(f"MODEL ACCURACY DIAGNOSTIC REPORT: {sport}")
    print("=" * 80)
    
    print(f"\n📊 OVERALL METRICS")
    print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")
    print(f"  Wins: {metrics.get('wins', 0)}")
    print(f"  Losses: {metrics.get('losses', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  vs Random (50%): {metrics.get('vs_random', 0):+.1%}")
    
    if metrics.get('sample_size_adequate'):
        print(f"  ✓ Sample size is adequate for analysis")
    else:
        print(f"  ⚠️  Sample size is small (< 30 predictions)")
    
    # By market
    if 'by_market' in metrics:
        print(f"\n📈 BY MARKET")
        for market, market_metrics in metrics['by_market'].items():
            print(f"  {market.upper()}:")
            print(f"    Win Rate: {market_metrics['win_rate']:.1%} ({market_metrics['wins']}/{market_metrics['total']})")
    
    # Confidence calibration
    if 'confidence' in metrics and metrics['confidence']:
        conf = metrics['confidence']
        print(f"\n🎯 CONFIDENCE CALIBRATION")
        print(f"  Average Confidence: {conf.get('average_confidence', 0):.1%}")
        if conf.get('ece') is not None:
            ece = conf['ece']
            print(f"  Expected Calibration Error (ECE): {ece:.4f}")
            if ece < 0.05:
                print(f"    ✓ Excellent calibration")
            elif ece < 0.10:
                print(f"    ⚠️  Good calibration (could be better)")
            else:
                print(f"    ✗ Poor calibration (needs recalibration)")
    
    # Edge statistics
    if 'edge' in metrics and metrics['edge']:
        edge = metrics['edge']
        print(f"\n💰 EDGE STATISTICS")
        print(f"  Mean Edge: {edge.get('mean', 0):.2%}")
        print(f"  Median Edge: {edge.get('median', 0):.2%}")
        print(f"  Min Edge: {edge.get('min', 0):.2%}")
        print(f"  Max Edge: {edge.get('max', 0):.2%}")
        print(f"  Std Dev: {edge.get('std', 0):.2%}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Diagnose model accuracy')
    parser.add_argument('--sport', type=str, required=True, help='Sport code (NFL, NHL, NBA, MLB)')
    parser.add_argument('--source', type=str, default='csv', choices=['csv', 'db'], help='Source: csv or db')
    parser.add_argument('--output-dir', type=str, default='exports', help='Output directory for results')
    
    args = parser.parse_args()
    sport = args.sport.upper()
    
    try:
        # Load predictions
        if args.source == 'csv':
            predictions_df = load_predictions_from_csv(sport)
        else:
            predictions_df = load_predictions_from_db(sport)
        
        if len(predictions_df) == 0:
            logger.error(f"No predictions found for {sport}")
            return
        
        logger.info(f"Loaded {len(predictions_df)} predictions")
        
        # Get game results
        game_ids = predictions_df['game_id'].unique().tolist()
        logger.info(f"Looking for {len(game_ids)} games in database...")
        games_df = get_game_results(sport, game_ids)
        
        if len(games_df) == 0:
            logger.warning(f"No completed games found for {sport}")
            logger.warning(f"  Searched for {len(game_ids)} game IDs from predictions")
            logger.warning(f"  This usually means:")
            logger.warning(f"    - Games haven't finished yet (check the date in CSV)")
            logger.warning(f"    - Games don't have final scores in database")
            logger.warning(f"    - Game IDs don't match between CSV and database")
            logger.warning(f"  Try running this after games have completed and scores are updated")
            return
        
        logger.info(f"Found {len(games_df)} completed games")
        
        # Merge predictions with game results
        merged_df = predictions_df.merge(
            games_df,
            on='game_id',
            how='inner'
        )
        
        if len(merged_df) == 0:
            logger.error("No matching predictions and game results found")
            return
        
        logger.info(f"Matched {len(merged_df)} predictions to game results")
        
        # Determine prediction results
        results = merged_df.apply(determine_prediction_result, axis=1)
        merged_df['is_correct'] = [r[0] for r in results]
        merged_df['actual_outcome'] = [r[1] for r in results]
        
        # Calculate metrics
        metrics = calculate_metrics(merged_df)
        
        # Print console report
        print_console_report(sport, metrics, merged_df)
        
        # Save detailed results CSV
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{sport}_accuracy_diagnostic_{timestamp}.csv"
        merged_df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")
        
        # Save summary JSON
        json_path = output_dir / f"{sport}_accuracy_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'sport': sport,
                'timestamp': timestamp,
                'metrics': metrics,
            }, f, indent=2)
        logger.info(f"Saved summary to {json_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

