"""
Comprehensive script to validate model probabilities and analyze edge distributions.
Answers: "Are high edges realistic? Is the model well-calibrated?"
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Prediction, Game
from app.utils.odds import calculate_moneyline_edge, american_odds_to_implied_prob
from app.prediction.validation import check_calibration_by_bucket


def analyze_edge_distribution(sport: str, market: str):
    """
    Analyze edge distribution for historical predictions.
    Shows how common different edge sizes are.
    """
    print(f"\n{'='*80}")
    print(f"Edge Distribution Analysis: {sport} {market}")
    print(f"{'='*80}\n")
    
    db = SessionLocal()
    try:
        # Get all predictions (settled and unsettled) with odds
        predictions = db.query(Prediction).filter(
            Prediction.sport == sport,
            Prediction.market == market
        ).all()
        
        if len(predictions) < 10:
            print(f"⚠️  Not enough predictions ({len(predictions)}) to analyze.")
            return
        
        results = []
        for pred in predictions:
            game = db.query(Game).filter(Game.id == pred.game_id).first()
            if not game:
                continue
            
            if market == "moneyline":
                home_prob = pred.home_win_prob
                if home_prob is None:
                    continue
                away_prob = 1 - home_prob
                
                home_odds = game.home_moneyline
                away_odds = game.away_moneyline
                
                if home_odds is None or away_odds is None:
                    continue
                
                # Calculate edges
                ml_edges = calculate_moneyline_edge(
                    home_prob, away_prob, home_odds, away_odds
                )
                
                best_edge = ml_edges.get("best_edge", 0)
                best_side = ml_edges.get("best_side")
                
                if best_side:
                    if best_side == "home":
                        implied_prob = american_odds_to_implied_prob(home_odds)
                        model_prob = home_prob
                        odds = home_odds
                    else:
                        implied_prob = american_odds_to_implied_prob(away_odds)
                        model_prob = away_prob
                        odds = away_odds
                    
                    # Determine if underdog
                    is_underdog = implied_prob < 0.5
                    
                    results.append({
                        "edge": best_edge,
                        "model_prob": model_prob,
                        "implied_prob": implied_prob,
                        "odds": odds,
                        "is_underdog": is_underdog,
                        "settled": pred.settled,
                        "side": best_side
                    })
        
        if len(results) < 10:
            print(f"⚠️  Not enough predictions with odds ({len(results)}) to analyze.")
            return
        
        df = pd.DataFrame(results)
        
        # Edge distribution statistics
        print("📊 Edge Distribution Statistics:")
        print(f"   Total predictions analyzed: {len(df)}")
        print(f"   Settled predictions: {df['settled'].sum()}")
        print(f"   Unsettled predictions: {(~df['settled']).sum()}")
        print(f"\n   Edge Statistics:")
        print(f"     Mean edge: {df['edge'].mean():.2%}")
        print(f"     Median edge: {df['edge'].median():.2%}")
        print(f"     Std dev: {df['edge'].std():.2%}")
        print(f"     Min edge: {df['edge'].min():.2%}")
        print(f"     Max edge: {df['edge'].max():.2%}")
        
        # Edge buckets
        print(f"\n   Edge Distribution by Bucket:")
        edge_buckets = [
            ("Negative (<0%)", df['edge'] < 0),
            ("Small (0-5%)", (df['edge'] >= 0) & (df['edge'] < 0.05)),
            ("Medium (5-10%)", (df['edge'] >= 0.05) & (df['edge'] < 0.10)),
            ("Large (10-15%)", (df['edge'] >= 0.10) & (df['edge'] < 0.15)),
            ("Very Large (15-20%)", (df['edge'] >= 0.15) & (df['edge'] < 0.20)),
            ("Extreme (20%+)", df['edge'] >= 0.20),
        ]
        
        for label, mask in edge_buckets:
            count = mask.sum()
            pct = (count / len(df)) * 100
            print(f"     {label:<25} {count:>5} ({pct:>5.1f}%)")
        
        # High edge analysis
        high_edges = df[df['edge'] >= 0.20]
        if len(high_edges) > 0:
            print(f"\n   ⚠️  High Edge Analysis (≥20%):")
            print(f"     Found {len(high_edges)} predictions with edge ≥20%")
            print(f"     This is {len(high_edges)/len(df)*100:.1f}% of all predictions")
            
            print(f"\n     Sample high-edge predictions:")
            for idx, row in high_edges.head(10).iterrows():
                print(f"       Edge: {row['edge']:.1%} | "
                      f"Model: {row['model_prob']:.1%} | "
                      f"Implied: {row['implied_prob']:.1%} | "
                      f"Odds: {row['odds']:+.0f} | "
                      f"Underdog: {row['is_underdog']}")
        
        # Underdog analysis
        underdog_high_prob = df[(df['is_underdog']) & (df['model_prob'] > 0.60)]
        if len(underdog_high_prob) > 0:
            print(f"\n   ⚠️  Underdog High Probability Analysis:")
            print(f"     Found {len(underdog_high_prob)} underdogs with model prob >60%")
            print(f"     This is {len(underdog_high_prob)/len(df)*100:.1f}% of all predictions")
            
            print(f"\n     Sample underdog high-prob predictions:")
            for idx, row in underdog_high_prob.head(10).iterrows():
                print(f"       Model: {row['model_prob']:.1%} | "
                      f"Implied: {row['implied_prob']:.1%} | "
                      f"Edge: {row['edge']:.1%} | "
                      f"Odds: {row['odds']:+.0f}")
        
        # Settled predictions ROI analysis
        settled_df = df[df['settled'] == True]
        if len(settled_df) > 0:
            print(f"\n   📈 Historical Performance (Settled Predictions):")
            print(f"     Total settled: {len(settled_df)}")
            
            # Get actual outcomes
            settled_results = []
            for pred in predictions:
                if not pred.settled:
                    continue
                game = db.query(Game).filter(Game.id == pred.game_id).first()
                if not game or game.home_score is None or game.away_score is None:
                    continue
                
                if market == "moneyline":
                    home_prob = pred.home_win_prob
                    if home_prob is None:
                        continue
                    
                    home_odds = game.home_moneyline
                    away_odds = game.away_moneyline
                    if home_odds is None or away_odds is None:
                        continue
                    
                    ml_edges = calculate_moneyline_edge(
                        home_prob, 1 - home_prob, home_odds, away_odds
                    )
                    
                    best_edge = ml_edges.get("best_edge", 0)
                    best_side = ml_edges.get("best_side")
                    
                    if best_side:
                        if best_side == "home":
                            won = game.home_score > game.away_score
                            odds = home_odds
                        else:
                            won = game.away_score > game.home_score
                            odds = away_odds
                        
                        settled_results.append({
                            "edge": best_edge,
                            "won": won,
                            "odds": odds
                        })
            
            if len(settled_results) > 0:
                settled_roi_df = pd.DataFrame(settled_results)
                
                # ROI by edge bucket
                print(f"\n     ROI by Edge Bucket:")
                roi_buckets = [
                    ("0-5%", (settled_roi_df['edge'] >= 0) & (settled_roi_df['edge'] < 0.05)),
                    ("5-10%", (settled_roi_df['edge'] >= 0.05) & (settled_roi_df['edge'] < 0.10)),
                    ("10-15%", (settled_roi_df['edge'] >= 0.10) & (settled_roi_df['edge'] < 0.15)),
                    ("15%+", settled_roi_df['edge'] >= 0.15),
                ]
                
                for label, mask in roi_buckets:
                    bucket_df = settled_roi_df[mask]
                    if len(bucket_df) == 0:
                        continue
                    
                    wins = bucket_df['won'].sum()
                    total = len(bucket_df)
                    win_rate = wins / total if total > 0 else 0
                    
                    # Calculate ROI (simplified: assume $1 bet per game)
                    profit = 0
                    for _, row in bucket_df.iterrows():
                        if row['won']:
                            if row['odds'] > 0:
                                profit += row['odds'] / 100
                            else:
                                profit += 100 / abs(row['odds'])
                        else:
                            profit -= 1
                    
                    roi = (profit / total) * 100 if total > 0 else 0
                    
                    print(f"       {label:<8} {total:>4} bets | "
                          f"Win Rate: {win_rate:.1%} | "
                          f"ROI: {roi:+.1f}%")
        
        print(f"\n{'='*80}\n")
        
    finally:
        db.close()


def validate_model(sport: str, market: str):
    """
    Comprehensive validation of model probabilities.
    """
    print(f"\n{'='*80}")
    print(f"Model Probability Validation: {sport} {market}")
    print(f"{'='*80}\n")
    
    # 1. Calibration check
    print("1️⃣  Calibration Check:")
    calibration_df = check_calibration_by_bucket(sport, market, n_bins=10)
    
    if calibration_df.empty:
        print("   ⚠️  Not enough settled predictions for calibration check.")
        print("   Need at least 50 settled predictions.\n")
    else:
        print(f"   ✓ Calibration data available ({len(calibration_df)} bins)\n")
        print("   Calibration by Probability Bucket:")
        print(f"   {'Bin':<6} {'Prob Range':<15} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Samples':<10}")
        print(f"   {'-'*80}")
        
        total_error = 0
        total_samples = 0
        
        for _, row in calibration_df.iterrows():
            error = row['calibration_error']
            samples = row['n_samples']
            total_error += error * samples
            total_samples += samples
            
            status = "✓" if error < 0.05 else "⚠️" if error < 0.10 else "✗"
            
            print(
                f"   {status} {int(row['prob_bin']):<4} "
                f"{row['prob_range']:<15} "
                f"{row['avg_predicted']:.1%}      "
                f"{row['avg_actual']:.1%}      "
                f"{error:.1%}        "
                f"{int(samples):<10}"
            )
        
        avg_calibration_error = total_error / total_samples if total_samples > 0 else 0
        
        print(f"\n   Overall Calibration Error: {avg_calibration_error:.2%}")
        if avg_calibration_error < 0.05:
            print("   ✓ Model is well-calibrated (error < 5%)")
        elif avg_calibration_error < 0.10:
            print("   ⚠️  Model calibration is acceptable (error < 10%)")
        else:
            print("   ✗ Model calibration is poor (error > 10%)")
    
    print()
    
    # 2. Edge distribution analysis
    print("2️⃣  Edge Distribution Analysis:")
    analyze_edge_distribution(sport, market)
    
    print(f"\n{'='*80}\n")
    print("✅ Validation Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate model probabilities and analyze edge distributions"
    )
    parser.add_argument("--sport", type=str, required=True, 
                       help="Sport code (NFL, NHL, NBA, MLB)")
    parser.add_argument("--market", type=str, required=True,
                       help="Market type (moneyline, spread, totals)")
    
    args = parser.parse_args()
    
    validate_model(args.sport, args.market)

