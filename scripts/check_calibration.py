"""
Script to check model calibration on historical predictions.
Validates that model probabilities match actual outcomes.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.prediction.validation import check_calibration_by_bucket
from app.database import SessionLocal
from app.models.db_models import Prediction, Game


def print_calibration_report(sport: str, market: str):
    """
    Print a calibration report for a model.
    
    Args:
        sport: Sport code
        market: Market type
    """
    print(f"\n{'='*80}")
    print(f"Calibration Report: {sport} {market}")
    print(f"{'='*80}\n")
    
    calibration_df = check_calibration_by_bucket(sport, market, n_bins=10)
    
    if calibration_df.empty:
        print(f"⚠️  Not enough settled predictions to check calibration.")
        print(f"   Need at least 50 settled predictions. Check back after more games settle.")
        return
    
    print(f"Calibration by Probability Bucket:")
    print(f"{'Bin':<6} {'Prob Range':<15} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Samples':<10}")
    print(f"{'-'*80}")
    
    total_error = 0
    total_samples = 0
    
    for _, row in calibration_df.iterrows():
        error = row['calibration_error']
        samples = row['n_samples']
        total_error += error * samples
        total_samples += samples
        
        status = "✓" if error < 0.05 else "⚠️" if error < 0.10 else "✗"
        
        print(
            f"{status} {int(row['prob_bin']):<4} "
            f"{row['prob_range']:<15} "
            f"{row['avg_predicted']:.1%}      "
            f"{row['avg_actual']:.1%}      "
            f"{error:.1%}        "
            f"{int(samples):<10}"
        )
    
    avg_calibration_error = total_error / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Overall Calibration Error: {avg_calibration_error:.2%}")
    
    if avg_calibration_error < 0.05:
        print("✓ Model is well-calibrated (error < 5%)")
    elif avg_calibration_error < 0.10:
        print("⚠️  Model calibration is acceptable (error < 10%)")
        print("   Consider recalibrating model probabilities.")
    else:
        print("✗ Model calibration is poor (error > 10%)")
        print("   Model probabilities are not reliable. Consider:")
        print("   1. Adding probability calibration (Platt scaling or isotonic regression)")
        print("   2. Checking for data leakage")
        print("   3. Reviewing feature engineering")
    
    print(f"{'='*80}\n")


def check_all_models():
    """Check calibration for all trained models."""
    sports = ["NFL", "NHL", "NBA", "MLB"]
    markets = ["moneyline", "spread", "totals"]
    
    print("\n" + "="*80)
    print("Model Calibration Check")
    print("="*80)
    print("\nThis script checks how well model probabilities match actual outcomes.")
    print("A well-calibrated model should have predicted probabilities close to actual win rates.\n")
    
    for sport in sports:
        for market in markets:
            print_calibration_report(sport, market)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check model calibration")
    parser.add_argument("--sport", type=str, help="Sport code (NFL, NHL, NBA, MLB)")
    parser.add_argument("--market", type=str, help="Market type (moneyline, spread, totals)")
    
    args = parser.parse_args()
    
    if args.sport and args.market:
        print_calibration_report(args.sport, args.market)
    else:
        check_all_models()

