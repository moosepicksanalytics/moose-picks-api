# Moose Picks - Model Accuracy Diagnostic & Fix Script
# Run this to understand your real model performance

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

class ModelAccuracyAnalyzer:
    """
    Analyzes and fixes model accuracy issues
    """
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.recommendations = []
    
    def analyze_confidence_calibration(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Check if model confidence scores match actual win rates
        
        Example: 80% confidence picks should win ~80% of the time
        """
        print("\n" + "="*80)
        print("CALIBRATION ANALYSIS: Do confidence scores match reality?")
        print("="*80)
        
        # Group by confidence bucket
        predictions_df['confidence_bucket'] = pd.cut(
            predictions_df['model_prob'],
            bins=[0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0],
            labels=['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']
        )
        
        calibration = predictions_df.groupby('confidence_bucket', observed=True).agg({
            'model_prob': ['count', 'mean'],
            'actual_result': ['sum', 'mean']  # Assuming 1 = hit, 0 = miss
        }).round(4)
        
        print("\nCalibration Table:")
        print("Confidence | Num Preds | Avg Confidence | Hit Rate | Expected | Difference")
        print("-" * 80)
        
        for bucket in ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']:
            if bucket in predictions_df['confidence_bucket'].values:
                count = len(predictions_df[predictions_df['confidence_bucket'] == bucket])
                avg_conf = predictions_df[predictions_df['confidence_bucket'] == bucket]['model_prob'].mean()
                hit_rate = predictions_df[predictions_df['confidence_bucket'] == bucket]['actual_result'].mean()
                print(f"{bucket:11} | {count:9} | {avg_conf:14.1%} | {hit_rate:8.1%} | {avg_conf:8.1%} | {hit_rate - avg_conf:+.1%}")
        
        # Calculate Expected Calibration Error (ECE)
        ece = np.mean(np.abs(
            predictions_df.groupby('confidence_bucket', observed=True)['actual_result'].mean() -
            predictions_df.groupby('confidence_bucket', observed=True)['model_prob'].mean()
        ))
        
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        print("  ECE < 0.05 is excellent")
        print("  ECE 0.05-0.10 is good")
        print("  ECE > 0.10 is poor (model needs recalibration)")
        
        if ece > 0.10:
            self.issues.append("MODEL OVERCONFIDENCE: ECE indicates poor calibration")
            self.recommendations.append(
                "Apply Platt scaling or isotonic regression to recalibrate model probabilities"
            )
        
        return {'ece': ece, 'calibration': calibration}
    
    def analyze_edge_calculations(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Verify edge calculations are realistic
        """
        print("\n" + "="*80)
        print("EDGE ANALYSIS: Are edges realistic?")
        print("="*80)
        
        print(f"\nEdge Statistics:")
        print(f"  Mean edge: {predictions_df['edge'].mean():.4f} ({predictions_df['edge'].mean()*100:.2f}%)")
        print(f"  Median edge: {predictions_df['edge'].median():.4f}")
        print(f"  Min edge: {predictions_df['edge'].min():.4f}")
        print(f"  Max edge: {predictions_df['edge'].max():.4f}")
        print(f"  Std dev: {predictions_df['edge'].std():.4f}")
        
        print("\nEdge Distribution:")
        print(f"  Picks with 0-5% edge: {len(predictions_df[predictions_df['edge'] < 0.05])}")
        print(f"  Picks with 5-10% edge: {len(predictions_df[(predictions_df['edge'] >= 0.05) & (predictions_df['edge'] < 0.10)])}")
        print(f"  Picks with 10%+ edge: {len(predictions_df[predictions_df['edge'] >= 0.10])}")
        
        # Red flag: if average edge is >15%, odds are wrong
        if predictions_df['edge'].mean() > 0.15:
            self.issues.append(f"UNREALISTIC EDGES: Average {predictions_df['edge'].mean():.1%} edge is too high")
            self.issues.append("  → Implied probabilities likely calculated incorrectly")
            self.issues.append("  → Or sportsbook odds are stale/wrong")
            self.recommendations.append(
                "Validate odds calculation logic - compare with live sportsbook odds"
            )
        
        return predictions_df['edge'].describe().to_dict()
    
    def analyze_win_rate_by_sport(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Calculate actual win rate by sport
        """
        print("\n" + "="*80)
        print("WIN RATE BY SPORT: Which sports actually work?")
        print("="*80)
        
        win_rates = predictions_df.groupby('sport').agg({
            'actual_result': ['count', 'sum', 'mean'],
            'model_prob': 'mean',
            'edge': 'mean'
        }).round(4)
        
        print("\nSport | Predictions | Hits | Win Rate | Avg Confidence | Avg Edge")
        print("-" * 80)
        
        results = {}
        for sport in predictions_df['sport'].unique():
            sport_data = predictions_df[predictions_df['sport'] == sport]
            count = len(sport_data)
            hits = sport_data['actual_result'].sum()
            win_rate = sport_data['actual_result'].mean()
            avg_conf = sport_data['model_prob'].mean()
            avg_edge = sport_data['edge'].mean()
            
            status = "✓ GOOD" if win_rate > 0.52 else "⚠ OKAY" if win_rate > 0.50 else "✗ BAD"
            
            print(f"{sport:8} | {count:11} | {int(hits):4} | {win_rate:8.1%} | {avg_conf:14.1%} | {avg_edge:8.1%} {status}")
            
            results[sport] = {
                'count': count,
                'hits': int(hits),
                'win_rate': float(win_rate),
                'avg_confidence': float(avg_conf),
                'avg_edge': float(avg_edge),
                'status': status
            }
            
            # Flag underperforming sports
            if win_rate < 0.50:
                self.issues.append(f"{sport} MODEL IS LOSING: {win_rate:.1%} win rate")
                self.recommendations.append(f"Pause {sport} predictions until model improves")
        
        return results
    
    def analyze_kelly_sizing(self, predictions_df: pd.DataFrame) -> None:
        """
        Check if recommended Kelly sizes are appropriate
        """
        print("\n" + "="*80)
        print("KELLY CRITERION SIZING: Are bet sizes optimal?")
        print("="*80)
        
        print(f"\nRecommended unit sizes (fractional Kelly):")
        print(f"  Mean: {predictions_df['recommended_unit_size'].mean():.2f}")
        print(f"  Min: {predictions_df['recommended_unit_size'].min():.2f}")
        print(f"  Max: {predictions_df['recommended_unit_size'].max():.2f}")
        
        # Kelly should be small (0.01-0.10 for fractional Kelly)
        if predictions_df['recommended_unit_size'].mean() > 0.25:
            self.issues.append("AGGRESSIVE SIZING: Recommended Kelly sizes are too large")
            self.recommendations.append("Use fractional Kelly (1/4 Kelly) to reduce bankroll risk")
    
    def generate_report(self, predictions_df: pd.DataFrame) -> str:
        """
        Generate complete accuracy report
        """
        print("\n\n" + "="*80)
        print("MOOSE PICKS - MODEL ACCURACY REPORT")
        print("="*80)
        
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Predictions Analyzed: {len(predictions_df)}")
        print(f"Date Range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        
        # Run analyses
        self.analyze_confidence_calibration(predictions_df)
        self.analyze_edge_calculations(predictions_df)
        sports_results = self.analyze_win_rate_by_sport(predictions_df)
        self.analyze_kelly_sizing(predictions_df)
        
        # Summary
        print("\n\n" + "="*80)
        print("ISSUES FOUND:")
        print("="*80)
        
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        else:
            print("✓ No major issues found!")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✓ Model appears to be performing well!")
        
        print("\n\n" + "="*80)
        print("ACTION ITEMS FOR LOVABLE FRONTEND:")
        print("="*80)
        print("""
1. Display ACTUAL win rates (not model confidence)
   - Show "Historical accuracy: 52% for NFL spreads"
   - Show "This model wins 51% vs random (50%)"

2. Add confidence calibration info
   - Show users: "82% confidence picks actually hit 78% historically"
   - This builds trust (shows you're honest about limitations)

3. Track user ROI if they follow picks
   - Store each user's bet placement
   - Calculate their actual ROI monthly
   - Show in dashboard

4. Focus on high-accuracy sports only
   - Pause sports with <50% win rate
   - Only publish picks you're confident in
   - Better to have 5 good picks than 20 mediocre ones

5. Real-time result settling
   - Auto-update prediction results within 1 hour of game end
   - Show hit/miss immediately
   - Calculate ROI in real-time
        """)
        
        return f"Analyzed {len(predictions_df)} predictions"


# Example usage
if __name__ == "__main__":
    # This would be used like:
    # analyzer = ModelAccuracyAnalyzer()
    # 
    # # Load your predictions from Supabase or CSV
    # predictions_df = pd.read_csv('exports/NFL_20251214.csv')
    # 
    # # Add actual results (you need to populate this)
    # predictions_df['actual_result'] = [1, 0, 1, ...]  # 1 = hit, 0 = miss
    # 
    # analyzer.generate_report(predictions_df)
    
    print("ModelAccuracyAnalyzer ready to use!")
    print("\nUsage:")
    print("  1. Load your predictions from Supabase")
    print("  2. Add actual game results (settled_correctly column)")
    print("  3. Call analyzer.generate_report(df)")
