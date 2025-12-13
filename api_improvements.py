# API IMPROVEMENTS FOR MOOSE PICKS
# Add these endpoints to your FastAPI app (app/api_endpoints.py)

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from datetime import datetime, timedelta
from sqlalchemy import func
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])

# ========================================
# 1. SETTLE PREDICTIONS ENDPOINT
# ========================================
# Auto-settle predictions when games finish

@router.post("/settle-predictions")
def settle_predictions(
    sport: str = Query(..., description="Sport to settle (NFL, NBA, NHL, MLB)"),
    background_tasks: BackgroundTasks = None
):
    """
    Automatically settle all predictions for games that have final scores.
    
    This endpoint:
    1. Fetches games with final status
    2. Matches against predictions
    3. Determines if prediction was correct
    4. Calculates ROI
    5. Updates prediction_results table
    
    Call this endpoint after games finish (e.g., every 30 minutes)
    """
    
    def settle_sport():
        from app.database import SessionLocal
        from app.models.db_models import Game, Prediction
        from sqlalchemy import and_, not_
        
        db = SessionLocal()
        try:
            # Get all games with final status
            finished_games = db.query(Game).filter(
                and_(
                    Game.sport == sport,
                    Game.status == 'final',
                    Game.home_score.isnot(None),
                    Game.away_score.isnot(None)
                )
            ).all()
            
            settled_count = 0
            
            for game in finished_games:
                # Get all unsettled predictions for this game
                predictions = db.query(Prediction).filter(
                    and_(
                        Prediction.game_id == game.id,
                        Prediction.settled == False
                    )
                ).all()
                
                for pred in predictions:
                    # Determine if prediction was correct
                    correct = evaluate_prediction(pred, game)
                    
                    # Calculate ROI
                    roi = 1.0 if correct else -1.0
                    
                    # Insert into prediction_results
                    from app.models.db_models import PredictionResult
                    result = PredictionResult(
                        prediction_id=pred.id,
                        game_id=game.id,
                        sport=sport,
                        market_type=pred.market,
                        predicted_side=get_predicted_side(pred),
                        predicted_probability=get_predicted_prob(pred),
                        actual_result='win' if correct else 'loss',
                        settled_at=datetime.utcnow(),
                        correct=correct,
                        units_won=pred.recommended_unit_size if correct else -pred.recommended_unit_size,
                        roi=roi,
                        confidence_at_prediction=get_predicted_prob(pred)
                    )
                    
                    db.add(result)
                    pred.settled = True
                    pred.settled_result = 'win' if correct else 'loss'
                    pred.settled_at = datetime.utcnow()
                    settled_count += 1
            
            db.commit()
            logger.info(f"Settled {settled_count} predictions for {sport}")
            return {"settled": settled_count, "sport": sport}
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error settling predictions: {e}")
            raise
        finally:
            db.close()
    
    if background_tasks:
        background_tasks.add_task(settle_sport)
        return {"status": "settling in background", "sport": sport}
    else:
        return settle_sport()


def evaluate_prediction(prediction, game) -> bool:
    """
    Determine if a prediction was correct given the final game result.
    """
    if prediction.market == 'moneyline':
        if prediction.home_win_prob > 0.5:
            return game.home_score > game.away_score
        else:
            return game.away_score > game.home_score
    
    elif prediction.market == 'spread':
        # Assuming spread is stored as "Home team spread"
        if prediction.spread_cover_prob > 0.5:
            return game.home_score - game.away_score > prediction.line
        else:
            return game.away_score - game.home_score > abs(prediction.line)
    
    elif prediction.market == 'over_under':
        total = game.home_score + game.away_score
        if prediction.over_prob > 0.5:
            return total > prediction.line
        else:
            return total < prediction.line
    
    return False


# ========================================
# 2. GET ACCURACY METRICS ENDPOINT
# ========================================
# Return real accuracy data for dashboard

@router.get("/metrics/accuracy/{sport}")
def get_accuracy_metrics(
    sport: str,
    days: int = Query(30, ge=1, le=365),
):
    """
    Get accuracy metrics for a sport over the last N days.
    
    Returns:
    - Win rate (correct / total)
    - ROI (units_won / units_placed)
    - Sample size
    - Confidence by percentile
    """
    
    from app.database import SessionLocal
    from sqlalchemy import func
    
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query prediction_results
        results = db.query(
            func.count(PredictionResult.id).label('total'),
            func.sum(func.cast(PredictionResult.correct, Integer)).label('wins'),
            func.avg(PredictionResult.roi).label('avg_roi'),
            func.avg(PredictionResult.confidence_at_prediction).label('avg_confidence')
        ).filter(
            and_(
                PredictionResult.sport == sport,
                PredictionResult.settled_at >= cutoff_date
            )
        ).first()
        
        if not results or results.total == 0:
            return {
                "sport": sport,
                "days": days,
                "total_predictions": 0,
                "win_rate": None,
                "avg_roi": None,
                "message": "No settled predictions found"
            }
        
        win_rate = results.wins / results.total if results.total > 0 else 0
        
        return {
            "sport": sport,
            "days": days,
            "total_predictions": results.total,
            "wins": results.wins,
            "losses": results.total - results.wins,
            "win_rate": round(win_rate, 4),
            "avg_roi": round(results.avg_roi, 4),
            "avg_confidence": round(results.avg_confidence, 4),
            "vs_random": round(win_rate - 0.5, 4),  # How much better than 50% (random)
            "sample_size_adequate": results.total >= 30  # Need 30+ for statistical significance
        }
        
    finally:
        db.close()


# ========================================
# 3. GET CALIBRATION DATA
# ========================================
# Show users that confidence scores match reality

@router.get("/metrics/calibration/{sport}")
def get_calibration_data(sport: str):
    """
    Returns calibration data showing if 80% confidence picks actually hit 80%.
    
    This is the most important trust-building metric.
    """
    
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Get predictions grouped by confidence bucket
        confidence_buckets = [
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.75),
            (0.75, 1.0)
        ]
        
        calibration = []
        
        for lower, upper in confidence_buckets:
            results = db.query(
                func.count(PredictionResult.id).label('count'),
                func.avg(PredictionResult.confidence_at_prediction).label('avg_conf'),
                func.avg(func.cast(PredictionResult.correct, Integer)).label('actual_win_rate')
            ).filter(
                and_(
                    PredictionResult.sport == sport,
                    PredictionResult.confidence_at_prediction >= lower,
                    PredictionResult.confidence_at_prediction < upper,
                    PredictionResult.settled_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).first()
            
            if results and results.count > 0:
                calibration.append({
                    "confidence_range": f"{int(lower*100)}-{int(upper*100)}%",
                    "num_predictions": results.count,
                    "avg_confidence": round(results.avg_conf, 3),
                    "actual_win_rate": round(results.actual_win_rate, 3),
                    "calibration_error": round(abs(results.actual_win_rate - results.avg_conf), 3)
                })
        
        # Calculate Expected Calibration Error
        ece = sum([b['calibration_error'] for b in calibration]) / len(calibration) if calibration else 0
        
        return {
            "sport": sport,
            "calibration": calibration,
            "overall_ece": round(ece, 4),
            "ece_rating": "Excellent" if ece < 0.05 else "Good" if ece < 0.10 else "Needs calibration",
            "interpretation": "If accurate, 80% confidence picks should hit 80% of the time"
        }
        
    finally:
        db.close()


# ========================================
# 4. USER PERFORMANCE TRACKING
# ========================================

@router.get("/user/performance")
def get_user_performance(user_id: str = Depends(get_current_user)):
    """
    Get logged-in user's performance relative to Moose Picks.
    
    Shows:
    - Picks they viewed/placed
    - Their win rate
    - Their ROI
    - How they compare to Moose Picks
    """
    
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Get user's picks
        user_picks_data = db.query(
            UserPick.sport,
            func.count(UserPick.id).label('picks_viewed'),
            func.sum(func.cast(UserPick.action == 'bet_placed', Integer)).label('picks_placed'),
            func.sum(func.cast(UserPick.settled_result == 'win', Integer)).label('wins'),
            func.sum(func.cast(UserPick.settled_result == 'loss', Integer)).label('losses'),
            func.sum(UserPick.bet_amount).label('total_bet'),
            func.sum(
                func.case(
                    (UserPick.settled_result == 'win', UserPick.bet_amount),
                    else_=func.cast(-UserPick.bet_amount, Float)
                )
            ).label('net_profit')
        ).filter(
            and_(
                UserPick.user_id == user_id,
                UserPick.settled == True
            )
        ).group_by(UserPick.sport).all()
        
        # Get Moose Picks accuracy for comparison
        moose_accuracy = {}
        for pred_result in db.query(PredictionResult).filter(
            PredictionResult.settled_at >= datetime.utcnow() - timedelta(days=30)
        ).all():
            if pred_result.sport not in moose_accuracy:
                moose_accuracy[pred_result.sport] = {'correct': 0, 'total': 0}
            moose_accuracy[pred_result.sport]['total'] += 1
            if pred_result.correct:
                moose_accuracy[pred_result.sport]['correct'] += 1
        
        performance = []
        for pick in user_picks_data:
            moose_wr = (moose_accuracy[pick.sport]['correct'] / moose_accuracy[pick.sport]['total']
                       if moose_accuracy.get(pick.sport) else 0)
            user_wr = (pick.wins / (pick.wins + pick.losses) 
                      if (pick.wins + pick.losses) > 0 else 0)
            
            performance.append({
                "sport": pick.sport,
                "picks_viewed": pick.picks_viewed,
                "picks_placed": pick.picks_placed,
                "user_win_rate": round(user_wr, 3),
                "moose_picks_win_rate": round(moose_wr, 3),
                "outperformance": round(user_wr - moose_wr, 3),
                "net_profit": round(pick.net_profit or 0, 2)
            })
        
        return {
            "user_id": user_id,
            "performance": performance,
            "note": "Positive outperformance means you're beating Moose Picks!"
        }
        
    finally:
        db.close()


# ========================================
# 5. DAILY AGGREGATION JOB
# ========================================
# Run this once per day to calculate daily_metrics

@router.post("/jobs/aggregate-daily-metrics")
def aggregate_daily_metrics(background_tasks: BackgroundTasks):
    """
    Aggregate prediction results into daily_metrics table.
    
    Run this once per day (e.g., at midnight via scheduler).
    """
    
    def calculate_metrics():
        from app.database import SessionLocal
        
        db = SessionLocal()
        try:
            yesterday = (datetime.utcnow() - timedelta(days=1)).date()
            
            # Get all predictions settled yesterday
            results = db.query(
                PredictionResult.sport,
                func.count(PredictionResult.id).label('total'),
                func.sum(func.cast(PredictionResult.correct, Integer)).label('correct'),
                func.avg(PredictionResult.roi).label('avg_roi'),
                func.avg(PredictionResult.confidence_at_prediction).label('avg_confidence')
            ).filter(
                func.date(PredictionResult.settled_at) == yesterday
            ).group_by(PredictionResult.sport).all()
            
            for sport_result in results:
                win_rate = (sport_result.correct / sport_result.total 
                           if sport_result.total > 0 else 0)
                
                # Insert or update daily_metrics
                metric = DailyMetric(
                    date=yesterday,
                    sport=sport_result.sport,
                    predictions_made=0,  # Could count from predictions table if needed
                    predictions_settled=sport_result.total,
                    correct_predictions=sport_result.correct,
                    incorrect_predictions=sport_result.total - sport_result.correct,
                    win_rate=win_rate,
                    avg_confidence=sport_result.avg_confidence,
                    total_units_won=sport_result.avg_roi * sport_result.total
                )
                db.add(metric)
            
            db.commit()
            logger.info(f"Aggregated daily metrics for {yesterday}")
            
        finally:
            db.close()
    
    background_tasks.add_task(calculate_metrics)
    return {"status": "aggregating daily metrics in background"}


# ========================================
# HELPER FUNCTIONS
# ========================================

def get_predicted_side(prediction) -> str:
    """Extract which side was predicted."""
    if prediction.market == 'moneyline':
        return 'home' if prediction.home_win_prob > 0.5 else 'away'
    elif prediction.market == 'spread':
        return 'cover' if prediction.spread_cover_prob > 0.5 else 'no_cover'
    elif prediction.market == 'over_under':
        return 'over' if prediction.over_prob > 0.5 else 'under'
    return None


def get_predicted_prob(prediction) -> float:
    """Extract confidence score."""
    if prediction.market == 'moneyline':
        return max(prediction.home_win_prob or 0, 0.5)
    elif prediction.market == 'spread':
        return prediction.spread_cover_prob or 0.5
    elif prediction.market == 'over_under':
        return prediction.over_prob or 0.5
    return 0.5


# DEPLOYMENT NOTE:
# 1. Add these endpoints to your app/api_endpoints.py
# 2. Update your database models to include PredictionResult, DailyMetric tables
# 3. Run Supabase migrations (see supabase_schema_updates.sql)
# 4. Deploy to Railway
# 5. Set up scheduled job to call /jobs/aggregate-daily-metrics daily at midnight UTC