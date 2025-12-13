from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Game(Base):
    __tablename__ = "games"
    
    id = Column(String, primary_key=True)  # ESPN game ID
    sport = Column(String, index=True)     # NFL, NBA, NHL, MLB
    league = Column(String)
    date = Column(DateTime, index=True)
    home_team = Column(String)
    away_team = Column(String)
    status = Column(String)  # scheduled, live, final
    
    # Odds + results
    home_moneyline = Column(Float, nullable=True)
    away_moneyline = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)
    over_under = Column(Float, nullable=True)  # Legacy - use closing_total instead
    
    # Over/Under data
    closing_total = Column(Float, nullable=True)  # Closing O/U line from ESPN
    actual_total = Column(Integer, nullable=True)  # Actual total points scored
    ou_result = Column(String, nullable=True)  # OVER, UNDER, or PUSH
    
    # Final scores
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    
    # Full ESPN JSON
    espn_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True)  # UUID
    game_id = Column(String, index=True)
    sport = Column(String)
    market = Column(String)  # moneyline, spread, over_under
    
    model_version = Column(String)
    
    # Predictions
    home_win_prob = Column(Float, nullable=True)
    spread_cover_prob = Column(Float, nullable=True)
    over_prob = Column(Float, nullable=True)
    
    # Bankroll management
    recommended_kelly = Column(Float, nullable=True)
    recommended_unit_size = Column(Float, default=1.0)
    
    # Settling
    settled = Column(Boolean, default=False)
    result = Column(String, nullable=True)  # win, loss, push (legacy)
    settled_result = Column(String, nullable=True)  # win, loss, push
    pnl = Column(Float, nullable=True)
    
    predicted_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime, nullable=True)
