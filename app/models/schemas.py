from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class GameBase(BaseModel):
    sport: str
    league: str
    home_team: str
    away_team: str
    status: str


class GameCreate(GameBase):
    id: str
    date: datetime
    espn_data: dict


class GameResponse(GameBase):
    id: str
    date: datetime
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None

    class Config:
        from_attributes = True


class PredictionCreate(BaseModel):
    game_id: str
    sport: str
    market: str
    model_version: str
    home_win_prob: Optional[float] = None
    spread_cover_prob: Optional[float] = None
    over_prob: Optional[float] = None


class PredictionResponse(PredictionCreate):
    id: str
    settled: bool
    result: Optional[str] = None
    pnl: Optional[float] = None

    class Config:
        from_attributes = True
