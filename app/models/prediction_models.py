from typing import List, Dict, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Models for sample_request.json

class ChallengeRequest(BaseModel):
    challenge_id: str = Field(..., alias="challengeId")
    home_team: str = Field(..., alias="homeTeam")
    away_team: str = Field(..., alias="awayTeam")
    venue: str
    league: str
    fixture_id: int = Field(..., alias="fixtureId")
    kickoff_time: datetime = Field(..., alias="kickoffTime")
    challenge_phase_minutes: int = Field(..., alias="challengePhaseMinutes")
    target_market: str = Field(..., alias="targetMarket")
    phase_identifier: str = Field(..., alias="phaseIdentifier")
    difficulty: float

# Models for sample_reply.json

class Prediction1X2(BaseModel):
    home: float
    draw: float
    away: float

class PredictionOutput(BaseModel):
    prediction_1X2: Prediction1X2 = Field(..., alias="1X2")

class ChallengeResponse(BaseModel):
    challenge_id: str = Field(..., alias="challengeId")
    prediction: PredictionOutput 