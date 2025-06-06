import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add scripts directory to path for importing Meta_pairwise
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

from Meta_pairwise import train_all_davidson_models, predict_davidson_match
from app.config import config

logger = logging.getLogger(__name__)

# Global variable to store trained Davidson models
_trained_models: Optional[Dict[str, Any]] = None

def get_trained_models() -> Dict[str, Any]:
    """Get or load trained Davidson models."""
    global _trained_models
    if _trained_models is None:
        logger.info("Loading Davidson models...")
        _trained_models = train_all_davidson_models()
        logger.info(f"Loaded Davidson models for leagues: {list(_trained_models.keys())}")
    return _trained_models

def predict_match(home_team: str, away_team: str, league_name: str) -> Optional[Dict[str, float]]:
    """
    Predict match outcome using Davidson model.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team  
        league_name: Name of the league
        
    Returns:
        Dictionary with home, draw, away probabilities or None if prediction cannot be made
    """
    try:
        # Get league code from mapping
        league_code = config.LEAGUE_NAME_TO_CODE_MAPPING.get(league_name)
        
        if not league_code:
            logger.warning(f"Unknown league: {league_name}. Available leagues: {list(config.LEAGUE_NAME_TO_CODE_MAPPING.keys())}")
            return None
        
        # Get trained Davidson models
        trained_models = get_trained_models()
        
        if not trained_models:
            logger.error("No Davidson models available")
            return None
        
        # Predict using Davidson model
        prediction = predict_davidson_match(
            home_team_name=home_team,
            away_team_name=away_team,
            league_code=league_code,
            all_trained_models=trained_models
        )
        
        if prediction:
            logger.info(f"Davidson prediction for {home_team} vs {away_team} in {league_name}: {prediction}")
            return prediction
        else:
            logger.warning(f"Davidson model could not predict {home_team} vs {away_team} in {league_name}. Teams may not be in training data.")
            return None
            
    except Exception as e:
        logger.error(f"Error predicting match {home_team} vs {away_team} in {league_name}: {e}")
        return None 