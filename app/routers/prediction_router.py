from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from app.models.prediction_models import ChallengeRequest, ChallengeResponse, Prediction1X2, PredictionOutput
from app.services.prediction_service import predict_match, get_trained_models
from app.services.auth_service import verify_api_key
from app.config import config

router = APIRouter(tags=["predictions"])
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=List[ChallengeResponse])
async def predict_challenges(
    challenges: List[ChallengeRequest],
    api_key: str = Depends(verify_api_key)
) -> List[ChallengeResponse]:
    """
    Predict outcomes for football match challenges using the Davidson model.
    
    Args:
        challenges: List of challenge requests containing match details
        api_key: Valid API key for authentication
        
    Returns:
        List of challenge responses with predictions
        
    Raises:
        HTTPException: If prediction cannot be made for any challenge
    """
    try:
        responses = []
        failed_predictions = []
        
        for challenge in challenges:
            # Get prediction for this match using Davidson model
            prediction_dict = predict_match(
                home_team=challenge.home_team,
                away_team=challenge.away_team,
                league_name=challenge.league
            )
            
            if prediction_dict is None:
                failed_predictions.append({
                    "challenge_id": challenge.challenge_id,
                    "home_team": challenge.home_team,
                    "away_team": challenge.away_team,
                    "league": challenge.league
                })
                continue
            
            # Create response objects
            prediction_1x2 = Prediction1X2(
                home=prediction_dict["home"],
                draw=prediction_dict["draw"],
                away=prediction_dict["away"]
            )
            
            prediction_output = PredictionOutput(**{"1X2": prediction_1x2})
            
            response = ChallengeResponse(
                **{"challengeId": challenge.challenge_id, "prediction": prediction_output}
            )
            
            responses.append(response)
        
        if failed_predictions:
            error_msg = f"Could not make predictions for {len(failed_predictions)} challenges. "
            error_msg += "This may be due to unknown teams or leagues not in the training data. "
            error_msg += f"Failed challenges: {failed_predictions}"
            logger.error(error_msg)
            raise HTTPException(status_code=422, detail=error_msg)
        
        logger.info(f"Successfully processed {len(challenges)} challenges using Davidson model")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing challenges: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing predictions: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint that returns API status and Davidson model availability.
    This endpoint does not require authentication.
    
    Returns:
        Dictionary with API status and model information
    """
    try:
        # Check if Davidson models are available
        models = get_trained_models()
        model_status = "available" if models else "unavailable"
        available_leagues = list(models.keys()) if models else []
        
        return {
            "status": "healthy",
            "api_version": config.API_VERSION,
            "model": {
                "type": "Davidson (meta_pairwise)",
                "status": model_status,
                "available_leagues": available_leagues,
                "supported_leagues": list(config.LEAGUE_NAME_TO_CODE_MAPPING.keys())
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "api_version": config.API_VERSION,
            "model": {
                "type": "Davidson (meta_pairwise)",
                "status": "error",
                "error": str(e)
            }
        } 