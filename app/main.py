from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.config import config
from app.routers.prediction_router import router as prediction_router
from app.services.prediction_service import get_trained_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    logger.info("Starting up Davidson Model API...")
    
    # Initialize Davidson models on startup
    try:
        logger.info("Loading/training Davidson models...")
        models = get_trained_models()
        if models:
            logger.info(f"Davidson models loaded successfully for leagues: {list(models.keys())}")
        else:
            logger.error("No Davidson models available - API will not function properly")
    except Exception as e:
        logger.error(f"Error loading Davidson models: {e}")
        logger.error("API may not function properly without models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Davidson Model API...")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=config.API_TITLE,
        description=f"""
        {config.API_DESCRIPTION}
        
        ## About the Davidson Model
        
        This API demonstrates the Davidson model (meta_pairwise) for predicting football match outcomes.
        The model estimates team strengths, home advantage, and draw propensity to calculate probabilities
        for Home Win, Draw, and Away Win.
        
        ## Authentication
        
        This API requires authentication using an API key. Include your API key in the Authorization header:
        
        ```
        Authorization: Bearer your-api-key-here
        ```
        
        ## Endpoints
        
        - **POST /api/v1/predict**: Make predictions for football matches using the Davidson model (requires authentication)
        - **GET /api/v1/health**: Check API health status and model availability (no authentication required)
        
        ## Supported Leagues
        
        The model supports predictions for teams in the following leagues (if training data is available):
        {', '.join(config.LEAGUE_NAME_TO_CODE_MAPPING.keys())}
        """,
        version=config.API_VERSION,
        lifespan=lifespan
    )
    
    # Include routers
    app.include_router(prediction_router, prefix="/api/v1")
    
    return app

app = create_app() 