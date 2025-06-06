import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application configuration
class Config:
    # Directory structure
    PROJECT_ROOT = Path(__file__).parent.parent  # Root of the project
    MODELS_DIR = PROJECT_ROOT / "models"
    FOOTBALL_DATA_DIR = PROJECT_ROOT / "football-data"
    
    # Model settings - Davidson model (meta_pairwise)
    MODEL_DATA_FILENAME = "davidson_model_data.joblib"
    MODEL_DATA_PATH = MODELS_DIR / MODEL_DATA_FILENAME
    
    # API settings
    API_TITLE = "Football Prediction API - Davidson Model Example"
    API_DESCRIPTION = "Example API demonstrating the Davidson model (meta_pairwise) for football match predictions. Built for machine learning engineers to learn and build upon."
    API_VERSION = "1.0.0"
    
    # Security settings
    API_KEY = os.getenv("API_KEY", "davidson-model-example-key")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # League mappings for Davidson model
    LEAGUE_NAME_TO_CODE_MAPPING: Dict[str, str] = {
        "Premier League": "E0",
        "Championship": "E1",
        "League One": "E2",
        "League Two": "E3",
        "La Liga": "SP1",
        "La Liga 2": "SP2",
        "Bundesliga": "D1",
        "2. Bundesliga": "D2",
        "Serie A": "I1",
        "Serie B": "I2",
        "Ligue 1": "F1",
        "Ligue 2": "F2",
        "Scottish Premiership": "SC0",
        "Scottish Championship": "SC1",
        "Scottish League One": "SC2",
        "Scottish League Two": "SC3",
        "Eredivisie": "N1",
        "Belgian Pro League": "B1",
        "Primeira Liga": "P1",
        "Süper Lig": "T1",
        "Super League 1": "G1",
    }

config = Config() 