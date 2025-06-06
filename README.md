# Football Match Prediction API - Davidson Model Example

This repository provides a complete example of implementing the **Davidson model (meta_pairwise)** for football match prediction. It's designed as an educational resource for machine learning engineers who want to understand and build upon sports prediction models.

## About the Davidson Model

The Davidson model is a probabilistic approach to predicting football match outcomes that accounts for:

- **Team Strengths**: Individual team abilities estimated from historical performance
- **Home Advantage**: The statistical advantage of playing at home
- **Draw Propensity**: The likelihood of matches ending in a draw

The model outputs probabilities for three outcomes: Home Win, Draw, and Away Win.

## Key Features

- **Complete Implementation**: Full Davidson model implementation in `scripts/Meta_pairwise.py`
- **REST API**: FastAPI-based web service for making predictions
- **Model Persistence**: Automatic saving/loading of trained model parameters
- **Multiple Leagues**: Support for 21+ football leagues
- **Production Ready**: Includes authentication, logging, and error handling

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)
- Football data (included in `football-data.zip`)

### Installation

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd predict-template
   uv sync
   ```

3. **Extract football data**:
   ```bash
   unzip football-data.zip
   ```

### Running the API

1. **Start the server**:

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test the API**:

   ```bash
   # Health check
   curl -X GET "http://localhost:8000/api/v1/health"

   # Make predictions (requires API key)
   curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer davidson-model-example-key" \
     -d @sample_request.json
   ```

3. **Interactive documentation**:
   Visit `http://localhost:8000/docs`

## Project Structure

```
├── app/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration settings
│   ├── models/
│   │   └── prediction_models.py   # Pydantic data models
│   ├── routers/
│   │   └── prediction_router.py   # API endpoints
│   └── services/
│       ├── prediction_service.py  # Business logic
│       └── auth_service.py        # Authentication
├── scripts/
│   └── Meta_pairwise.py          # Davidson model implementation
├── models/
│   └── davidson_model_data.joblib # Trained model parameters
├── football-data/                # Historical match data
├── sample_request.json           # Example API request
├── sample_reply.json            # Example API response
└── pyproject.toml               # Dependencies
```

## Understanding the Davidson Model

### Core Components

1. **Team Strength Estimation** (`fit_initial_davidson_params`):

   - Uses maximum likelihood estimation to fit initial team strengths
   - Estimates home advantage and draw parameters

2. **Dynamic Updates** (`update_strengths_elo_style`):

   - Updates team strengths after each match using Elo-style learning
   - Adapts to recent team performance changes

3. **Probability Calculation** (`davidson_prob`):
   - Converts team strengths to match outcome probabilities
   - Handles the mathematical complexity of the Davidson model

### Key Functions

```python
# Train models for all leagues
trained_models = train_all_davidson_models()

# Make a prediction
prediction = predict_davidson_match(
    home_team_name="Arsenal",
    away_team_name="Chelsea",
    league_code="E0",  # Premier League
    all_trained_models=trained_models
)
# Returns: {"home": 0.45, "draw": 0.30, "away": 0.25}
```

## Supported Leagues

The model supports 21+ leagues including:

- **England**: Premier League, Championship, League One, League Two
- **Spain**: La Liga, La Liga 2
- **Germany**: Bundesliga, 2. Bundesliga
- **Italy**: Serie A, Serie B
- **France**: Ligue 1, Ligue 2
- **Scotland**: Premiership, Championship, League One, League Two
- **Others**: Eredivisie, Belgian Pro League, Primeira Liga, Süper Lig, Super League 1

## API Usage

### Authentication

Include your API key in requests:

```bash
Authorization: Bearer davidson-model-example-key
```

### Request Format

```json
[
  {
    "challengeId": "match_001",
    "homeTeam": "Arsenal",
    "awayTeam": "Chelsea",
    "league": "Premier League",
    "venue": "Emirates Stadium",
    "fixtureId": 12345,
    "kickoffTime": "2024-01-15T15:00:00Z",
    "challengePhaseMinutes": 90,
    "targetMarket": "1X2",
    "phaseIdentifier": "FT",
    "difficulty": 0.5
  }
]
```

### Response Format

```json
[
  {
    "challengeId": "match_001",
    "prediction": {
      "1X2": {
        "home": 0.45,
        "draw": 0.3,
        "away": 0.25
      }
    }
  }
]
```

## Extending the Model

### Adding New Leagues

1. Add league mapping in `app/config.py`:

   ```python
   LEAGUE_NAME_TO_CODE_MAPPING = {
       "New League": "NL1",
       # ... existing leagues
   }
   ```

2. Ensure corresponding data files exist in `football-data/`

### Customizing the Model

The Davidson model implementation in `scripts/Meta_pairwise.py` can be modified:

- **Learning Rate**: Adjust `learning_rate` in `update_strengths_elo_style`
- **Initial Seasons**: Change `DEFAULT_INIT_SEASONS_COUNT` for parameter fitting
- **Constraints**: Modify bounds and constraints in `fit_initial_davidson_params`

### Model Retraining

```python
# Force retrain all models
trained_models = train_all_davidson_models(force_retrain=True)

# Or delete the model file to trigger retraining
# rm models/davidson_model_data.joblib
```

## Development

### Using uv

```bash
# Install dependencies
uv sync

# Add new dependency
uv add scikit-learn

# Run in development
uv run uvicorn main:app --reload

```

### Environment Variables

Create a `.env` file:

```bash
API_KEY=your-secure-api-key-here
ENVIRONMENT=development
```

## Contributing

This is an educational example. Feel free to:

1. **Fork** the repository
2. **Experiment** with different model parameters
3. **Add** new features or models
4. **Share** your improvements

## License

MIT License - see LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{davidson_model_example,
  title={Football Match Prediction API - Davidson Model Example},
  author={Your Name},
  year={2024},
  url={https://github.com/score-technologies/predict-template}
}
```

## Further Reading

- Davidson, R. R. (1970). On extending the Bradley-Terry model to accommodate ties in paired comparison experiments
- Elo rating system applications in sports
- Maximum likelihood estimation for sports prediction models
