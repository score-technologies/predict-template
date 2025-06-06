#!/bin/bash

# Davidson Model API Test Script
# This script demonstrates how to use the Davidson model prediction API

API_BASE_URL="http://localhost:8000/api/v1"
API_KEY="davidson-model-example-key"

echo "🏈 Davidson Model Prediction API Test"
echo "====================================="

# Test 1: Health Check (no auth required)
echo ""
echo "1. Testing Health Check (no authentication required)..."
curl -X GET "${API_BASE_URL}/health" \
  -H "Content-Type: application/json" \
  | python -m json.tool

echo ""
echo "====================================="

# Test 2: Prediction with Authentication (two games)
echo ""
echo "2. Testing Davidson Model Predictions with Authentication..."

# Create JSON payload with two games from supported leagues
PAYLOAD='[
  {
    "challengeId": "test-game-001",
    "homeTeam": "Man United",
    "awayTeam": "Liverpool",
    "league": "Premier League",
    "venue": "Old Trafford",
    "fixtureId": 12345,
    "kickoffTime": "2024-01-15T15:00:00Z",
    "challengePhaseMinutes": 90,
    "targetMarket": "1X2",
    "phaseIdentifier": "FT",
    "difficulty": 0.7
  },
  {
    "challengeId": "test-game-002", 
    "homeTeam": "Barcelona",
    "awayTeam": "Real Madrid",
    "league": "La Liga",
    "venue": "Camp Nou",
    "fixtureId": 67890,
    "kickoffTime": "2024-01-15T20:00:00Z",
    "challengePhaseMinutes": 90,
    "targetMarket": "1X2",
    "phaseIdentifier": "FT",
    "difficulty": 0.8
  }
]'

curl -X POST "${API_BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "${PAYLOAD}" \
  | python -m json.tool

echo ""
echo "====================================="

# Test 3: Prediction without Authentication (should fail)
echo ""
echo "3. Testing Predictions without Authentication (should fail)..."

curl -X POST "${API_BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  | python -m json.tool

echo ""
echo "====================================="
echo "✅ Test completed!"
echo ""
echo "Expected results:"
echo "- Health check: Should show Davidson model status and available leagues"
echo "- Authenticated prediction: Should return Davidson model predictions for both games"
echo "- Unauthenticated prediction: Should return 'Not authenticated' error"
echo ""
echo "Note: If teams are not in the training data, predictions may fail with 422 error" 