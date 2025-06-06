import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
import joblib # Added for saving/loading model data
from typing import Dict, Any, Optional, List, Tuple, Union

# --- Constants ---
MODEL_DATA_FILENAME = "davidson_model_data.joblib"
DEFAULT_INIT_SEASONS_COUNT = 1 # Number of initial seasons to use for parameter fitting

# --- Data Loading (existing) ---
def load_data() -> Dict[str, pd.DataFrame]:
    leagues = ['E0', 'E1', 'E2', 'E3', 'SC0', 'SC1', 'SC2', 'SC3',
               'D1', 'D2', 'I1', 'I2', 'SP1', 'SP2', 'F1', 'F2',
               'N1', 'B1', 'P1', 'T1', 'G1']
    # Limiting seasons for faster example processing, adjust as needed
    seasons = [str(1718 + 101 * i) for i in range(3)] # Reduced from 7 for speed
    data = {}
    
    # Find the project root directory (where football-data should be)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from scripts/ to project root
    base_path = project_root / "football-data"
    
    print(f"Looking for football data in: {base_path}")
    
    for league in leagues:
        for season in seasons:
            file = base_path / f'{league}_{season}.csv'
            if file.exists():
                try:
                    data[f'{league}_{season}'] = pd.read_csv(file, parse_dates=['Date'], dayfirst=True, encoding='ISO-8859-1')
                except Exception as e:
                    print(f"Error reading {file}: {e}. Trying with 'latin1'.")
                    try:
                        data[f'{league}_{season}'] = pd.read_csv(file, parse_dates=['Date'], dayfirst=True, encoding='latin1')
                    except Exception as e_latin1:
                         print(f"Error reading {file} with 'latin1': {e_latin1}. Skipping.")
            else:
                print(f'Skipping missing file {file}')
    return data

# --- Model Persistence ---
def save_model_data(data: Dict[str, Any], file_path: Path = None) -> None:
    if file_path is None:
        # Find the project root and use models directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        file_path = project_root / "models" / MODEL_DATA_FILENAME
    
    # Ensure the models directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model data to {file_path}...")
    joblib.dump(data, file_path)
    print("Model data saved.")

def load_model_data(file_path: Path = None) -> Optional[Dict[str, Any]]:
    if file_path is None:
        # Find the project root and use models directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        file_path = project_root / "models" / MODEL_DATA_FILENAME
    
    if not file_path.exists():
        print(f"Model data file {file_path} not found.")
        return None
    print(f"Loading model data from {file_path}...")
    data = joblib.load(file_path)
    print("Model data loaded.")
    return data

# --- Core Davidson Model Utilities (mostly existing, slightly adapted) ---
def davidson_prob(h_strength_eff: Union[float, np.ndarray], 
                  a_strength_eff: Union[float, np.ndarray], 
                  draw_param: float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute 1X2 probabilities under Davidson model.
    h_strength_eff: Effective home strength (e.g., strength_h + home_adv). Can be scalar or array.
    a_strength_eff: Effective away strength. Can be scalar or array.
    draw_param: Draw affinity parameter (scalar).
    """
    
    h = np.exp(np.clip(h_strength_eff, -20, 20))
    a = np.exp(np.clip(a_strength_eff, -20, 20))

    denom = h + a + 2 * draw_param * np.sqrt(h * a)
    denom = np.maximum(denom, 1e-10) 

    pH = h / denom
    pA = a / denom
    # Initial calculation of pD for consistency.
    # This pD, along with pH and pA, will be used to check if normalization is needed.
    pD_initial = 1.0 - pH - pA

    # --- Normalization to ensure probabilities sum to 1.0 robustly ---
    sum_probs = pH + pA + pD_initial

    # Default to original pH, pA, and consistently calculated pD
    pH_final = pH
    pA_final = pA
    pD_final = pD_initial

    if np.isscalar(sum_probs):
        if not np.isclose(sum_probs, 1.0) and sum_probs != 0.0:
            # If sum_probs is not 1.0 and not 0, normalize
            pH_final = pH / sum_probs
            pA_final = pA / sum_probs
            # pD is recalculated to ensure the sum is exactly 1.0
            pD_final = 1.0 - pH_final - pA_final
    else: # sum_probs is an array
        # Identify elements where sum_probs is not close to 1.0 AND sum_probs is not zero
        needs_normalization_mask = ~np.isclose(sum_probs, 1.0) & (sum_probs != 0.0)

        # Ensure pH_final, pA_final, pD_final are arrays if input was array, to allow masked assignment
        if not isinstance(pH_final, np.ndarray) and np.any(needs_normalization_mask): pH_final = np.array(pH_final, dtype=float)
        if not isinstance(pA_final, np.ndarray) and np.any(needs_normalization_mask): pA_final = np.array(pA_final, dtype=float)
        if not isinstance(pD_final, np.ndarray) and np.any(needs_normalization_mask): pD_final = np.array(pD_final, dtype=float)


        if np.any(needs_normalization_mask):
            sum_probs_selected = sum_probs[needs_normalization_mask]
            
            pH_final[needs_normalization_mask] = pH[needs_normalization_mask] / sum_probs_selected
            pA_final[needs_normalization_mask] = pA[needs_normalization_mask] / sum_probs_selected
            # Recalculate pD for the normalized elements specifically
            pD_final[needs_normalization_mask] = 1.0 - pH_final[needs_normalization_mask] - pA_final[needs_normalization_mask]
        
        # For all elements in the array case, ensure pD makes the sum 1,
        # even for those initially close to 1, to correct minor floating point issues.
        # This must be done after potential normalization of masked elements.
        if isinstance(pH_final, np.ndarray): # Check if it's an array before trying to assign pD broadly
             pD_final = 1.0 - pH_final - pA_final
            
    return pH_final, pD_final, pA_final

def davidson_loss(params: np.ndarray, h_idx: np.ndarray, a_idx: np.ndarray, targets: np.ndarray, n_teams: int) -> float:
    strengths = params[:n_teams]
    home_adv = params[n_teams] # Adjusted index
    draw_param = params[n_teams+1] # Adjusted index
    
    # Clip values to prevent overflow in exp
    eff_h_strength = np.clip(strengths[h_idx] + home_adv, -20, 20)
    eff_a_strength = np.clip(strengths[a_idx], -20, 20)

    # Calculate probabilities using exponentiated strengths internally in davidson_prob
    # Here we pass the linear strengths, davidson_prob will handle exp
    pH, pD, pA = davidson_prob(eff_h_strength, eff_a_strength, draw_param)
    
    probs = np.vstack([pH, pD, pA]).T
    pt = probs[np.arange(len(targets)), targets]
    pt = np.clip(pt, 1e-10, 1.0) # Clip probabilities to avoid log(0)
    return -np.sum(np.log(pt))

def fit_initial_davidson_params(h_idx: np.ndarray, a_idx: np.ndarray, targets: np.ndarray, n_teams: int) -> Tuple[np.ndarray, float, float]:
    params0 = np.zeros(n_teams + 2)
    params0[n_teams] = 0.1  # Initial guess for home advantage
    params0[n_teams+1] = 0.3  # Initial guess for draw parameter (gamma in some literature)

    bounds = [(None, None)] * n_teams + [(None, None), (1e-6, None)] # Strengths, HA, Draw Param > 0
    cons = ({'type': 'eq', 'fun': lambda p: np.sum(p[:n_teams])}) # Sum of strengths = 0 for identifiability
    
    res = minimize(
        davidson_loss,
        params0,
        args=(h_idx, a_idx, targets, n_teams),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'disp': False} # Changed disp to False
    )
    if not res.success:
        print(f"Warning: Initial parameter fitting did not converge: {res.message}")

    fitted_strengths = res.x[:n_teams]
    fitted_home_adv = res.x[n_teams]
    fitted_draw_param = res.x[n_teams+1]
    return fitted_strengths, fitted_home_adv, fitted_draw_param

def update_strengths_elo_style(strengths: np.ndarray, h_idx: int, a_idx: int, result: int, pH: float, pD: float, learning_rate: float = 0.05) -> np.ndarray:
    if result == 0: S = 1.0  # Home win
    elif result == 1: S = 0.5  # Draw
    else: S = 0.0  # Away win

    E = pH + 0.5 * pD # Expected score for home team
    delta = learning_rate * (S - E)

    strengths[h_idx] += delta
    strengths[a_idx] -= delta
    # Optional: Re-normalize strengths to sum to 0 if desired, though not strictly necessary if only diff matters for davidson_prob
    # strengths -= np.mean(strengths) 
    return strengths

# --- Training Functions ---
def _train_single_league_model(league_df: pd.DataFrame, init_seasons_count: int = DEFAULT_INIT_SEASONS_COUNT) -> Optional[Dict[str, Any]]:
    if league_df.empty:
        print("Empty DataFrame provided for single league training. Skipping.")
        return None

    league_df = league_df.sort_values('Date').reset_index(drop=True)
    
    # Prepare team encoding for this league
    unique_teams = pd.unique(league_df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    if len(unique_teams) < 2:
        print(f"Not enough unique teams ({len(unique_teams)}) in league data. Skipping.")
        return None
        
    encoder = LabelEncoder().fit(unique_teams)
    league_df['H_ID'] = encoder.transform(league_df['HomeTeam'])
    league_df['A_ID'] = encoder.transform(league_df['AwayTeam'])
    league_df['target'] = league_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Drop rows where target is NaN (e.g. if FTR was invalid)
    league_df = league_df.dropna(subset=['H_ID', 'A_ID', 'target'])
    league_df['target'] = league_df['target'].astype(int)


    n_teams = len(encoder.classes_)
    
    # Determine split point for initial fit vs. updates
    # This requires 'Season' column to be present in league_df
    if 'Season' not in league_df.columns:
        print("Error: 'Season' column not found in league_df. Cannot split for training.")
        # Fallback: use first 20% of matches for init, rest for update
        # This is a heuristic if 'Season' is missing. Proper season handling is better.
        # For this to work, `league_df` must come from `train_all_davidson_models` where 'Season' is added.
        # If 'Season' is guaranteed, this fallback is not strictly needed.
        # For now, let's assume 'Season' is present as per `train_all_davidson_models` structure.
        print("Attempting to use 'Season' column for splitting.")
        
    available_seasons = sorted(league_df['Season'].unique())
    if len(available_seasons) < 1:
        print("Not enough seasons available in the league data. Skipping.")
        return None

    init_fit_seasons = set(available_seasons[:init_seasons_count])
    
    df_init = league_df[league_df['Season'].isin(init_fit_seasons)]
    df_update = league_df[~league_df['Season'].isin(init_fit_seasons)]

    if df_init.empty:
        print(f"No data for initial parameter fitting after season split (init_seasons_count={init_seasons_count}). Skipping league.")
        return None

    # 1. Initial Parameter Fit
    h_init, a_init, y_init = df_init['H_ID'].values, df_init['A_ID'].values, df_init['target'].values
    
    current_strengths, home_adv, draw_param = fit_initial_davidson_params(h_init, a_init, y_init, n_teams)
    print(f"  Initial fit: HA={home_adv:.4f}, DrawP={draw_param:.4f}")

    # 2. Iterative Updates
    if not df_update.empty:
        print(f"  Updating strengths with {len(df_update)} matches...")
        for _, row in df_update.iterrows():
            h_id, a_id, target_outcome = int(row['H_ID']), int(row['A_ID']), int(row['target'])
            
            eff_h_strength = current_strengths[h_id] + home_adv
            eff_a_strength = current_strengths[a_id]
            
            pH, pD, _ = davidson_prob(eff_h_strength, eff_a_strength, draw_param)
            
            current_strengths = update_strengths_elo_style(current_strengths, h_id, a_id, target_outcome, pH, pD)
    else:
        print("  No matches available for iterative updates after initial fit.")
        
    return {
        'strengths': current_strengths,
        'home_advantage': home_adv,
        'draw_parameter': draw_param,
        'encoder': encoder,
        'team_names': list(encoder.classes_) # Store for convenience
    }

def train_all_davidson_models(force_retrain: bool = False) -> Dict[str, Any]:
    # Use the correct path resolution
    if not force_retrain:
        loaded_data = load_model_data()  # This will use the correct path
        if loaded_data is not None:
            return loaded_data
    
    print("Starting training for all Davidson models...")
    raw_data_by_league_season = load_data()
    if not raw_data_by_league_season:
        print("No data loaded. Cannot train models.")
        return {}

    all_leagues_model_params: Dict[str, Any] = {}
    
    # Group data by league code first
    league_dfs: Dict[str, List[pd.DataFrame]] = {}
    for league_season_key, df_season in raw_data_by_league_season.items():
        league_code = league_season_key.split('_')[0]
        season_code = league_season_key.split('_')[1]
        df_season_copy = df_season.copy() # Avoid SettingWithCopyWarning
        df_season_copy['Season'] = season_code # Add Season column
        
        # Ensure required columns exist and drop NaNs early for FTR
        required_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'Date']
        if not all(col in df_season_copy.columns for col in required_cols):
            print(f"League-season {league_season_key} missing one of required columns: {required_cols}. Skipping.")
            continue
        df_season_copy = df_season_copy.dropna(subset=['FTR'])


        if league_code not in league_dfs:
            league_dfs[league_code] = []
        league_dfs[league_code].append(df_season_copy)

    for league_code, dfs_for_league_seasons in league_dfs.items():
        if not dfs_for_league_seasons:
            continue
        
        print(f"Training model for league: {league_code}")
        # Concatenate all seasons for the current league
        full_league_df = pd.concat(dfs_for_league_seasons, ignore_index=True)
        full_league_df = full_league_df.sort_values('Date').reset_index(drop=True)
        
        # Filter out matches with NaN FTR, HomeTeam, AwayTeam (should be done earlier too)
        full_league_df = full_league_df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR', 'Date'])
        if full_league_df.empty:
            print(f"  No valid matches for league {league_code} after cleaning. Skipping.")
            continue

        league_model_trained_data = _train_single_league_model(full_league_df, init_seasons_count=DEFAULT_INIT_SEASONS_COUNT)
        
        if league_model_trained_data:
            all_leagues_model_params[league_code] = league_model_trained_data
            print(f"  Successfully trained model for league: {league_code}")
        else:
            print(f"  Failed to train model for league: {league_code}")
            
    if all_leagues_model_params:
        save_model_data(all_leagues_model_params)  # This will use the correct path
    else:
        print("No models were trained successfully. No data saved.")
        
    return all_leagues_model_params

# --- Prediction Function ---
def predict_davidson_match(
    home_team_name: str, 
    away_team_name: str, 
    league_code: str, 
    all_trained_models: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    
    if league_code not in all_trained_models:
        print(f"No trained model found for league_code: {league_code}")
        return None
    
    league_model_data = all_trained_models[league_code]
    encoder: LabelEncoder = league_model_data['encoder']
    strengths: np.ndarray = league_model_data['strengths']
    home_adv: float = league_model_data['home_advantage']
    draw_param: float = league_model_data['draw_parameter']
    
    try:
        home_team_id = encoder.transform([home_team_name])[0]
        away_team_id = encoder.transform([away_team_name])[0]
    except ValueError:
        # One or both teams not found in the encoder for this league
        unknown_teams = []
        if home_team_name not in encoder.classes_: unknown_teams.append(home_team_name)
        if away_team_name not in encoder.classes_: unknown_teams.append(away_team_name)
        print(f"Cannot predict: Team(s) {unknown_teams} not found in trained model for league {league_code}.")
        return None

    eff_home_strength = strengths[home_team_id] + home_adv
    eff_away_strength = strengths[away_team_id]
    
    pH, pD, pA = davidson_prob(eff_home_strength, eff_away_strength, draw_param)
    
    return {"home": pH, "draw": pD, "away": pA}


# --- Main execution for testing ---
if __name__ == '__main__':
    print("Attempting to train Davidson models (force_retrain=True for this example run)...")
    # Ensure football-data directory is present and populated relative to this script's location.
    # The load_data() function assumes 'football-data/*.csv'.
    
    # First, run training. force_retrain=True will make it always retrain.
    # In a real scenario, you might set force_retrain=False or use other logic.
    trained_models = train_all_davidson_models(force_retrain=True)
    
    if trained_models:
        print("\nSuccessfully trained and/or loaded models.")
        print(f"Trained models for leagues: {list(trained_models.keys())}")

        # Example prediction (replace with actual team names and league codes from your trained model)
        # To find valid teams/leagues, you can inspect the 'trained_models' dictionary
        # For example, if 'E0' (English Premier League) was trained:
        if 'E0' in trained_models and len(trained_models['E0']['team_names']) >= 2:
            sample_league = 'E0'
            team1 = trained_models[sample_league]['team_names'][0]
            team2 = trained_models[sample_league]['team_names'][1]
            
            print(f"\nExample prediction for a match in league '{sample_league}': {team1} vs {team2}")
            prediction = predict_davidson_match(team1, team2, sample_league, trained_models)
            
            if prediction:
                print(f"  Predicted probabilities: Home={prediction['home']:.4f}, Draw={prediction['draw']:.4f}, Away={prediction['away']:.4f}")
            else:
                print("  Could not make an example prediction (e.g., teams not found after training).")
        else:
            print("\nSkipping example prediction as league 'E0' not trained or has insufficient teams.")
            
        # Example of unknown team prediction
        print("\nExample prediction for an unknown team:")
        unknown_team_prediction = predict_davidson_match("Unknown Team FC", "Another Unknown FC", "E0", trained_models)
        if not unknown_team_prediction:
            print("  Correctly failed to predict for unknown team (as expected).")

    else:
        print("\nNo models were trained or loaded. Check data and paths.")

    # The old elo() and davidson() functions are removed as their functionality is now part of the new structure.
    # Keep other utility functions if they are general or used by other models not refactored yet.
    # For example, elo_update, build_ratings, etc., related to the Elo model are still here but not directly used
    # by the new Davidson training/prediction flow unless explicitly integrated.