import json
import os
import numpy as np
from typing import Dict, Any, Optional

BASELINE_FILE = "data/baseline.json"

def save_baseline(features: Dict[str, float]):
    with open(BASELINE_FILE, "w") as f:
        json.dump(features, f)

def load_baseline() -> Optional[Dict[str, float]]:
    if not os.path.exists(BASELINE_FILE):
        return None
    try:
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    except:
        return None

def calculate_z_scores(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """
    Returns Z-Score for each feature.
    Z = (Current - Mean) / SD.
    Since we only have a 'single' baseline session (which is just means),
    we must estimate specific feature standard deviations (or store SDs in baseline).
    
    If 'baseline' was a single point (mean), we cannot calculate Z-score without SD.
    Assumption: The 'calibration' session captures the standard deviation of user behavior?
    OR we use heuristic population SDs.
    
    Better approach for v1: store both MEAN and STD from the calibration session.
    But our FeatureExtractor returns single floats (means/stds of that session).
    
    So:
    Baseline = { 'mouse_vel_mean': X, 'mouse_vel_std': Y } (from calibration)
    Current = { 'mouse_vel_mean': A, 'mouse_vel_std': B }
    
    Comparison:
    Z_score_of_mean = (Current Mean - Baseline Mean) / Baseline STD.
    """
    z_scores = {}
    
    # Mapping: Which "STD" feature corresponds to which "Mean" feature
    pairs = {
        'mouse_vel_mean': 'mouse_vel_std',
        'key_flight_mean': 'key_flight_std',
        'key_dwell_mean': 'key_dwell_std'
    }
    
    for metric, std_metric in pairs.items():
        if metric in current and metric in baseline and std_metric in baseline:
            mu = baseline[metric]
            sigma = baseline[std_metric]
            val = current[metric]
            
            if sigma > 0:
                z_scores[f"z_{metric}"] = (val - mu) / sigma
            else:
                z_scores[f"z_{metric}"] = 0.0
                
    # specialized z-scores for scalar features without natural SD in the list
    # e.g. 'mouse_path_efficiency', 'key_error_rate'.
    # We will assume a heuristic SD (e.g. 20% variance) if we don't have historical data.
    
    heuristics = [
        'mouse_path_efficiency', 
        'key_error_rate', 
        'key_backspace_count',
        'mouse_click_latency'
    ]
    
    for h in heuristics:
        if h in current and h in baseline:
            mu = baseline[h]
            val = current[h]
            # Assume 30% of mean as standard deviation for robustness
            sigma = max(mu * 0.3, 0.001) 
            z_scores[f"z_{h}"] = (val - mu) / sigma
            
    return z_scores
