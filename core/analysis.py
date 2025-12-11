from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import time
import logging
from .features import FeatureExtractor
from .agent import StressManagementAgent
from . import utils
from . import config

# Configure Logging
logger = logging.getLogger("StressTrackerAnalysis")
logging.basicConfig(level=logging.INFO)

# --- FastAPI Setup ---
app = FastAPI(title="Stress Mouse Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MouseData(BaseModel):
    x: float
    y: float
    timestamp: float

class SessionData(BaseModel):
    movements: List[MouseData]
    clicks: List[Dict[str, Any]] = []
    keystrokes: List[Dict[str, Any]] = []
    analyze_with_llm: bool = False

# --- Core Logic Initialization ---
logger.info("Initializing StressManagementAgent...")
try:
    stress_agent = StressManagementAgent(model_name=config.OLLAMA_MODEL)
    logger.info("Agent initialized.")
except Exception as e:
    logger.error(f"Failed to initialize AI Agent: {e}")
    stress_agent = None

# --- Shared Logic Functions ---
def calibrate_session_logic(data: SessionData):
    """
    Saves the current session as the 'Baseline' for this user.
    """
    if not data.movements and not data.keystrokes:
        return {"status": "error", "message": "No data to calibrate"}
        
    mouse_feats = FeatureExtractor.extract_mouse_features([m.model_dump() for m in data.movements])
    key_feats = FeatureExtractor.extract_keystroke_features(data.keystrokes)
    
    # Calculate new research features
    mouse_efficiency = FeatureExtractor.extract_mouse_efficiency([m.model_dump() for m in data.movements])
    click_latency = FeatureExtractor.extract_click_latency([m.model_dump() for m in data.movements], data.clicks)
    
    baseline = {**mouse_feats, **key_feats}
    baseline['mouse_path_efficiency'] = mouse_efficiency
    baseline['mouse_click_latency'] = click_latency
    
    utils.save_baseline(baseline)
    
    return {"status": "calibrated", "baseline": baseline}


def submit_session_logic(data: SessionData):
    session_id = str(int(time.time()))
    filename = os.path.join(config.DATA_DIR, f"session_{session_id}.json")
    
    # Dump raw data
    with open(filename, "w") as f:
        json.dump(data.model_dump(), f)
    
    # Extract features
    mouse_feats = FeatureExtractor.extract_mouse_features([m.model_dump() for m in data.movements])
    key_feats = FeatureExtractor.extract_keystroke_features(data.keystrokes)
    
    # Research Features
    mouse_efficiency = FeatureExtractor.extract_mouse_efficiency([m.model_dump() for m in data.movements])
    click_latency = FeatureExtractor.extract_click_latency([m.model_dump() for m in data.movements], data.clicks)
    
    current_features = {
        **mouse_feats, 
        **key_feats,
        "mouse_path_efficiency": mouse_efficiency,
        "mouse_click_latency": click_latency
    }
    
    # Load Baseline & Calculate Z-Scores
    baseline = utils.load_baseline()
    z_scores = {}
    stress_score = 0.5 # Default neutral
    
    if baseline:
        z_scores = utils.calculate_z_scores(current_features, baseline)
        
        # Calculate Stress Score based on Z-Scores
        z_sum = 0.0
        weights = {
            'z_mouse_acc_std': 0.3, 
            'z_key_flight_std': 0.4, 
            'z_mouse_path_efficiency': 0.3, 
            'z_mouse_click_latency': 0.2
        }
        
        for k, w in weights.items():
            val = z_scores.get(k, 0)
            z_sum += val * w
            
        # Sigmoid to map Z-sum to 0.0-1.0
        import math
        stress_score = 1 / (1 + math.exp(-z_sum))
        
    else:
        # Fallback to Heuristics if no baseline
        if mouse_feats.get('mouse_acc_std', 0) > 0.5: 
            stress_score += 0.2
        if mouse_feats.get('mouse_vel_mean', 0) > 2.0:
            stress_score += 0.1
        if key_feats.get('key_flight_std', 0) > 50:
            stress_score += 0.20
        if key_feats.get('key_error_rate', 0) > 0.05:
            stress_score += 0.15
        stress_score = min(max(stress_score, 0.0), 1.0)
    
    llm_analysis = {}
    if data.analyze_with_llm:
        if stress_agent:
            # Pass Z-scores to Agent
            analysis_input = {**current_features, **z_scores}
            llm_analysis = stress_agent.analyze_session(analysis_input, stress_score)
        else:
            llm_analysis = {
                "stress_level": stress_score,
                "clinical_assessment": "AI Agent unavailable. Please check Ollama installation.",
            }

    return {
        "status": "received", 
        "data_points": len(data.movements) + len(data.keystrokes), 
        "filename": filename,
        "features": current_features,
        "z_scores": z_scores,
        "stress_score": stress_score,
        "llm_analysis": llm_analysis
    }

# --- API Routes ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Stress Mouse Tracker API is running"}

@app.post("/calibrate")
def calibrate_endpoint(data: SessionData):
    return calibrate_session_logic(data)

@app.post("/submit-session")
def submit_session_endpoint(data: SessionData):
    return submit_session_logic(data)
