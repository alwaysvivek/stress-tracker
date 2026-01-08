from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Stress Mouse Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Stress Mouse Tracker API is running"}

class MouseData(BaseModel):
    x: float
    y: float
    timestamp: float

class SessionData(BaseModel):
    movements: List[MouseData]
    keystrokes: List[Dict[str, Any]] = []

import os
import json
import time

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

from features import FeatureExtractor

@app.post("/submit-session")
def submit_session(data: SessionData):
    session_id = str(int(time.time()))
    filename = os.path.join(DATA_DIR, f"session_{session_id}.json")
    
    # Dump raw data
    with open(filename, "w") as f:
        json.dump(data.model_dump(), f)
    
    # Extract features
    mouse_feats = FeatureExtractor.extract_mouse_features([m.model_dump() for m in data.movements])
    key_feats = FeatureExtractor.extract_keystroke_features(data.keystrokes)
    
    # Heuristic Stress Score (Prototype)
    # High velocity + high jitter (accel std) might indicate stress/frustration
    # Fast typing with erratic flight times might indicate stress
    
    stress_score = 0.5 # Neutral baseline
    if mouse_feats:
        # Arbitrary thresholds for demo
        if mouse_feats.get('mouse_acc_std', 0) > 0.5: 
            stress_score += 0.2
        if mouse_feats.get('mouse_vel_mean', 0) > 2.0:
             stress_score += 0.1
             
    if key_feats:
        # Fast typing but consistent? Or erratic?
        if key_feats.get('key_flight_std', 0) > 100: # High variance in typing rhythm
             stress_score += 0.2
             
    stress_score = min(max(stress_score, 0.0), 1.0)
    
    return {
        "status": "received", 
        "data_points": len(data.movements) + len(data.keystrokes), 
        "filename": filename,
        "features": {**mouse_feats, **key_feats},
        "stress_score": stress_score
    }
