import os

# App Settings
SESSION_DURATION = 60  # seconds
LOG_DIR = "logs"
LOG_FILE = "stress_tracker.log"

# AI Settings
OLLAMA_MODEL = "llama3.2"

# Data Settings
DATA_DIR = "data"
BASELINE_FILE = os.path.join(DATA_DIR, "baseline.json")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
