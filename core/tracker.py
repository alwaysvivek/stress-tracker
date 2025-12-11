import time
import json
import requests
import threading
import logging
import os
from pydantic import BaseModel
from typing import List, Dict, Any
from pynput import mouse, keyboard

from core import config

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, config.LOG_FILE)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SESSION_DURATION = config.SESSION_DURATION

class MouseData(BaseModel):
    x: float
    y: float
    timestamp: float

class SessionData(BaseModel):
    movements: List[MouseData]
    keystrokes: List[Dict[str, Any]] = []
    analyze_with_llm: bool = True

class BackgroundTracker:
    def __init__(self):
        self.mouse_data = []
        self.clicks = []
        self.keystrokes = []
        self.active_keys = {}
        self.running = False
        self.listeners = []

    def on_move(self, x, y):
        if self.running:
            self.mouse_data.append({
                "x": x,
                "y": y,
                "timestamp": time.time()
            })

    def on_click(self, x, y, button, pressed):
        if self.running and pressed:
            self.clicks.append({
                "x": x,
                "y": y,
                "timestamp": time.time(),
                "button": str(button)
            })

    def on_press(self, key):
        if not self.running: return
        key_str = str(key)
        if key_str not in self.active_keys:
            self.active_keys[key_str] = time.time()

    def on_release(self, key):
        if not self.running: return
        key_str = str(key)
        if key_str in self.active_keys:
            press_time = self.active_keys.pop(key_str)
            duration = time.time() - press_time
            self.keystrokes.append({
                "key": key_str,
                "hold_time": duration,
                "timestamp": press_time
            })

    def start(self):
        self.running = True
        self.mouse_data = []
        self.clicks = []
        self.keystrokes = []
        self.active_keys = {}
        
        try:
            # Listeners
            self.mouse_listener = mouse.Listener(
                on_move=self.on_move,
                on_click=self.on_click)
            self.key_listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release)

            self.mouse_listener.start()
            self.key_listener.start()
            self.listeners = [self.mouse_listener, self.key_listener]
            logger.info("Tracker started successfully in background.")
        except Exception as e:
            logger.error(f"Failed to start listeners: {e}")
            self.running = False

    def stop(self):
        self.running = False
        for listener in self.listeners:
            try:
                listener.stop()
            except Exception as e:
                logger.error(f"Error stopping listener: {e}")
        logger.info("Tracker stopped.")
        
        return {
            "movements": self.mouse_data,
            "keystrokes": self.keystrokes,
            "clicks": self.clicks
        }

if __name__ == "__main__":
    tracker = BackgroundTracker()
    tracker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        data = tracker.stop()
        logger.info(f"Captured {len(data['movements'])} movements")
