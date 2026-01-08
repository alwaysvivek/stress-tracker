import numpy as np
import pandas as pd
from typing import List, Dict, Any

class FeatureExtractor:
    @staticmethod
    def extract_mouse_features(movements: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(movements) < 2:
            return {}
        
        df = pd.DataFrame(movements)
        df = df.sort_values('timestamp')
        
        # Calculate deltas
        df['dt'] = df['timestamp'].diff().fillna(0)
        df['dx'] = df['x'].diff().fillna(0)
        df['dy'] = df['y'].diff().fillna(0)
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        # Filter out zero time differences to avoid div by zero
        valid = df['dt'] > 0
        
        velocities = df.loc[valid, 'dist'] / df.loc[valid, 'dt']
        
        # Acceleration
        accels = velocities.diff().fillna(0) / df.loc[valid, 'dt']
        
        # Jitter / Tremor (Movement Efficiency)
        # Calculate path length vs displacement (if we had specific tasks)
        # Here we can use "curvature" or "straightness" of segments
        
        # Simple stats
        features = {
            "mouse_vel_mean": float(velocities.mean()),
            "mouse_vel_std": float(velocities.std()),
            "mouse_vel_max": float(velocities.max()),
            "mouse_acc_mean": float(accels.abs().mean()),
            "mouse_acc_std": float(accels.std()),
            "mouse_total_dist": float(df['dist'].sum()),
            "mouse_points": len(df)
        }
        
        return {k: (0.0 if np.isnan(v) else v) for k, v in features.items()}

    @staticmethod
    def extract_keystroke_features(keystrokes: List[Dict[str, Any]]) -> Dict[str, float]:
        if not keystrokes:
            return {}
            
        # Separate Downs and Ups
        downs = [k for k in keystrokes if k['action'] == 'down']
        ups = [k for k in keystrokes if k['action'] == 'up']
        
        # Map by key and timestamp to find pairs? 
        # For simplicity in this raw stream, we rely on the component or just aggregate stats.
        # But wait, providing dwell time needs matching up/down.
        
        # Simpler approach: Calculate flight times (inter-key intervals)
        timestamps = sorted([k['timestamp'] for k in keystrokes if k['action'] == 'down'])
        flight_times = np.diff(timestamps) if len(timestamps) > 1 else []
        
        # Dwell times: match recent down with up
        dwell_times = []
        pending_downs = {} # key -> timestamp
        
        for k in keystrokes:
            if k['action'] == 'down':
                pending_downs[k['key']] = k['timestamp']
            elif k['action'] == 'up':
                if k['key'] in pending_downs:
                    start = pending_downs[k['key']]
                    dwell_times.append(k['timestamp'] - start)
                    del pending_downs[k['key']]
        
        features = {
            "key_dwell_mean": float(np.mean(dwell_times)) if dwell_times else 0.0,
            "key_dwell_std": float(np.std(dwell_times)) if dwell_times else 0.0,
            "key_flight_mean": float(np.mean(flight_times)) if len(flight_times) > 0 else 0.0,
            "key_flight_std": float(np.std(flight_times)) if len(flight_times) > 0 else 0.0,
            "key_cpm": (len(timestamps) / (timestamps[-1] - timestamps[0]) * 60000) if len(timestamps) > 1 else 0.0
        }
        
        return features
