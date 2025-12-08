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
    def extract_mouse_efficiency(movements: List[Dict[str, Any]]) -> float:
        """
        Calculates Straightness Index (Euclidean Dist / Actual Path Dist).
        Ideal = 1.0. High Stress/Hesitation < 1.0 (more curved).
        Note: Research often inverts this (Actual/Euclidean). Let's use Actual/Euclidean -> Ideal 1.0, Jitter > 1.0.
        """
        if len(movements) < 4:
            return 1.0
        
        df = pd.DataFrame(movements)
        df['dx'] = df['x'].diff().fillna(0)
        df['dy'] = df['y'].diff().fillna(0)
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        actual_dist = df['dist'].sum()
        
        # Euclidean distance from start to end (simplified - assumes one continuous task)
        # Better: Sum of euclidean distances of 'strokes' if we could segment.
        # For continuous monitoring, this might be noisy. 
        # Let's try to calculate it over sliding windows or segments ideally.
        # Simplified approach: Start point to End point of the whole session is WRONG.
        # Correct approach for Unsupervised: Sum of segment straightness?
        # Let's implement 'Tortuosity': Sum of angles?
        # Reverting to the Epp et al. proxy: "Movement Efficiency" = Sum(Dist) / Sum(Displacement of sub-segments).
        # We'll take 1-second windows to simulate "sub-tasks".
        
        df['time_sec'] = df['timestamp'].astype(int)
        efficiencies = []
        
        for t in df['time_sec'].unique():
            seg = df[df['time_sec'] == t]
            if len(seg) < 2: continue
            
            seg_actual = seg['dist'].sum()
            seg_dx = seg.iloc[-1]['x'] - seg.iloc[0]['x']
            seg_dy = seg.iloc[-1]['y'] - seg.iloc[0]['y']
            seg_euclidean = np.sqrt(seg_dx**2 + seg_dy**2)
            
            if seg_euclidean > 5: # Ignore micro-movements
                efficiencies.append(seg_actual / seg_euclidean)
                
        if not efficiencies:
            return 1.0
            
        return float(np.mean(efficiencies))

    @staticmethod
    def extract_click_latency(movements: List[Dict[str, Any]], clicks: List[Dict[str, Any]]) -> float:
        """
        Time between the last mouse movement and the click.
        High latency + errors = "Choking".
        """
        if not clicks or not movements:
            return 0.0
            
        latencies = []
        move_times = sorted([m['timestamp'] for m in movements])
        
        for c in clicks:
            ct = c['timestamp']
            # Find last movement before this click
            # Efficient way: bisect_right
            import bisect
            idx = bisect.bisect_right(move_times, ct)
            if idx > 0:
                last_move_time = move_times[idx-1]
                lat = (ct - last_move_time) * 1000 # ms
                if lat < 2000: # Filter outliers > 2s (user probably thinking)
                    latencies.append(lat)
                    
        if not latencies:
            return 0.0
            
        return float(np.mean(latencies))

    @staticmethod
    def extract_keystroke_features(keystrokes: List[Dict[str, Any]]) -> Dict[str, float]:
        if not keystrokes:
            return {}
            
        # Check format: simple (background tracker) vs raw (frontend)
        is_raw = 'action' in keystrokes[0]
        
        dwell_times = []
        flight_times = []
        timestamps = []
        
        if is_raw:
            # Frontend format: raw 'down' and 'up' events
            timestamps = sorted([k['timestamp'] for k in keystrokes if k['action'] == 'down'])
            flight_times = np.diff(timestamps) if len(timestamps) > 1 else []
            
            pending_downs = {} # key -> timestamp
            for k in keystrokes:
                if k['action'] == 'down':
                    pending_downs[k['key']] = k['timestamp']
                elif k['action'] == 'up':
                    if k['key'] in pending_downs:
                        start = pending_downs[k['key']]
                        dwell_times.append(k['timestamp'] - start)
                        del pending_downs[k['key']]
        else:
            # Background tracker format: pre-processed 'hold_time'
            # We treat each event as a 'down' for flight time purposes
            timestamps = sorted([k['timestamp'] for k in keystrokes])
            flight_times = np.diff(timestamps) if len(timestamps) > 1 else []
            
            # Dwell times are directly available
            dwell_times = [k['hold_time'] for k in keystrokes if 'hold_time' in k]

        # Error Rate (Backspace/Delete usage)
        special_keys = {'Backspace', 'Delete'}
        backspace_count = sum(1 for k in keystrokes if k['key'] in special_keys)
        total_keys = len(keystrokes)
        error_rate = backspace_count / total_keys if total_keys > 0 else 0.0

        features = {
            "key_dwell_mean": float(np.mean(dwell_times)) if dwell_times else 0.0,
            "key_dwell_std": float(np.std(dwell_times)) if dwell_times else 0.0,
            "key_flight_mean": float(np.mean(flight_times)) if len(flight_times) > 0 else 0.0,
            "key_flight_std": float(np.std(flight_times)) if len(flight_times) > 0 else 0.0,
            "key_cpm": (len(timestamps) / (timestamps[-1] - timestamps[0]) * 60000) if len(timestamps) > 1 else 0.0,
            "key_error_rate": float(error_rate),
            "key_backspace_count": backspace_count
        }
        
        return features
