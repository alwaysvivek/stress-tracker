
import time
import random
import numpy as np
from core.features import FeatureExtractor

def generate_synthetic_data(num_points=100000):
    print(f"Generating {num_points} synthetic mouse events...")
    data = []
    start_time = time.time()
    
    # Vectorized generation for speed in setup, though list comprehension is fine for this scale
    # We simulate a chaotic mouse movement
    for i in range(num_points):
        data.append({
            "x": random.randint(0, 1920),
            "y": random.randint(0, 1080),
            "timestamp": start_time + (i * 0.01) # 10ms intervals -> 100Hz
        })
    return data

def run_benchmark():
    num_events = 100000
    data = generate_synthetic_data(num_events)
    
    print("-" * 40)
    print("Benchmarking FeatureExtractor.extract_mouse_features...")
    
    start = time.perf_counter()
    features = FeatureExtractor.extract_mouse_features(data)
    end = time.perf_counter()
    
    duration = end - start
    events_per_sec = num_events / duration
    
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Throughput: {events_per_sec:,.2f} events/second")
    print("-" * 40)
    
    # Also benchmark efficiency calculation
    print("Benchmarking extract_mouse_efficiency...")
    start_eff = time.perf_counter()
    eff = FeatureExtractor.extract_mouse_efficiency(data)
    end_eff = time.perf_counter()
    duration_eff = end_eff - start_eff
    print(f"Efficiency Calc Time: {duration_eff:.4f} seconds")
    print("-" * 40)
    
    return events_per_sec

if __name__ == "__main__":
    run_benchmark()
