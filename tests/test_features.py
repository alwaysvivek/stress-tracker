import unittest
import numpy as np
from core.features import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    
    def test_mouse_features_basic(self):
        # 1. Simulate 10px movement in 1 second
        movements = [
            {"x": 0, "y": 0, "timestamp": 1000},
            {"x": 10, "y": 0, "timestamp": 1001}
        ]
        
        feats = FeatureExtractor.extract_mouse_features(movements)
        
        # Velocity = Dist / Time = 10 / 1 = 10
        self.assertAlmostEqual(feats['mouse_vel_mean'], 10.0, places=2)
        self.assertAlmostEqual(feats['mouse_total_dist'], 10.0, places=2)

    def test_mouse_features_empty(self):
        feats = FeatureExtractor.extract_mouse_features([])
        self.assertEqual(feats, {})

    def test_keystroke_flight_time(self):
        # 2. Simulate 0.5s between keypresses
        keystrokes = [
            {"key": "a", "hold_time": 0.1, "timestamp": 1000},
            {"key": "b", "hold_time": 0.1, "timestamp": 1000.5},
            {"key": "c", "hold_time": 0.1, "timestamp": 1001.0}
        ]
        
        feats = FeatureExtractor.extract_keystroke_features(keystrokes)
        
        # Flight times: [0.5, 0.5]
        # Mean = 0.5
        self.assertAlmostEqual(feats['key_flight_mean'], 0.5, places=2)
        self.assertAlmostEqual(feats['key_flight_std'], 0.0, places=2)

    def test_keystroke_errors(self):
        keystrokes = [
            {"key": "a", "timestamp": 1},
            {"key": "Backspace", "timestamp": 2},
            {"key": "b", "timestamp": 3},
            {"key": "Backspace", "timestamp": 4}
        ]
        
        feats = FeatureExtractor.extract_keystroke_features(keystrokes)
        # 2 backspaces out of 4 keys = 0.5 error rate
        self.assertEqual(feats['key_backspace_count'], 2)
        self.assertEqual(feats['key_error_rate'], 0.5)

if __name__ == '__main__':
    unittest.main()
