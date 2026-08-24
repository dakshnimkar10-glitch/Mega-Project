import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.data_loader import load_example_dc, load_oxford_dataset
from src.preprocess import extract_soc_features
from src.models.soc_model import SOCEstimator
from src.models.soh_model import SOHEstimator
from src.models.validation_model import BatterySensorValidator

class TestBatteryMLPipeline(unittest.TestCase):
    
    def test_01_data_loader(self):
        ex = load_example_dc('ExampleDC_C1.mat')
        self.assertIn('ch', ex)
        self.assertIn('dc', ex)
        self.assertGreater(len(ex['ch']), 0)
        self.assertGreater(len(ex['dc']), 0)
        self.assertIn('soc', ex['ch'].columns)
        self.assertIn('soc', ex['dc'].columns)
        
    def test_02_feature_extraction(self):
        ex = load_example_dc('ExampleDC_C1.mat')
        X, y, _ = extract_soc_features(ex['dc'])
        self.assertEqual(len(X), len(ex['dc']))
        self.assertIn('voltage_v', X.columns)
        self.assertIn('current_ma', X.columns)
        self.assertIn('temp_c', X.columns)
        
    def test_03_sensor_validator_physical_rules(self):
        validator = BatterySensorValidator()
        
        # Valid test sample (3.8V, 740mA, 40.0C)
        res_valid = validator.validate_sample(3.8, 740.0, 40.0)
        self.assertTrue(res_valid['is_valid'])
        self.assertEqual(res_valid['status'], 'RIGHT')
        self.assertEqual(len(res_valid['faults']), 0)
        
        # Invalid Over-voltage test sample (4.8V)
        res_high_v = validator.validate_sample(4.8, 740.0, 40.0)
        self.assertFalse(res_high_v['is_valid'])
        self.assertEqual(res_high_v['status'], 'NOT RIGHT')
        self.assertTrue(any('Over-voltage' in f for f in res_high_v['faults']))
        
        # Invalid Over-temperature test sample (65C)
        res_high_t = validator.validate_sample(3.8, 740.0, 65.0)
        self.assertFalse(res_high_t['is_valid'])
        self.assertEqual(res_high_t['status'], 'NOT RIGHT')
        self.assertTrue(any('Over-temperature' in f for f in res_high_t['faults']))

    def test_04_model_instantiation(self):
        soc_model = SOCEstimator(n_estimators=10)
        soh_model = SOHEstimator(n_estimators=10)
        self.assertIsNotNone(soc_model)
        self.assertIsNotNone(soh_model)

if __name__ == '__main__':
    unittest.main()
