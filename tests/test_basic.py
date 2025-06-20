"""
Basic tests for AeroForge modules
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airfoil.airfoil_manager import AirfoilManager
from wing.llt_solver import LiftingLineTheory, WingSection
from optimizer.evaluator import DesignParameters, DesignEvaluator

class TestBasicFunctionality(unittest.TestCase):
    
    def setUp(self):
        self.airfoil_mgr = AirfoilManager()
        self.llt = LiftingLineTheory()
    
    def test_naca_generation(self):
        """Test NACA airfoil generation"""
        coords = self.airfoil_mgr.create_naca_airfoil("2412")
        self.assertGreater(len(coords), 50)
        self.assertEqual(coords.shape[1], 2)
    
    def test_design_parameters(self):
        """Test design parameter creation"""
        design = DesignParameters(
            span=2.0,
            root_chord=0.3,
            tip_chord=0.2,
            cruise_speed=25.0
        )
        self.assertEqual(design.span, 2.0)
        self.assertGreater(design.wing_area, 0)
    
    def test_llt_solver(self):
        """Test LLT solver basic functionality"""
        sections = [
            WingSection(y=0.0, chord=0.3, twist=0.0, airfoil_data={}),
            WingSection(y=1.0, chord=0.2, twist=-2.0, airfoil_data={})
        ]
        
        flight_condition = {
            'alpha': 5.0,
            'velocity': 20.0,
            'density': 1.225
        }
        
        results = self.llt.analyze_wing(sections, flight_condition)
        self.assertIn('CL', results)
        self.assertIn('CDi', results)

if __name__ == '__main__':
    unittest.main()
