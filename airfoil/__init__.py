"""
AeroForge Airfoil Analysis Module
Provides 2D airfoil analysis using XFOIL integration
"""

from .xfoil_driver import XFOILDriver
from .airfoil_manager import AirfoilManager

__all__ = ['XFOILDriver', 'AirfoilManager']

