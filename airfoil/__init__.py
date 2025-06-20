"""
AeroForge Airfoil Analysis Module
Handles 2D airfoil analysis and management
"""

from .xfoil_driver import XFOILDriver
from .airfoil_manager import AirfoilManager

__all__ = ['XFOILDriver', 'AirfoilManager']
