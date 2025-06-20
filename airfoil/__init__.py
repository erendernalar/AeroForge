"""
AeroForge Airfoil Analysis Module
Provides 2D airfoil analysis using XFOIL integration
"""

from .xfoil_driver import XfoilDriver
from .airfoil_manager import AirfoilManager

__all__ = ['XfoilDriver', 'AirfoilManager']