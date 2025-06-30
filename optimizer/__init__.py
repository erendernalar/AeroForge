"""
AeroForge Optimization Module
Provides design optimization and evaluation capabilities
"""

from .evaluator import DesignEvaluator
from .design_loop import AeroForgeOptimizer as DesignLoop  # 👈 Alias here

__all__ = ['DesignEvaluator', 'DesignLoop']


