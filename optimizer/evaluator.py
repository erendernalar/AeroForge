"""
AeroForge Design Evaluator and Constraint System
Evaluates aircraft designs and applies constraints
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

@dataclass
class DesignParameters:
    """Container for aircraft design parameters"""
    # Wing geometry
    span: float
    root_chord: float
    tip_chord: float
    sweep: float = 0.0          # Leading edge sweep (degrees)
    dihedral: float = 0.0       # Dihedral angle (degrees) 
    twist: float = 0.0          # Washout (degrees)
    taper_ratio: float = 1.0    # tip_chord / root_chord
    
    # Airfoil
    airfoil_root: str = "NACA2412"
    airfoil_tip: str = "NACA2412"
    
    # Flight conditions
    cruise_speed: float = 20.0   # m/s
    cruise_altitude: float = 100.0  # m
    design_load_factor: float = 3.5
    
    # Mass properties
    empty_weight: float = 2.0    # kg
    payload_weight: float = 0.5  # kg
    
    def __post_init__(self):
        """Calculate derived parameters"""
        self.taper_ratio = self.tip_chord / self.root_chord if self.root_chord > 0 else 1.0
        self.wing_area = 0.5 * (self.root_chord + self.tip_chord) * self.span
        self.aspect_ratio = self.span**2 / self.wing_area if self.wing_area > 0 else 0
        self.total_weight = self.empty_weight + self.payload_weight

@dataclass 
class PerformanceMetrics:
    """Container for aircraft performance metrics"""
    # Aerodynamic performance
    CL_cruise: float = 0.0
    CD_total: float = 0.0
    CDi: float = 0.0           # Induced drag coefficient
    CDp: float = 0.0           # Profile drag coefficient  
    L_D_ratio: float = 0.0     # Lift-to-drag ratio
    
    # Stability margins
    static_margin: float = 0.0
    CL_max: float = 0.0
    stall_margin: float = 0.0
    
    # Mission performance
    max_range: float = 0.0     # km
    endurance: float = 0.0     # hours
    climb_rate: float = 0.0    # m/s
    
    # Structural
    wing_loading: float = 0.0  # N/m²
    power_loading: float = 0.0 # N/W
    
    # Overall scores
    mission_score: float = 0.0
    constraint_penalty: float = 0.0
    total_score: float = 0.0

class Constraint(ABC):
    """Abstract base class for design constraints"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def evaluate(self, design: DesignParameters, performance: PerformanceMetrics) -> float:
        """
        Evaluate constraint violation
        
        Returns:
            Penalty value (0 = satisfied, >0 = violated)
        """
        pass

class GeometryConstraint(Constraint):
    """Geometric design constraints"""
    
    def __init__(self, 
                 max_span: Optional[float] = None,
                 min_span: Optional[float] = None,
                 max_wing_area: Optional[float] = None,
                 min_wing_area: Optional[float] = None,
                 max_aspect_ratio: Optional[float] = None,
                 min_aspect_ratio: Optional[float] = None,
                 weight: float = 1.0):
        super().__init__("Geometry", weight)
        self.max_span = max_span
        self.min_span = min_span
        self.max_wing_area = max_wing_area
        self.min_wing_area = min_wing_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio
    
    def evaluate(self, design: DesignParameters, performance: PerformanceMetrics) -> float:
        """Evaluate geometry constraint violations"""
        penalty = 0.0
        
        if self.max_span and design.span > self.max_span:
            penalty += (design.span - self.max_span) / self.max_span
        
        if self.min_aspect_ratio and design.aspect_ratio < self.min_aspect_ratio:
            penalty += (self.min_aspect_ratio - design.aspect_ratio) / self.min_aspect_ratio
        
        return penalty * self.weight

class PerformanceConstraint(Constraint):
    """Performance-based constraints"""
    
    def __init__(self,
                 min_L_D_ratio: Optional[float] = None,
                 max_wing_loading: Optional[float] = None,
                 min_CL_max: Optional[float] = None,
                 min_stall_margin: Optional[float] = None,
                 min_climb_rate: Optional[float] = None,
                 weight: float = 1.0):
        super().__init__("Performance", weight)
        self.min_L_D_ratio = min_L_D_ratio
        self.max_wing_loading = max_wing_loading
        self.min_CL_max = min_CL_max
        self.min_stall_margin = min_stall_margin
        self.min_climb_rate = min_climb_rate
    
    def evaluate(self, design: DesignParameters, performance: PerformanceMetrics) -> float:
        """Evaluate performance constraint violations"""
        penalty = 0.0
        
        if self.min_L_D_ratio and performance.L_D_ratio < self.min_L_D_ratio:
            penalty += (self.min_L_D_ratio - performance.L_D_ratio) / self.min_L_D_ratio
            
        if self.max_wing_loading and performance.wing_loading > self.max_wing_loading:
            penalty += (performance.wing_loading - self.max_wing_loading) / self.max_wing_loading
            
        if self.min_CL_max and performance.CL_max < self.min_CL_max:
            penalty += (self.min_CL_max - performance.CL_max) / self.min_CL_max
            
        if self.min_stall_margin and performance.stall_margin < self.min_stall_margin:
            penalty += (self.min_stall_margin - performance.stall_margin) / self.min_stall_margin
            
        if self.min_climb_rate and performance.climb_rate < self.min_climb_rate:
            penalty += (self.min_climb_rate - performance.climb_rate) / self.min_climb_rate
        
        return penalty * self.weight

class ObjectiveFunction:
    """Multi-objective function for design optimization"""
    
    def __init__(self, 
                 maximize_objectives: List[str] = None,
                 minimize_objectives: List[str] = None,
                 weights: Dict[str, float] = None):
        """
        Initialize objective function
        
        Args:
            maximize_objectives: List of metrics to maximize
            minimize_objectives: List of metrics to minimize  
            weights: Relative weights for each objective
        """
        self.maximize_objectives = maximize_objectives or ['L_D_ratio']
        self.minimize_objectives = minimize_objectives or ['CD_total']
        self.weights = weights or {}
        
        # Default weights
        for obj in self.maximize_objectives + self.minimize_objectives:
            if obj not in self.weights:
                self.weights[obj] = 1.0
    
    def evaluate(self, performance: PerformanceMetrics) -> float:
        """
        Calculate objective function value
        
        Returns:
            Objective score (higher is better)
        """
        score = 0.0
        
        # Maximize objectives (positive contribution)
        for obj in self.maximize_objectives:
            value = getattr(performance, obj, 0.0)
            weight = self.weights.get(obj, 1.0)
            score += weight * value
        
        # Minimize objectives (negative contribution)
        for obj in self.minimize_objectives:
            value = getattr(performance, obj, 0.0)
            weight = self.weights.get(obj, 1.0)
            # Use reciprocal to convert minimization to maximization
            if value > 0:
                score += weight / value
            else:
                score -= weight * 1000  # Heavy penalty for invalid values
        
        return score

class DesignEvaluator:
    """Main design evaluation system"""
    
    def __init__(self, 
                 constraints: List[Constraint] = None,
                 objective_function: ObjectiveFunction = None):
        """
        Initialize design evaluator
        
        Args:
            constraints: List of design constraints
            objective_function: Objective function for optimization
        """
        self.constraints = constraints or []
        self.objective_function = objective_function or ObjectiveFunction()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_design(self, 
                       design: DesignParameters,
                       aerodynamic_data: Dict) -> PerformanceMetrics:
        """
        Evaluate a complete aircraft design
        
        Args:
            design: Design parameters
            aerodynamic_data: Results from aerodynamic analysis
            
        Returns:
            Performance metrics and scores
        """
        # Initialize performance metrics
        performance = PerformanceMetrics()
        
        # Extract aerodynamic data
        if 'CL' in aerodynamic_data:
            performance.CL_cruise = aerodynamic_data['CL']
        if 'CDi' in aerodynamic_data:
            performance.CDi = aerodynamic_data['CDi']
        if 'CD_total' in aerodynamic_data:
            performance.CD_total = aerodynamic_data['CD_total']
        if 'CL_max' in aerodynamic_data:
            performance.CL_max = aerodynamic_data['CL_max']
        
        # Calculate L/D ratio
        if performance.CD_total > 0:
            performance.L_D_ratio = performance.CL_cruise / performance.CD_total
        
        # Calculate wing loading
        g = 9.81  # m/s²
        performance.wing_loading = (design.total_weight * g) / design.wing_area
        
        # Calculate stall margin
        if performance.CL_max > 0:
            performance.stall_margin = performance.CL_max / performance.CL_cruise - 1.0
        
        # Estimate mission performance
        performance = self._calculate_mission_performance(design, performance)
        
        # Evaluate constraints
        constraint_penalty = 0.0
        for constraint in self.constraints:
            penalty = constraint.evaluate(design, performance)
            constraint_penalty += penalty
        
        performance.constraint_penalty = constraint_penalty
        
        # Calculate mission score
        performance.mission_score = self.objective_function.evaluate(performance)
        
        # Total score (mission score penalized by constraints)
        performance.total_score = performance.mission_score - constraint_penalty * 100
        
        return performance
    
    def _calculate_mission_performance(self, 
                                     design: DesignParameters,
                                     performance: PerformanceMetrics) -> PerformanceMetrics:
        """Calculate mission-specific performance metrics"""
        
        # Simplified range estimation (Breguet range equation)
        if performance.L_D_ratio > 0:
            # Assume electric propulsion for simplicity
            specific_energy = 150  # Wh/kg for Li-ion battery
            propulsive_efficiency = 0.8
            
            # Rough range calculation
            performance.max_range = (specific_energy * 3600 * propulsive_efficiency * 
                                   performance.L_D_ratio) / (design.cruise_speed * 9.81)  # km
        
        # Simplified climb rate calculation
        # Assumes excess power available
        if design.wing_area > 0:
            power_to_weight = 100  # W/kg assumed power-to-weight ratio
            power_available = design.total_weight * power_to_weight
            power_required = (design.total_weight * 9.81 * design.cruise_speed) / performance.L_D_ratio
            
            if power_available > power_required:
                excess_power = power_available - power_required
                performance.climb_rate = excess_power / (design.total_weight * 9.81)
        
        # Endurance estimation
        if design.cruise_speed > 0:
            performance.endurance = performance.max_range / design.cruise_speed  # hours
        
        return performance
    
    def compare_designs(self, 
                       designs: List[Tuple[DesignParameters, PerformanceMetrics]]) -> List[int]:
        """
        Rank designs by total score
        
        Args:
            designs: List of (design, performance) tuples
            
        Returns:
            List of indices sorted by performance (best first)
        """
        scores = [perf.total_score for _, perf in designs]
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices
    
    def is_design_feasible(self, 
                          design: DesignParameters,
                          performance: PerformanceMetrics,
                          tolerance: float = 0.01) -> bool:
        """Check if design satisfies all constraints within tolerance"""
        total_penalty = 0.0
        for constraint in self.constraints:
            penalty = constraint.evaluate(design, performance)
            total_penalty += penalty
        
        return total_penalty <= tolerance