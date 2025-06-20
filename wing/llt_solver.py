"""
AeroForge Lifting Line Theory (LLT) Solver
Implements classical LLT for 3D wing analysis
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass

@dataclass
class WingSection:
    """Represents a wing section at a spanwise location"""
    y: float           # Spanwise position
    chord: float       # Local chord length
    twist: float       # Local twist angle (degrees)
    airfoil_data: Dict # 2D airfoil polar data
    dihedral: float = 0.0  # Dihedral angle (degrees)

class LiftingLineTheory:
    """Classical Lifting Line Theory implementation for 3D wing analysis"""
    
    def __init__(self, n_stations: int = 25):
        """
        Initialize LLT solver
        
        Args:
            n_stations: Number of spanwise stations for analysis
        """
        self.n_stations = n_stations
        self.logger = logging.getLogger(__name__)
        
    def analyze_wing(self, 
                    wing_sections: List[WingSection],
                    flight_condition: Dict,
                    n_terms: int = 15) -> Dict:
        """
        Perform lifting line analysis on wing
        
        Args:
            wing_sections: List of wing sections from root to tip
            flight_condition: Flight conditions (alpha, V, rho, etc.)
            n_terms: Number of Fourier terms for circulation distribution
            
        Returns:
            Dictionary containing 3D wing performance data
        """
        # Extract flight conditions
        alpha_wing = flight_condition.get('alpha', 0.0)  # Wing angle of attack (deg)
        velocity = flight_condition.get('velocity', 1.0)
        density = flight_condition.get('density', 1.225)
        
        # Create spanwise stations
        span = max(section.y for section in wing_sections) * 2  # Full span
        y_stations = self._create_spanwise_stations(span)
        
        # Interpolate wing properties at stations
        stations_data = self._interpolate_wing_properties(wing_sections, y_stations)
        
        # Solve for circulation distribution
        circulation = self._solve_circulation_distribution(
            stations_data, alpha_wing, n_terms
        )
        
        # Calculate forces and moments
        results = self._calculate_forces_and_moments(
            stations_data, circulation, velocity, density, alpha_wing
        )
        
        # Add spanwise distributions
        results['spanwise'] = {
            'y': y_stations,
            'circulation': circulation,
            'chord': [station['chord'] for station in stations_data],
            'twist': [station['twist'] for station in stations_data],
            'cl_local': results.get('cl_distribution', []),
            'cd_local': results.get('cd_distribution', [])
        }
        
        return results
    
    def _create_spanwise_stations(self, span: float) -> np.ndarray:
        """Create spanwise stations using cosine distribution"""
        # Use cosine distribution for better convergence
        theta = np.linspace(0, np.pi, self.n_stations)
        y_stations = -(span/2) * np.cos(theta)  # Symmetric about y=0
        return y_stations
    
    def _interpolate_wing_properties(self, 
                                   wing_sections: List[WingSection],
                                   y_stations: np.ndarray) -> List[Dict]:
        """Interpolate wing properties at analysis stations"""
        # Sort sections by spanwise position
        sections = sorted(wing_sections, key=lambda s: s.y)
        
        # Extract arrays for interpolation
        y_sections = np.array([s.y for s in sections])
        chords = np.array([s.chord for s in sections])
        twists = np.array([s.twist for s in sections])
        
        # Handle symmetry - extend to negative y values
        if y_sections[0] >= 0:  # Only positive y values provided
            y_sections = np.concatenate([-y_sections[::-1], y_sections])
            chords = np.concatenate([chords[::-1], chords])
            twists = np.concatenate([twists[::-1], twists])
        
        # Interpolate properties at stations
        stations_data = []
        for y in y_stations:
            chord = np.interp(abs(y), y_sections[y_sections >= 0], chords[y_sections >= 0])
            twist = np.interp(abs(y), y_sections[y_sections >= 0], twists[y_sections >= 0])
            
            # For now, use the first section's airfoil data
            # In a full implementation, you'd interpolate airfoil properties too
            airfoil_data = sections[0].airfoil_data
            
            stations_data.append({
                'y': y,
                'chord': chord,
                'twist': twist,
                'airfoil_data': airfoil_data
            })
        
        return stations_data
    
    def _solve_circulation_distribution(self,
                                      stations_data: List[Dict],
                                      alpha_wing: float,
                                      n_terms: int) -> np.ndarray:
        """Solve for circulation distribution using Fourier series method"""
        
        n_stations = len(stations_data)
        y_stations = np.array([station['y'] for station in stations_data])
        chords = np.array([station['chord'] for station in stations_data])
        twists = np.array([station['twist'] for station in stations_data])
        
        # Get semi-span
        b = max(abs(y_stations))
        
        # Calculate theta coordinates
        theta = np.arccos(-y_stations / b)
        
        # Set up system of equations for Fourier coefficients
        def equations(A_coeffs):
            A = np.array(A_coeffs)
            residuals = np.zeros(n_stations)
            
            for i in range(n_stations):
                # Calculate induced angle at station i
                alpha_induced = 0.0
                for n in range(1, n_terms + 1):
                    if n == 1:
                        alpha_induced += A[n-1] / (4 * b)
                    else:
                        alpha_induced += A[n-1] * n * np.sin(n * theta[i]) / np.sin(theta[i])
                
                # Local angle of attack
                alpha_local = alpha_wing + twists[i] - np.degrees(alpha_induced)
                
                # Get local lift slope (simplified)
                # In practice, you'd use the airfoil data here
                a_local = 2 * np.pi  # per radian
                
                # Calculate circulation from Kutta-Joukowski
                circulation_local = 0.0
                for n in range(1, n_terms + 1):
                    circulation_local += A[n-1] * np.sin(n * theta[i])
                
                # Residual equation
                residuals[i] = (circulation_local - 
                               a_local * chords[i] * np.radians(alpha_local) / 2)
            
            return residuals
        
        # Initial guess
        A_initial = np.ones(n_terms) * 0.1
        
        # Solve system
        try:
            A_solution = fsolve(equations, A_initial)
            
            # Calculate circulation at each station
            circulation = np.zeros(n_stations)
            for i in range(n_stations):
                for n in range(1, n_terms + 1):
                    circulation[i] += A_solution[n-1] * np.sin(n * theta[i])
            
            return circulation
            
        except Exception as e:
            self.logger.error(f"LLT solver failed: {e}")
            return np.zeros(n_stations)
    
    def _calculate_forces_and_moments(self,
                                    stations_data: List[Dict],
                                    circulation: np.ndarray,
                                    velocity: float,
                                    density: float,
                                    alpha_wing: float) -> Dict:
        """Calculate integrated forces and moments"""
        
        y_stations = np.array([station['y'] for station in stations_data])
        chords = np.array([station['chord'] for station in stations_data])
        
        # Calculate local forces
        dy = np.diff(y_stations)
        dy = np.append(dy, dy[-1])  # Extend for integration
        
        # Lift per unit span (Kutta-Joukowski theorem)
        lift_per_span = density * velocity * circulation
        
        # Induced drag per unit span
        # Simplified calculation - would need more sophisticated method
        induced_drag_per_span = np.zeros_like(circulation)
        for i in range(len(circulation)):
            if i > 0 and i < len(circulation) - 1:
                dgamma_dy = (circulation[i+1] - circulation[i-1]) / (y_stations[i+1] - y_stations[i-1])
                induced_drag_per_span[i] = density * dgamma_dy * circulation[i] / velocity
        
        # Integrate forces
        total_lift = np.trapz(lift_per_span, y_stations)
        total_induced_drag = np.trapz(induced_drag_per_span, y_stations)
        
        # Wing reference area (simplified)
        wing_area = np.trapz(chords, y_stations)
        
        # Coefficients
        q_inf = 0.5 * density * velocity**2
        CL = total_lift / (q_inf * wing_area)
        CDi = total_induced_drag / (q_inf * wing_area)
        
        # Aspect ratio
        span = max(y_stations) - min(y_stations)
        aspect_ratio = span**2 / wing_area
        
        # Theoretical minimum induced drag (elliptical distribution)
        CDi_min = CL**2 / (np.pi * aspect_ratio)
        efficiency_factor = CDi_min / CDi if CDi > 0 else 1.0
        
        results = {
            'CL': CL,
            'CDi': CDi,
            'CDi_min': CDi_min,
            'efficiency_factor': efficiency_factor,
            'aspect_ratio': aspect_ratio,
            'wing_area': wing_area,
            'span': span,
            'total_lift': total_lift,
            'total_induced_drag': total_induced_drag,
            'cl_distribution': lift_per_span / (q_inf * chords),
            'cd_distribution': induced_drag_per_span / (q_inf * chords)
        }
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the LLT solver
    llt = LiftingLineTheory(n_stations=21)
    
    # Create simple wing sections
    sections = [
        WingSection(y=0.0, chord=0.3, twist=2.0, airfoil_data={}),
        WingSection(y=0.5, chord=0.25, twist=0.0, airfoil_data={}),
        WingSection(y=1.0, chord=0.2, twist=-2.0, airfoil_data={})
    ]
    
    # Flight condition
    flight_condition = {
        'alpha': 5.0,      # degrees
        'velocity': 20.0,  # m/s
        'density': 1.225   # kg/m³
    }
    
    # Analyze wing
    results = llt.analyze_wing(sections, flight_condition)
    
    print("LLT Analysis Results:")
    print(f"CL: {results['CL']:.3f}")
    print(f"CDi: {results['CDi']:.4f}")
    print(f"Aspect Ratio: {results['aspect_ratio']:.2f}")
    print(f"Efficiency Factor: {results['efficiency_factor']:.3f}")
    print(f"Wing Area: {results['wing_area']:.3f} m²")