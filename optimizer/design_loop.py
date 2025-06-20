"""
AeroForge Main Design Loop and Optimization Engine
Coordinates the complete aircraft design optimization process
"""

import numpy as np
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import asdict
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Import AeroForge modules (assuming they're in the same project)
from airfoil.xfoil_driver import XFOILDriver
from airfoil.airfoil_manager import AirfoilManager
from wing.llt_solver import LiftingLineTheory, WingSection
from optimizer.evaluator import (DesignEvaluator, DesignParameters, PerformanceMetrics,
                                GeometryConstraint, PerformanceConstraint, ObjectiveFunction)

class DesignSpace:
    """Defines the parameter space for aircraft design optimization"""
    
    def __init__(self, config: Dict):
        """Initialize design space from configuration"""
        self.config = config
        
        # Define parameter bounds
        self.bounds = {
            'span': config.get('span_range', [1.0, 3.0]),
            'root_chord': config.get('root_chord_range', [0.15, 0.4]),
            'tip_chord': config.get('tip_chord_range', [0.1, 0.3]),
            'sweep': config.get('sweep_range', [0.0, 15.0]),
            'twist': config.get('twist_range', [-5.0, 5.0]),
            'cruise_speed': config.get('speed_range', [15.0, 30.0])
        }
        
        # Parameter names for optimization
        self.param_names = list(self.bounds.keys())
        self.param_bounds = [self.bounds[name] for name in self.param_names]
    
    def vector_to_design(self, x: np.ndarray) -> DesignParameters:
        """Convert optimization vector to design parameters"""
        params = {}
        for i, name in enumerate(self.param_names):
            params[name] = x[i]
        
        # Calculate tip chord from taper ratio if specified
        if 'taper_ratio' in self.config:
            taper = self.config['taper_ratio']
            params['tip_chord'] = params['root_chord'] * taper
        
        # Set fixed parameters
        for key, value in self.config.get('fixed_parameters', {}).items():
            params[key] = value
        
        return DesignParameters(**params)
    
    def design_to_vector(self, design: DesignParameters) -> np.ndarray:
        """Convert design parameters to optimization vector"""
        x = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            x[i] = getattr(design, name)
        return x

class OptimizationResult:
    """Container for optimization results"""
    
    def __init__(self):
        self.best_designs = []
        self.convergence_history = []
        self.evaluation_count = 0
        self.total_time = 0.0
        self.success = False

class AeroForgeOptimizer:
    """Main optimization engine for aircraft design"""
    
    def __init__(self, config_file: str):
        """
        Initialize optimizer from configuration file
        
        Args:
            config_file: Path to JSON configuration file
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.xfoil_driver = XFOILDriver()
        self.airfoil_manager = AirfoilManager()
        self.llt_solver = LiftingLineTheory()
        self.design_space = DesignSpace(self.config.get('design_space', {}))
        
        # Set up constraints and objectives
        self.evaluator = self._setup_evaluator()
        
        # Optimization settings
        self.opt_config = self.config.get('optimization', {})
        self.max_iterations = self.opt_config.get('max_iterations', 100)
        self.population_size = self.opt_config.get('population_size', 50)
        self.convergence_tolerance = self.opt_config.get('convergence_tolerance', 1e-6)
        
        # Results storage
        self.results = OptimizationResult()
        
    def _setup_evaluator(self) -> DesignEvaluator:
        """Set up the design evaluator from configuration"""
        
        # Create constraints
        constraints = []
        
        # Geometry constraints
        geom_config = self.config.get('constraints', {}).get('geometry', {})
        if geom_config:
            geom_constraint = GeometryConstraint(
                max_span=geom_config.get('max_span'),
                min_span=geom_config.get('min_span'),
                max_wing_area=geom_config.get('max_wing_area'),
                min_wing_area=geom_config.get('min_wing_area'),
                max_aspect_ratio=geom_config.get('max_aspect_ratio'),
                min_aspect_ratio=geom_config.get('min_aspect_ratio')
            )
            constraints.append(geom_constraint)
        
        # Performance constraints
        perf_config = self.config.get('constraints', {}).get('performance', {})
        if perf_config:
            perf_constraint = PerformanceConstraint(
                min_L_D_ratio=perf_config.get('min_L_D_ratio'),
                min_CL_max=perf_config.get('min_CL_max'),
                min_stall_margin=perf_config.get('min_stall_margin')
            )
            constraints.append(perf_constraint)
        
        # Objective function
        obj_config = self.config.get('objectives', {})
        objective = ObjectiveFunction(
            maximize_objectives=obj_config.get('maximize', ['L_D_ratio']),
            minimize_objectives=obj_config.get('minimize', ['CD_total']),
            weights=obj_config.get('weights', {})
        )
        
        return DesignEvaluator(constraints=constraints, objective_function=objective)
    
    def analyze_design(self, design: DesignParameters) -> Dict:
        """
        Perform complete aerodynamic analysis of a design
        
        Args:
            design: Aircraft design parameters
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # 1. Get airfoil configuration
            airfoil_config = self.config.get('airfoil_configuration', {})
            root_airfoil = airfoil_config.get('root_airfoil', {}).get('name', 'NACA2412')
            
            # 2. Generate/get airfoil and analyze with XFOIL
            try:
                airfoil_polar = self.xfoil_driver.analyze_airfoil_by_name(
                    root_airfoil,
                    self.airfoil_manager,
                    reynolds=self.config.get('analysis_settings', {}).get('reynolds_number', 500000),
                    alpha_range=tuple(self.config.get('analysis_settings', {}).get('angle_of_attack_range', [-4, 14]))
                )
            except:
                # Fallback to NACA generation
                airfoil_coords = self.airfoil_manager.create_naca_airfoil("2412")
                airfoil_polar = self.xfoil_driver.analyze_airfoil(
                    airfoil_coords,
                    reynolds=self.config.get('analysis_settings', {}).get('reynolds_number', 500000),
                    alpha_range=tuple(self.config.get('analysis_settings', {}).get('angle_of_attack_range', [-4, 14]))
                )
            
            if len(airfoil_polar['alpha']) == 0:
                self.logger.warning("XFOIL analysis failed")
                return {'CL': 0, 'CD_total': 1.0, 'CL_max': 0}
            
            # 3. Create wing sections for LLT analysis
            wing_sections = [
                WingSection(y=0.0, chord=design.root_chord, twist=design.twist, 
                           airfoil_data=airfoil_polar),
                WingSection(y=design.span/2, chord=design.tip_chord, twist=0.0,
                           airfoil_data=airfoil_polar)
            ]
            
            # 4. Set up flight condition
            flight_condition = {
                'alpha': self.config.get('analysis_settings', {}).get('cruise_alpha', 5.0),
                'velocity': design.cruise_speed,
                'density': 1.225  # Sea level density
            }
            
            # 5. Perform LLT analysis
            llt_results = self.llt_solver.analyze_wing(wing_sections, flight_condition)
            
            # 6. Combine results
            analysis_results = {
                'CL': llt_results.get('CL', 0),
                'CDi': llt_results.get('CDi', 0),
                'CL_max': airfoil_polar.get('CL_max', 0),
                'L_D_max_airfoil': airfoil_polar.get('L_D_max', 0),
                'efficiency_factor': llt_results.get('efficiency_factor', 0)
            }
            
            # Estimate total drag (induced + profile)
            # Simple approximation: CD_total = CDi + CDp
            # CDp estimated from airfoil data
            cruise_alpha_idx = np.argmin(np.abs(airfoil_polar['alpha'] - 5.0))
            if cruise_alpha_idx < len(airfoil_polar['CD']):
                CDp = airfoil_polar['CD'][cruise_alpha_idx]
            else:
                CDp = 0.02  # Default estimate
            
            analysis_results['CD_total'] = analysis_results['CDi'] + CDp
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {'CL': 0, 'CD_total': 1.0, 'CL_max': 0}
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization
        
        Args:
            x: Design parameter vector
            
        Returns:
            Negative total score (for minimization)
        """
        self.results.evaluation_count += 1
        
        # Convert vector to design
        design = self.design_space.vector_to_design(x)
        
        # Analyze design
        aero_results = self.analyze_design(design)
        
        # Evaluate performance
        performance = self.evaluator.evaluate_design(design, aero_results)
        
        # Store for convergence tracking
        self.results.convergence_history.append({
            'iteration': self.results.evaluation_count,
            'score': performance.total_score,
            'L_D_ratio': performance.L_D_ratio,
            'constraint_penalty': performance.constraint_penalty
        })
        
        # Return negative score for minimization
        return -performance.total_score
    
    def optimize(self, method: str = 'differential_evolution') -> OptimizationResult:
        """
        Run the optimization process
        
        Args:
            method: Optimization method ('differential_evolution', 'scipy_minimize')
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Starting optimization with method: {method}")
        start_time = time.time()
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds=self.design_space.param_bounds,
                    maxiter=self.max_iterations,
                    popsize=15,  # Population size multiplier
                    seed=42,
                    disp=True,
                    callback=self._optimization_callback
                )
                
                self.results.success = result.success
                best_x = result.x
                
            elif method == 'scipy_minimize':
                # Use a simple initial guess
                x0 = np.array([(bounds[0] + bounds[1]) / 2 
                              for bounds in self.design_space.param_bounds])
                
                result = minimize(
                    self.objective_function,
                    x0,
                    method='L-BFGS-B',
                    bounds=self.design_space.param_bounds,
                    options={'maxiter': self.max_iterations}
                )
                
                self.results.success = result.success
                best_x = result.x
            
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Analyze best design
            best_design = self.design_space.vector_to_design(best_x)
            best_aero = self.analyze_design(best_design)
            best_performance = self.evaluator.evaluate_design(best_design, best_aero)
            
            self.results.best_designs = [(best_design, best_performance)]
            self.results.total_time = time.time() - start_time
            
            self.logger.info(f"Optimization completed in {self.results.total_time:.1f}s")
            self.logger.info(f"Best score: {best_performance.total_score:.2f}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.results.success = False
            return self.results
    
    def _optimization_callback(self, x, convergence):
        """Callback for monitoring optimization progress"""
        if self.results.evaluation_count % 10 == 0:
            latest_score = self.results.convergence_history[-1]['score']
            self.logger.info(f"Iteration {self.results.evaluation_count}, "
                           f"Score: {latest_score:.2f}")
        return False  # Continue optimization
    
    def save_results(self, output_dir: str = "output"):
        """Save optimization results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best designs
        if self.results.best_designs:
            design, performance = self.results.best_designs[0]
            
            design_dict = asdict(design)
            performance_dict = asdict(performance)
            
            results_data = {
                'design': design_dict,
                'performance': performance_dict,
                'convergence_history': self.results.convergence_history,
                'optimization_info': {
                    'evaluations': self.results.evaluation_count,
                    'time': self.results.total_time,
                    'success': self.results.success
                }
            }
            
            with open(f"{output_dir}/best_design.json", 'w') as f:
                json.dump(results_data, f, indent=4, default=str)
        
        # Plot convergence
        self.plot_convergence(f"{output_dir}/convergence.png")
        
        self.logger.info(f"Results saved to {output_dir}/")
    
    def plot_convergence(self, filename: str = None):
        """Plot optimization convergence"""
        if not self.results.convergence_history:
            return
        
        iterations = [h['iteration'] for h in self.results.convergence_history]
        scores = [h['score'] for h in self.results.convergence_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'b-', linewidth=2)
        plt.xlabel('Evaluation Number')
        plt.ylabel('Total Score')
        plt.title('Optimization Convergence')
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()