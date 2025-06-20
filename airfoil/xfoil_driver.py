"""
AeroForge XFOIL Driver Module
Handles direct communication with XFOIL for 2D airfoil analysis
"""

import subprocess
import os
import tempfile
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class XFOILDriver:
    """Interface for running XFOIL analysis on airfoils"""
    
    def __init__(self, xfoil_path: str = "xfoil"):
        """
        Initialize XFOIL driver
        
        Args:
            xfoil_path: Path to XFOIL executable
        """
        self.xfoil_path = xfoil_path
        self.logger = logging.getLogger(__name__)
        
    def check_xfoil_available(self) -> bool:
        """Check if XFOIL is available in system PATH"""
        try:
            result = subprocess.run([self.xfoil_path], 
                                  capture_output=True, 
                                  timeout=5,
                                  input="\nquit\n",
                                  text=True)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_naca_airfoil(self, naca_code: str, n_points: int = 160) -> np.ndarray:
        """
        Generate NACA airfoil coordinates using XFOIL
        
        Args:
            naca_code: NACA 4-digit code (e.g., "2412")
            n_points: Number of points to generate
            
        Returns:
            Array of (x, y) coordinates
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_file = os.path.join(temp_dir, "airfoil.dat")
            
            # XFOIL commands
            commands = [
                f"naca {naca_code}",
                f"ppar",
                f"n {n_points}",
                "",
                "",
                f"save {coord_file}",
                "",
                "quit"
            ]
            
            try:
                process = subprocess.run(
                    [self.xfoil_path],
                    input="\n".join(commands),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if os.path.exists(coord_file):
                    coords = np.loadtxt(coord_file, skiprows=1)
                    return coords
                else:
                    raise RuntimeError("XFOIL failed to generate airfoil coordinates")
                    
            except subprocess.TimeoutExpired:
                raise RuntimeError("XFOIL timed out during airfoil generation")
    
    def analyze_airfoil_by_name(self, 
                               airfoil_name: str,
                               airfoil_manager,
                               reynolds: float,
                               mach: float = 0.0,
                               alpha_range: Tuple[float, float] = (-4, 14),
                               alpha_step: float = 0.5) -> Dict:
        """
        Analyze airfoil by name using airfoil manager
        
        Args:
            airfoil_name: Name of airfoil (NACA code, custom name, etc.)
            airfoil_manager: AirfoilManager instance
            reynolds: Reynolds number
            mach: Mach number
            alpha_range: (min_alpha, max_alpha) in degrees
            alpha_step: Step size for alpha sweep
            
        Returns:
            Dictionary containing polar data
        """
        # Try to get airfoil from manager
        coords = airfoil_manager.get_airfoil(airfoil_name)
        
        if coords is None:
            # Try to generate NACA airfoil if it looks like NACA code
            if airfoil_name.upper().startswith('NACA') and len(airfoil_name) == 8:
                naca_code = airfoil_name[4:]
                coords = airfoil_manager.create_naca_airfoil(naca_code)
            else:
                # Try to download from online database
                try:
                    coords = airfoil_manager.download_airfoil(airfoil_name.lower())
                except:
                    raise ValueError(f"Airfoil '{airfoil_name}' not found and cannot be generated/downloaded")
        
        return self.analyze_airfoil(coords, reynolds, mach, alpha_range, alpha_step)
    
    def analyze_airfoil(self, 
                       coordinates: np.ndarray,
                       reynolds: float,
                       mach: float = 0.0,
                       alpha_range: Tuple[float, float] = (-4, 14),
                       alpha_step: float = 0.5,
                       n_critical: float = 9.0) -> Dict:
        """
        Perform XFOIL analysis on airfoil
        
        Args:
            coordinates: Airfoil (x,y) coordinates
            reynolds: Reynolds number
            mach: Mach number
            alpha_range: (min_alpha, max_alpha) in degrees
            alpha_step: Step size for alpha sweep
            n_critical: Critical amplification factor
            
        Returns:
            Dictionary containing polar data
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_file = os.path.join(temp_dir, "airfoil.dat")
            polar_file = os.path.join(temp_dir, "polar.dat")
            
            # Save coordinates
            with open(coord_file, 'w') as f:
                f.write("Airfoil\n")
                for x, y in coordinates:
                    f.write(f"{x:.6f} {y:.6f}\n")
            
            # Generate alpha sequence
            alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step, alpha_step)
            
            # XFOIL commands
            commands = [
                f"load {coord_file}",
                "",
                "oper",
                f"visc {reynolds}",
                f"mach {mach}",
                f"pacc",
                f"{polar_file}",
                "",
                f"iter 200"
            ]
            
            # Add alpha sweep commands
            for alpha in alphas:
                commands.append(f"alfa {alpha}")
            
            commands.extend([
                "pacc",
                "",
                "quit"
            ])
            
            try:
                process = subprocess.run(
                    [self.xfoil_path],
                    input="\n".join(commands),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if os.path.exists(polar_file):
                    return self._parse_polar_file(polar_file)
                else:
                    self.logger.warning("XFOIL analysis failed - no polar file generated")
                    return self._empty_polar_data()
                    
            except subprocess.TimeoutExpired:
                self.logger.error("XFOIL analysis timed out")
                return self._empty_polar_data()
    
    def _parse_polar_file(self, polar_file: str) -> Dict:
        """Parse XFOIL polar output file"""
        try:
            # Skip header lines and read data
            data = np.loadtxt(polar_file, skiprows=12)
            
            if data.size == 0:
                return self._empty_polar_data()
            
            # Ensure data is 2D
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            polar_data = {
                'alpha': data[:, 0],      # Angle of attack
                'CL': data[:, 1],         # Lift coefficient  
                'CD': data[:, 2],         # Drag coefficient
                'CDp': data[:, 3],        # Pressure drag coefficient
                'CM': data[:, 4],         # Moment coefficient
                'Top_Xtr': data[:, 5],    # Top transition location
                'Bot_Xtr': data[:, 6]     # Bottom transition location
            }
            
            # Calculate derived quantities
            polar_data['L_D'] = np.divide(polar_data['CL'], polar_data['CD'], 
                                        out=np.zeros_like(polar_data['CL']), 
                                        where=polar_data['CD']!=0)
            
            # Find key performance points
            polar_data['CL_max'] = np.max(polar_data['CL'])
            polar_data['alpha_CL_max'] = polar_data['alpha'][np.argmax(polar_data['CL'])]
            polar_data['L_D_max'] = np.max(polar_data['L_D'])
            polar_data['alpha_L_D_max'] = polar_data['alpha'][np.argmax(polar_data['L_D'])]
            
            return polar_data
            
        except Exception as e:
            self.logger.error(f"Error parsing polar file: {e}")
            return self._empty_polar_data()
    
    def _empty_polar_data(self) -> Dict:
        """Return empty polar data structure"""
        return {
            'alpha': np.array([]),
            'CL': np.array([]),
            'CD': np.array([]),
            'CDp': np.array([]),
            'CM': np.array([]),
            'Top_Xtr': np.array([]),
            'Bot_Xtr': np.array([]),
            'L_D': np.array([]),
            'CL_max': 0.0,
            'alpha_CL_max': 0.0,
            'L_D_max': 0.0,
            'alpha_L_D_max': 0.0
        }

# Example usage
if __name__ == "__main__":
    # Test the XFOIL driver
    driver = XFOILDriver()
    
    if driver.check_xfoil_available():
        print("✅ XFOIL is available")
        
        # Generate NACA 2412 airfoil
        coords = driver.generate_naca_airfoil("2412")
        print(f"Generated airfoil with {len(coords)} points")
        
        # Analyze the airfoil
        polar = driver.analyze_airfoil(coords, reynolds=500000)
        
        if len(polar['alpha']) > 0:
            print(f"Analysis complete:")
            print(f"  CL_max: {polar['CL_max']:.3f} at α = {polar['alpha_CL_max']:.1f}°")
            print(f"  (L/D)_max: {polar['L_D_max']:.1f} at α = {polar['alpha_L_D_max']:.1f}°")
        else:
            print("❌ Analysis failed")
    else:
        print("❌ XFOIL not found. Please install XFOIL and add to PATH")