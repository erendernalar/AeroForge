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
        self.xfoil_path = xfoil_path
        self.logger = logging.getLogger(__name__)

    def check_xfoil_available(self) -> bool:
        try:
            subprocess.run([self.xfoil_path], capture_output=True, timeout=5, input="\nquit\n", text=True)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def generate_naca_airfoil(self, naca_code: str, n_points: int = 160) -> np.ndarray:
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_file = os.path.join(temp_dir, "airfoil.dat")
            commands = [
                f"naca {naca_code}",
                "ppar",
                f"n {n_points}",
                "",
                "",
                "pane",  # 🔧 Ensure airfoil is paneled before saving
                f"save {coord_file}",
                "",
                "quit"
            ]
            try:
                subprocess.run([self.xfoil_path], input="\n".join(commands), capture_output=True, text=True, timeout=30)
                if os.path.exists(coord_file):
                    return np.loadtxt(coord_file, skiprows=1)
                raise RuntimeError("XFOIL failed to generate airfoil coordinates")
            except subprocess.TimeoutExpired:
                raise RuntimeError("XFOIL timed out during airfoil generation")

    def analyze_airfoil_by_name(self, airfoil_name: str, airfoil_manager, reynolds: float,
                                mach: float = 0.0, alpha_range: Tuple[float, float] = (-4, 14),
                                alpha_step: float = 0.5) -> Dict:
        coords = airfoil_manager.get_airfoil(airfoil_name)
        if coords is None:
            if airfoil_name.upper().startswith('NACA') and len(airfoil_name) == 8:
                coords = airfoil_manager.create_naca_airfoil(airfoil_name[4:])
            else:
                try:
                    coords = airfoil_manager.download_airfoil(airfoil_name.lower())
                except:
                    raise ValueError(f"Airfoil '{airfoil_name}' not found and cannot be generated/downloaded")
        return self.analyze_airfoil(coords, reynolds, mach, alpha_range, alpha_step)

    def analyze_airfoil(self, coordinates: np.ndarray, reynolds: float, mach: float = 0.0,
                        alpha_range: Tuple[float, float] = (-4, 14), alpha_step: float = 0.5,
                        n_critical: float = 9.0) -> Dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_file = os.path.join(temp_dir, "airfoil.dat")
            polar_file = os.path.join(temp_dir, "polar.dat")

            with open(coord_file, 'w') as f:
                f.write("Airfoil\n")
                for x, y in coordinates:
                    f.write(f"{x:.6f} {y:.6f}\n")

            alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step, alpha_step)
            commands = [
                f"load {coord_file}",
                "",  # Enter name prompt
                "pane",  # 🔧 Ensure the paneling is set up
                "oper",
                f"visc {reynolds}",
                f"mach {mach}",
                f"pacc",
                f"{polar_file}",
                "",
                f"iter 200"
            ] + [f"alfa {alpha}" for alpha in alphas] + ["pacc", "", "quit"]

            try:
                proc = subprocess.run([self.xfoil_path], input="\n".join(commands),
                                      capture_output=True, text=True, timeout=60)
                if os.path.exists(polar_file):
                    return self._parse_polar_file(polar_file)
                self.logger.warning("XFOIL analysis failed - no polar file generated")
                self.logger.debug(f"XFOIL Output:\n{proc.stdout}")
                return self._empty_polar_data()
            except subprocess.TimeoutExpired:
                self.logger.error("XFOIL analysis timed out")
                return self._empty_polar_data()

    def _parse_polar_file(self, polar_file: str) -> Dict:
        try:
            data = np.loadtxt(polar_file, skiprows=12)
            if data.size == 0:
                return self._empty_polar_data()
            if data.ndim == 1:
                data = data.reshape(1, -1)
            polar_data = {
                'alpha': data[:, 0],
                'CL': data[:, 1],
                'CD': data[:, 2],
                'CDp': data[:, 3],
                'CM': data[:, 4],
                'Top_Xtr': data[:, 5],
                'Bot_Xtr': data[:, 6],
                'L_D': np.divide(data[:, 1], data[:, 2], out=np.zeros_like(data[:, 1]), where=data[:, 2] != 0),
                'CL_max': np.max(data[:, 1]),
                'alpha_CL_max': data[:, 0][np.argmax(data[:, 1])],
                'L_D_max': np.max(np.divide(data[:, 1], data[:, 2], out=np.zeros_like(data[:, 1]), where=data[:, 2] != 0)),
                'alpha_L_D_max': data[:, 0][np.argmax(np.divide(data[:, 1], data[:, 2], out=np.zeros_like(data[:, 1]), where=data[:, 2] != 0))]
            }
            return polar_data
        except Exception as e:
            self.logger.error(f"Error parsing polar file: {e}")
            return self._empty_polar_data()

    def _empty_polar_data(self) -> Dict:
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = XFOILDriver()
    if driver.check_xfoil_available():
        print("✅ XFOIL is available")
        coords = driver.generate_naca_airfoil("2412")
        print(f"Generated airfoil with {len(coords)} points")
        polar = driver.analyze_airfoil(coords, reynolds=500000)
        if len(polar['alpha']) > 0:
            print(f"Analysis complete:")
            print(f"  CL_max: {polar['CL_max']:.3f} at α = {polar['alpha_CL_max']:.1f}°")
            print(f"  (L/D)_max: {polar['L_D_max']:.1f} at α = {polar['alpha_L_D_max']:.1f}°")
        else:
            print("❌ Analysis failed")
    else:
        print("❌ XFOIL not found. Please install XFOIL and add to PATH")
