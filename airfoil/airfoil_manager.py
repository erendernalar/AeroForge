"""
AeroForge Airfoil Import and Management System
Supports importing custom airfoils from various file formats
"""

import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import requests
from urllib.parse import urlparse

class AirfoilManager:
    """Manages airfoil coordinate import and database"""
    
    def __init__(self, airfoil_dir: str = "airfoils"):
        """
        Initialize airfoil manager
        
        Args:
            airfoil_dir: Directory to store airfoil files
        """
        self.airfoil_dir = Path(airfoil_dir)
        self.airfoil_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Airfoil database
        self.airfoil_database = {}
        self._load_airfoil_database()
    
    def import_airfoil_file(self, 
                           file_path: str, 
                           name: Optional[str] = None,
                           file_format: str = 'auto') -> np.ndarray:
        """
        Import airfoil coordinates from file
        
        Args:
            file_path: Path to airfoil coordinate file
            name: Custom name for the airfoil
            file_format: File format ('dat', 'txt', 'csv', 'selig', 'auto')
            
        Returns:
            Array of (x, y) coordinates
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Airfoil file not found: {file_path}")
        
        # Auto-detect format if needed
        if file_format == 'auto':
            file_format = self._detect_file_format(file_path)
        
        # Import based on format
        if file_format in ['dat', 'selig']:
            coords = self._import_selig_format(file_path)
        elif file_format == 'txt':
            coords = self._import_text_format(file_path)
        elif file_format == 'csv':
            coords = self._import_csv_format(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Validate and process coordinates
        coords = self._validate_coordinates(coords)
        
        # Store in database
        airfoil_name = name or file_path.stem
        self.airfoil_database[airfoil_name] = {
            'coordinates': coords,
            'source_file': str(file_path),
            'format': file_format
        }
        
        # Save to airfoil directory
        self._save_airfoil(airfoil_name, coords)
        
        self.logger.info(f"Imported airfoil '{airfoil_name}' with {len(coords)} points")
        return coords
    
    def download_airfoil(self, 
                        airfoil_name: str,
                        source: str = 'uiuc') -> np.ndarray:
        """
        Download airfoil from online database
        
        Args:
            airfoil_name: Name of airfoil (e.g., 'naca2412', 'clarky')
            source: Database source ('uiuc', 'airfoiltools')
            
        Returns:
            Array of (x, y) coordinates
        """
        if source == 'uiuc':
            coords = self._download_from_uiuc(airfoil_name)
        elif source == 'airfoiltools':
            coords = self._download_from_airfoiltools(airfoil_name)
        else:
            raise ValueError(f"Unknown airfoil source: {source}")
        
        # Store in database
        self.airfoil_database[airfoil_name] = {
            'coordinates': coords,
            'source_file': f"{source}:{airfoil_name}",
            'format': 'downloaded'
        }
        
        # Save locally
        self._save_airfoil(airfoil_name, coords)
        
        self.logger.info(f"Downloaded airfoil '{airfoil_name}' from {source}")
        return coords
    
    def get_airfoil(self, name: str) -> Optional[np.ndarray]:
        """Get airfoil coordinates by name"""
        if name in self.airfoil_database:
            return self.airfoil_database[name]['coordinates']
        
        # Try to load from file
        airfoil_file = self.airfoil_dir / f"{name}.dat"
        if airfoil_file.exists():
            coords = self.import_airfoil_file(airfoil_file, name)
            return coords
        
        return None
    
    def list_airfoils(self) -> List[str]:
        """List all available airfoils"""
        # Airfoils in memory
        airfoils = list(self.airfoil_database.keys())
        
        # Airfoils in directory
        for file_path in self.airfoil_dir.glob("*.dat"):
            name = file_path.stem
            if name not in airfoils:
                airfoils.append(name)
        
        return sorted(airfoils)
    
    def create_naca_airfoil(self, naca_code: str) -> np.ndarray:
        """
        Generate NACA 4-digit airfoil coordinates analytically
        
        Args:
            naca_code: 4-digit NACA code (e.g., '2412')
            
        Returns:
            Array of (x, y) coordinates
        """
        if len(naca_code) != 4 or not naca_code.isdigit():
            raise ValueError("NACA code must be 4 digits")
        
        # Extract NACA parameters
        m = int(naca_code[0]) / 100.0  # Maximum camber
        p = int(naca_code[1]) / 10.0   # Position of maximum camber
        t = int(naca_code[2:4]) / 100.0  # Maximum thickness
        
        # Generate x coordinates (cosine distribution)
        n_points = 200
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Thickness distribution
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                      0.2843 * x**3 - 0.1015 * x**4)
        
        # Camber line
        if m == 0 or p == 0:  # Symmetric airfoil
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.where(x <= p,
                         m / p**2 * (2 * p * x - x**2),
                         m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            
            dyc_dx = np.where(x <= p,
                             2 * m / p**2 * (p - x),
                             2 * m / (1 - p)**2 * (p - x))
        
        # Surface coordinates
        theta = np.arctan(dyc_dx)
        
        # Upper surface
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        
        # Lower surface
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        # Combine coordinates (upper surface from TE to LE, then lower surface LE to TE)
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
        
        coords = np.column_stack([x_coords, y_coords])
        
        # Store in database
        name = f"NACA{naca_code}"
        self.airfoil_database[name] = {
            'coordinates': coords,
            'source_file': 'generated',
            'format': 'naca_analytical'
        }
        
        return coords
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Auto-detect airfoil file format"""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        # Check if first line is a title (Selig format)
        try:
            float(first_line.split()[0])
            return 'txt'  # First line is numeric
        except (ValueError, IndexError):
            return 'selig'  # First line is title
    
    def _import_selig_format(self, file_path: Path) -> np.ndarray:
        """Import Selig format (.dat) airfoil file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip title line
        data_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith('#'):
                data_lines.append(line)
        
        # Parse coordinates
        coords = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append([x, y])
                except ValueError:
                    continue
        
        return np.array(coords)
    
    def _import_text_format(self, file_path: Path) -> np.ndarray:
        """Import plain text format airfoil file"""
        return np.loadtxt(file_path)
    
    def _import_csv_format(self, file_path: Path) -> np.ndarray:
        """Import CSV format airfoil file"""
        return np.loadtxt(file_path, delimiter=',')
    
    def _download_from_uiuc(self, airfoil_name: str) -> np.ndarray:
        """Download airfoil from UIUC database"""
        # UIUC airfoil database URL pattern
        base_url = "https://m-selig.ae.illinois.edu/ads/coord"
        url = f"{base_url}/{airfoil_name.lower()}.dat"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse response text
            lines = response.text.strip().split('\n')
            coords = []
            
            # Skip title line
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue
            
            return np.array(coords)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download airfoil {airfoil_name}: {e}")
    
    def _download_from_airfoiltools(self, airfoil_name: str) -> np.ndarray:
        """Download airfoil from AirfoilTools.com"""
        # This would require web scraping or API access
        # For now, raise not implemented
        raise NotImplementedError("AirfoilTools download not yet implemented")
    
    def _validate_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Validate and clean airfoil coordinates"""
        if len(coords) < 10:
            raise ValueError("Airfoil must have at least 10 coordinate points")
        
        # Remove duplicate points
        coords = np.unique(coords, axis=0)
        
        # Check if coordinates are reasonable
        x = coords[:, 0]
        y = coords[:, 1]
        
        if np.min(x) < -0.1 or np.max(x) > 1.1:
            self.logger.warning("X coordinates outside expected range [0, 1]")
        
        if np.min(y) < -0.5 or np.max(y) > 0.5:
            self.logger.warning("Y coordinates outside expected range [-0.5, 0.5]")
        
        # Ensure leading edge is at x=0 or close to it
        le_idx = np.argmin(x)
        if x[le_idx] > 0.05:
            self.logger.warning("Leading edge not at x=0")
        
        # Sort coordinates: upper surface from TE to LE, lower surface from LE to TE
        coords = self._sort_coordinates(coords)
        
        return coords
    
    def _sort_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Sort coordinates in proper order for XFOIL"""
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Find leading edge (minimum x)
        le_idx = np.argmin(x)
        
        # Split into upper and lower surfaces
        upper_surface = []
        lower_surface = []
        
        for i, (xi, yi) in enumerate(coords):
            if i == le_idx:
                continue
            
            if yi >= y[le_idx]:  # Upper surface
                upper_surface.append([xi, yi])
            else:  # Lower surface
                lower_surface.append([xi, yi])
        
        # Sort upper surface by x descending (TE to LE)
        upper_surface = sorted(upper_surface, key=lambda p: p[0], reverse=True)
        
        # Sort lower surface by x ascending (LE to TE)
        lower_surface = sorted(lower_surface, key=lambda p: p[0])
        
        # Combine: upper surface + LE + lower surface
        sorted_coords = upper_surface + [[x[le_idx], y[le_idx]]] + lower_surface
        
        return np.array(sorted_coords)
    
    def _save_airfoil(self, name: str, coords: np.ndarray):
        """Save airfoil coordinates to file"""
        file_path = self.airfoil_dir / f"{name}.dat"
        
        with open(file_path, 'w') as f:
            f.write(f"{name}\n")
            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f}\n")
    
    def _load_airfoil_database(self):
        """Load existing airfoils from directory"""
        for file_path in self.airfoil_dir.glob("*.dat"):
            try:
                name = file_path.stem
                coords = self._import_selig_format(file_path)
                self.airfoil_database[name] = {
                    'coordinates': coords,
                    'source_file': str(file_path),
                    'format': 'dat'
                }
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
    
    def plot_airfoil(self, name: str, save_path: Optional[str] = None):
        """Plot airfoil geometry"""
        coords = self.get_airfoil(name)
        if coords is None:
            raise ValueError(f"Airfoil '{name}' not found")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(f'Airfoil: {name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()