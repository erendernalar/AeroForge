# AeroForge User Guide

## Getting Started

### Basic Usage
```bash
# Check installation
python main.py --check

# Run demonstration
python main.py --demo

# Optimize with default config
python main.py --config config/mission1.json
```

### Configuration Files
AeroForge uses JSON configuration files to define:
- Mission objectives
- Design constraints
- Optimization parameters
- Airfoil specifications

### Custom Airfoils
```python
from airfoil.airfoil_manager import AirfoilManager

mgr = AirfoilManager()

# Import from file
mgr.import_airfoil_file("airfoil.dat", "my_airfoil")

# Download from database
mgr.download_airfoil("clarky")

# Generate NACA
mgr.create_naca_airfoil("2412")
```

## Advanced Usage

### Custom Optimization
Modify configuration files to:
- Set design variable ranges
- Define constraints
- Specify objectives
- Configure optimization algorithm

### Results Analysis
Results are saved to `output/` directory:
- `best_design.json`: Optimal design parameters
- `convergence.png`: Optimization progress
- Analysis logs and reports

### Extending AeroForge
The modular structure allows easy extension:
- Add new airfoil sources
- Implement advanced analysis methods
- Create custom constraint functions
- Integrate with CAD systems
