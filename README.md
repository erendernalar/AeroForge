# AeroForge 🚀  
**AI-Powered Aircraft Design and Optimization Engine**

AeroForge is an intelligent aerodynamic design and optimization system that evaluates and evolves aircraft designs. It uses XFOIL for airfoil analysis and implements Lifting Line Theory for full-wing performance evaluation.

## ✈️ Key Features

- ✅ Direct integration with XFOIL for 2D airfoil analysis
- ✅ Custom-built 3D analysis using Lifting Line Theory (LLT)
- ✅ Fully automated design and optimization pipeline
- ✅ Multi-objective scoring and constraint filtering
- ✅ Support for custom airfoil import (NACA, files, online download)
- ✅ Professional Python project structure

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install XFOIL (external dependency)
# Download from: https://web.mit.edu/drela/Public/web/xfoil/

# Check installation
python main.py --check

# Run demo
python main.py --demo

# Run optimization
python main.py --config config/enhanced_config.json
```

## 📊 Usage Examples

### Basic Optimization
```bash
python main.py --config config/mission1.json
```

### Custom Airfoil Import
```python
from airfoil.airfoil_manager import AirfoilManager

mgr = AirfoilManager()
mgr.import_airfoil_file("path/to/airfoil.dat", "my_airfoil")
mgr.download_airfoil("clarky")  # Download Clark Y
```

## 📁 Project Structure

```
AeroForge/
├── airfoil/          # 2D airfoil analysis
├── wing/             # 3D wing analysis (LLT)
├── optimizer/        # Optimization engine
├── config/           # Mission configurations
├── airfoils/         # Airfoil database
├── output/           # Results and plots
└── main.py           # Application entry point
```

## 🔧 Dependencies

- Python 3.8+
- XFOIL (external)
- NumPy, SciPy, Matplotlib
- See requirements.txt for complete list

## 📜 License

MIT License – free to use, contribute, and modify.
