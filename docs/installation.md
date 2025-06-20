# AeroForge Installation Guide

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 1GB free space

## Dependencies

### Python Packages
All Python dependencies are listed in `requirements.txt` and can be installed with pip.

### External Dependencies
- **XFOIL**: Required for 2D airfoil analysis
  - Download: https://web.mit.edu/drela/Public/web/xfoil/
  - Must be accessible via command line

## Installation Steps

### 1. Clone/Download Project
```bash
# Using the setup script
chmod +x setup_aeroforge.sh
./setup_aeroforge.sh
```

### 2. Install Python Dependencies
```bash
cd AeroForge
pip install -r requirements.txt
```

### 3. Install XFOIL
- Download XFOIL from MIT website
- Extract and compile (or use pre-compiled version)
- Add to system PATH

### 4. Verify Installation
```bash
python main.py --check
python scripts/validate_xfoil.py
```

### 5. Run Tests
```bash
python -m pytest tests/
```

## Troubleshooting

### XFOIL Not Found
- Ensure XFOIL is in system PATH
- Test with: `xfoil` command in terminal
- Check executable permissions

### Python Package Issues
- Use virtual environment: `python -m venv aeroforge_env`
- Activate: `source aeroforge_env/bin/activate`
- Install packages: `pip install -r requirements.txt`

### Permission Issues
- Make scripts executable: `chmod +x scripts/*.py`
- Check directory permissions

## Optional Components

### GUI Dependencies (Future)
- PyQt5 for graphical interface
- pyvista for 3D visualization

### Development Tools
- pytest for testing
- black for code formatting
- flake8 for linting
