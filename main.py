#!/usr/bin/env python3
"""
AeroForge - AI-Powered Aircraft Design and Optimization Engine
Main application entry point
"""

import argparse
import sys
import json
import subprocess
import os
import numpy as np
from pathlib import Path

def check_xfoil_installation():
    try:
        subprocess.run(['xfoil'], input='\nquit\n', text=True, capture_output=True, timeout=10)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def check_dependencies():
    required_modules = ['numpy', 'scipy', 'matplotlib', 'json']
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing

def check_project_structure():
    required_dirs = ['airfoil', 'wing', 'optimizer', 'config', 'airfoils', 'output']
    required_files = ['requirements.txt']
    missing = []

    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(f"Directory: {dir_name}/")

    for file_name in required_files:
        if not Path(file_name).exists():
            missing.append(f"File: {file_name}")

    return missing

def run_system_check():
    print("🔍 AeroForge System Check")
    print("=" * 40)

    print("Checking XFOIL installation...", end=" ")
    if check_xfoil_installation():
        print("✅ OK")
    else:
        print("❌ FAILED")
        print("  └─ XFOIL not found. Please install from: https://web.mit.edu/drela/Public/web/xfoil/")
        return False

    print("Checking Python dependencies...", end=" ")
    missing_deps = check_dependencies()
    if not missing_deps:
        print("✅ OK")
    else:
        print("❌ FAILED")
        print(f"  └─ Missing: {', '.join(missing_deps)}")
        print("  └─ Run: pip install -r requirements.txt")
        return False

    print("Checking project structure...", end=" ")
    missing_structure = check_project_structure()
    if not missing_structure:
        print("✅ OK")
    else:
        print("❌ FAILED")
        for item in missing_structure:
            print(f"  └─ Missing: {item}")
        return False

    print("Checking module imports...", end=" ")
    try:
        from airfoil.xfoil_driver import XFOILDriver
        from airfoil.airfoil_manager import AirfoilManager
        from wing import LiftingLineTheory
        from optimizer.evaluator import DesignEvaluator
        print("✅ OK")
    except ImportError as e:
        print("❌ FAILED")
        print(f"  └─ Import error: {e}")
        return False

    print("\n✅ All checks passed! AeroForge is ready to use.")
    return True

def run_demo():
    print("🚀 Running AeroForge Demo")
    print("=" * 30)

    try:
        from airfoil import AirfoilManager
        from optimizer import DesignEvaluator

        Path("output").mkdir(exist_ok=True)

        print("Initializing airfoil manager...")
        airfoil_mgr = AirfoilManager()

        print("Loading NACA 2412 airfoil...")
        coords = airfoil_mgr.create_naca_airfoil("2412")
        np.savetxt("output/demo_airfoil.dat", coords, fmt="%.6f")

        print("Demo completed successfully! ✅")
        print("Check output/ directory for results.")
        return True

    except Exception as e:
        print(f"Demo failed: {e} ❌")
        return False

def run_optimization(config_file):
    print(f"🔄 Running optimization with config: {config_file}")
    print("=" * 50)

    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False

    try:
        with open(config_file, 'r') as f:
            json.load(f)
        print("✅ Configuration loaded successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

    try:
        from optimizer import DesignLoop
        Path("output").mkdir(exist_ok=True)

        print("Initializing optimization...")
        optimizer = DesignLoop(config_file)  # Your DesignLoop expects file path

        print("Starting optimization loop...")
        results = optimizer.optimize()

        optimizer.save_results()

        print("✅ Optimization completed successfully!")
        print("Results saved to output/")
        return True

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="AeroForge - AI-Powered Aircraft Design and Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --check                           # System check
  python main.py --demo                            # Run demo
  python main.py --config config/mission1.json     # Run optimization
        """
    )

    parser.add_argument('--check', action='store_true', help='Run system check to verify installation')
    parser.add_argument('--demo', action='store_true', help='Run a simple demonstration')
    parser.add_argument('--config', type=str, help='Path to configuration file for optimization')
    parser.add_argument('--version', action='version', version='AeroForge 1.0.0')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return 0

    success = True

    if args.check:
        success = run_system_check()

    if args.demo:
        if not run_system_check():
            print("\n❌ System check failed. Please fix issues before running demo.")
            return 1
        success = run_demo()

    if args.config:
        if not run_system_check():
            print("\n❌ System check failed. Please fix issues before running optimization.")
            return 1
        success = run_optimization(args.config)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
