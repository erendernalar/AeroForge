"""
AeroForge Main Application
Entry point for aircraft design optimization
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from optimizer.design_loop import AeroForgeOptimizer
from airfoil.xfoil_driver import XFOILDriver
from airfoil.airfoil_manager import AirfoilManager

def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('aeroforge.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check XFOIL
    xfoil_driver = XFOILDriver()
    if not xfoil_driver.check_xfoil_available():
        print("❌ XFOIL not found. Please install XFOIL and add to PATH")
        print("   Download from: https://web.mit.edu/drela/Public/web/xfoil/")
        return False
    else:
        print("✅ XFOIL is available")
    
    # Check Python packages
    try:
        import numpy, scipy, matplotlib
        print("✅ Core Python packages available")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        return False
    
    return True

def create_project_directories():
    """Create necessary project directories"""
    directories = [
        "config",
        "airfoils", 
        "output",
        "output/logs",
        "docs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='AeroForge Aircraft Design Optimization')
    parser.add_argument('--config', '-c', default='config/mission1.json',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration optimization')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AeroForge")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        print("✅ All dependencies satisfied")
        sys.exit(0)
    
    # Create directories
    create_project_directories()
    
    # Demo mode
    if args.demo:
        print("🎯 Running AeroForge demonstration...")
        run_demo()
        return
    
    # Check config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize optimizer
        optimizer = AeroForgeOptimizer(args.config)
        
        # Run optimization
        print("🚀 Starting aircraft design optimization...")
        results = optimizer.optimize()
        
        if results.success and results.best_designs:
            design, performance = results.best_designs[0]
            
            print("\n✅ Optimization completed successfully!")
            print(f"📊 Evaluations: {results.evaluation_count}")
            print(f"⏱️  Time: {results.total_time:.1f}s")
            print(f"🏆 Best Score: {performance.total_score:.2f}")
            print(f"✈️  L/D Ratio: {performance.L_D_ratio:.1f}")
            print(f"📏 Span: {design.span:.2f}m")
            print(f"📐 Wing Area: {design.wing_area:.3f}m²")
            print(f"📈 Wing Loading: {performance.wing_loading:.1f} N/m²")
            
            # Save results
            optimizer.save_results()
            print("💾 Results saved to output/")
            
        else:
            print("❌ Optimization failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

def run_demo():
    """Run a simple demonstration"""
    from airfoil.airfoil_manager import AirfoilManager
    
    print("🛩️  AeroForge Demo")
    
    # Initialize airfoil manager
    airfoil_mgr = AirfoilManager()
    
    # Generate NACA airfoil
    naca_coords = airfoil_mgr.create_naca_airfoil("2412")
    print(f"✅ Generated NACA 2412 with {len(naca_coords)} points")
    
    # List available airfoils
    available = airfoil_mgr.list_airfoils()
    print(f"📋 Available airfoils: {available}")
    
    print("\n📝 To run full optimization:")
    print("   python main.py --config config/enhanced_config.json")

if __name__ == "__main__":
    main()
