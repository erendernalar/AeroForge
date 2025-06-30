# xfoil_debug_runner.py
"""
Standalone test script for debugging XFOIL airfoil analysis
"""
import logging
from airfoil.xfoil_driver import XFOILDriver

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("XFOILTest")

# Instantiate driver
driver = XFOILDriver()

# Check if XFOIL is available
if not driver.check_xfoil_available():
    logger.error("XFOIL is not available in the system PATH. Please install and ensure it's accessible.")
    exit(1)

# Try generating and analyzing NACA 2412
try:
    logger.info("Generating NACA 2412 airfoil")
    coords = driver.generate_naca_airfoil("2412")
    print(f"Generated airfoil with {len(coords)} points")

    logger.info("Analyzing airfoil with Reynolds = 500000")
    result = driver.analyze_airfoil(
        coordinates=coords,
        reynolds=500000,
        mach=0.0,
        alpha_range=(-4, 14),
        alpha_step=0.5
    )

    if result['CL'].size > 0:
        print("\n✅ XFOIL Analysis Results:")
        print(f"CL_max: {result['CL_max']:.3f} at α = {result['alpha_CL_max']:.1f}°")
        print(f"(L/D)_max: {result['L_D_max']:.1f} at α = {result['alpha_L_D_max']:.1f}°")
    else:
        print("❌ XFOIL returned no valid data. Check log for details.")

except Exception as e:
    logger.exception(f"Exception during analysis: {e}")
