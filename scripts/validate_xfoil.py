#!/usr/bin/env python3
"""
XFOIL Installation Validator
Checks if XFOIL is properly installed and accessible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airfoil.xfoil_driver import XFOILDriver

def main():
    print("🔍 Validating XFOIL installation...")
    
    driver = XFOILDriver()
    
    if driver.check_xfoil_available():
        print("✅ XFOIL is available and working")
        
        # Test airfoil generation
        try:
            coords = driver.generate_naca_airfoil("0012")
            print(f"✅ Generated NACA 0012 with {len(coords)} points")
            
            # Test analysis
            polar = driver.analyze_airfoil(coords, reynolds=500000)
            if len(polar['alpha']) > 0:
                print(f"✅ Analysis successful: CL_max = {polar['CL_max']:.3f}")
            else:
                print("⚠️  Analysis returned no data")
                
        except Exception as e:
            print(f"❌ XFOIL test failed: {e}")
            return False
            
    else:
        print("❌ XFOIL not found or not working")
        print("📝 Installation instructions:")
        print("   1. Download XFOIL from: https://web.mit.edu/drela/Public/web/xfoil/")
        print("   2. Extract and compile (or use pre-compiled version)")
        print("   3. Add XFOIL executable to system PATH")
        print("   4. Test with: xfoil")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
