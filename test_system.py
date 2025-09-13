#!/usr/bin/env python3
"""
Quick test of the ROM system to verify all components work
"""

import sys
import os
sys.path.append('.')

def test_import():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from main import HeatROMSystem, hertz_source, TUIMenu
        print("✅ Main system imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_rom_system():
    """Test basic ROM system functionality"""
    print("\nTesting ROM system...")
    
    try:
        from main import HeatROMSystem, hertz_source
        
        # Initialize system
        rom = HeatROMSystem()
        print("✅ ROM system initialized")
        
        # Check if matrices exist or need training
        if not rom.check_rom_matrices_exist():
            print("No existing ROM matrices found. Testing training...")
            # Quick training with smaller parameters for testing
            rom.train_rom_model(Nx=101, Nt=100, q0=1e3, Lx=1, Lt=10)
            print("✅ ROM training completed")
        else:
            rom.load_rom_matrices()
            print("✅ Existing ROM matrices loaded")
        
        # Setup ROM
        if rom.setup_rom(Nr=5):
            print("✅ ROM setup successful")
        else:
            print("❌ ROM setup failed")
            return False
        
        # Test prediction
        import numpy as np
        test_params = {'velocity': 0.1, 'width': 0.1, 'intensity': 1000}
        t_test = np.linspace(0, 5, 20)
        
        T, pred_time = rom.predict(hertz_source, t_test, test_params, verbose=False)
        
        if T is not None:
            print(f"✅ ROM prediction successful")
            print(f"   Prediction time: {pred_time:.4f} seconds")
            print(f"   Max temperature: {np.max(T):.1f} K")
            print(f"   Temperature shape: {T.shape}")
            return True
        else:
            print("❌ ROM prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ ROM system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tui_initialization():
    """Test TUI system initialization"""
    print("\nTesting TUI initialization...")
    
    try:
        from main import TUIMenu
        
        # Just test initialization, not the interactive loop
        menu = TUIMenu()
        print("✅ TUI system initialized")
        
        # Test header generation
        print("Testing TUI components...")
        menu.print_header()
        menu.print_status()
        
        print("✅ TUI components working")
        return True
        
    except Exception as e:
        print(f"❌ TUI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("1D HEAT EQUATION ROM SYSTEM - INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_import),
        ("ROM System Test", test_rom_system),
        ("TUI Initialization Test", test_tui_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! System is ready for use.")
        print("\nTo run the full system:")
        print("python main.py")
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())