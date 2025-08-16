"""
Basic tests for GTZAN analysis
"""

import sys
import os
sys.path.append('..')

def test_imports():
    """Test that we can import the main analyzer."""
    try:
        from gtzan_analysis import GTZANAnalyzer
        print("✅ GTZANAnalyzer import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_analyzer_initialization():
    """Test analyzer can be initialized."""
    try:
        from gtzan_analysis import GTZANAnalyzer
        analyzer = GTZANAnalyzer()
        print("✅ GTZANAnalyzer initialization successful")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running basic tests...")
    test1 = test_imports()
    test2 = test_analyzer_initialization()
    
    if test1 and test2:
        print("✅ All basic tests passed!")
    else:
        print("❌ Some tests failed")
