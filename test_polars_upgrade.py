#!/usr/bin/env python3
"""
Test script to verify Polars upgrade compatibility
"""

def test_imports():
    """Test if all modules can be imported without polars/aiosqlite"""
    print("Testing imports...")
    
    try:
        # Test basic imports that don't require polars/aiosqlite
        from src.utils.config import get_config
        print("✅ Config import successful")
        
        from src.data_generation.distributions import StatisticalDistributions
        print("✅ StatisticalDistributions import successful")
        
        print("✅ Basic imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_polars_availability():
    """Test if polars is available"""
    try:
        import polars as pl
        print(f"✅ Polars {pl.__version__} is available")
        return True
    except ImportError:
        print("⚠️  Polars not installed - run: pip install polars>=0.20.0")
        return False

def test_aiosqlite_availability():
    """Test if aiosqlite is available"""
    try:
        import aiosqlite
        print("✅ aiosqlite is available")
        return True
    except ImportError:
        print("⚠️  aiosqlite not installed - run: pip install aiosqlite>=0.19.0")
        return False

def main():
    """Run all tests"""
    print("=== Polars Upgrade Compatibility Test ===\n")
    
    basic_imports = test_imports()
    polars_available = test_polars_availability()
    aiosqlite_available = test_aiosqlite_availability()
    
    print(f"\n=== Test Results ===")
    print(f"Basic imports: {'✅' if basic_imports else '❌'}")
    print(f"Polars available: {'✅' if polars_available else '⚠️'}")
    print(f"aiosqlite available: {'✅' if aiosqlite_available else '⚠️'}")
    
    if basic_imports and polars_available and aiosqlite_available:
        print("\n🎉 All tests passed! Ready to use the upgraded codebase.")
        
        # Test a simple polars operation
        try:
            import polars as pl
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            result = df.with_columns((pl.col("a") * 2).alias("a_doubled"))
            print(f"✅ Polars test operation successful: {len(result)} rows")
        except Exception as e:
            print(f"❌ Polars operation failed: {e}")
    else:
        print("\n⚠️  Some dependencies missing. Install with:")
        print("pip install polars>=0.20.0 aiosqlite>=0.19.0")

if __name__ == "__main__":
    main()