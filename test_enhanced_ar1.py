#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Enhanced AR(1) Trend Analysis tool

This script performs basic validation and testing of the trend analysis tool
to ensure it works correctly with your data.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'xarray', 'numpy', 'scipy', 'matplotlib', 
        'cartopy', 'geopandas', 'shapely', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages available")
        return True

def test_tool_import():
    """Test that the main tool can be imported."""
    print("\n🔍 Testing tool import...")
    
    try:
        from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
        print("  ✅ Enhanced AR(1) tool imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Cannot import tool: {e}")
        return False

def create_test_data(filename="test_data.nc"):
    """Create a simple test dataset for validation."""
    print(f"\n🔧 Creating test dataset: {filename}")
    
    # Create synthetic time series data
    np.random.seed(42)  # For reproducible results
    
    # Define dimensions
    time = pd.date_range('2000-01-01', '2022-12-31', freq='MS')  # Monthly data
    lat = np.arange(-20, -10, 0.5)  # Bolivia-like latitudes
    lon = np.arange(-70, -60, 0.5)  # Bolivia-like longitudes
    
    # Create base precipitation field
    base_precip = 50 + 30 * np.sin(2 * np.pi * np.arange(len(time)) / 12)  # Seasonal cycle
    
    # Add spatial and temporal variations
    data = np.zeros((len(time), len(lat), len(lon)))
    
    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            # Add spatial gradient
            spatial_factor = 1 + 0.3 * (lat_val + 15) / 5  # Gradient with latitude
            
            # Add trend (some pixels increasing, some decreasing, some no trend)
            if (i + j) % 3 == 0:
                trend = 0.1 * np.arange(len(time)) / 12  # Increasing trend
            elif (i + j) % 3 == 1:
                trend = -0.05 * np.arange(len(time)) / 12  # Decreasing trend
            else:
                trend = 0  # No trend
            
            # Add noise and autocorrelation
            noise = np.random.normal(0, 10, len(time))
            # Simple AR(1) process for autocorrelation
            for t in range(1, len(time)):
                noise[t] += 0.3 * noise[t-1]
            
            data[:, i, j] = (base_precip + trend + noise) * spatial_factor
    
    # Ensure no negative values
    data = np.maximum(data, 0)
    
    # Create xarray dataset
    ds = xr.Dataset({
        'precipitation': (['time', 'lat', 'lon'], data)
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon
    })
    
    # Add attributes
    ds['precipitation'].attrs = {
        'units': 'mm/month',
        'long_name': 'Monthly Precipitation',
        'description': 'Synthetic precipitation data for testing'
    }
    
    # Save to NetCDF
    ds.to_netcdf(filename)
    print(f"  ✅ Test dataset created: {filename}")
    print(f"     Dimensions: {len(time)} time steps, {len(lat)} lats, {len(lon)} lons")
    print(f"     Data range: {np.min(data):.1f} to {np.max(data):.1f} mm/month")
    
    return filename

def test_basic_analysis(test_file):
    """Test basic analysis functionality."""
    print(f"\n🧪 Testing basic analysis with {test_file}")
    
    try:
        from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
        
        # Initialize analyzer
        analyzer = EnhancedAR1TrendAnalyzer(
            input_file=test_file,
            variable_name="precipitation",
            output_dir="test_results"
        )
        
        print("  ✅ Analyzer initialized")
        
        # Load and validate data
        analyzer.load_and_validate_data()
        print("  ✅ Data loaded and validated")
        
        # Run analysis (on small subset for speed)
        print("  🔄 Running analysis...")
        summary = analyzer.run_analysis()
        print("  ✅ Analysis completed")
        
        # Print summary
        print(f"     Total pixels: {summary['total_pixels']}")
        print(f"     Valid pixels: {summary['valid_pixels']}")
        print(f"     Significant pixels: {summary['significant_pixels']} ({summary['percent_significant']:.1f}%)")
        
        return True, analyzer
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        return False, None

def test_outputs(analyzer):
    """Test output generation."""
    print("\n📤 Testing output generation...")
    
    try:
        # Test shapefile creation
        shp_path = analyzer.create_shapefile("test_trends.shp")
        print(f"  ✅ Shapefile created: {shp_path}")
        
        # Test visualization creation
        plot_path = analyzer.create_visualization("test_map.png")
        print(f"  ✅ Visualization created: {plot_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Output generation failed: {e}")
        return False

def test_config_functionality():
    """Test configuration functionality."""
    print("\n⚙️ Testing configuration functionality...")
    
    try:
        from config import get_config, print_available_configs
        
        # Test getting ERA5 config
        era5_config = get_config('era5', 'precipitation')
        print(f"  ✅ ERA5 config retrieved: {era5_config['variable_name']}")
        
        # Test getting CHIRPS config
        chirps_config = get_config('chirps', 'precipitation')
        print(f"  ✅ CHIRPS config retrieved: {chirps_config['variable_name']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config functionality failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    files_to_remove = ['test_data.nc']
    dirs_to_remove = ['test_results']
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"  🗑️ Removed: {file}")
    
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"  🗑️ Removed: {dir_path}/")

def main():
    """Run all tests."""
    print("Enhanced AR(1) Trend Analysis - Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False
        print("\n❌ Cannot proceed without required packages")
        return
    
    # Test 2: Tool import
    if not test_tool_import():
        all_tests_passed = False
        print("\n❌ Cannot proceed without main tool")
        return
    
    # Test 3: Create test data
    try:
        test_file = create_test_data()
    except Exception as e:
        print(f"\n❌ Cannot create test data: {e}")
        all_tests_passed = False
        return
    
    # Test 4: Basic analysis
    analysis_success, analyzer = test_basic_analysis(test_file)
    if not analysis_success:
        all_tests_passed = False
    
    # Test 5: Output generation
    if analyzer is not None:
        if not test_outputs(analyzer):
            all_tests_passed = False
    
    # Test 6: Configuration functionality
    if not test_config_functionality():
        all_tests_passed = False
    
    # Final results
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The Enhanced AR(1) tool is ready for use.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above.")
    
    # Cleanup
    cleanup_test_files()
    
    print("\n📚 Next steps:")
    print("1. Check the README.md for detailed usage instructions")
    print("2. Look at examples.py for usage examples")
    print("3. Use config.py for pre-configured dataset settings")
    print("4. Run your analysis with: python enhanced_ar1_trend_analysis.py --help")

if __name__ == "__main__":
    main()