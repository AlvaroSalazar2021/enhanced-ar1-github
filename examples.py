#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the Enhanced AR(1) Trend Analysis tool

This script shows different ways to use the Enhanced AR(1) analyzer
with various types of climate data.
"""

from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
import os

def example_basic_usage():
    """Basic usage example with auto-detection."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Auto-Detection")
    print("=" * 60)
    
    # Initialize with minimal parameters (auto-detect variable and dimensions)
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="your_data.nc",
        output_dir="basic_analysis_results"
    )
    
    # Run complete analysis
    try:
        analyzer.load_and_validate_data()
        analyzer.run_analysis()
        analyzer.create_shapefile()
        analyzer.create_visualization()
        analyzer.print_summary()
    except Exception as e:
        print(f"Error in basic example: {e}")

def example_era5_precipitation():
    """Example for ERA5 precipitation data."""
    print("=" * 60)
    print("EXAMPLE 2: ERA5 Precipitation Analysis")
    print("=" * 60)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="era5_precipitation.nc",
        variable_name="tp",           # ERA5 total precipitation variable
        data_scaling=0.1,             # ERA5 comes as mm*10
        units="mm/month",             # Units after scaling
        output_dir="era5_results"
    )
    
    try:
        analyzer.load_and_validate_data()
        analyzer.run_analysis()
        
        # Create shapefile with custom filename
        analyzer.create_shapefile("ERA5_precipitation_trends.shp")
        
        # Create visualization with custom extent (Bolivia)
        bolivia_extent = [-70, -57, -23, -9]  # [lon_min, lon_max, lat_min, lat_max]
        analyzer.create_visualization(
            filename="ERA5_precipitation_map.png",
            extent=bolivia_extent,
            title="ERA5 Precipitation Trends (1980-2022)"
        )
        
        analyzer.print_summary()
    except Exception as e:
        print(f"Error in ERA5 example: {e}")

def example_chirps_precipitation():
    """Example for CHIRPS precipitation data."""
    print("=" * 60)
    print("EXAMPLE 3: CHIRPS Precipitation Analysis")
    print("=" * 60)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="chirps_precipitation.nc",
        variable_name="precip",       # CHIRPS precipitation variable
        data_scaling=1.0,             # CHIRPS doesn't need scaling
        units="mm/month",
        output_dir="chirps_results"
    )
    
    try:
        analyzer.load_and_validate_data()
        analyzer.run_analysis()
        analyzer.create_shapefile("CHIRPS_precipitation_trends.shp")
        analyzer.create_visualization("CHIRPS_precipitation_map.png")
        analyzer.print_summary()
    except Exception as e:
        print(f"Error in CHIRPS example: {e}")

def example_temperature_data():
    """Example for temperature data analysis."""
    print("=" * 60)
    print("EXAMPLE 4: Temperature Data Analysis")
    print("=" * 60)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="temperature_data.nc",
        variable_name="t2m",          # 2-meter temperature
        data_scaling=1.0,
        units="°C/month",             # Temperature units
        output_dir="temperature_results"
    )
    
    try:
        analyzer.load_and_validate_data()
        analyzer.run_analysis()
        analyzer.create_shapefile("temperature_trends.shp")
        analyzer.create_visualization(
            filename="temperature_trends_map.png",
            title="Temperature Trends Analysis"
        )
        analyzer.print_summary()
    except Exception as e:
        print(f"Error in temperature example: {e}")

def example_custom_dimensions():
    """Example with custom dimension names."""
    print("=" * 60)
    print("EXAMPLE 5: Custom Dimension Names")
    print("=" * 60)
    
    # For datasets with non-standard dimension names
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="custom_data.nc",
        variable_name="my_variable",
        time_dim="datetime",          # Custom time dimension name
        lat_dim="latitude",           # Custom latitude dimension name
        lon_dim="longitude",          # Custom longitude dimension name
        data_scaling=1.0,
        units="units/month",
        output_dir="custom_results"
    )
    
    try:
        analyzer.load_and_validate_data()
        analyzer.run_analysis()
        analyzer.create_shapefile()
        analyzer.create_visualization()
        analyzer.print_summary()
    except Exception as e:
        print(f"Error in custom dimensions example: {e}")

def example_python_api_advanced():
    """Advanced example using Python API with custom processing."""
    print("=" * 60)
    print("EXAMPLE 6: Advanced Python API Usage")
    print("=" * 60)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file="advanced_data.nc",
        variable_name="precipitation",
        output_dir="advanced_results"
    )
    
    try:
        # Load data
        analyzer.load_and_validate_data()
        
        # Run analysis and get summary statistics
        summary = analyzer.run_analysis()
        
        # Print detailed statistics
        print(f"\\nDetailed Analysis Results:")
        print(f"Total pixels analyzed: {summary['total_pixels']:,}")
        print(f"Valid pixels: {summary['valid_pixels']:,}")
        print(f"Significant trends: {summary['significant_pixels']:,} ({summary['percent_significant']:.1f}%)")
        print(f"Increasing trends: {summary['increasing_trends']:,} ({summary['percent_increasing']:.1f}%)")
        print(f"Decreasing trends: {summary['decreasing_trends']:,} ({summary['percent_decreasing']:.1f}%)")
        
        # Conditional output creation based on results
        if summary['significant_pixels'] > 0:
            print("\\nCreating outputs (significant trends found)...")
            shapefile_path = analyzer.create_shapefile("significant_trends.shp")
            plot_path = analyzer.create_visualization("trend_map.png")
            print(f"Shapefile: {shapefile_path}")
            print(f"Plot: {plot_path}")
        else:
            print("\\nNo significant trends found. Skipping detailed outputs.")
        
        # Access raw results for custom processing
        if hasattr(analyzer, 'slopes'):
            import numpy as np
            print(f"\\nRaw Results Available:")
            print(f"Slope range: {np.nanmin(analyzer.slopes):.6f} to {np.nanmax(analyzer.slopes):.6f}")
            print(f"P-value range: {np.nanmin(analyzer.pvalues):.6f} to {np.nanmax(analyzer.pvalues):.6f}")
        
    except Exception as e:
        print(f"Error in advanced example: {e}")

def example_batch_processing():
    """Example for processing multiple files."""
    print("=" * 60)
    print("EXAMPLE 7: Batch Processing Multiple Files")
    print("=" * 60)
    
    # List of files to process
    files_to_process = [
        {"file": "era5_data.nc", "variable": "tp", "scaling": 0.1, "name": "ERA5"},
        {"file": "chirps_data.nc", "variable": "precip", "scaling": 1.0, "name": "CHIRPS"},
        {"file": "mswep_data.nc", "variable": "precipitation", "scaling": 1.0, "name": "MSWEP"}
    ]
    
    results_summary = []
    
    for file_info in files_to_process:
        print(f"\\nProcessing {file_info['name']}...")
        
        try:
            analyzer = EnhancedAR1TrendAnalyzer(
                input_file=file_info["file"],
                variable_name=file_info["variable"],
                data_scaling=file_info["scaling"],
                output_dir=f"batch_results_{file_info['name'].lower()}"
            )
            
            # Check if file exists
            if not os.path.exists(file_info["file"]):
                print(f"  ⚠️ File not found: {file_info['file']}")
                continue
            
            analyzer.load_and_validate_data()
            summary = analyzer.run_analysis()
            analyzer.create_shapefile(f"{file_info['name']}_trends.shp")
            analyzer.create_visualization(f"{file_info['name']}_map.png")
            
            # Store summary for comparison
            summary['dataset'] = file_info['name']
            results_summary.append(summary)
            
            print(f"  ✅ {file_info['name']} completed")
            
        except Exception as e:
            print(f"  ❌ Error processing {file_info['name']}: {e}")
    
    # Print comparison summary
    if results_summary:
        print("\\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"{'Dataset':<10} {'Total':<8} {'Significant':<12} {'Increasing':<12} {'Decreasing':<12}")
        print("-" * 60)
        
        for result in results_summary:
            print(f"{result['dataset']:<10} "
                  f"{result['total_pixels']:<8,} "
                  f"{result['significant_pixels']:<12,} "
                  f"{result['increasing_trends']:<12,} "
                  f"{result['decreasing_trends']:<12,}")

def main():
    """Run all examples (modify as needed)."""
    print("Enhanced AR(1) Trend Analysis - Example Usage")
    print("=" * 60)
    
    # Note: Comment out examples that don't have corresponding data files
    
    # Basic examples
    # example_basic_usage()
    # example_era5_precipitation()
    # example_chirps_precipitation()
    # example_temperature_data()
    # example_custom_dimensions()
    
    # Advanced examples
    # example_python_api_advanced()
    # example_batch_processing()
    
    print("\\n🎉 All examples completed!")
    print("\\nNote: Uncomment the example functions you want to run")
    print("and make sure you have the corresponding data files.")

if __name__ == "__main__":
    main()