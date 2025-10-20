#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AR(1) Trend Analysis for Time Series Raster Data
=========================================================

A robust implementation of Enhanced AR(1) methodology for detecting trends 
in time series raster data with improved autocorrelation treatment.

Features:
- Enhanced AR(1) method with conservative autocorrelation correction
- Supports any NetCDF raster with time series data
- Automatic variable and dimension detection
- Robust statistical significance testing
- Shapefile and visualization export

Mathematical Approach:
1. Conservative sample size correction using multi-lag autocorrelation
2. Ultra-conservative variance correction for residual autocorrelation
3. Strict significance criteria with effective degrees of freedom

Usage:
    python enhanced_ar1_trend_analysis.py --input data.nc --variable precipitation
    
Requirements:
- xarray, numpy, scipy, matplotlib, cartopy, geopandas, shapely

Author: Climate Analysis Team
License: MIT
Version: 1.0
"""

import xarray as xr
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os
import time
import argparse
import sys
from pathlib import Path
from scipy import stats
from scipy.stats import t
from shapely.geometry import Point

# Configure matplotlib for non-interactive backend
mpl.use('Agg')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class EnhancedAR1TrendAnalyzer:
    """
    Enhanced AR(1) Trend Analysis for raster time series data.
    
    This class implements a robust methodology for detecting trends in 
    spatiotemporal data with proper treatment of temporal autocorrelation.
    """
    
    def __init__(self, input_file, variable_name=None, time_dim='time', 
                 lat_dim='lat', lon_dim='lon', output_dir=None, 
                 data_scaling=1.0, units='mm/month'):
        """
        Initialize the Enhanced AR(1) Trend Analyzer.
        
        Parameters:
        -----------
        input_file : str
            Path to input NetCDF file
        variable_name : str, optional
            Name of variable to analyze. If None, will auto-detect
        time_dim : str, default 'time'
            Name of time dimension
        lat_dim : str, default 'lat'
            Name of latitude dimension
        lon_dim : str, default 'lon'
            Name of longitude dimension
        output_dir : str, optional
            Output directory. If None, creates based on input filename
        data_scaling : float, default 1.0
            Scaling factor to apply to data (e.g., 0.1 for ERA5)
        units : str, default 'mm/month'
            Units for the data after scaling
        """
        self.input_file = Path(input_file)
        self.variable_name = variable_name
        self.time_dim = time_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.data_scaling = data_scaling
        self.units = units
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = Path(f"Enhanced_AR1_Results_{self.input_file.stem}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.dataset = None
        self.data = None
        self.slopes = None
        self.pvalues = None
        self.trends = None
        
        print("🚀 Enhanced AR(1) Trend Analyzer Initialized")
        print("=" * 60)
    
    def load_and_validate_data(self):
        """Load and validate the input dataset."""
        print(f"📁 Loading dataset: {self.input_file}")
        
        # Check if file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Load dataset
        try:
            self.dataset = xr.open_dataset(self.input_file)
            print(f"✅ Dataset loaded successfully")
            print(f"   Available variables: {list(self.dataset.variables.keys())}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
        
        # Auto-detect variable if not specified
        if self.variable_name is None:
            self.variable_name = self._detect_main_variable()
        
        if self.variable_name not in self.dataset.variables:
            raise ValueError(f"Variable '{self.variable_name}' not found in dataset")
        
        print(f"🌧️ Analyzing variable: '{self.variable_name}'")
        self.data = self.dataset[self.variable_name]
        
        # Validate and rename dimensions
        self._validate_and_rename_dimensions()
        
        # Apply scaling if needed
        if self.data_scaling != 1.0:
            print(f"⚙️ Applying data scaling factor: {self.data_scaling}")
            self.data = self.data * self.data_scaling
        
        # Display dataset information
        self._display_dataset_info()
        
        print(f"✅ Data validation completed successfully")
    
    def _detect_main_variable(self):
        """Auto-detect the main data variable to analyze."""
        # Common variable names for different data types
        common_vars = [
            'precipitation', 'precip', 'pr', 'tp', 'total_precipitation',
            'temperature', 'temp', 'tas', 't2m', 'air_temperature',
            'wind_speed', 'wspd', 'ws', 'wind',
            'humidity', 'hum', 'rh', 'relative_humidity'
        ]
        
        # Look for common variable names
        for var in common_vars:
            if var in self.dataset.variables:
                return var
        
        # If no common names found, look for variables with time dimension
        time_vars = []
        for var_name, var in self.dataset.variables.items():
            if len(var.dims) >= 3:  # At least 3D (time, lat, lon)
                # Check if it has time-like dimension
                for dim in var.dims:
                    if any(time_word in dim.lower() for time_word in ['time', 'date']):
                        time_vars.append(var_name)
                        break
        
        if len(time_vars) == 1:
            return time_vars[0]
        elif len(time_vars) > 1:
            print(f"Multiple time series variables found: {time_vars}")
            print(f"Using first one: {time_vars[0]}")
            return time_vars[0]
        else:
            raise ValueError("Could not auto-detect main variable. Please specify variable_name parameter.")
    
    def _validate_and_rename_dimensions(self):
        """Validate and standardize dimension names."""
        print(f"📊 Dataset information:")
        print(f"   Dimensions: {self.data.dims}")
        print(f"   Shape: {self.data.shape}")
        print(f"   Coordinates: {list(self.data.coords.keys())}")
        
        # Check for required dimensions and rename if necessary
        dim_mapping = {
            # Time dimension variants
            'valid_time': 'time',
            'datetime': 'time',
            'date': 'time',
            # Latitude dimension variants
            'latitude': 'lat',
            'y': 'lat',
            'north': 'lat',
            # Longitude dimension variants
            'longitude': 'lon',
            'x': 'lon',
            'east': 'lon'
        }
        
        # Apply dimension renaming
        for old_name, new_name in dim_mapping.items():
            if old_name in self.data.dims:
                self.data = self.data.rename({old_name: new_name})
                print(f"   Renamed dimension: {old_name} -> {new_name}")
        
        # Verify required dimensions exist
        required_dims = {'time', 'lat', 'lon'}
        available_dims = set(self.data.dims)
        
        if not required_dims.issubset(available_dims):
            missing = required_dims - available_dims
            raise ValueError(f"Missing required dimensions: {missing}. Available: {available_dims}")
    
    def _display_dataset_info(self):
        """Display information about the loaded dataset."""
        lats = self.data.coords['lat'].values
        lons = self.data.coords['lon'].values
        times = self.data.coords['time'].values
        
        print(f"📐 Dataset dimensions:")
        print(f"   Time steps: {len(times)} ({times[0]} to {times[-1]})")
        print(f"   Latitudes: {len(lats)} points ({np.nanmin(lats):.3f}° to {np.nanmax(lats):.3f}°)")
        print(f"   Longitudes: {len(lons)} points ({np.nanmin(lons):.3f}° to {np.nanmax(lons):.3f}°)")
        print(f"   Total pixels: {len(lats) * len(lons):,}")
        print(f"   Data units: {self.units}")
    
    def enhanced_autocorrelation_treatment(self, ts):
        """
        Enhanced AR(1) treatment with improved autocorrelation correction.
        
        This method implements multiple approaches to handle high autocorrelation:
        1. Conservative effective sample size correction
        2. Ultra-conservative variance correction  
        3. Strict significance criteria
        
        Parameters:
        -----------
        ts : array-like
            Time series data
            
        Returns:
        --------
        slope : float
            Trend slope per time step
        p_value : float
            Statistical significance (p-value)
        trend_category : str
            Categorical trend classification
        method_info : str
            Information about the method used
        """
        try:
            n = len(ts)
            if n < 24:
                return 0, 1, 'no_trend', 'insufficient_data'
            
            # Remove missing values
            valid_mask = ~np.isnan(ts)
            ts_clean = ts[valid_mask]
            n_clean = len(ts_clean)
            
            if n_clean < 24:
                return 0, 1, 'no_trend', 'insufficient_data'
            
            # ---- APPROACH 1: Conservative effective sample size correction ----
            # Calculate autocorrelation up to lag 12 to detect seasonal patterns
            max_lag = min(12, n_clean // 4)
            autocorr_values = []
            
            for lag in range(1, max_lag + 1):
                if n_clean > lag:
                    corr = np.corrcoef(ts_clean[:-lag], ts_clean[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            
            if len(autocorr_values) == 0:
                effective_n = n_clean
            else:
                # Use average autocorrelation of first lags (more conservative)
                avg_autocorr = np.mean(autocorr_values[:min(3, len(autocorr_values))])
                
                # Apply Bretherton et al. (1999) formula with conservative correction
                if avg_autocorr > 0.1:
                    # Stricter correction factor
                    correction_factor = (1 - avg_autocorr) / (1 + avg_autocorr)
                    effective_n = max(10, n_clean * correction_factor * 0.5)  # Additional 0.5 factor
                else:
                    effective_n = n_clean
            
            # ---- APPROACH 2: Ultra-conservative variance correction ----
            # Trend analysis with robust correction
            x = np.arange(n_clean)
            
            # Basic linear regression
            slope, intercept, r_value, p_value_raw, std_err = stats.linregress(x, ts_clean)
            
            # Calculate variance with autocorrelation effects
            residuals = ts_clean - (slope * x + intercept)
            
            # Estimate residual autocorrelation (more robust)
            if n_clean > 3:
                residual_lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                if np.isnan(residual_lag1_corr):
                    residual_lag1_corr = 0
            else:
                residual_lag1_corr = 0
            
            # Ultra-conservative variance correction
            variance_correction = 1.0
            if abs(residual_lag1_corr) > 0.1:
                # More conservative formula including higher order terms
                variance_correction = (1 + abs(residual_lag1_corr)) / (1 - abs(residual_lag1_corr))
                variance_correction = min(variance_correction, 5.0)  # Limit extreme correction
            
            # Recalculate standard error with correction
            corrected_std_err = std_err * np.sqrt(variance_correction)
            
            # ---- APPROACH 3: Strict significance criteria ----
            # Use effective_n for degrees of freedom
            df = max(8, effective_n - 2)
            
            # Calculate t-statistic with corrected error
            if corrected_std_err > 0:
                t_stat = slope / corrected_std_err
                # Two-tailed p-value (more conservative)
                p_value = 2 * (1 - t.cdf(abs(t_stat), df))
            else:
                p_value = 1.0
            
            # ---- ULTRA-CONSERVATIVE CATEGORIZATION ----
            # Strict thresholds for trend classification
            abs_slope = abs(slope)
            
            # Multiple criteria for categorization
            is_significant = p_value < 0.05
            is_strong = abs_slope > 0.02  # per time step (stricter)
            is_very_strong = abs_slope > 0.05
            
            # Categorization with multiple criteria
            if not is_significant:
                trend_category = 'no_trend'
            else:
                if slope > 0:
                    if is_very_strong:
                        trend_category = 'very_strong_increasing'
                    elif is_strong:
                        trend_category = 'strong_increasing'
                    else:
                        trend_category = 'weak_increasing'
                else:
                    if is_very_strong:
                        trend_category = 'very_strong_decreasing'
                    elif is_strong:
                        trend_category = 'strong_decreasing'
                    else:
                        trend_category = 'weak_decreasing'
            
            return slope, p_value, trend_category, f'enhanced_ar1_n{int(effective_n)}'
            
        except Exception as e:
            print(f"⚠️ Error in time series analysis: {e}")
            return 0, 1, 'no_trend', 'error'
    
    def run_analysis(self):
        """Run the Enhanced AR(1) trend analysis on the entire dataset."""
        print("\n🔄 Starting trend analysis...")
        start_time = time.time()
        
        # Get dimensions
        lats = self.data.coords['lat'].values
        lons = self.data.coords['lon'].values
        n_lat, n_lon = len(lats), len(lons)
        
        # Initialize result arrays
        self.slopes = np.full((n_lat, n_lon), np.nan)
        self.pvalues = np.full((n_lat, n_lon), np.nan)
        self.trends = np.full((n_lat, n_lon), 'no_trend', dtype=object)
        
        # Pixel-by-pixel analysis
        total_pixels = n_lat * n_lon
        processed = 0
        
        print("\n🧮 Processing pixels...")
        print("Progress: [          ] 0%", end='', flush=True)
        
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series
                ts = self.data.isel(lat=i, lon=j).values
                
                # Enhanced AR(1) analysis
                slope, pvalue, trend, method = self.enhanced_autocorrelation_treatment(ts)
                
                self.slopes[i, j] = slope
                self.pvalues[i, j] = pvalue
                self.trends[i, j] = trend
                
                processed += 1
                
                # Update progress every 5%
                if processed % (total_pixels // 20) == 0:
                    percent = int(100 * processed / total_pixels)
                    progress_bar = '█' * (percent // 10) + '░' * (10 - percent // 10)
                    print(f"\rProgress: [{progress_bar}] {percent}%", end='', flush=True)
        
        analysis_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {analysis_time:.1f} seconds")
        
        # Create xarray DataArrays for easier handling
        self.slope_data = xr.DataArray(self.slopes, coords=[('lat', lats), ('lon', lons)])
        self.pvalue_data = xr.DataArray(self.pvalues, coords=[('lat', lats), ('lon', lons)])
        self.trend_data = xr.DataArray(self.trends, coords=[('lat', lats), ('lon', lons)])
        
        return self._get_analysis_summary()
    
    def _get_analysis_summary(self):
        """Generate summary statistics of the analysis."""
        valid_mask = np.isfinite(self.slopes) & np.isfinite(self.pvalues)
        n_valid = np.sum(valid_mask)
        n_significant = np.sum((self.pvalues < 0.05) & valid_mask)
        n_increasing = np.sum((self.pvalues < 0.05) & (self.slopes > 0) & valid_mask)
        n_decreasing = np.sum((self.pvalues < 0.05) & (self.slopes < 0) & valid_mask)
        
        total_pixels = len(self.data.coords['lat']) * len(self.data.coords['lon'])
        
        summary = {
            'total_pixels': total_pixels,
            'valid_pixels': n_valid,
            'significant_pixels': n_significant,
            'increasing_trends': n_increasing,
            'decreasing_trends': n_decreasing,
            'percent_significant': 100 * n_significant / n_valid if n_valid > 0 else 0,
            'percent_increasing': 100 * n_increasing / n_valid if n_valid > 0 else 0,
            'percent_decreasing': 100 * n_decreasing / n_valid if n_valid > 0 else 0
        }
        
        return summary
    
    def create_shapefile(self, filename=None):
        """
        Create a shapefile with trend analysis results.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename. If None, auto-generates based on variable name
        """
        if self.slopes is None:
            raise ValueError("Analysis must be run before creating shapefile")
        
        print(f"📍 Creating shapefile...")
        
        if filename is None:
            filename = f"Enhanced_AR1_Trends_{self.variable_name}.shp"
        
        filepath = self.output_dir / filename
        
        # Prepare data lists
        geometries = []
        slopes = []
        pvalues = []
        trends = []
        lons = []
        lats = []
        significance = []
        trend_direction = []
        slope_annual = []
        
        lat_coords = self.data.coords['lat'].values
        lon_coords = self.data.coords['lon'].values
        
        # Iterate over all pixels
        for i in range(len(lat_coords)):
            for j in range(len(lon_coords)):
                # Extract coordinates as numeric values
                lon = float(lon_coords[j] if hasattr(lon_coords[j], 'item') 
                          else lon_coords[j])
                lat = float(lat_coords[i] if hasattr(lat_coords[i], 'item') 
                          else lat_coords[i])
                
                # Extract analysis data
                slope = (self.slopes[i, j] if hasattr(self.slopes[i, j], 'item') 
                        else self.slopes[i, j])
                pvalue = (self.pvalues[i, j] if hasattr(self.pvalues[i, j], 'item') 
                         else self.pvalues[i, j])
                trend_raw = self.trends[i, j]
                
                # Skip invalid values
                if not (np.isfinite(lon) and np.isfinite(lat) and 
                       np.isfinite(slope) and np.isfinite(pvalue)):
                    continue
                
                # Convert trend to shorter format for shapefile
                trend = self._format_trend_for_shapefile(trend_raw)
                
                # Create point geometry
                point = Point(lon, lat)
                geometries.append(point)
                
                # Add basic data
                slopes.append(float(slope))
                pvalues.append(float(pvalue))
                trends.append(trend)
                lons.append(float(lon))
                lats.append(float(lat))
                
                # Calculate annual trend (assuming monthly data, multiply by 12)
                slope_annual.append(float(slope * 12))
                
                # Categorize significance
                is_significant = float(pvalue) < 0.05
                significance.append('Significant' if is_significant else 'Not_Signif')
                
                # Categorize trend direction
                if is_significant:
                    trend_direction.append('Increasing' if float(slope) > 0 else 'Decreasing')
                else:
                    trend_direction.append('No_Trend')
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'longitude': lons,
            'latitude': lats,
            'slope': slopes,
            'p_value': pvalues,
            'slope_year': slope_annual,
            'trend': trends,
            'significan': significance,  # Max 10 chars for shapefile
            'direction': trend_direction,
            'method': ['Enhanced_AR1'] * len(geometries),
            'variable': [self.variable_name] * len(geometries),
            'units': [self.units] * len(geometries)
        })
        
        # Set coordinate reference system
        gdf.crs = "EPSG:4326"
        
        # Save shapefile
        gdf.to_file(filepath, driver='ESRI Shapefile')
        
        print(f"✅ Shapefile saved: {filepath}")
        print(f"   Total points: {len(gdf):,}")
        
        if len(gdf) > 0:
            self._print_shapefile_summary(gdf)
        
        return filepath
    
    def _format_trend_for_shapefile(self, trend_raw):
        """Format trend category for shapefile (max 10 characters)."""
        if isinstance(trend_raw, str):
            trend_mapping = {
                'very_strong_increasing': 'VStrong+',
                'very_strong_decreasing': 'VStrong-',
                'strong_increasing': 'Strong+',
                'strong_decreasing': 'Strong-',
                'weak_increasing': 'Weak+',
                'weak_decreasing': 'Weak-',
                'no_trend': 'NoTrend'
            }
            return trend_mapping.get(trend_raw, str(trend_raw)[:10])
        else:
            return str(trend_raw)[:10]
    
    def _print_shapefile_summary(self, gdf):
        """Print summary statistics for the created shapefile."""
        n_significant = len(gdf[gdf['significan'] == 'Significant'])
        n_increase = len(gdf[gdf['direction'] == 'Increasing'])
        n_decrease = len(gdf[gdf['direction'] == 'Decreasing'])
        n_no_trend = len(gdf[gdf['direction'] == 'No_Trend'])
        
        print(f"   Significant pixels: {n_significant:,} ({100*n_significant/len(gdf):.1f}%)")
        print(f"   Increasing trends: {n_increase:,}")
        print(f"   Decreasing trends: {n_decrease:,}")
        print(f"   No trend: {n_no_trend:,}")
    
    def create_visualization(self, filename=None, extent=None, title=None):
        """
        Create a visualization of the trend analysis results.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename for the plot
        extent : list, optional
            Map extent as [lon_min, lon_max, lat_min, lat_max]
        title : str, optional
            Custom title for the plot
        """
        if self.slopes is None:
            raise ValueError("Analysis must be run before creating visualization")
        
        print(f"🎨 Creating visualization...")
        
        if filename is None:
            filename = f"Enhanced_AR1_Trends_{self.variable_name}.png"
        
        filepath = self.output_dir / filename
        
        # Configure figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Convert to numpy arrays
        slope_np = (self.slope_data.values if hasattr(self.slope_data, 'values') 
                   else self.slope_data)
        pvalue_np = (self.pvalue_data.values if hasattr(self.pvalue_data, 'values') 
                    else self.pvalue_data)
        lon_np = self.data.coords['lon'].values
        lat_np = self.data.coords['lat'].values
        
        # Create significance mask
        significance_mask = (pvalue_np < 0.05) & np.isfinite(pvalue_np) & np.isfinite(slope_np)
        
        # Create array for visualization (only significant values)
        trend_visual = np.full_like(slope_np, np.nan)
        trend_visual[significance_mask] = slope_np[significance_mask]
        
        # Create meshgrids for plotting
        lon_mesh, lat_mesh = np.meshgrid(lon_np, lat_np)
        
        # Main plot (only finite values)
        finite_mask = np.isfinite(trend_visual)
        if np.any(finite_mask):
            # Determine color scale based on data
            data_range = np.nanpercentile(trend_visual[finite_mask], [5, 95])
            vmax = max(abs(data_range[0]), abs(data_range[1]))
            vmin = -vmax
            
            im = ax.pcolormesh(lon_mesh, lat_mesh, trend_visual, 
                              cmap='RdBu_r', 
                              vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
        else:
            print("⚠️ No finite data to visualize")
            im = None
        
        # Configure map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.7)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        
        # Configure grid
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set extent
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            # Auto-determine extent from data
            lon_margin = (np.max(lon_np) - np.min(lon_np)) * 0.05
            lat_margin = (np.max(lat_np) - np.min(lat_np)) * 0.05
            auto_extent = [np.min(lon_np) - lon_margin, np.max(lon_np) + lon_margin,
                          np.min(lat_np) - lat_margin, np.max(lat_np) + lat_margin]
            ax.set_extent(auto_extent, crs=ccrs.PlateCarree())
        
        # Set title
        if title is None:
            title = f'{self.variable_name.title()} Trends (Enhanced AR1)\nSignificant values only (p < 0.05)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        if im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.05)
            cbar.set_label(f'Trend ({self.units} per time step)', fontsize=12)
        
        # Add statistical information
        total_pixels = len(lat_np) * len(lon_np)
        significant_pixels = np.sum(significance_mask)
        percent_significant = 100 * significant_pixels / total_pixels if total_pixels > 0 else 0
        
        ax.text(0.02, 0.98, f'Significant pixels: {significant_pixels:,} ({percent_significant:.1f}%)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis results."""
        if self.slopes is None:
            print("❌ No analysis results available. Run analysis first.")
            return
        
        summary = self._get_analysis_summary()
        
        print("\n📊 ANALYSIS SUMMARY:")
        print("=" * 50)
        print(f"Dataset: {self.input_file.name}")
        print(f"Variable: {self.variable_name}")
        print(f"Units: {self.units}")
        print(f"Method: Enhanced AR(1)")
        print(f"\nSpatial Coverage:")
        print(f"  Total pixels: {summary['total_pixels']:,}")
        print(f"  Valid pixels: {summary['valid_pixels']:,}")
        print(f"\nTrend Statistics:")
        print(f"  Significant pixels: {summary['significant_pixels']:,} ({summary['percent_significant']:.1f}%)")
        print(f"  Increasing trends: {summary['increasing_trends']:,} ({summary['percent_increasing']:.1f}%)")
        print(f"  Decreasing trends: {summary['decreasing_trends']:,} ({summary['percent_decreasing']:.1f}%)")
        print(f"\nOutput Directory: {self.output_dir}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Enhanced AR(1) Trend Analysis for Time Series Raster Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_ar1_trend_analysis.py --input data.nc --variable precipitation
  python enhanced_ar1_trend_analysis.py --input era5.nc --variable tp --scaling 0.1
  python enhanced_ar1_trend_analysis.py --input chirps.nc --output chirps_trends
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input NetCDF file path')
    parser.add_argument('--variable', '-v',
                       help='Variable name to analyze (auto-detect if not specified)')
    parser.add_argument('--output', '-o',
                       help='Output directory (auto-generate if not specified)')
    parser.add_argument('--scaling', '-s', type=float, default=1.0,
                       help='Data scaling factor (default: 1.0)')
    parser.add_argument('--units', '-u', default='mm/month',
                       help='Data units after scaling (default: mm/month)')
    parser.add_argument('--time-dim', default='time',
                       help='Time dimension name (default: time)')
    parser.add_argument('--lat-dim', default='lat',
                       help='Latitude dimension name (default: lat)')
    parser.add_argument('--lon-dim', default='lon',
                       help='Longitude dimension name (default: lon)')
    parser.add_argument('--extent', nargs=4, type=float, metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
                       help='Map extent for visualization')
    parser.add_argument('--no-shapefile', action='store_true',
                       help='Skip shapefile creation')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization creation')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = EnhancedAR1TrendAnalyzer(
            input_file=args.input,
            variable_name=args.variable,
            time_dim=args.time_dim,
            lat_dim=args.lat_dim,
            lon_dim=args.lon_dim,
            output_dir=args.output,
            data_scaling=args.scaling,
            units=args.units
        )
        
        # Load and validate data
        analyzer.load_and_validate_data()
        
        # Run analysis
        analyzer.run_analysis()
        
        # Create outputs
        if not args.no_shapefile:
            analyzer.create_shapefile()
        
        if not args.no_plot:
            analyzer.create_visualization(extent=args.extent)
        
        # Print summary
        analyzer.print_summary()
        
        print("\n🎉 Enhanced AR(1) analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()