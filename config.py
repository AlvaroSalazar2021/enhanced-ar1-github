#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration templates for common climate datasets

This file contains pre-configured settings for popular climate datasets
to make it easier to use the Enhanced AR(1) trend analysis tool.
"""

# Common dataset configurations
DATASET_CONFIGS = {
    'era5': {
        'description': 'ERA5 Reanalysis Data',
        'common_variables': {
            'precipitation': {
                'variable_name': 'tp',
                'scaling': 0.1,  # ERA5 precipitation comes as mm*10
                'units': 'mm/month',
                'long_name': 'Total Precipitation'
            },
            'temperature': {
                'variable_name': 't2m',
                'scaling': 1.0,
                'units': '°C',
                'long_name': '2m Temperature'
            },
            'wind_speed': {
                'variable_name': ['u10', 'v10'],  # Need to calculate magnitude
                'scaling': 1.0,
                'units': 'm/s',
                'long_name': '10m Wind Speed'
            }
        },
        'dimension_mapping': {
            'time': 'valid_time',  # ERA5 uses 'valid_time'
            'lat': 'latitude',
            'lon': 'longitude'
        }
    },
    
    'chirps': {
        'description': 'CHIRPS Precipitation Dataset',
        'common_variables': {
            'precipitation': {
                'variable_name': 'precip',
                'scaling': 1.0,
                'units': 'mm/month',
                'long_name': 'Precipitation'
            }
        },
        'dimension_mapping': {
            'time': 'time',
            'lat': 'latitude',
            'lon': 'longitude'
        }
    },
    
    'mswep': {
        'description': 'MSWEP Precipitation Dataset',
        'common_variables': {
            'precipitation': {
                'variable_name': 'precipitation',
                'scaling': 1.0,
                'units': 'mm/month',
                'long_name': 'Precipitation'
            }
        },
        'dimension_mapping': {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon'
        }
    },
    
    'gpcp': {
        'description': 'GPCP Precipitation Dataset',
        'common_variables': {
            'precipitation': {
                'variable_name': 'precip',
                'scaling': 1.0,
                'units': 'mm/month',
                'long_name': 'Precipitation'
            }
        },
        'dimension_mapping': {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon'
        }
    },
    
    'berkeley_earth': {
        'description': 'Berkeley Earth Temperature Dataset',
        'common_variables': {
            'temperature': {
                'variable_name': 'temperature',
                'scaling': 1.0,
                'units': '°C',
                'long_name': 'Temperature Anomaly'
            }
        },
        'dimension_mapping': {
            'time': 'time',
            'lat': 'latitude',
            'lon': 'longitude'
        }
    },
    
    'cru': {
        'description': 'CRU TS Dataset',
        'common_variables': {
            'temperature': {
                'variable_name': 'tmp',
                'scaling': 1.0,
                'units': '°C',
                'long_name': 'Temperature'
            },
            'precipitation': {
                'variable_name': 'pre',
                'scaling': 1.0,
                'units': 'mm/month',
                'long_name': 'Precipitation'
            }
        },
        'dimension_mapping': {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon'
        }
    }
}

# Regional extents for common study areas
REGIONAL_EXTENTS = {
    'global': None,  # Auto-detect
    'bolivia': [-70, -57, -23, -9],
    'south_america': [-85, -30, -60, 15],
    'amazon': [-80, -45, -20, 10],
    'andes': [-80, -60, -25, 10],
    'caribbean': [-90, -55, 10, 25],
    'central_america': [-92, -75, 5, 20],
    'north_america': [-170, -50, 15, 75],
    'europe': [-15, 40, 35, 75],
    'africa': [-20, 55, -40, 40],
    'asia': [60, 150, -10, 55],
    'australia': [110, 160, -45, -10]
}

def get_config(dataset_name, variable_type='precipitation'):
    """
    Get configuration for a specific dataset and variable.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'era5', 'chirps', 'mswep')
    variable_type : str
        Type of variable (e.g., 'precipitation', 'temperature')
    
    Returns:
    --------
    dict : Configuration dictionary
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    if variable_type not in dataset_config['common_variables']:
        available = list(dataset_config['common_variables'].keys())
        raise ValueError(f"Variable '{variable_type}' not found for {dataset_name}. Available: {available}")
    
    var_config = dataset_config['common_variables'][variable_type]
    dim_mapping = dataset_config['dimension_mapping']
    
    return {
        'variable_name': var_config['variable_name'],
        'data_scaling': var_config['scaling'],
        'units': var_config['units'],
        'time_dim': dim_mapping.get('time', 'time'),
        'lat_dim': dim_mapping.get('lat', 'lat'),
        'lon_dim': dim_mapping.get('lon', 'lon'),
        'description': f"{dataset_config['description']} - {var_config['long_name']}"
    }

def print_available_configs():
    """Print all available dataset configurations."""
    print("Available Dataset Configurations:")
    print("=" * 50)
    
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n{dataset_name.upper()}: {config['description']}")
        print("  Variables:")
        for var_type, var_config in config['common_variables'].items():
            print(f"    {var_type}: {var_config['variable_name']} ({var_config['units']})")

def print_regional_extents():
    """Print all available regional extents."""
    print("Available Regional Extents:")
    print("=" * 50)
    
    for region_name, extent in REGIONAL_EXTENTS.items():
        if extent is None:
            print(f"{region_name}: Auto-detect from data")
        else:
            print(f"{region_name}: Lon [{extent[0]}°, {extent[1]}°], Lat [{extent[2]}°, {extent[3]}°]")

# Example usage functions
def create_era5_analyzer(input_file, variable_type='precipitation', region='bolivia', output_dir=None):
    """Create an analyzer for ERA5 data with pre-configured settings."""
    from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
    
    config = get_config('era5', variable_type)
    extent = REGIONAL_EXTENTS.get(region)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file=input_file,
        variable_name=config['variable_name'],
        time_dim=config['time_dim'],
        lat_dim=config['lat_dim'],
        lon_dim=config['lon_dim'],
        data_scaling=config['data_scaling'],
        units=config['units'],
        output_dir=output_dir
    )
    
    return analyzer, extent

def create_chirps_analyzer(input_file, region='bolivia', output_dir=None):
    """Create an analyzer for CHIRPS data with pre-configured settings."""
    from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
    
    config = get_config('chirps', 'precipitation')
    extent = REGIONAL_EXTENTS.get(region)
    
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file=input_file,
        variable_name=config['variable_name'],
        time_dim=config['time_dim'],
        lat_dim=config['lat_dim'],
        lon_dim=config['lon_dim'],
        data_scaling=config['data_scaling'],
        units=config['units'],
        output_dir=output_dir
    )
    
    return analyzer, extent

def quick_analysis(input_file, dataset_type, variable_type='precipitation', 
                  region='global', output_dir=None, create_outputs=True):
    """
    Run a quick analysis with pre-configured settings.
    
    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file
    dataset_type : str
        Type of dataset (e.g., 'era5', 'chirps', 'mswep')
    variable_type : str
        Type of variable (default: 'precipitation')
    region : str
        Region name for map extent (default: 'global')
    output_dir : str
        Output directory (optional)
    create_outputs : bool
        Whether to create shapefile and visualization
    
    Returns:
    --------
    dict : Analysis summary
    """
    from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer
    
    # Get configuration
    config = get_config(dataset_type, variable_type)
    extent = REGIONAL_EXTENTS.get(region)
    
    # Create analyzer
    analyzer = EnhancedAR1TrendAnalyzer(
        input_file=input_file,
        variable_name=config['variable_name'],
        time_dim=config['time_dim'],
        lat_dim=config['lat_dim'],
        lon_dim=config['lon_dim'],
        data_scaling=config['data_scaling'],
        units=config['units'],
        output_dir=output_dir
    )
    
    # Run analysis
    print(f"Running {config['description']} analysis...")
    analyzer.load_and_validate_data()
    summary = analyzer.run_analysis()
    
    # Create outputs if requested
    if create_outputs:
        analyzer.create_shapefile()
        analyzer.create_visualization(extent=extent)
    
    analyzer.print_summary()
    
    return summary

if __name__ == "__main__":
    # Print available configurations
    print_available_configs()
    print("\n")
    print_regional_extents()
    
    # Example usage
    print("\n" + "=" * 50)
    print("Example Usage:")
    print("=" * 50)
    print("""
# Quick analysis examples:
from config import quick_analysis, create_era5_analyzer

# ERA5 precipitation analysis for Bolivia
summary = quick_analysis('era5_data.nc', 'era5', 'precipitation', 'bolivia')

# CHIRPS precipitation analysis for South America  
summary = quick_analysis('chirps_data.nc', 'chirps', 'precipitation', 'south_america')

# Custom analyzer creation
analyzer, extent = create_era5_analyzer('era5_temp.nc', 'temperature', 'europe')
analyzer.load_and_validate_data()
analyzer.run_analysis()
analyzer.create_visualization(extent=extent)
""")