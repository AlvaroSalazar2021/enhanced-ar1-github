# Enhanced AR(1) Trend Analysis

A robust Python implementation for detecting trends in time series raster data using Enhanced AR(1) methodology with improved autocorrelation treatment.

## Features

- **Enhanced AR(1) Methodology**: Implements conservative autocorrelation correction for robust trend detection
- **Universal Compatibility**: Works with any NetCDF raster file containing time series data
- **Automatic Detection**: Auto-detects variable names and dimension structures
- **Flexible Configuration**: Supports custom scaling, units, and dimension names
- **Multiple Outputs**: Generates shapefiles and visualizations
- **Statistical Rigor**: Conservative significance testing with effective degrees of freedom

## Mathematical Approach

The Enhanced AR(1) method implements three key improvements over standard trend analysis:

1. **Conservative Sample Size Correction**: Uses multi-lag autocorrelation to calculate effective sample size using Bretherton et al. (1999) formula with additional conservative factors.

2. **Ultra-Conservative Variance Correction**: Applies robust variance correction for residual autocorrelation effects.

3. **Strict Significance Criteria**: Uses effective degrees of freedom for more reliable statistical significance testing.

## Installation

### Requirements

```bash
pip install xarray numpy scipy matplotlib cartopy geopandas shapely pandas
```

### Optional (for better performance)
```bash
pip install netcdf4 dask
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python enhanced_ar1_trend_analysis.py --input data.nc --variable precipitation
```

#### ERA5 Data Example
```bash
python enhanced_ar1_trend_analysis.py --input era5.nc --variable tp --scaling 0.1 --units "mm/month"
```

#### CHIRPS Data Example
```bash
python enhanced_ar1_trend_analysis.py --input chirps.nc --variable precip --output chirps_results
```

#### Full Configuration Example
```bash
python enhanced_ar1_trend_analysis.py \
    --input data.nc \
    --variable precipitation \
    --output results \
    --scaling 1.0 \
    --units "mm/month" \
    --time-dim time \
    --lat-dim latitude \
    --lon-dim longitude \
    --extent -70 -57 -23 -9
```

### Python API

```python
from enhanced_ar1_trend_analysis import EnhancedAR1TrendAnalyzer

# Initialize analyzer
analyzer = EnhancedAR1TrendAnalyzer(
    input_file="data.nc",
    variable_name="precipitation",  # Optional: auto-detects if None
    data_scaling=1.0,              # Apply scaling if needed (e.g., 0.1 for ERA5)
    units="mm/month"               # Units after scaling
)

# Load and validate data
analyzer.load_and_validate_data()

# Run analysis
summary = analyzer.run_analysis()

# Create outputs
shapefile_path = analyzer.create_shapefile()
plot_path = analyzer.create_visualization()

# Print summary
analyzer.print_summary()
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input NetCDF file path | Required |
| `--variable` | `-v` | Variable name to analyze | Auto-detect |
| `--output` | `-o` | Output directory | Auto-generate |
| `--scaling` | `-s` | Data scaling factor | 1.0 |
| `--units` | `-u` | Data units after scaling | mm/month |
| `--time-dim` | | Time dimension name | time |
| `--lat-dim` | | Latitude dimension name | lat |
| `--lon-dim` | | Longitude dimension name | lon |
| `--extent` | | Map extent [lon_min, lon_max, lat_min, lat_max] | Auto-detect |
| `--no-shapefile` | | Skip shapefile creation | False |
| `--no-plot` | | Skip visualization creation | False |

## Supported Data Formats

### Automatic Variable Detection
The tool automatically detects common variable names:
- **Precipitation**: `precipitation`, `precip`, `pr`, `tp`, `total_precipitation`
- **Temperature**: `temperature`, `temp`, `tas`, `t2m`, `air_temperature`
- **Wind**: `wind_speed`, `wspd`, `ws`, `wind`
- **Humidity**: `humidity`, `hum`, `rh`, `relative_humidity`

### Automatic Dimension Mapping
The tool automatically maps common dimension names:
- **Time**: `valid_time`, `datetime`, `date` → `time`
- **Latitude**: `latitude`, `y`, `north` → `lat`
- **Longitude**: `longitude`, `x`, `east` → `lon`

### Common Data Sources
- **ERA5**: Use `--scaling 0.1` (data comes as mm*10)
- **CHIRPS**: Usually no scaling needed
- **MSWEP**: Usually no scaling needed
- **Custom datasets**: Specify scaling as needed

## Output Files

### Shapefile Attributes
- `longitude`, `latitude`: Coordinates
- `slope`: Trend slope per time step
- `p_value`: Statistical significance
- `slope_year`: Annual trend (slope × 12 for monthly data)
- `trend`: Categorical trend classification
- `significan`: Significance flag
- `direction`: Trend direction
- `method`: Analysis method used
- `variable`: Variable name
- `units`: Data units

### Trend Categories
- `VStrong+/VStrong-`: Very strong increasing/decreasing (|slope| > 0.05)
- `Strong+/Strong-`: Strong increasing/decreasing (|slope| > 0.02)
- `Weak+/Weak-`: Weak increasing/decreasing (significant but |slope| ≤ 0.02)
- `NoTrend`: No significant trend

## Examples for Different Data Types

### ERA5 Precipitation
```bash
python enhanced_ar1_trend_analysis.py \
    --input era5_precip.nc \
    --variable tp \
    --scaling 0.1 \
    --units "mm/month"
```

### CHIRPS Precipitation
```bash
python enhanced_ar1_trend_analysis.py \
    --input chirps.nc \
    --variable precip
```

### Temperature Data
```bash
python enhanced_ar1_trend_analysis.py \
    --input temperature.nc \
    --variable t2m \
    --units "°C/month"
```

### Custom Dataset with Non-Standard Dimensions
```bash
python enhanced_ar1_trend_analysis.py \
    --input custom_data.nc \
    --variable my_variable \
    --time-dim datetime \
    --lat-dim y \
    --lon-dim x
```

## Troubleshooting

### Common Issues

1. **Variable not found**: Use `--variable` to specify the exact variable name
2. **Dimension errors**: Check dimension names with `ncdump -h file.nc` and use appropriate `--*-dim` arguments
3. **Memory issues**: For large datasets, consider using `dask` for lazy loading
4. **No significant trends**: Normal for some datasets; check data quality and time series length

### Debugging Tips

```bash
# Check dataset structure
ncdump -h your_file.nc

# Check variable names
python -c "import xarray as xr; print(list(xr.open_dataset('your_file.nc').variables.keys()))"

# Check dimensions
python -c "import xarray as xr; ds=xr.open_dataset('your_file.nc'); print(ds.dims)"
```

## Scientific Background

The Enhanced AR(1) methodology addresses limitations of standard trend analysis in the presence of temporal autocorrelation:

1. **Autocorrelation Impact**: Temporal autocorrelation inflates significance and reduces effective sample size
2. **Conservative Approach**: Multiple correction factors ensure robust statistical inference
3. **Seasonal Patterns**: Multi-lag autocorrelation analysis captures seasonal effects
4. **Effective Sample Size**: Bretherton et al. (1999) correction with additional conservative factors

### Key References
- Bretherton, C. S., et al. (1999). The effective number of spatial degrees of freedom of a time-varying field. Journal of Climate.
- von Storch, H., & Zwiers, F. W. (1999). Statistical analysis in climate research.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Citation

If you use this code in your research, please cite:

```
Enhanced AR(1) Trend Analysis Tool
Climate Analysis Team
https://github.com/your-repo/enhanced-ar1-trend-analysis
```