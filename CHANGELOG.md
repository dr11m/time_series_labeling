# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2025-08-23

### Added
- **Enhanced Display Settings**: New configuration options for improved visualization
  - `SHOW_TIMESTAMPS_AS_DATES`: Display timestamps as readable dates instead of numbers
  - `SHOW_CURRENT_DATE`: Add current date as virtual point on plots for real-time reference
  - Automatic date formatting with matplotlib's DateFormatter and AutoDateLocator
- **Labeled Values Visualization**: Display existing labeled values on plots
  - Horizontal lines showing labeled prices with values in legend
  - Works in both main plot and similar patterns subplots
  - Automatic normalization support for labeled values
- **Improved Settings Window**: Enhanced configuration interface
  - New "Display Settings" section with checkboxes for date and current date options
  - Better organized layout with clear section headers
  - Increased window size to accommodate new settings

### Changed
- **Plot Visualization**: Enhanced matplotlib integration
  - Automatic date axis formatting when timestamps are displayed as dates
  - Better legend management for multiple labeled values
  - Improved color scheme: orange for labeled values, red for current date
- **User Experience**: More intuitive interface
  - Clear visual feedback for labeled values
  - Better date representation for time series analysis
  - Enhanced settings organization

### Technical Improvements
- **Code Structure**: Cleaner implementation of display options
- **Configuration Management**: Direct access to mandatory settings without getattr fallbacks
- **Documentation**: Updated README with new configuration options

## [2.0.0] - 2025-08-22

### Added
- **Automatic Configuration Loader**: `cfg_loader.py` automatically creates `cfg.py` from `cfg.example.py` if missing
- **Universal Data Format**: Complete migration to JSON format with Pydantic models (TimeSeriesDataset, TimeSeries, TimeSeriesPoint)
  - Support for time series of any length (no longer limited to 15 points)
  - Flexible metadata structure and multiple labeled values per series
  - Backward compatibility with existing labeled data

- **Advanced Data Filtering System** : `examples/filtering_data_example.ipynb` - Comprehensive filtering capabilities for data preprocessing
  - **Duplicate Removal**: Automatic removal of duplicate timestamps within series
  - **Length-based Filtering**: Filter series by minimum/maximum length requirements
  - **Sale Time Analysis**: Sophisticated filtering based on sales interval statistics
    - Trimmed mean analysis (removing outliers while preserving current date interval)
    - Coefficient of variation (CV) filtering for stability assessment
    - Range ratio analysis for distribution uniformity
    - Gap ratio detection for clustering identification
    - Outlier ratio analysis for data quality assessment
  - **Current Date Integration**: All metrics include interval to current date for real-time analysis
  - **Visual Analysis Tools**: Detailed visualization of rejected series with borderline case analysis

- **Enhanced Data Processing Pipeline**:
  - Multi-step filtering workflow (duplicates → length → sale time analysis)
  - Configurable filtering parameters with detailed documentation
  - Comprehensive statistics and metadata tracking
  - Export capabilities for filtered datasets

- **Improved Documentation**:
  - Detailed README with data format specifications
  - Example notebook for data filtering (`examples/filtering_data_example.ipynb`)
  - Comprehensive parameter explanations and recommendations
  - Integration examples for database adapters

### Changed
- **Data Format Migration**: Moved from CSV-based format to universal JSON format
- **Application Architecture**: Updated to support flexible data sources and formats
- **Filtering Logic**: Enhanced filtering system with multiple criteria and visual feedback
- **Configuration Management**: Automated configuration setup and validation

### Technical Improvements
- **Performance**: Optimized data processing for large datasets
- **Flexibility**: Modular design allowing easy addition of new filtering criteria
- **User Experience**: Better error handling, progress reporting, and automatic setup
- **Maintainability**: Cleaner code structure with comprehensive documentation and type safety

## [1.0.0] - 2025-04-24

### Initial Release
- Basic time series labeling functionality
- CSV-based data format support
- Interactive matplotlib-based interface
- DTW similarity search
- Basic filtering by series length

---

## Versioning

This project uses [Semantic Versioning](http://semver.org/). For the versions available, see the [tags on this repository](https://github.com/dr11m/time_series_labeling/tags).

## Migration Guide

### From v1.0 to v2.0

1. **Data Format Migration**:
   - Convert your CSV data to the new JSON format using the provided examples
   - Use the `csv_to_json_format` function in the README for automatic conversion

2. **Configuration Updates**:
   - Update your `cfg/cfg.py` to point to JSON files instead of CSV
   - Review and adjust filtering parameters in the example notebook

3. **New Features**:
   - Explore the new filtering capabilities in `examples/filtering_data_example.ipynb`
   - Configure filtering parameters based on your specific requirements

### Breaking Changes
- CSV format is no longer supported as the primary data format
- Some configuration parameters have been renamed or restructured
- Filtering logic has been completely redesigned for better performance and flexibility
