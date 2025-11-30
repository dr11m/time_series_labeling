# Changelog

All notable changes to this project will be documented in this file.

## [3.0.0] - 2025-11-29

### Added
- **NumPy Data Format**: Complete migration to NumPy-based data format for improved performance and compatibility
  - Support for loading datasets from individual `.npy` files (prices, timestamps, ids, cluster_ids, labels)
  - Flexible metadata support via JSON files
  - Automatic detection and loading of label files with multiple naming conventions
- **Anomaly Detection Mode**: New labeling type for detecting anomalies in time series
  - Click-based anomaly point marking
  - Per-point labeling (0 = normal, 1 = anomaly)
  - Index-based auto-backup system
  - Resume functionality from last processed index
- **Enhanced Settings Window**: Improved configuration interface
  - Tabbed interface with organized sections (General, Labeling Settings, Similarity Search)
  - Real-time dataset information display
  - Metadata JSON editor with validation
  - Automatic settings loading from dataset metadata
- **Improved Similarity Search**: Enhanced Soft-DTW implementation
  - Processed data API for consistent similarity calculations
  - Visual comparison with labeled values display
  - Support for all labeling types (predict, classify, anomaly detection)
- **Advanced Visualization Features**:
  - Right padding visualization for future price prediction
  - Highlighting of last N points
  - Zoom region selection with rectangle selector
  - Predicted prices visualization (magenta markers)
  - Y-axis padding configuration
- **Export Functionality**: 
  - X/y data export for classification mode
  - Separate export for labeled and unlabeled data
  - Metadata snapshot with settings and statistics

### Changed
- **Data Format Migration**: Moved from JSON-based format to NumPy arrays for better performance
  - Individual `.npy` files for each data component
  - Backward compatibility maintained through automatic file detection
- **Application Architecture**: Complete refactoring with modular design
  - Separated labeling types into individual modules (`predict.py`, `classify.py`, `anomaly_detection.py`)
  - Base class architecture for extensibility
  - Plugin-based similarity search system
  - Settings management with JSON persistence
- **Labeling Workflow**: Improved user experience
  - Automatic navigation to first unlabeled sample
  - Periodic auto-backups (every 15 labels for predict/classify, every 10 indices for anomaly detection)
  - Backup file naming with dataset prefix
  - Progress tracking and resume functionality
- **Settings Management**: Enhanced configuration system
  - Settings saved to both `settings.json` and dataset `metadata.json`
  - Automatic restoration of settings from dataset metadata
  - Last metadata path tracking

### Technical Improvements
- **Code Structure**: Modular architecture with clear separation of concerns
  - `src/dataset_loader.py` - NumPy dataset loading
  - `src/labeling_types/` - Labeling implementations
  - `src/similarity/` - Similarity search algorithms
  - `src/settings/` - Settings management
- **Performance**: Optimized data loading and processing
  - Direct NumPy array operations
  - Efficient similarity search with processed data API
- **Error Handling**: Improved error messages and validation
  - File existence checks
  - Data shape validation
  - Clear error messages for common issues
- **Type Safety**: Added type hints throughout the codebase
- **Dependencies**: Cleaned up unused dependencies (removed seaborn, plotly, dash, ujson)

### Breaking Changes
- **Data Format**: NumPy format is now the primary data format
  - Old JSON format is no longer directly supported
  - Migration required: convert JSON datasets to NumPy format
- **Settings Structure**: Settings format has changed
  - New file-based configuration system
  - Settings stored per-dataset in metadata.json
- **API Changes**: Internal APIs have been refactored
  - Similarity finder now uses processed data API
  - Labeling types use base class architecture

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

### Bug Fixes
- Fixed syntax error in settings window (duplicate font parameter)
- Improved error handling for timestamp conversion

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
