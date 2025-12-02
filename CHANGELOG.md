# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2025-12-02

### Added
- **Python 3.13 support**: Full compatibility with Python 3.13.x
- **Enhanced scikit-learn compatibility**: Automatic compatibility layer for models trained with sklearn 0.22+ to work with sklearn 1.3-1.5
- **UV package manager support**: Optimized for use with UV for faster dependency resolution and builds
- **Comprehensive compatibility fixes**: 
  - IsotonicRegression attribute migration (`_necessary_X_`/`_necessary_y_` → `X_thresholds_`/`y_thresholds_`)
  - TfidfVectorizer unpickling fixes for newer sklearn versions
  - CalibratedClassifierCV attribute updates (`base_estimator` → `estimator`)

### Improved
- **CI/CD performance**: Migrated from pip to UV for 3x faster dependency installation
- **Code quality**: Fixed all linting issues and improved code formatting
- **Build system**: Enhanced GitHub Actions workflows with latest action versions
- **Development status**: Promoted from Alpha to Beta reflecting stability improvements

### Changed
- **scikit-learn version range**: Now supports sklearn 1.3.0 to 1.5.x (was pinned to 1.3.x)
- **Build backend**: Optimized hatchling configuration for UV compatibility
- **Dependency management**: Streamlined dependency groups and version constraints

### Technical Details
- **Backward compatibility**: Models trained with sklearn 0.22.2 now work seamlessly with sklearn 1.5.x
- **Performance**: Significantly faster package builds and CI runs
- **Robustness**: Enhanced error handling for sklearn version mismatches
- **Future-proof**: Compatibility layer designed to handle future sklearn updates

### Migration Notes
- **No breaking changes**: Existing code continues to work without modifications
- **Version warnings**: sklearn version warnings are expected and harmless
- **Performance boost**: Consider using UV for faster development workflows

## [0.2.3] - 2024-xx-xx

### Previous releases
- Initial implementations of soft news classification
- UK and US model support
- URL pattern classification
- CLI tools and Python API