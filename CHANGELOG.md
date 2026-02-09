# Changelog

All notable changes to the Predictive Maintenance MCP Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-01-15

### Fixed
- Package layout: added `[tool.setuptools.package-dir]` and `[tool.setuptools.packages.find]` for correct `src/` layout resolution
- Version sync: aligned `src/__init__.py` version with `pyproject.toml` (0.3.2)
- Fixed `test_suite.py` sys.path (was pointing to wrong directory)
- Fixed `test_reports.py` calling async functions without await
- Fixed `validate_server.py` checking for obsolete tool names
- Fixed `setup_claude.ps1` hardcoded placeholder paths (now uses `$PSScriptRoot`)

### Added
- `smithery.yaml` for Smithery MCP registry listing
- `.vscode/mcp.json` example configuration for VS Code
- `Dockerfile` for containerized deployment
- `SECURITY.md` security policy
- GitHub issue templates (bug report, feature request) and PR template
- Comprehensive CHANGELOG entries for all versions

### Changed
- Updated CHANGELOG.md with complete version history
- Improved README with PyPI badge, npx quickstart, and corrected documentation

## [0.3.1] - 2025-01-10

### Fixed
- Improved error handling in signal loading
- Better validation of sampling rate parameters
- Fixed edge cases in ISO 20816-3 zone classification

### Changed
- Enhanced PCA visualization report generation
- Improved bearing catalog search accuracy

## [0.3.0] - 2025-01-05

### Added
- **Machine Documentation Reader** (production-ready)
  - `extract_manual_specs()` - Extract bearing/gear specs from PDF/TXT manuals
  - `calculate_bearing_characteristic_frequencies()` - ISO 15243:2017 calculations
  - `read_manual_excerpt()` - Configurable page-limit manual reading
  - `search_bearing_catalog()` - Local catalog with 20+ common bearings
  - MCP Resources: `manual://list` and `manual://read/{filename}`
  - JSON caching system for repeated queries
- **PCA Visualization** - `generate_pca_visualization_report()` for feature space exploration
- **Signal Plotting** - `plot_signal()` and `plot_spectrum()` tools
- **Unit Conversion** - Automatic acceleration-to-velocity conversion in ISO evaluation

### Changed
- Upgraded from PoC documentation reader (v0.2.1) to production-ready module
- Improved ML pipeline with better feature normalization

## [0.2.1] - 2025-11-15

### Added
- **Machine Documentation Reader (Beta)**
  - PDF/TXT manual text extraction via `document_reader.py`
  - Regex-based bearing and RPM extraction
  - Basic bearing frequency calculation
  - Initial caching system

### Changed
- Improved test coverage for envelope analysis
- Better error messages for missing data files

## [0.2.0] - 2025-11-11

### Added
- **Professional HTML Report Generation System**
  - Interactive Plotly visualizations with modern, responsive design
  - `generate_fft_report()` - FFT spectrum analysis with peak detection
  - `generate_envelope_report()` - Bearing fault detection with frequency markers
  - `generate_iso_report()` - ISO 20816-3 compliance evaluation with zone charts
  - `list_html_reports()` - List all generated reports with metadata
  - `get_report_info()` - Extract metadata without loading full HTML

- **Real Bearing Vibration Dataset**
  - 20 production-quality signals from real machinery tests (train: 14, test: 6)
  - 3 healthy baselines, 7 inner race faults, 10 outer race faults
  - Sampling rates: 48.8-97.7 kHz, durations: 3-6 seconds (varies by signal)
  - Complete metadata with bearing frequencies (BPFO, BPFI, BSF, FTF)

- **Advanced Diagnostics**
  - Evidence-based bearing diagnostic workflow (`diagnose_bearing`)
  - Gear fault detection workflow (`diagnose_gear`)
  - ISO 20816-3 vibration severity assessment
  - Automatic acceleration→velocity conversion

- **Machine Learning Tools**
  - `extract_features_from_signal()` - 17+ statistical features
  - `train_anomaly_model()` - OneClassSVM/LocalOutlierFactor training
  - `predict_anomalies()` - Anomaly detection with confidence scores

- **Comprehensive Test Suite**
  - 80%+ test coverage
  - Real data validation tests
  - CI/CD pipeline with GitHub Actions
  - Automated code quality checks (pytest, flake8, mypy, black)

### Changed
- Migrated from inline HTML artifacts to file-based reports
- Optimized signal processing algorithms for accuracy and performance
- Enhanced documentation with step-by-step tutorials
- Improved diagnostic accuracy with evidence-based workflows

### Fixed
- Signal processing edge cases
- Peak detection accuracy
- ISO 20816-3 zone classification
- Metadata handling for various signal formats

## [0.1.0] - 2025-11-01

### Added
- Initial release of Predictive Maintenance MCP Server
- Core vibration analysis tools (FFT, envelope, statistics)
- Basic MCP server implementation with FastMCP
- Sample signal generation
- Initial documentation and examples

---

## Roadmap

### Planned for v0.4.0
- **🔍 Vector Search for Large Documents** - ChromaDB/FAISS integration for semantic manual search
- **📷 OCR Support** - Tesseract integration for scanned PDF manuals
- **🌐 Online Bearing Catalog** - Optional web-based bearing lookup (privacy-first)
- Multi-signal comparison and trending tools
- Advanced diagnostic workflows (pumps, motors, gearboxes)
- Enhanced ML models with deep learning approaches

### Future Enhancements
- Real-time signal streaming support
- Cloud integration options (Azure/AWS)
- Dashboard for multi-asset fleet monitoring
- Mobile-friendly report viewing
- Integration with industrial IoT platforms
- **Multimodal diagnostics**: Combine vibration, temperature, acoustic data
