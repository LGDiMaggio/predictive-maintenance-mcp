# Changelog

All notable changes to the Predictive Maintenance MCP Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - 15 production-quality signals from real machinery tests
  - 3 healthy baselines, 7 outer race faults, 5 inner race faults
  - Sampling rate: 97.7 kHz, duration: 6 seconds each
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

### Planned for v0.3.0
- **🤖 AI-Powered Machine Documentation Reader**
  - Automatic extraction of bearing/gear specifications from equipment manuals (PDF)
  - Parse technical datasheets to identify characteristic frequencies (BPFO, BPFI, BSF, FTF)
  - Extract machine parameters: power rating, operating speeds, bearing types
  - Integration with LLM for natural language understanding of technical specs
  - Support for common formats: maintenance manuals, bearing catalogs, OEM documentation
- Multi-signal comparison tools
- Advanced trending and monitoring
- Additional diagnostic workflows (pumps, motors, gearboxes)
- Enhanced ML models with hyperparameter tuning
- Extended dataset with more fault types

### Future Enhancements
- Real-time signal streaming support
- Cloud integration options
- Dashboard for multi-asset monitoring
- Mobile-friendly report viewing
- Integration with industrial IoT platforms
- **Multimodal diagnostics**: Combine vibration, temperature, acoustic data
