# Changelog

All notable changes to the Predictive Maintenance MCP Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-15

### Added
- **Persona-Based Documentation System**
  - New `docs/QUICKSTART_ENGINEER.md` — Zero-code guide for maintenance and reliability engineers
  - New `docs/QUICKSTART_DEVELOPER.md` — Architecture guide for AI/software developers with tutorial on creating new MCP tools
  - "Choose Your Path" section in README with two clear entry points

- **"Our Mission" Section in README**
  - Project vision and purpose integrated directly into the repository (previously only on external blog post)
  - Explains the "why" of MCP for industrial diagnostics

- **Ecosystem Architecture Overview**
  - Visual diagram explaining the MCP flow: User → LLM → MCP Server → Data
  - Explanation of MCP as "USB port for AI" — plug-and-play tool integration
  - Clarifies the Resource vs Tool pattern

- **GitHub Issue Templates**
  - Bug Report template with environment details
  - Feature Request template with impact assessment
  - Good First Issue template with effort estimates and mentorship links
  - Domain Validation template for engineers to provide expert feedback (no code required)
  - Pull Request template with standardized checklist
  - Issue template config with contact links to Discussions and guides

- **Revamped CONTRIBUTING.md with Four Contribution Paths**
  - Path 1: Domain Expert (no code required — validate results, provide datasets, review diagnostics)
  - Path 2: Software Developer (add tools, improve architecture, build Docker support)
  - Path 3: Technical Writer (tutorials, translations, case studies)
  - Path 4: Tester / QA (edge cases, cross-platform, ground truth validation)

- **Actionable Roadmap**
  - Roadmap items now link to GitHub Issues/Discussions
  - Priority-based table with Get Involved column
  - Docker image for zero-install setup added as high-priority item

### Changed
- **README.md completely restructured** — Mission → Architecture → Choose Your Path → Content
  - Moved from purely technical README to narrative + technical hybrid
  - Added Documentation table linking all guides by audience
  - Consolidated support links (Issues, Discussions, Blog post) in dedicated section
- **CONTRIBUTING.md rewritten** — From generic PR guide to persona-based contribution manifesto
- Version bump from 0.3.2 to 0.4.0

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

### Planned for v0.5.0
- **📦 Docker image** for zero-install setup
- **📂 Parquet/HDF5 data format support**
- **📏 Customizable ISO report thresholds**
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
