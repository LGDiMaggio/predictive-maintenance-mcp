# Contributing to Predictive Maintenance MCP Server

Thank you for your interest in contributing! This project thrives on **diverse contributions** — not just code. Whether you're a maintenance engineer, a Python developer, a technical writer, or a student exploring open source for the first time, there's a meaningful way for you to help.

---

## Table of Contents

- [Choose Your Contribution Path](#choose-your-contribution-path)
- [Code of Conduct](#code-of-conduct)
- [Getting Started (All Contributors)](#getting-started-all-contributors)
- [Path 1: Domain Expert (No Coding Required)](#path-1-domain-expert-no-coding-required)
- [Path 2: Software Developer](#path-2-software-developer)
- [Path 3: Technical Writer / Translator](#path-3-technical-writer--translator)
- [Path 4: Tester / QA](#path-4-tester--qa)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Adding New Analysis Tools](#adding-new-analysis-tools)
- [Recognition](#recognition)
- [Questions?](#questions)

---

## Choose Your Contribution Path

Not everyone contributes in the same way, and **all contributions are equally valued**. Find your path:

<table>
<tr>
<td width="50%" valign="top">

### 🔧 Path 1: Domain Expert
*Maintenance engineer, reliability analyst, vibration specialist*

**You don't need to write code.** Your industrial knowledge is the most valuable asset this project needs.

- Validate diagnostic accuracy against real-world experience
- Provide real vibration datasets (anonymized)
- Review whether fault detection results make engineering sense
- Suggest new diagnostic workflows from your daily practice

➡️ [Jump to Path 1 details](#path-1-domain-expert-no-coding-required)

</td>
<td width="50%" valign="top">

### 💻 Path 2: Software Developer
*Python developer, AI/ML engineer, full-stack dev*

**The architecture is clean and extensible.** Adding a new MCP tool is as simple as writing a Python function with a decorator.

- Add new diagnostic tools
- Improve ML models
- Build Docker support
- Enhance the report system

➡️ [Jump to Path 2 details](#path-2-software-developer)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📖 Path 3: Technical Writer
*Documentation specialist, translator, educator*

**Clear documentation saves hours** for every future user. Help us explain complex concepts simply.

- Improve tutorials and examples
- Translate documentation
- Write case studies
- Create video walkthroughs

➡️ [Jump to Path 3 details](#path-3-technical-writer--translator)

</td>
<td width="50%" valign="top">

### 🧪 Path 4: Tester / QA
*Quality assurance, edge case hunter, validation specialist*

**Break things on purpose.** Find the inputs that make the system fail — that's how we make it robust.

- Test with unusual data formats
- Validate results against ground truth
- Report edge cases and unexpected behaviors
- Cross-platform testing (Windows/macOS/Linux)

➡️ [Jump to Path 4 details](#path-4-tester--qa)

</td>
</tr>
</table>

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Respecting differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what's best for the community

**Unacceptable behaviors:**
- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## Getting Started (All Contributors)

### 1. Find Something to Work On

Browse [**Issues**](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) and look for these labels:

| Label | Meaning |
|-------|---------|
| `good first issue` | Small, well-defined tasks perfect for newcomers |
| `help wanted` | Tasks where we specifically need community help |
| `documentation` | Writing, editing, or translating docs |
| `bug` | Something isn't working correctly |
| `enhancement` | New feature or improvement |
| `domain-knowledge` | Requires industrial/engineering expertise, not necessarily code |

**Can't find a matching issue?** [Open a discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) to propose your idea first.

### 2. Claim the Issue

Comment on the issue: *"I'd like to work on this!"* — this prevents duplicate effort.

### 3. Set Up Your Environment

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp

# Add upstream remote
git remote add upstream https://github.com/LGDiMaggio/predictive-maintenance-mcp.git

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS/Linux

# Install development dependencies
pip install -e ".[dev]"

# Verify everything works
pytest -v
```

---

## Path 1: Domain Expert (No Coding Required)

Your industrial experience is **irreplaceable**. Here's how to contribute without writing a single line of code:

### Validate Diagnostic Results

Run the server with the included test data and verify:
- Do the bearing fault frequencies make engineering sense?
- Are the ISO 20816-3 zone classifications correct for the given vibration levels?
- Would the envelope analysis results lead to the right maintenance decision?

**How to provide feedback**: Open an issue titled *"Validation: [what you tested]"* and describe what you found. Include screenshots of the HTML reports if helpful.

### Provide Real-World Datasets

We need anonymized vibration data from real machinery:
- Healthy baseline signals for different machine types
- Known fault conditions (bearing, gear, imbalance, misalignment)
- Metadata: sampling rate, shaft speed, bearing type, fault description

**Format**: CSV with one column of acceleration values + a JSON metadata file. See `data/signals/real_train/` for examples.

**Privacy**: Anonymize all data. Remove any proprietary machine identifiers, company names, or location information.

### Suggest New Diagnostic Workflows

You know what matters in the field. Open a discussion or issue describing:
- What diagnostic procedure do you follow daily?
- What information do you need that the current tools don't provide?
- What standards or guidelines should we support (VDI 3832, ISO 10816, etc.)?

### Review the Bearing Catalog

The file `resources/bearing_catalogs/common_bearings_catalog.json` contains a small set of bearings whose geometry is traceable to a public source — every entry carries a mandatory `source` citation, and `tests/test_bearing_catalog_validation.py` enforces physical validity. You can:
- Verify the geometry data against the cited sources
- Suggest additional bearing models **with a verifiable public source** (manufacturer datasheet, dataset documentation, or peer-reviewed paper)
- Provide corrections for contact angles, ball counts, or pitch diameters (with source)

This is a JSON file — you can edit it directly or provide the data in any format and we'll add it.

---

## Path 2: Software Developer

### Development Setup

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, then test
pytest -v
black src/ --check
flake8 src/ --max-line-length=120
mypy src/ --ignore-missing-imports

# Commit with clear messages
git commit -m "feat: Add parquet file reading support

- Add load_parquet() function in signal loading
- Support both single and multi-column parquet files
- Add unit tests with synthetic data
- Update EXAMPLES.md with parquet usage

Closes #42"
```

### Good First Issues for Developers

| Task | What You'll Learn | Issue |
|------|-------------------|-------|
| ~~**Add Parquet data format support**~~ | ~~Signal loading pipeline, pandas I/O~~ | ✅ Done in v0.5.0 |
| **Make ISO thresholds configurable** | Tool parameter design, ISO standard | [Browse issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| **Add unit conversion tool** | MCP tool pattern, unit systems | [Browse issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| **Improve error messages** | Error handling best practices | [Browse issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| **Create Dockerfile** | Docker, deployment, DevOps | [Browse issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |

### How to Create a New MCP Tool

See the complete template in the [Developer Quickstart](docs/QUICKSTART_DEVELOPER.md#step-4-create-your-first-tool).

Summary:
1. Add a `@mcp.tool()` decorated function in `machinery_diagnostics_server.py`
2. Write comprehensive docstring (the LLM reads this!)
3. Add tests in `tests/`
4. Update `EXAMPLES.md` if the tool adds a new workflow
5. Submit PR

### Architecture Contributions

For bigger features, **open a discussion first** to align on approach:

- **Docker containerization** — Package everything into `docker run predictive-maintenance-mcp`
- **Vector search** — ChromaDB/FAISS for semantic search over large equipment manuals
- **Streaming support** — Real-time vibration signal monitoring via Server-Sent Events
- **Plugin system** — Allow external Python packages to register tools

---

## Path 3: Technical Writer / Translator

### Documentation Improvements

| Task | Impact |
|------|--------|
| Add a **video walkthrough** of the engineer quickstart | Reduces barrier for non-technical users |
| Write a **case study** using real (anonymized) data | Shows the tool's value to decision-makers |
| Improve **inline code comments** in `machinery_diagnostics_server.py` | Helps new developers understand the codebase |
| **Translate** the engineer quickstart to other languages | Expands reach globally |
| Create a **glossary** of vibration analysis terms used in the project | Bridges the gap between domains |

### Documentation Standards

- Use clear, simple language. Assume the reader may not be a native English speaker.
- Include the **"why"** alongside the **"how"** — people remember reasons better than steps.
- Every code example should be copy-pasteable and testable.
- Use screenshots or diagrams for visual concepts (reports, architecture).

---

## Path 4: Tester / QA

### What to Test

| Test Category | What to Try | Why It Matters |
|---------------|-------------|----------------|
| **Edge cases** | Very short signals (10 samples), very long signals (10M+), zero values, NaN values | Robustness |
| **Format variations** | CSV with headers, without headers, different delimiters, numeric formats | Real-world data is messy |
| **Cross-platform** | Run all tests on Windows, macOS, and Linux | Consistent experience |
| **Integration** | Full workflow: load → analyze → report → ML train → predict | End-to-end validation |
| **Ground truth** | Compare results against known reference values from textbooks or certified tools | Accuracy guarantee |

### How to Report Issues

Use this template when filing bugs:

```markdown
**Bug Description:**
Brief description of what went wrong

**Steps to Reproduce:**
1. Load signal file X
2. Run tool Y with parameters Z
3. Observe error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens (include full error message)

**Environment:**
- OS: Windows 11 / macOS 14 / Ubuntu 22.04
- Python: 3.11.x / 3.12.x
- Server version: 0.7.1
```

---

## Coding Standards

### Python Style

- **Formatter**: `black` (run before every commit)
- **Linter**: `flake8` (max line length: 120)
- **Type checker**: `mypy` (use type hints for all function signatures)
- **Docstrings**: Google style
- **Imports**: Standard library → Third-party → Local

```bash
# Quick check before committing
black src/ tests/
flake8 src/ --max-line-length=120
mypy src/ --ignore-missing-imports
pytest -v
```

### Testing Requirements

- **>80% coverage** for new code
- **100% coverage** for critical analysis functions (FFT, envelope, ISO)
- All tests must pass before PR submission

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Pull Request Process

### Before Submitting

1. Update your fork: `git fetch upstream && git merge upstream/main`
2. Run all checks: `pytest -v && black --check src/ && flake8 src/`
3. Write clear commit messages (see format below)

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example:**
```
feat: Add bearing frequency calculator tool

- Implements calculate_bearing_frequencies() function
- Adds validation for bearing parameters
- Includes comprehensive unit tests
- Updates README with usage examples

Closes #42
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
```

### Review Timeline

PRs are typically reviewed on a **weekly basis**. Critical bug fixes may receive faster attention.

---

## Adding New Analysis Tools

### Template

```python
@mcp.tool()
def new_analysis_tool(
    file_path: str,
    sampling_rate: float,
    parameter1: float,
    parameter2: str | None = None
) -> dict:
    """
    Brief description of what the tool does.
    
    Args:
        file_path: Path to signal file (required)
        sampling_rate: Sampling frequency in Hz (required)
        parameter1: Description (required)
        parameter2: Description (optional, default: None)
    
    Returns:
        Dictionary with analysis results
    """
    # 1. Load and validate signal
    # 2. Validate parameters
    # 3. Perform analysis
    # 4. Return structured results
```

### Checklist for New Tools

- [ ] Function implements core algorithm correctly
- [ ] Input validation for all parameters
- [ ] Clear error messages (the LLM relays these to users!)
- [ ] Comprehensive docstring (the LLM reads this to decide when to call the tool!)
- [ ] Unit tests with >80% coverage
- [ ] Integration test with sample data
- [ ] EXAMPLES.md updated (if new workflow)

---

## Recognition

Contributors will be:
- **Acknowledged** in release notes for the version including their contribution
- **Mentioned** in relevant documentation sections they contributed to
- **Credited** in commit history with proper attribution

**Major Contributors** (significant features or sustained contributions) may receive:
- Highlighted mention in README.md
- Co-authorship on research publications using this work (if applicable)
- Invitation to join as project collaborator

---

## Questions?

- **General questions**: [Open a Discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions)
- **Bug reports**: [Open an Issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues)
- **Security issues**: Email directly to the repository owner (contact in GitHub profile)
- **Contribution ideas**: [Start a Discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) — we'll help you find the right approach

---

**Every contribution matters. The smallest fix improves the experience for every future user.** 🚀
