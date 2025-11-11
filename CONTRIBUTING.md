# Contributing to Predictive Maintenance MCP Server

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

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

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of signal processing (helpful but not required)
- Familiarity with MCP protocol (can be learned along the way)

### Initial Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mcp.git
   cd predictive-maintenance-mcp
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
   ```

4. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Environment Configuration

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Development Dependencies

The `[dev]` extra includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `flake8` - Code linting
- `mypy` - Type checking
- `black` - Code formatting

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 1. Bug Reports
- Use GitHub Issues
- Include clear description
- Provide steps to reproduce
- Include error messages/logs
- Specify your environment

**Template:**
```markdown
**Bug Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Load signal file X
2. Run tool Y with parameters Z
3. Observe error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: Windows/macOS/Linux
- Python version: 3.11
- Server version: 0.2.0
```

#### 2. Feature Requests
- Describe the problem the feature would solve
- Provide use cases
- Suggest implementation approach (optional)

#### 3. Code Contributions
- Bug fixes
- New analysis tools
- Performance improvements
- Documentation improvements

#### 4. Documentation
- Fix typos or unclear explanations
- Add examples
- Translate documentation
- Improve API documentation

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some specific conventions:

#### Imports
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
from scipy import signal

# Local
from machinery_diagnostics_server import analyze_fft
```

#### Function Documentation
```python
def analyze_signal(
    file_path: str,
    sampling_rate: float,
    segment_duration: float = 1.0
) -> dict:
    """
    Analyze vibration signal using FFT.
    
    Args:
        file_path: Path to CSV signal file
        sampling_rate: Sampling frequency in Hz
        segment_duration: Segment duration in seconds (default: 1.0)
    
    Returns:
        Dictionary containing analysis results with keys:
        - frequencies: Frequency array
        - magnitudes: Magnitude array
        - peaks: List of detected peaks
    
    Raises:
        FileNotFoundError: If signal file doesn't exist
        ValueError: If sampling_rate <= 0
    
    Example:
        >>> result = analyze_signal("motor.csv", 10000)
        >>> print(result['peaks'])
        [{'frequency': 50.0, 'magnitude': 0.85}]
    """
    pass
```

#### Type Hints
Always use type hints for function signatures:
```python
from typing import List, Dict, Optional, Tuple

def detect_peaks(
    spectrum: np.ndarray,
    threshold: float = 0.1
) -> List[Tuple[float, float]]:
    """Detect peaks in spectrum."""
    pass
```

#### Error Handling
```python
try:
    signal_data = load_signal(file_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Signal file not found: {file_path}. "
        "Please check the file path and try again."
    )
except Exception as e:
    raise RuntimeError(f"Error loading signal: {e}")
```

### Code Formatting

We use **Black** for consistent formatting:
```bash
# Format your code
black src/

# Check formatting
black --check src/
```

### Linting

Use **flake8** to check code quality:
```bash
flake8 src/ --max-line-length=120 --max-complexity=20
```

### Type Checking

Use **mypy** for type checking:
```bash
mypy src/ --ignore-missing-imports
```

## Testing Guidelines

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_fft_analysis.py

import pytest
import numpy as np
from machinery_diagnostics_server import analyze_fft

def test_analyze_fft_basic():
    """Test basic FFT analysis."""
    # Arrange
    signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    
    # Act
    result = analyze_fft(signal, sampling_rate=1000)
    
    # Assert
    assert result['peak_frequency'] == pytest.approx(50.0, rel=0.1)
    assert len(result['frequencies']) > 0

def test_analyze_fft_invalid_sampling_rate():
    """Test that invalid sampling rate raises error."""
    signal = np.zeros(100)
    
    with pytest.raises(ValueError):
        analyze_fft(signal, sampling_rate=-1)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fft_analysis.py

# Run specific test
pytest tests/test_fft_analysis.py::test_analyze_fft_basic
```

### Test Coverage

Aim for:
- **>80% coverage** for new code
- **100% coverage** for critical analysis functions

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def calculate_bearing_frequencies(
    shaft_speed: float,
    num_balls: int,
    ball_diameter: float,
    pitch_diameter: float,
    contact_angle: float = 0.0
) -> dict:
    """
    Calculate bearing characteristic frequencies.
    
    Uses standard bearing kinematic formulas to compute:
    - BPFO (Ball Pass Frequency Outer race)
    - BPFI (Ball Pass Frequency Inner race)
    - BSF (Ball Spin Frequency)
    - FTF (Fundamental Train Frequency)
    
    Args:
        shaft_speed: Shaft rotation speed in Hz
        num_balls: Number of rolling elements
        ball_diameter: Diameter of rolling element in mm
        pitch_diameter: Bearing pitch diameter in mm
        contact_angle: Contact angle in degrees (default: 0.0)
    
    Returns:
        Dictionary with keys:
        - BPFO: Outer race frequency
        - BPFI: Inner race frequency
        - BSF: Ball spin frequency
        - FTF: Cage frequency
    
    Example:
        >>> freqs = calculate_bearing_frequencies(
        ...     shaft_speed=30.0,
        ...     num_balls=8,
        ...     ball_diameter=12.7,
        ...     pitch_diameter=58.0
        ... )
        >>> print(f"BPFO: {freqs['BPFO']:.2f} Hz")
        BPFO: 99.75 Hz
    
    References:
        Harris, T. A. (2001). Rolling Bearing Analysis (4th ed.).
        ISO 281:2007 - Rolling bearings — Dynamic load ratings
    """
    pass
```

### Updating README

When adding new features:
1. Update "Features" section
2. Add tool documentation to "Available Tools"
3. Update examples if applicable
4. Add to roadmap if partial implementation

### API Documentation

For new tools, add to API reference:
```markdown
#### `new_analysis_tool`
Brief description of what the tool does.

**Parameters:**
- `param1` (type, required) - Description
- `param2` (type, optional) - Description (default: value)

**Returns:**
- Description of return value

**Example:**
\```
new_analysis_tool
param1: value1
param2: value2
\```
```

## Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   flake8 src/
   mypy src/
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add feature: brief description
   
   - Detailed change 1
   - Detailed change 2
   - Fixes #123"
   ```

### Commit Message Format

Use this format:
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code restructuring without behavior change
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example:**
```
feat: Add bearing frequency calculator tool

- Implements calculate_bearing_frequencies() function
- Adds validation for bearing parameters
- Includes comprehensive unit tests
- Updates README with usage examples

Closes #42
```

### Submitting Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in PR template

3. **PR Description Template**
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
   - [ ] Code coverage maintained/improved
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-reviewed the code
   - [ ] Commented complex sections
   - [ ] Updated documentation
   - [ ] No breaking changes (or documented)
   
   ## Related Issues
   Fixes #123
   ```

### Review Process

1. **Automated checks** run automatically (pytest, flake8, mypy, black)
2. **Maintainer review** (typically weekly, may be faster for urgent fixes)
3. **Address feedback** if requested
4. **Approval and merge** once approved

> 📅 **Review Timeline**: PRs are typically reviewed on a weekly basis. Critical bug fixes may receive faster attention.

## Adding New Analysis Tools

### Template for New Tool

```python
@mcp.tool()
def new_analysis_tool(
    file_path: str,
    sampling_rate: float,
    parameter1: float,
    parameter2: Optional[str] = None
) -> dict:
    """
    Brief description of what the tool does.
    
    Args:
        file_path: Path to signal file (required)
        sampling_rate: Sampling frequency in Hz (required)
        parameter1: Description of parameter1 (required)
        parameter2: Description of parameter2 (optional, default: None)
    
    Returns:
        Dictionary with analysis results
    
    Example:
        >>> result = new_analysis_tool("signal.csv", 10000, 1.5)
    """
    # 1. Load and validate signal
    try:
        signal_data = load_signal(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Signal file not found: {file_path}")
    
    # 2. Validate parameters
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
    
    # 3. Perform analysis
    result = perform_analysis(signal_data, parameter1, parameter2)
    
    # 4. Format and return results
    return {
        "analysis_type": "new_analysis",
        "file_path": file_path,
        "sampling_rate": sampling_rate,
        "results": result,
        "timestamp": datetime.now().isoformat()
    }
```

### Checklist for New Tools

- [ ] Function implements core algorithm correctly
- [ ] Input validation for all parameters
- [ ] Clear error messages
- [ ] Comprehensive docstring
- [ ] Unit tests with >80% coverage
- [ ] Integration test with sample data
- [ ] README documentation updated
- [ ] Example usage in EXAMPLES.md
- [ ] Type hints for all parameters

## Questions?

- **General questions**: Open a [Discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) (enable in Settings if not available)
- **Bug reports**: Open an [Issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues)
- **Security issues**: Email directly to the repository owner (contact information in GitHub profile)

## Recognition

Contributors will be:
- Acknowledged in release notes for the version including their contribution
- Mentioned in relevant documentation sections they contributed to
- Credited in commit history with proper attribution

**Major Contributors** (significant features or sustained contributions) may receive:
- Highlighted mention in README.md
- Co-authorship on research publications using this work (if applicable)
- Invitation to join as project collaborator

Thank you for contributing to Predictive Maintenance MCP Server! 🚀
