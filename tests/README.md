# Test Suite

This directory contains the test suite for the Predictive Maintenance MCP Server.

## Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_fft_analysis.py     # Tests for FFT analysis
├── test_envelope_iso.py     # Tests for envelope analysis and ISO 20816-3
├── test_ml_tools.py         # Tests for ML tools (features, training, prediction)
└── README.md               # This file
```

## Running Tests

### Install Development Dependencies

```bash
pip install -e .[dev]
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-asyncio` - Async test support
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_fft_analysis.py
```

### Run Specific Test Class

```bash
pytest tests/test_fft_analysis.py::TestFFTAnalysis
```

### Run Specific Test Function

```bash
pytest tests/test_fft_analysis.py::TestFFTAnalysis::test_fft_synthetic_sine_50hz
```

### Run with Coverage Report

```bash
pytest --cov=src --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

### Run with Verbose Output

```bash
pytest -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only ML tests
pytest -m ml
```

## Test Markers

Tests are organized with markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, use real data)
- `@pytest.mark.slow` - Slow tests (can be skipped)
- `@pytest.mark.ml` - Machine learning tests
- `@pytest.mark.real_data` - Tests using real bearing data samples

## Coverage Target

Target: **>80% code coverage**

Check current coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

## Test Data

Tests use:
1. **Synthetic signals** - Generated on-the-fly for controlled testing
2. **Real bearing data** - From `data/signals/real_train/` directory

Fixtures in `conftest.py`:
- `sample_healthy_signal` - Loads `baseline_1.csv`
- `sample_faulty_signal` - Loads `OuterRaceFault_1.csv`
- `sample_metadata` - Loads `baseline_1_metadata.json`
- `synthetic_sine_signal` - Generates 50 Hz sine wave
- `temp_csv_file` - Creates temporary CSV file

## Writing New Tests

### Test File Naming

- File: `test_<module>.py`
- Class: `Test<Feature>`
- Function: `test_<scenario>`

### Example Test

```python
import pytest
import numpy as np

class TestMyFeature:
    """Test suite for my feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = [1, 2, 3]
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes >1 second."""
        # ... slow test code ...
```

### Using Fixtures

```python
def test_with_real_data(sample_healthy_signal, sample_metadata):
    """Test using real bearing data."""
    signal = sample_healthy_signal
    fs = sample_metadata['sampling_rate']
    
    # ... test code ...
```

## CI/CD Integration

Tests run automatically on GitHub Actions:

- **On Push**: To `main` and `develop` branches
- **On Pull Request**: To `main` and `develop` branches
- **Matrix**: Python 3.11 and 3.12

CI Pipeline includes:
1. **Test** - Run pytest with coverage
2. **Lint** - Run flake8
3. **Type Check** - Run mypy
4. **Format Check** - Run black

## Common Issues

### Missing Sample Data

If tests fail with "Sample data not found":
```bash
# Verify data files exist
ls data/signals/real_train/
```

### Import Errors

If you see "Module not found":
```bash
# Reinstall package in development mode
pip install -e .
```

### Coverage Too Low

If coverage is below 80%:
1. Check which lines are not covered:
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```
2. Add tests for uncovered code
3. Or mark code as `# pragma: no cover` if untestable

## Best Practices

1. **Arrange-Act-Assert** - Structure tests clearly
2. **One assertion per test** - Makes failures easier to debug
3. **Descriptive names** - `test_fft_detects_50hz_sine_wave` not `test_1`
4. **Use fixtures** - Don't repeat setup code
5. **Mock external dependencies** - Don't make HTTP requests in unit tests
6. **Test edge cases** - Empty input, negative values, very large numbers
7. **Document tests** - Add docstrings explaining what's being tested

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/pytest-python-testing/)

---

**Questions?** Open an issue or discussion on GitHub!
