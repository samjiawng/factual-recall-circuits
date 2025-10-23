# Contributing to Factual Recall Circuits

Thank you for your interest in contributing to this project. This document provides guidelines for contributing code, documentation, and other improvements.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/factual-recall-circuits.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and verify functionality
6. Commit with clear messages
7. Push to your fork
8. Open a pull request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/factual-recall-circuits.git
cd factual-recall-circuits

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Standards

### Style Guide

This project follows PEP 8 style guidelines with the following specifications:
- Line length: 88 characters (Black formatter default)
- Indentation: 4 spaces
- Imports: Organized using isort
- Docstrings: NumPy style

### Formatting

Code should be formatted using Black:

```bash
black src/ main.py
```

### Type Hints

Use type hints for all function signatures:

```python
def discover_circuit(
    self,
    fact_prompts: List[Dict[str, str]],
    fact_type: str,
    threshold: float = 0.01
) -> Circuit:
    """Function implementation"""
```

### Documentation

All public functions and classes must include docstrings:

```python
def method_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of method.
    
    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2
        Description of param2
    
    Returns
    -------
    ReturnType
        Description of return value
    
    Examples
    --------
    >>> result = method_name(arg1, arg2)
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_circuit_discovery.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

Tests should be placed in the `tests/` directory with names matching `test_*.py`:

```python
import pytest
from circuit_discovery import CircuitDiscovery

def test_circuit_discovery_initialization():
    """Test that CircuitDiscovery initializes correctly."""
    discovery = CircuitDiscovery(model_name="gpt2", device="cpu")
    assert discovery.model is not None
    assert discovery.tokenizer is not None
```

## Pull Request Process

1. Update documentation for any changed functionality
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md with notable changes
5. Reference any related issues in the PR description

### PR Guidelines

- One feature or fix per pull request
- Clear, descriptive commit messages
- Link to related issues
- Include motivation and context
- Update relevant documentation

## Areas for Contribution

### High Priority

- Additional model architecture support (Llama, Mistral, etc.)
- Performance optimizations for CPU execution
- Enhanced visualization capabilities
- Expanded test coverage

### Medium Priority

- Alternative attribution methods
- Improved SAE architectures
- Additional fact types
- Batch processing capabilities

### Documentation

- Tutorial expansions
- Code examples
- API documentation improvements
- Usage guides

## Reporting Issues

When reporting bugs or requesting features, please include:

1. Clear description of the issue
2. Steps to reproduce (for bugs)
3. Expected vs actual behavior
4. Environment details (Python version, OS, GPU/CPU)
5. Relevant code snippets or error messages

Use the GitHub issue tracker and apply appropriate labels.

## Code Review Process

All submissions require review before merging:

1. Automated checks must pass (formatting, tests)
2. At least one maintainer approval required
3. Discussion and feedback addressed
4. Documentation updated as needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

## Recognition

Contributors will be acknowledged in the project README and release notes.
