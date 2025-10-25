# Contributing

## Setup

```bash
git clone https://github.com/yourusername/factual-recall-circuits.git
cd factual-recall-circuits
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

## Workflow

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes
4. Run tests: `pytest tests/`
5. Format code: `black src/ main.py`
6. Commit and push
7. Open pull request

## Code Standards

**Style**: PEP 8, 88 character line length

**Type hints required**:
```python
def discover_circuit(
    self,
    fact_prompts: List[Dict[str, str]],
    fact_type: str,
    threshold: float = 0.01
) -> Circuit:
    pass
```

**Docstrings required** (NumPy style):
```python
def method_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description.
    
    Parameters
    ----------
    param1 : Type1
        Description
    param2 : Type2
        Description
    
    Returns
    -------
    ReturnType
        Description
    """
```

## Testing

```bash
pytest tests/
pytest --cov=src tests/
```

Place tests in `tests/` directory with `test_*.py` naming:
```python
def test_circuit_discovery_initialization():
    discovery = CircuitDiscovery(model_name="gpt2", device="cpu")
    assert discovery.model is not None
```

## Pull Requests

**Requirements**:
- All tests pass
- Code formatted with Black
- Documentation updated
- One feature per PR
- Clear commit messages

**Process**:
1. Automated checks pass
2. Maintainer approval
3. Address feedback
4. Merge

## Issues

Include:
- Clear description
- Reproduction steps (bugs)
- Expected vs actual behavior
- Environment (Python version, OS, hardware)
- Error messages or code snippets

## Priority Areas

**High**: Additional model support, performance optimization, test coverage

**Medium**: Alternative attribution methods, improved SAE architectures, batch processing

**Documentation**: Tutorials, examples, API docs

## License

Contributions licensed under MIT License.