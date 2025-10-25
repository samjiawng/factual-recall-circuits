# File Index

## Quick Reference

| File | Purpose |
|------|---------|
| `README.md` | Full documentation |
| `QUICKSTART.md` | Fast-track guide |
| `main.py` | Example pipeline |
| `verify_install.py` | Installation check |

## Core Modules

### circuit_discovery.py
Circuit discovery implementation.

**Classes**:
- `CircuitDiscovery` - Main orchestration
- `NeuropediaAttributionGraph` - Attribution methods
- `SparseAutoencoder` - Feature extraction
- `Circuit` - Circuit data structure

**Key methods**:
- `discover_all_circuits()` - Run full discovery
- `discover_circuit()` - Single circuit discovery
- `train_sparse_autoencoder()` - SAE training

### testing_pipeline.py
Validation and hypothesis testing.

**Classes**:
- `HypothesisValidator` - Feature hypothesis testing
- `CircuitTester` - Circuit validation
- `FeatureHypothesis` - Hypothesis structure
- `TestResult` - Test results

**Key methods**:
- `test_hypothesis()` - Test single hypothesis
- `batch_test()` - Batch testing
- `run_full_validation()` - Complete validation

### utils.py
Visualization and analysis.

**Functions**:
- `visualize_circuit()` - Circuit graphs
- `plot_attribution_heatmap()` - Attribution visualization
- `compare_circuits()` - Multi-circuit comparison
- `plot_testing_results()` - Test dashboards
- `plot_circuit_overlap()` - Overlap analysis
- `export_circuit_summary()` - Export results

### main.py
Complete pipeline example.

**Functions**:
- `create_factual_dataset()` - Dataset construction
- `create_hypotheses()` - Hypothesis generation
- `export_results()` - Result export
- `main()` - Full pipeline execution

## Documentation

### README.md
Complete technical documentation covering installation, methodology, API reference, and examples.

### QUICKSTART.md
Fast-track guide with quick installation and usage patterns.

### PROJECT_SUMMARY.md
High-level project overview and feature summary.

### tutorial.ipynb
Interactive Jupyter notebook with step-by-step walkthrough.

## Configuration

### requirements.txt
Python dependencies:
- torch>=2.0.0
- transformers>=4.35.0
- numpy, pandas, matplotlib, seaborn
- scikit-learn, einops, tqdm
- jupyter

## Usage Patterns

**Quick start**:
```bash
python verify_install.py
python main.py
```

**Interactive learning**:
```bash
jupyter notebook tutorial.ipynb
```

**Custom pipeline**:
```python
from circuit_discovery import CircuitDiscovery
discovery = CircuitDiscovery(device='cuda')
circuits = discovery.discover_all_circuits(dataset)
```

## Modification Guide

**Add fact types**: Modify `create_factual_dataset()` in `main.py`

**Customize attribution**: Edit `NeuropediaAttributionGraph` in `circuit_discovery.py`

**New validation metrics**: Extend `CircuitTester` in `testing_pipeline.py`

**Additional visualizations**: Add functions to `utils.py`

**Change SAE architecture**: Modify `SparseAutoencoder` in `circuit_discovery.py`

## File Statistics

| Category | Files | LOC |
|----------|-------|-----|
| Core modules | 4 | ~2000 |
| Documentation | 4 | - |
| Configuration | 1 | - |
| Utilities | 1 | - |