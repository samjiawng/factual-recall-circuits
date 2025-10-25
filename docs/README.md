# Factual Recall Circuits in Gemma 2B

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mechanistic interpretability toolkit for discovering circuits that encode factual knowledge in transformer language models.

## Overview

Discovers interpretable computational subgraphs responsible for factual recall using:
- Neuropedia-style attribution graphs (integrated gradients, activation patching)
- Sparse autoencoders for feature extraction
- Automated validation and hypothesis testing

Supports multiple fact types:
- Entity-location relationships
- Capital-country relationships
- Historical dates
- Person-occupation relationships

## Installation

```bash
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn einops tqdm networkx
```

For Gemma models:
```bash
huggingface-cli login
```

## Quick Start

```python
from circuit_discovery import CircuitDiscovery

discovery = CircuitDiscovery(device='cuda')

dataset = {
    'entity_location': [
        {'clean': 'The Eiffel Tower is in Paris', 
         'corrupted': 'The Eiffel Tower is in London'},
    ]
}

circuits = discovery.discover_all_circuits(dataset, train_sae=True)

for circuit in circuits:
    print(f"{circuit.name}: {len(circuit.nodes)} nodes")
```

### Full Pipeline

```bash
python main.py
```

Runs complete discovery, validation, and visualization. Results saved to `outputs/circuit_discovery_[timestamp]/`

## Project Structure

```
├── circuit_discovery.py    # Core algorithms
├── testing_pipeline.py     # Validation framework
├── utils.py                # Visualization tools
├── main.py                 # Example pipeline
└── requirements.txt        # Dependencies
```

## Architecture

### CircuitDiscovery
Main discovery engine.

**Methods**:
- `train_sparse_autoencoder()` - Train SAE on activations
- `discover_circuit()` - Find circuit for fact type
- `discover_all_circuits()` - Batch discovery

### NeuropediaAttributionGraph
Attribution methods.

**Methods**:
- `integrated_gradients()` - Compute attribution scores
- `activation_patching()` - Measure causal importance

### CircuitTester
Validation framework.

**Methods**:
- `test_hypothesis()` - Test feature hypothesis
- `test_circuit_faithfulness()` - Verify necessity/sufficiency
- `run_full_validation()` - Complete testing pipeline

## Methodology

**Attribution Analysis**
- Integrated gradients identify important activations
- Activation patching measures causal importance
- Focus on components affecting factual predictions

**Sparse Autoencoding**
- Train SAEs on model activations (4x expansion)
- L1 sparsity penalty for interpretability
- Extract features corresponding to factual knowledge

**Circuit Extraction**
- Connect features across layers via attribution scores
- Build directed graph of dependencies
- Identify circuits for specific fact types

**Validation**
- Generate hypotheses about feature behavior
- Test on positive and negative examples
- Measure precision and specificity
- Verify faithfulness through ablation

## Output

Example results:
```
entity_location_circuit: 12 nodes, score = 0.234
capital_country_circuit: 15 nodes, score = 0.198
historical_date_circuit: 10 nodes, score = 0.167
person_occupation_circuit: 8 nodes, score = 0.145
```

Generated files:
- `circuit_summary.txt` - Detailed descriptions
- `circuits.json` - Machine-readable data
- `*.png` - Visualizations
- `*_results.csv` - Test results
- `REPORT.md` - Analysis report

## Customization

### Add Fact Types

```python
dataset = {
    'custom_type': [
        {'clean': 'correct statement',
         'corrupted': 'incorrect version'},
    ]
}

circuits = discovery.discover_all_circuits(dataset)
```

### Custom Hypotheses

```python
from testing_pipeline import FeatureHypothesis

hypothesis = FeatureHypothesis(
    feature_id=(layer, feature_idx),
    hypothesis="Feature description",
    test_prompts=["test", "examples"],
    control_prompts=["control", "examples"],
    expected_activation_threshold=0.5
)

result = tester.validator.test_hypothesis(hypothesis)
```

### Adjust SAE Parameters

```python
sae = SparseAutoencoder(
    d_model=model_dim,
    d_hidden=model_dim * 8,
    sparsity_coef=5e-3
)
```

## Performance

- GPU: 30-60 minutes for full pipeline (A100)
- CPU: Several hours (supported but slow)
- Memory: ~10GB GPU RAM for Gemma 2B
- Model download: ~5GB (cached)

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~10GB GPU memory

## Troubleshooting

**Out of memory**:
```python
sae = SparseAutoencoder(d_model, d_model * 2)
```

**Slow execution**:
```python
circuits = discovery.discover_all_circuits(dataset, train_sae=False)
```

**Import errors**:
```bash
pip install -r requirements.txt --upgrade
```

## Applications

- Mechanistic interpretability of LLMs
- Targeted model editing
- Fact-checking systems
- Safety research
- Educational demonstrations

## Documentation

- `QUICKSTART.md` - Fast-track guide
- `PROJECT_SUMMARY.md` - Detailed overview
- `tutorial.ipynb` - Interactive tutorial
- `CONTRIBUTING.md` - Development guidelines

## References

- [Mechanistic Interpretability](https://transformer-circuits.pub/)
- [Neuropedia](https://neuropedia.ai/)
- [Sparse Autoencoders](https://arxiv.org/abs/2309.08600)
- [Attribution Patching](https://arxiv.org/abs/2310.10348)

## Contributing

Contributions welcome. Priority areas:
- Additional model support
- Performance optimization
- Enhanced attribution methods
- Test coverage expansion

See `CONTRIBUTING.md` for guidelines.

## License

MIT License