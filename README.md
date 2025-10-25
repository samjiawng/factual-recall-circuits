# Factual Recall Circuits

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mechanistic interpretability toolkit for discovering circuits that encode factual knowledge in transformer language models.

## Overview

This implementation discovers interpretable computational subgraphs in LLMs using attribution analysis (integrated gradients, activation patching), sparse autoencoders for feature extraction, automated circuit validation, and a hypothesis testing framework

Tested on Gemma 2B and GPT-2 variants.

## Installation

Requirements: Python 3.11+, PyTorch 2.0+

```bash
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn einops tqdm networkx
```

For Gemma models, authenticate with HuggingFace:
```bash
huggingface-cli login
```

## Usage

Basic circuit discovery:

```python
from circuit_discovery import CircuitDiscovery

discovery = CircuitDiscovery(model_name="google/gemma-2b", device='cuda')

dataset = {
    'entity_location': [
        {'clean': 'The Eiffel Tower is located in Paris',
         'corrupted': 'The Eiffel Tower is located in London'},
    ]
}

circuits = discovery.discover_all_circuits(dataset, train_sae=True)

for circuit in circuits:
    print(f"{circuit.name}: {len(circuit.nodes)} nodes, score: {circuit.attribution_score:.4f}")
```

Run full pipeline:
```bash
python main.py
```

Results saved to `outputs/circuit_discovery_[timestamp]/`

## Project Structure

```
├── src/
│   ├── circuit_discovery.py    # Core algorithms
│   ├── testing_pipeline.py     # Validation framework
│   └── utils.py                # Visualization tools
├── examples/tutorial.ipynb     # Interactive walkthrough
└── main.py                     # Full pipeline
```

## Methods

**Attribution Analysis**
- Integrated gradients for feature importance
- Activation patching for causal analysis

**Sparse Autoencoders**
- 4x expansion over hidden dimension
- L1 sparsity penalty
- Layer-specific training

**Circuit Discovery**
1. Attribution scoring across layers
2. Feature extraction via SAE
3. Node importance ranking
4. Edge construction
5. Validation testing

## Performance

- GPU recommended (30-60 min on A100)
- CPU supported but slower (several hours)
- Memory: ~10GB GPU RAM for Gemma 2B
- Model weights: ~5GB download (cached)

## Documentation

- `docs/QUICKSTART.md` - Fast-track guide
- `docs/PROJECT_SUMMARY.md` - Detailed overview
- `examples/tutorial.ipynb` - Interactive tutorial

## License

MIT License. See LICENSE file.

## References

- [Mechanistic Interpretability](https://transformer-circuits.pub/)
- [Neuropedia](https://neuropedia.ai/)
- [Sparse Autoencoders](https://arxiv.org/abs/2309.08600)
- [Attribution Patching](https://arxiv.org/abs/2310.10348)