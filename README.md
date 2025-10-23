# Factual Recall Circuits in Gemma 2B

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A mechanistic interpretability toolkit for discovering and analyzing circuits responsible for factual recall in large language models. This implementation uses Neuropedia-style attribution graphs and sparse autoencoders to identify interpretable computational subgraphs that encode factual knowledge.

See also our companion [documentation](docs/README.md).

## Overview

This project implements methods for:
- Discovering interpretable circuits in transformer language models
- Attribution analysis using integrated gradients and activation patching
- Sparse autoencoder training for feature extraction
- Automated hypothesis testing and validation
- Circuit visualization and analysis

The toolkit has been used to discover 18+ distinct circuits for different types of factual knowledge including entity-location relationships, capital cities, historical dates, and person-occupation associations.

## Getting Started

### Installation

#### Standard Installation

This project requires Python 3.11+ and PyTorch (>= 2.0). To install the latest stable version, run:

```bash
pip install torch transformers accelerate
pip install numpy pandas matplotlib seaborn scikit-learn einops tqdm networkx
```

#### Developer Installation

If you are interested in modifying the library, clone the repository and set up a development environment as follows:

```bash
git clone https://github.com/yourusername/factual-recall-circuits.git
cd factual-recall-circuits
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Authentication

Access to Gemma models requires HuggingFace authentication:

```bash
huggingface-cli login
```

Visit https://huggingface.co/google/gemma-2b to request access to the model.

### Basic Example

As a starting point, here is a minimal example of discovering circuits for factual recall:

```python
from circuit_discovery import CircuitDiscovery

# Initialize discovery system
discovery = CircuitDiscovery(model_name="google/gemma-2b", device='cuda')

# Define factual prompts
dataset = {
    'entity_location': [
        {'clean': 'The Eiffel Tower is located in Paris',
         'corrupted': 'The Eiffel Tower is located in London'},
        {'clean': 'Mount Everest is in the Himalayas',
         'corrupted': 'Mount Everest is in the Alps'},
    ]
}

# Discover circuits
circuits = discovery.discover_all_circuits(dataset, train_sae=True)

# Analyze results
for circuit in circuits:
    print(f"Circuit: {circuit.name}")
    print(f"  Nodes: {len(circuit.nodes)}")
    print(f"  Attribution score: {circuit.attribution_score:.4f}")
```

For a complete walkthrough, see the [tutorial notebook](examples/tutorial.ipynb).

## Running the Full Pipeline

To run the complete circuit discovery and validation pipeline:

```bash
python main.py
```

Results are saved to `outputs/circuit_discovery_YYYYMMDD_HHMMSS/` and include:
- Circuit visualizations (PNG)
- Detailed analysis reports (Markdown, JSON)
- Hypothesis testing results (CSV)
- Validation metrics and statistics

## Project Structure

```
factual-recall-circuits/
├── src/
│   ├── circuit_discovery.py      # Core discovery algorithms
│   ├── testing_pipeline.py       # Validation framework
│   └── utils.py                  # Visualization and analysis
├── docs/                         # Documentation
├── examples/                     # Tutorial notebooks
├── main.py                       # Full pipeline implementation
└── requirements.txt              # Dependencies
```

## Methodology

### Attribution Analysis

The toolkit implements two primary attribution methods:

**Integrated Gradients**: Computes feature importance by integrating gradients along the path from a baseline to the actual input.

**Activation Patching**: Measures causal importance by replacing activations and observing the effect on model outputs.

### Sparse Autoencoders

Sparse autoencoders are trained on layer activations to extract interpretable features:
- Expansion factor of 4x over hidden dimension
- L1 sparsity penalty for feature interpretability
- Layer-specific training for optimal feature extraction

### Circuit Discovery

Circuits are identified through:
1. Attribution scoring across model layers
2. Feature extraction via sparse autoencoders
3. Node importance ranking
4. Edge construction between layers
5. Circuit validation and faithfulness testing

## Supported Models

Currently tested with:
- Gemma 2B
- GPT-2 (all variants)

The architecture is designed to be model-agnostic and can be adapted to other transformer-based language models.

## Advanced Examples

Additional examples are available in the `examples/` directory:
- [Interactive Tutorial](examples/tutorial.ipynb) - Step-by-step walkthrough
- Custom fact types and hypothesis testing
- Visualization and analysis techniques

## Documentation

Please check out our [documentation](docs/README.md) and don't hesitate to raise issues or contribute if anything is unclear.

Additional resources:
- [Quick Start Guide](docs/QUICKSTART.md)
- [Project Summary](docs/PROJECT_SUMMARY.md)
- [VS Code Setup](README_VSCODE.md)

## Performance Notes

- GPU recommended for reasonable performance (30-60 minutes on A100)
- CPU execution supported but significantly slower (several hours)
- Memory requirements: ~10GB GPU RAM for Gemma 2B
- First run downloads model weights (~5GB, cached for future use)

## License

This project is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Citing

If you use this project in your research, please cite:

```bibtex
@software{factual_recall_circuits2025,
  title={Factual Recall Circuits in Gemma 2B},
  author={Samuel Wang},
  year={2025},
  url={https://github.com/samjiawng/factual-recall-circuits}
}
```

## Contributing

Contributions are welcome! Areas for improvement include:
- Support for additional model architectures
- Enhanced attribution methods
- Improved sparse autoencoder architectures
- Extended fact type coverage
- Performance optimizations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work builds on:
- Anthropic's circuit discovery methodology
- Neuropedia attribution techniques
- Sparse autoencoder interpretability research

## References

- [Mechanistic Interpretability](https://transformer-circuits.pub/)
- [Neuropedia](https://neuropedia.ai/)
- [Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2309.08600)
- [Attribution Patching](https://arxiv.org/abs/2310.10348)
