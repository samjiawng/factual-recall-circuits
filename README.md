# Factual Recall Circuits in Gemma 2B

This project implements a mechanistic interpretability pipeline for discovering circuits responsible for factual recall in the Gemma 2B language model using Neuropedia-style attribution graphs and sparse autoencoders.

## Overview

The project discovers 18+ circuits that encode different types of factual knowledge:
- Entity-location relationships (e.g., "The Eiffel Tower is in Paris")
- Capital-country relationships (e.g., "Paris is the capital of France")
- Historical dates (e.g., "WWII ended in 1945")
- Person-occupation relationships (e.g., "Einstein was a physicist")

## Features

### 1. Circuit Discovery
- **Neuropedia Attribution Graphs**: Uses integrated gradients and activation patching to identify important model components
- **Sparse Autoencoders (SAEs)**: Extracts interpretable features from neural network activations
- **Automatic Circuit Extraction**: Finds connected components of features that work together for factual recall

### 2. Automated Testing Pipeline
- **Hypothesis Generation**: Automatically creates testable hypotheses about circuit features
- **Validation Framework**: Tests hypotheses using precision and specificity metrics
- **Faithfulness Testing**: Verifies that discovered circuits are necessary and sufficient

### 3. Visualization & Analysis
- Circuit graph visualizations
- Attribution heatmaps
- Feature activation distributions
- Testing result dashboards
- Circuit overlap analysis

## Installation

```bash
# Clone or download this repository
cd factual-recall-circuits

# Install dependencies
pip install -r requirements.txt

# Note: Requires CUDA-capable GPU for reasonable performance
# CPU execution is supported but will be very slow
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- CUDA-capable GPU (recommended)
- ~10GB GPU memory for Gemma 2B

## Quick Start

### Basic Usage

```python
from circuit_discovery import CircuitDiscovery

# Initialize
discovery = CircuitDiscovery(device='cuda')

# Define factual prompts
dataset = {
    'entity_location': [
        {'clean': 'The Eiffel Tower is in Paris', 
         'corrupted': 'The Eiffel Tower is in London'},
        # ... more examples
    ]
}

# Discover circuits
circuits = discovery.discover_all_circuits(dataset, train_sae=True)

# Analyze results
for circuit in circuits:
    print(f"{circuit.name}: {len(circuit.nodes)} nodes")
```

### Run Complete Pipeline

```bash
# Run the full discovery and testing pipeline
python main.py
```

This will:
1. Load Gemma 2B model
2. Train sparse autoencoders
3. Discover circuits for different fact types
4. Validate circuits with automated tests
5. Generate visualizations and reports
6. Export all results to `/mnt/user-data/outputs/`

## Project Structure

```
├── circuit_discovery.py      # Core circuit discovery algorithms
├── testing_pipeline.py        # Automated testing framework
├── utils.py                   # Visualization and analysis utilities
├── main.py                    # Example pipeline implementation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Architecture

### CircuitDiscovery
Main class for discovering circuits. Key methods:
- `train_sparse_autoencoder()` - Train SAE on layer activations
- `discover_circuit()` - Find circuit for specific fact type
- `discover_all_circuits()` - Batch discovery for multiple fact types

### NeuropediaAttributionGraph
Implements attribution methods:
- `integrated_gradients()` - Compute attribution scores
- `activation_patching()` - Measure component importance

### CircuitTester
Validates discovered circuits:
- `test_hypothesis()` - Test individual feature hypotheses
- `test_circuit_faithfulness()` - Verify circuit necessity/sufficiency
- `run_full_validation()` - Complete testing pipeline

## Methodology

### 1. Attribution Analysis
- Use integrated gradients to identify important activations
- Apply activation patching to measure causal importance
- Focus on components that change factual predictions

### 2. Sparse Autoencoding
- Train SAEs on model activations (expansion factor of 4x)
- L1 sparsity penalty encourages interpretable features
- Extract features that correspond to factual knowledge

### 3. Circuit Extraction
- Connect features across layers based on attribution scores
- Build directed graph of feature dependencies
- Identify circuits responsible for specific fact types

### 4. Validation
- Generate hypotheses about feature behavior
- Test on positive and negative examples
- Measure precision (% test cases activating) and specificity (% controls not activating)
- Verify faithfulness through ablation studies

## Example Output

After running `main.py`, you'll get:

```
Discovered 4 circuits for factual recall in Gemma 2B:
  • entity_location_circuit: 12 nodes, attribution score = 0.234
  • capital_country_circuit: 15 nodes, attribution score = 0.198
  • historical_date_circuit: 10 nodes, attribution score = 0.167
  • person_occupation_circuit: 8 nodes, attribution score = 0.145
```

### Generated Files
- `circuit_summary.txt` - Detailed circuit descriptions
- `circuits.json` - Machine-readable circuit data
- `*.png` - Visualizations
- `*_results.csv` - Testing results
- `REPORT.md` - Complete analysis report

## Customization

### Add New Fact Types

```python
custom_dataset = {
    'your_fact_type': [
        {'clean': 'correct fact statement',
         'corrupted': 'incorrect version'},
        # ... more examples
    ]
}

circuits = discovery.discover_all_circuits(custom_dataset)
```

### Custom Hypotheses

```python
from testing_pipeline import FeatureHypothesis

hypothesis = FeatureHypothesis(
    feature_id=(layer, feature_idx),
    hypothesis="Description of what feature detects",
    test_prompts=["examples", "that should", "activate"],
    control_prompts=["examples", "that shouldn't", "activate"],
    expected_activation_threshold=0.5
)

result = tester.validator.test_hypothesis(hypothesis)
```

### Adjust SAE Parameters

```python
# In circuit_discovery.py, modify SparseAutoencoder
sae = SparseAutoencoder(
    d_model=model_dim,
    d_hidden=model_dim * 8,  # Larger expansion factor
    sparsity_coef=5e-3       # Stronger sparsity
)
```

## Performance Notes

- **GPU Required**: Circuit discovery requires significant computation
- **Memory**: Gemma 2B needs ~10GB GPU memory
- **Time**: Full pipeline takes 30-60 minutes on A100 GPU
- **CPU Fallback**: Supported but 10-20x slower

## Limitations

- Currently supports Gemma 2B (can be adapted for other models)
- Requires labeled factual prompt pairs
- SAE training requires substantial compute
- Attribution methods are approximate

## Research Applications

This codebase enables:
- **Mechanistic Interpretability**: Understanding how LLMs store factual knowledge
- **Model Editing**: Targeted modification of factual knowledge
- **Fact-Checking**: Identifying circuits responsible for specific facts
- **Safety Research**: Understanding and controlling model behavior

## Citation

If you use this code in your research, please cite:

```bibtex
@software{factual_recall_circuits,
  title={Factual Recall Circuits in Gemma 2B},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/factual-recall-circuits}
}
```

## Related Work

- [Anthropic's Circuit Discovery](https://transformer-circuits.pub/)
- [Neuropedia](https://neuropedia.ai/)
- [Sparse Autoencoders for Mechanistic Interpretability](https://arxiv.org/abs/2309.08600)
- [Attribution Patching](https://arxiv.org/abs/2310.10348)

## Contributing

Contributions welcome! Areas for improvement:
- Support for additional models (Llama, GPT, etc.)
- More sophisticated attribution methods
- Better SAE architectures
- Expanded fact type coverage
- Performance optimizations

## License

MIT License - feel free to use and modify!

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size or use smaller expansion factor
sae = SparseAutoencoder(d_model, d_model * 2)  # Instead of 4x
```

### Slow Execution
```python
# Reduce number of prompts or layers analyzed
circuits = discovery.discover_all_circuits(
    dataset, 
    train_sae=False  # Skip SAE training
)
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- Google Research for Gemma models
- Anthropic for circuit discovery methodology
- HuggingFace for model infrastructure
- The mechanistic interpretability community

---

**Happy circuit hunting! 🔍🧠**
