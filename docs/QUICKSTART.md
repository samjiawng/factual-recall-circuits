# Quick Start

## Installation

```bash
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn einops tqdm
```

Verify CUDA (optional):
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Full Pipeline
```bash
python main.py
```

Runs complete discovery, generates visualizations, exports results.

Time: 30-60 minutes (GPU), several hours (CPU)

### Minimal Example
```python
from circuit_discovery import CircuitDiscovery

discovery = CircuitDiscovery(device='cuda')

dataset = {
    'location': [
        {'clean': 'Paris is in France', 
         'corrupted': 'Paris is in Germany'}
    ]
}

circuits = discovery.discover_all_circuits(dataset, train_sae=False)
```

Time: 5-10 minutes

### Interactive Tutorial
```bash
jupyter notebook tutorial.ipynb
```

## Output Structure

```
outputs/circuit_discovery_[timestamp]/
├── REPORT.md
├── circuit_summary.txt
├── circuits.json
├── circuit_*.png
└── *_results.csv
```

## Common Tasks

### Test Feature Hypothesis
```python
from testing_pipeline import FeatureHypothesis

hyp = FeatureHypothesis(
    feature_id=(8, 100),
    hypothesis="Detects country names",
    test_prompts=["France", "Japan", "Brazil"],
    control_prompts=["apple", "seven", "quickly"]
)

result = tester.validator.test_hypothesis(hyp)
```

### Add Custom Fact Types
```python
dataset = {
    'math_facts': [
        {'clean': '2 + 2 = 4', 'corrupted': '2 + 2 = 5'},
    ]
}

circuits = discovery.discover_all_circuits(dataset)
```

### Visualize Circuit
```python
from utils import visualize_circuit

visualize_circuit(circuits[0], save_path='circuit.png')
```

## Troubleshooting

**Out of memory**:
```python
discovery = CircuitDiscovery(device='cpu')
```

**Model download fails**:
```python
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

**Too slow**:
```python
dataset = {k: v[:2] for k, v in dataset.items()}
circuits = discovery.discover_all_circuits(dataset, train_sae=False)
```

**Import errors**:
```bash
pip install -r requirements.txt --upgrade
```

## Performance

**Speed up**:
- Use GPU (10-20x faster)
- Reduce epochs in SAE training
- Use fewer prompt pairs
- Set `train_sae=False`

**Improve accuracy**:
- Add more prompt pairs
- Increase SAE training epochs
- Tune threshold parameter
- Average multiple runs

## Key Components

**Classes**:
- `CircuitDiscovery` - Main engine
- `CircuitTester` - Validation
- `FeatureHypothesis` - Hypothesis structure
- `Circuit` - Circuit data

**Functions**:
- `discover_all_circuits()` - Run discovery
- `test_hypothesis()` - Validate feature
- `visualize_circuit()` - Generate graph
- `compare_circuits()` - Compare multiple

**Parameters**:
- `device`: 'cuda' or 'cpu'
- `threshold`: Minimum attribution (default: 0.01)
- `epochs`: SAE training iterations (default: 10)
- `train_sae`: Enable SAE training (default: True)

## Results Interpretation

**Circuit attributes**:
- `circuit.name` - Circuit identifier
- `circuit.nodes` - (layer, feature) tuples
- `circuit.edges` - Node connections
- `circuit.attribution_score` - Importance score

**Test metrics**:
- Precision: % test activations (target: >80%)
- Specificity: % control non-activations (target: >70%)
- Pass: Both thresholds met

## Documentation

- README.md - Complete documentation
- tutorial.ipynb - Interactive examples
- PROJECT_SUMMARY.md - Overview
