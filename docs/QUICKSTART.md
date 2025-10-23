# Quick Start Guide

Get started with circuit discovery in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn einops tqdm

# 2. Verify CUDA (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Run Your First Discovery

### Option 1: Run Complete Pipeline (Recommended)

```bash
python main.py
```

This will:
- Load Gemma 2B
- Discover 4+ circuits
- Generate visualizations
- Export all results

**Time:** 30-60 minutes on GPU, several hours on CPU

### Option 2: Quick Test (Minimal)

```python
from circuit_discovery import CircuitDiscovery

# Initialize
discovery = CircuitDiscovery(device='cuda')

# Simple example
dataset = {
    'location': [
        {'clean': 'Paris is in France', 
         'corrupted': 'Paris is in Germany'}
    ]
}

# Discover
circuits = discovery.discover_all_circuits(dataset, train_sae=False)
print(f"Found {len(circuits)} circuits!")
```

**Time:** 5-10 minutes

### Option 3: Interactive Notebook

```bash
jupyter notebook tutorial.ipynb
```

Great for learning step-by-step!

## What You'll Get

After running `main.py`:

```
/mnt/user-data/outputs/circuit_discovery_[timestamp]/
├── REPORT.md                      # 📊 Main findings
├── circuit_summary.txt            # 📝 Detailed descriptions  
├── circuits.json                  # 💾 Machine-readable data
├── circuit_0_entity_location.png  # 🖼️ Visualizations
├── circuit_1_capital_country.png
├── circuit_comparison.png
├── circuit_overlap.png
└── *_results.csv                  # 📈 Test results
```

## Understanding Results

### Circuit Structure
```python
circuit.name              # e.g., "entity_location_circuit"
circuit.nodes             # List of (layer, feature) tuples
circuit.edges             # Connections between nodes
circuit.attribution_score # Importance score (higher = more important)
```

### Test Results
- **Precision**: % of test cases where feature activates (want >80%)
- **Specificity**: % of control cases where it doesn't (want >70%)
- **Pass/Fail**: Hypothesis validated if both thresholds met

## Common Use Cases

### 1. Find What Features Detect
```python
# Create hypothesis about a feature
from testing_pipeline import FeatureHypothesis

hyp = FeatureHypothesis(
    feature_id=(8, 100),
    hypothesis="Detects country names",
    test_prompts=["France", "Japan", "Brazil"],
    control_prompts=["apple", "seven", "quickly"]
)

# Test it
result = tester.validator.test_hypothesis(hyp)
print(f"Passed: {result.passed}")
```

### 2. Add New Fact Types
```python
my_dataset = {
    'math_facts': [
        {'clean': '2 + 2 = 4', 'corrupted': '2 + 2 = 5'},
        {'clean': '10 / 2 = 5', 'corrupted': '10 / 2 = 4'},
    ]
}

circuits = discovery.discover_all_circuits(my_dataset)
```

### 3. Visualize Specific Circuit
```python
from utils import visualize_circuit

fig = visualize_circuit(circuits[0], save_path='my_circuit.png')
```

## Troubleshooting

### "Out of memory"
```python
# Use CPU or smaller expansion
discovery = CircuitDiscovery(device='cpu')
```

### "Model download fails"
```python
# Set HuggingFace cache
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

### "Takes too long"
```python
# Reduce dataset size
dataset = {k: v[:2] for k, v in dataset.items()}  # Only 2 examples per type

# Or skip SAE training
circuits = discovery.discover_all_circuits(dataset, train_sae=False)
```

### "Import errors"
```bash
pip install -r requirements.txt --upgrade
```

## Performance Tips

### Speed Up Discovery
1. **Use GPU**: 10-20x faster than CPU
2. **Reduce epochs**: Set `epochs=3` in SAE training
3. **Sample data**: Use fewer prompt pairs
4. **Skip SAEs**: Set `train_sae=False`

### Improve Accuracy
1. **More data**: Add more prompt pairs per fact type
2. **More epochs**: Train SAEs longer (10-20 epochs)
3. **Tune threshold**: Adjust `threshold` parameter
4. **Multiple runs**: Average results across runs

## Next Steps

After getting results:

1. **Read REPORT.md** - Understand what was found
2. **Check visualizations** - See circuit structure
3. **Analyze results CSVs** - Dive into details
4. **Modify code** - Customize for your needs
5. **Share findings** - Contribute back!

## Quick Reference

### Key Classes
- `CircuitDiscovery` - Main discovery engine
- `CircuitTester` - Validation framework
- `FeatureHypothesis` - Testable hypothesis
- `Circuit` - Discovered circuit structure

### Key Functions
- `discover_all_circuits()` - Find circuits
- `test_hypothesis()` - Validate feature
- `visualize_circuit()` - Create graph
- `compare_circuits()` - Compare multiple

### Important Parameters
- `device`: 'cuda' or 'cpu'
- `threshold`: Minimum attribution score (default: 0.01)
- `epochs`: SAE training iterations (default: 10)
- `train_sae`: Whether to train SAEs (default: True)

## Getting Help

- 📖 Read full README.md for details
- 💻 Check tutorial.ipynb for examples  
- 🐛 GitHub issues for bugs
- 💬 Discussions for questions

## Example Output

```
=== DISCOVERED CIRCUITS ===

Circuit: entity_location_circuit
  Nodes: 12
  Edges: 8
  Attribution: 0.234
  
Circuit: capital_country_circuit
  Nodes: 15
  Edges: 11
  Attribution: 0.198
  
=== TESTING SUMMARY ===
Total hypotheses: 20
Passed: 16 (80.0%)
Average precision: 85.2%
Average specificity: 78.9%
```

---

**Ready to discover some circuits? Run `python main.py` now!** 🚀
