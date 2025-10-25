# START HERE

## Overview

Complete implementation of circuit discovery using Neuropedia-style attribution graphs and sparse autoencoders for factual recall in Gemma 2B.

## Quick Start

### Installation
```bash
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn einops tqdm jupyter
python verify_install.py
```

### Run
```bash
python main.py
```

### Interactive Tutorial
```bash
jupyter notebook tutorial.ipynb
```

## Core Components

**Modules** (~2000 lines)
- `circuit_discovery.py` - Attribution graphs and circuit extraction
- `testing_pipeline.py` - Automated validation framework
- `utils.py` - Visualization and analysis
- `main.py` - Example pipeline

**Documentation**
- `README.md` - Full technical documentation
- `QUICKSTART.md` - Fast-track guide
- `PROJECT_SUMMARY.md` - Project overview
- `tutorial.ipynb` - Interactive walkthrough

## Methods

Discovers interpretable circuits encoding factual knowledge using:
- Integrated gradients
- Activation patching
- Sparse autoencoders with L1 regularization
- Automated hypothesis testing

## Project Structure

```
factual-recall-circuits/
├── circuit_discovery.py
├── testing_pipeline.py
├── utils.py
├── main.py
├── tutorial.ipynb
├── verify_install.py
├── requirements.txt
└── docs/
    ├── README.md
    ├── QUICKSTART.md
    ├── PROJECT_SUMMARY.md
    └── FILE_INDEX.md
```

## Expected Output

Discovery results:
- Circuit graphs with node attributions
- Validation metrics (precision, recall, specificity)
- Visualization plots
- Structured reports (JSON/CSV)

Results saved to: `outputs/circuit_discovery_[timestamp]/`

## Documentation Guide

| Purpose | Document |
|---------|----------|
| Quick execution | QUICKSTART.md |
| Full reference | README.md |
| Overview | PROJECT_SUMMARY.md |
| Learning | tutorial.ipynb |
| File navigation | FILE_INDEX.md |

## Requirements

**Minimum**: Python 3.8+, CPU
**Recommended**: GPU (10-20x faster)

## Implementation Notes

Implements methods from CV research:
- Circuit discovery pipeline
- Attribution patching
- Sparse autoencoders
- Automated testing framework
- PyTorch-based implementation

## Next Steps

First time: Run `verify_install.py`, then `main.py`
Learning: Open `tutorial.ipynb`
Customization: Modify `main.py` for new datasets