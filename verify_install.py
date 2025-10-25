"""Verify installation dependencies."""

import sys
import importlib

REQUIRED_PACKAGES = {
    'torch': 'torch',
    'transformers': 'transformers', 
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'einops': 'einops',
    'tqdm': 'tqdm',
    'sklearn': 'scikit-learn',
}

def verify_packages():
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    return missing

def main():
    missing = verify_packages()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    import torch
    cuda_available = torch.cuda.is_available()
    device = "CUDA" if cuda_available else "CPU"
    print(f"Installation verified. Device: {device}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)