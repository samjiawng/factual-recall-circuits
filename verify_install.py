"""
Installation Verification Script
Run this to verify that all dependencies are correctly installed
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name:20s} installed")
        return True
    except ImportError:
        print(f"✗ {package_name:20s} MISSING - install with: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("Checking Installation")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Core dependencies
    print("Core Dependencies:")
    all_ok &= check_import("torch", "torch")
    all_ok &= check_import("transformers", "transformers")
    all_ok &= check_import("numpy", "numpy")
    all_ok &= check_import("pandas", "pandas")
    
    print()
    print("Visualization:")
    all_ok &= check_import("matplotlib", "matplotlib")
    all_ok &= check_import("seaborn", "seaborn")
    
    print()
    print("Utilities:")
    all_ok &= check_import("einops", "einops")
    all_ok &= check_import("tqdm", "tqdm")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("networkx", "networkx")
    
    print()
    print("=" * 60)
    
    # Check CUDA
    print("\nHardware Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    
    print()
    print("=" * 60)
    
    # Check project files
    print("\nProject Files:")
    import os
    expected_files = [
        'circuit_discovery.py',
        'testing_pipeline.py',
        'utils.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'tutorial.ipynb'
    ]
    
    for filename in expected_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} MISSING")
            all_ok = False
    
    print()
    print("=" * 60)
    
    # Final verdict
    print()
    if all_ok:
        print("🎉 Everything looks good! You're ready to go!")
        print()
        print("Next steps:")
        print("  1. Read QUICKSTART.md")
        print("  2. Run: python main.py")
        print("  3. Or try: jupyter notebook tutorial.ipynb")
    else:
        print("⚠️  Some dependencies are missing.")
        print()
        print("To install all dependencies:")
        print("  pip install -r requirements.txt")
    
    print()
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
