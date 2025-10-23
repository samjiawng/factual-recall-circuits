# Factual Recall Circuits - VS Code Setup

## 📁 Project Structure

```
factual-recall-circuits/
├── .vscode/                    # VS Code configuration
│   ├── settings.json          # Python & editor settings
│   └── launch.json            # Debug configurations
├── src/                       # Core source code
│   ├── __init__.py
│   ├── circuit_discovery.py   # Circuit discovery engine
│   ├── testing_pipeline.py    # Validation framework
│   └── utils.py               # Visualization tools
├── docs/                      # Documentation
│   ├── README.md             # Full documentation
│   ├── QUICKSTART.md         # Quick start guide
│   ├── PROJECT_SUMMARY.md    # Project overview
│   └── FILE_INDEX.md         # File reference
├── examples/                  # Examples and tutorials
│   └── tutorial.ipynb        # Interactive notebook
├── main.py                    # Main pipeline
├── verify_install.py          # Installation checker
├── requirements.txt           # Dependencies
├── START_HERE.md             # Getting started guide
├── .gitignore                # Git ignore rules
└── *.code-workspace          # VS Code workspace file
```

## 🚀 Quick Start in VS Code

### 1. Open in VS Code

**Option A: Open Workspace File (Recommended)**
```bash
code factual-recall-circuits.code-workspace
```

**Option B: Open Folder**
```bash
code .
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

Press `F5` or click **Run > Start Debugging** and select "Run Verify Install"

Or from terminal:
```bash
python verify_install.py
```

### 4. Run Circuit Discovery

Press `F5` and select "Run Main Pipeline"

Or from terminal:
```bash
python main.py
```

## 🎯 VS Code Features Configured

### Debugging
- Press `F5` to run with debugger
- Set breakpoints by clicking left of line numbers
- Inspect variables in Debug view
- Step through code with F10 (step over) / F11 (step into)

### Available Debug Configurations
1. **Python: Main Pipeline** - Run the full discovery process
2. **Python: Verify Install** - Check installation
3. **Python: Current File** - Debug any open Python file
4. **Python: Jupyter Notebook** - Launch tutorial notebook

### Code Navigation
- `Ctrl+Click` (or `Cmd+Click` on Mac) - Go to definition
- `F12` - Go to definition
- `Shift+F12` - Find all references
- `Ctrl+P` - Quick open files
- `Ctrl+Shift+F` - Search across all files

### IntelliSense
- Auto-completion as you type
- Hover over functions to see documentation
- `Ctrl+Space` - Trigger IntelliSense manually

### Testing
- Tests will appear in the Testing view (beaker icon)
- Run/debug individual tests
- See test coverage

## 📝 Common Tasks in VS Code

### Run Main Pipeline
```bash
python main.py
```
Or use the debug configuration "Python: Main Pipeline"

### Edit Core Code
Navigate to `src/` folder:
- `circuit_discovery.py` - Modify discovery algorithms
- `testing_pipeline.py` - Change validation methods
- `utils.py` - Add new visualizations

### Run Jupyter Notebook
```bash
# From VS Code terminal
jupyter notebook examples/tutorial.ipynb
```
Or click on `tutorial.ipynb` and VS Code will open it in the interactive notebook editor

### View Documentation
All documentation is in the `docs/` folder:
- Start with `START_HERE.md` (in root)
- Read `docs/QUICKSTART.md` for fast start
- Check `docs/README.md` for full details

## 🔧 Customizing Your Setup

### Change Python Interpreter
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose your virtual environment

### Install Additional Extensions (Recommended)
- **Python** (Microsoft) - Already configured
- **Jupyter** - For notebook support
- **Pylance** - Enhanced Python language support
- **Python Docstring Generator** - Auto-generate docstrings
- **GitLens** - Git supercharged
- **Better Comments** - Improved comment highlighting

### Modify Linting Rules
Edit `.vscode/settings.json` to change:
- Line length (default: 88 for Black)
- Linting rules
- Formatting options

## 🐛 Debugging Tips

### Set Breakpoints
Click left of line number to set breakpoint (red dot appears)

### Debug Variables
- Hover over variables while debugging to see values
- Use Debug Console to evaluate expressions
- Watch window to track specific variables

### Common Debug Scenarios

**1. Debug Circuit Discovery**
```python
# Set breakpoint in circuit_discovery.py
# In discover_circuit() method
# Step through to see how circuits are built
```

**2. Debug Testing Pipeline**
```python
# Set breakpoint in testing_pipeline.py
# In test_hypothesis() method
# Inspect activation values
```

**3. Debug Visualizations**
```python
# Set breakpoint in utils.py
# In visualize_circuit() function
# See how plots are generated
```

## 📊 Working with Results

Results are saved to `outputs/` directory (created on first run):
```
outputs/
└── circuit_discovery_YYYYMMDD_HHMMSS/
    ├── REPORT.md                     # Main findings
    ├── circuit_summary.txt           # Detailed info
    ├── circuits.json                 # Machine-readable data
    ├── *.png                         # Visualizations
    └── *_results.csv                 # Test results
```

### View Results in VS Code
- PNG files open in image preview
- JSON/CSV files have syntax highlighting
- Markdown files render with preview (`Ctrl+K V`)

## 🎓 Learning the Code

### Start Here
1. Open `main.py` - See the full pipeline
2. Open `src/circuit_discovery.py` - Core algorithms
3. Open `examples/tutorial.ipynb` - Interactive walkthrough

### Use Code Navigation
- Click on class/function names to jump to definition
- See all usages with `Shift+F12`
- Hover for inline documentation

### Read Docstrings
Every function has documentation:
```python
def discover_circuit(self, fact_prompts, fact_type):
    """
    Discover a circuit for a specific type of factual recall
    
    Parameters
    ----------
    fact_prompts : List[Dict[str, str]]
        List of dicts with 'clean' and 'corrupted' prompts
    fact_type : str
        Type of fact (e.g., 'entity_attribute')
    
    Returns
    -------
    Circuit
        Discovered circuit structure
    """
```

## 🚀 Next Steps

1. **Run the pipeline** - `python main.py` or press F5
2. **Explore the tutorial** - Open `examples/tutorial.ipynb`
3. **Modify the code** - Add your own datasets in `main.py`
4. **Read the docs** - Check `docs/` folder
5. **Experiment** - Try different fact types and hypotheses

## 💡 Pro Tips

### Terminal in VS Code
- `` Ctrl+` `` to open integrated terminal
- Multiple terminals supported
- Automatically activates virtual environment

### Split Editor
- `Ctrl+\` to split editor
- View code and docs side-by-side
- Drag tabs to rearrange

### Markdown Preview
- `Ctrl+K V` to preview Markdown files
- Side-by-side preview
- Auto-updates as you edit

### Search and Replace
- `Ctrl+F` - Find in file
- `Ctrl+H` - Find and replace
- `Ctrl+Shift+F` - Find in all files

### Git Integration
- Source Control view (left sidebar)
- Stage, commit, push all in VS Code
- View diffs inline

## ❓ Troubleshooting

### "Module not found" error
```bash
# Make sure you're in the right directory
cd factual-recall-circuits

# Verify PYTHONPATH includes src/
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or run from VS Code (already configured)
```

### Debugger not working
1. Check Python interpreter is selected
2. Verify virtual environment is activated
3. Restart VS Code

### Imports not resolving
1. Open Command Palette (`Ctrl+Shift+P`)
2. Run "Python: Clear Cache and Reload Window"
3. Check `.vscode/settings.json` has correct paths

## 📚 Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [VS Code Debugging](https://code.visualstudio.com/docs/editor/debugging)
- [Python in VS Code](https://code.visualstudio.com/docs/languages/python)

## 🎉 You're Ready!

Everything is configured and ready to go. Just:

1. Open this folder in VS Code
2. Press `F5` to run
3. Start discovering circuits!

Happy coding! 🧠✨
