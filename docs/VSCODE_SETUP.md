# VS Code Setup

## Project Structure

```
factual-recall-circuits/
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── src/
│   ├── circuit_discovery.py
│   ├── testing_pipeline.py
│   └── utils.py
├── docs/
├── examples/
│   └── tutorial.ipynb
├── main.py
├── verify_install.py
└── requirements.txt
```

## Setup

### Open Project

```bash
code .
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Verify

```bash
python verify_install.py
```

Or press `F5` and select "Run Verify Install"

## Debugging

**Start debugging**: `F5`

**Available configurations**:
- Python: Main Pipeline
- Python: Verify Install
- Python: Current File
- Python: Jupyter Notebook

**Breakpoints**: Click left of line numbers

**Inspection**:
- Hover over variables
- Use Debug Console
- Watch window for tracking

**Navigation**:
- `F10` - Step over
- `F11` - Step into
- `Shift+F11` - Step out

## Code Navigation

- `Ctrl+Click` - Go to definition
- `F12` - Go to definition
- `Shift+F12` - Find references
- `Ctrl+P` - Quick open
- `Ctrl+Shift+F` - Search all files
- `Ctrl+Space` - Trigger IntelliSense

## Common Tasks

**Run pipeline**:
```bash
python main.py
```

**Open notebook**:
```bash
jupyter notebook examples/tutorial.ipynb
```
Or click `tutorial.ipynb` in VS Code

**Edit core code**: Navigate to `src/` folder

**View documentation**: Check `docs/` folder

## Configuration

**Select Python interpreter**:
1. `Ctrl+Shift+P`
2. "Python: Select Interpreter"
3. Choose venv

**Recommended extensions**:
- Python (Microsoft)
- Jupyter
- Pylance
- GitLens

**Modify settings**: Edit `.vscode/settings.json`

## Output Structure

```
outputs/circuit_discovery_[timestamp]/
├── REPORT.md
├── circuit_summary.txt
├── circuits.json
├── *.png
└── *_results.csv
```

**View in VS Code**:
- PNG: Image preview
- JSON/CSV: Syntax highlighting
- Markdown: Preview with `Ctrl+K V`

## Terminal

**Open**: `` Ctrl+` ``

**Features**:
- Multiple terminals
- Auto-activates venv
- Integrated with debugger

## Editor Features

**Split editor**: `Ctrl+\`

**Search**:
- `Ctrl+F` - Find in file
- `Ctrl+H` - Replace
- `Ctrl+Shift+F` - Find in all files

**Markdown preview**: `Ctrl+K V`

## Troubleshooting

**Module not found**:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**Debugger issues**:
1. Verify Python interpreter selected
2. Check venv activated
3. Restart VS Code

**Import resolution**:
1. `Ctrl+Shift+P`
2. "Python: Clear Cache and Reload Window"

## Learning the Code

1. Start with `main.py`
2. Explore `src/circuit_discovery.py`
3. Open `examples/tutorial.ipynb`

Use `F12` for definitions, `Shift+F12` for references.

## Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [VS Code Debugging](https://code.visualstudio.com/docs/editor/debugging)
- [Python in VS Code](https://code.visualstudio.com/docs/languages/python)