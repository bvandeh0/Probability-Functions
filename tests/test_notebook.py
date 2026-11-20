"""Notebook integrity and execution smoke test.

Run the tutorial cells from the repository root, using a headless plotting backend.
This detects broken examples without requiring a Jupyter server.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "probability_distributions.ipynb"


def test_notebook_format_and_code_syntax():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"<notebook cell {index}>", "exec")


def test_notebook_runs_end_to_end():
    runner = """
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

doc = json.loads(Path('notebooks/probability_distributions.ipynb').read_text())
scope = {'__name__': '__notebook__'}
for index, cell in enumerate(doc['cells']):
    if cell['cell_type'] != 'code':
        continue
    code = compile(''.join(cell['source']), f'<cell {index}>', 'exec')
    exec(code, scope)
    plt.close('all')
"""
    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
