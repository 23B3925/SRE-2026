# Run it in VS Code (Windows / macOS / Linux)

## One-time setup
1. Install Python 3.10+ from python.org (tick "Add Python to PATH" on Windows) and VS Code with the **Python** extension.
2. Unzip `mmi_project_full.zip`. In VS Code: **File > Open Folder** > the `mmi_project` folder (the one containing `mmi/`, `scripts/`, `data/`).
3. Open a terminal (**Terminal > New Terminal**) and create an environment:
   - Windows:  `python -m venv .venv`  then  `.venv\Scripts\activate`
   - macOS/Linux:  `python3 -m venv .venv`  then  `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Press **Ctrl+Shift+P > Python: Select Interpreter** > choose the `.venv` one.
6. `python scripts/00_check_setup.py`  -> must end with "SETUP OK".

## Showing it (suggested order for the presentation)
| Step | Command (or Run and Debug panel, pick the same-named entry) | Time | What to show |
|---|---|---|---|
| 1 | `python scripts/01_audit_real_data.py` | 10 s | Real CST data audit: `outputs/real_data_audit.png` (mild/severe = healthy x constant) |
| 2 | `python scripts/06_present.py` | instant | All 4 reconstructions side by side (uses the saved results, so it is safe live) |
| 3 | `python scripts/02_run_demo.py severe` | ~4 min | Live run: the DBIM reconstruction happens in front of the audience. Output: `outputs/demo_severe_part1.png`, `..._part2.png` |
| 4 | `python scripts/02_run_demo.py mild` | ~4 min | Same for mild |

Tip: start step 3 before you begin talking about the method, and show the figure when it finishes. Results are
identical on every run (fixed noise seeds). Set `DR_DB=110` (Windows PowerShell: `$env:DR_DB=110`) to rerun with less noise.

## Optional: the generative (Pix2pix) stage
`pip install torch`, then `python scripts/03_make_dataset.py 200 0` (hours; can run several seeds in parallel),
`python scripts/04_train_pix2pix.py`, `python scripts/05_evaluate.py`. The Pix2pix script has not been executed yet.

## Folder map
`mmi/` algorithm modules | `scripts/` runnable steps | `data/` your .s8p, voxel and tissue files | `outputs/` figures and saved results | `.vscode/launch.json` one-click run entries
