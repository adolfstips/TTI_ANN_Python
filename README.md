# TTI_ANN_Python — Deep Learning Emulation of the Baltic Sea Trophic Transfer Index

Companion Python/PyTorch implementation for **Stips et al. (2026)**, *"A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index"* (*Environmental Modelling & Software*; primary R repository: https://github.com/adolfstips/TTI_ANN_FIG).

## Methodological Highlights (Revision 2)
- **Zero-Leakage Spatial & Temporal Validation:**
  - `01_Kfold_MLP_SPAT.py` & `02_Kfold_LSTM_SPAT.py`: 10-fold spatial block cross-validation (`10×10` grid cells; 80% train / 10% validation / 10% held-out test).
  - `03_Rolling_MLP_TIME.py` & `04_Rolling_LSTM_TIME.py`: Expanding 30-year rolling-origin out-of-sample temporal extrapolation (`1995–2020`, 6 unseen forward assessment pentads conditional on contemporaneous predictors, separated from training windows by a 5-year validation buffer).
- **Controlled Common-Period & Recurrence Experiment (`11_Controlled_SPAT_Comparison_SM.py`):**
  - Evaluates `MLP_SPAT_Common8` (`1-pentad` input), `MLP_SEQ7_Windowed` (`7-pentad` flattened feedforward control), and `LSTM_SPAT_Common8` (`7-pentad` sequential recurrent network) on the identical held-out spatial blocks and identical 8 target pentads (`1985–2020`), reporting paired fold-level differences and two-sided paired *t*-tests (**Supplementary Table S1**).
- **Pure-GPU 2D & 3D Trajectory-Level Monte Carlo Shapley (`tti_common.py`):**
  - Implements paired Monte Carlo Shapley estimation (Štrumbelj & Kononenko, 2014) directly on GPU tensors—swapping complete 7-pentad chronological trajectories (`Steps 1–7`) for `LSTM_SPAT` and `LSTM_TIME` alongside an exact 2-block temporal inertia decomposition (`Steps 1–6` antecedent history vs. `Step 7` target pentad).

- **Automatic Pre-Flight Check, Sleep Guard & Transcript Logging (`00_Run_All_Workflow.py`):**
  - Verifies package dependencies up front, prevents Windows/WSL2 host sleep strictly while running (`SetThreadExecutionState`), and saves the complete console log to `output_collection.txt`.

## Quick Start
```bash
# 1. Install dependencies (Python >= 3.10 with CUDA-enabled PyTorch)
pip install -r requirements.txt

# 2. Place GES_Baltic_5year_ALLE_Combined_4D.nc.zip (or .nc) in the root folder and run all steps:
python 00_Run_All_Workflow.py

# 3. Regenerate tables and figures only (skipping neural network training):
python 00_Run_All_Workflow.py --skip-training
```

## License
GNU General Public License v3.0 (GPL-3.0).
