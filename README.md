# A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index (Python/PyTorch Pipeline)

**Authors:** Adolf Stips*, Ove Pärn, Luca Polimene, Diego Macias-Moy*, Svetla Miladinova, Elisa Garcia-Gorriz, Olaf Duteil  
**Journal:** *Environmental Modelling & Software* (2026)  
**Contact:** `adolf.stips@ext.ec.europa.eu` | `diego.macias-moy@ec.europa.eu` | `adolf.stips@gmail.com`  
**Primary R/Torch Repository:** https://github.com/adolfstips/TTI_ANN_FIG/

---

## Overview

This repository (`TTI_ANN_Python`) provides the companion Python/PyTorch implementation of the data-processing, neural network training, Explainable AI (pure-GPU 2D and 3D Monte Carlo Shapley), controlled spatial evaluation, and publication visualization workflow for **"A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index"**.

All spatial block partitions (`10 × 10` and `20 × 20` grid blocks), expanding rolling-origin temporal splits (with a 5-year validation buffer), neural network architectures (`128–64` MLP and stacked LSTM), controlled 1985–2020 experiments (`MLP_SPAT_Common8`, `MLP_SEQ7_Windowed`, `LSTM_SPAT_Common8`), and 3D trajectory-level Shapley estimators mirror the primary R implementation (`TTI_ANN_FIG`).

---

## Repository Structure

```text
TTI_ANN_Python/
├── README.md                                 # Complete workflow documentation
├── LICENSE                                   # GNU General Public License v3.0 (GPL-3.0)
├── requirements.txt                          # Python package dependencies
├── GES_Baltic_5year_ALLE_Combined_4D.nc.zip  # Zipped NetCDF input dataset (14 pentads, 1955-2020)
├── tti_common.py                             # Shared VRAM PyTorch trainers & GPU 2D/3D Shapley engine
├── 00_Run_All_Workflow.py                    # Master pipeline runner (package check, sleep guard & logging)
├── 01_Kfold_MLP_SPAT.py                      # 10-Fold Spatial Block CV (MLP_SPAT)
├── 02_Kfold_LSTM_SPAT.py                     # 10-Fold Spatial Block CV & 3D SHAP (LSTM_SPAT)
├── 03_Rolling_MLP_TIME.py                    # Rolling-Origin Temporal Extrapolation (MLP_TIME)
├── 04_Rolling_LSTM_TIME.py                   # Rolling-Origin Temporal Extrapolation & 3D SHAP (LSTM_TIME)
├── 05_Spatial_Ensemble_MLP_SPAT.py           # 10-Fold Spatial Bootstrap Ensemble Reconstruction
├── 06_Generate_Manuscript_Tables.py          # Compiles Manuscript Tables 2, 3, and 4 (.csv)
├── 07_Plot_Fig1_Spatial_Block_Map.py         # Generates Figure 1
├── 08_Plot_Figs2_3_4_SHAP_Analysis.py        # Generates Figures 2, 3, and 4
├── 09_Plot_Figs5_6_S3_Spatial_Ensemble.py    # Generates Figures 5, 6, and Supplementary Figure S3
├── 10_Plot_FigsS1_S2_Confusion_Matrices.py   # Generates Supplementary Figures S1 and S2
├── 11_Controlled_SPAT_Comparison_SM.py       # Controlled 1985-2020 Spatial Evaluation (Supplementary Table S1)
├── output_collection.txt                     # Full timestamped console execution transcript
├── Output/                                   # Pre-computed & generated .pkl model outputs
├── Tables/                                   # Generated Manuscript & Supplementary Tables (.csv)
└── Figures/                                  # Generated Publication Figures (.png, 500 DPI)
```

---

## System Requirements & Installation

- **Python Environment:** Python >= 3.10 with PyTorch >= 2.2 (CUDA 12.x recommended; automatically falls back to CPU if CUDA is unavailable)
- **Installation:**
```bash
pip install -r requirements.txt
```

---

## Reproducing the Workflow

### Option A: Fast Regeneration of All Tables and Figures (< 20 Seconds, No Retraining)
Using the pre-computed `.pkl` archives in `./Output/`:
```bash
python 00_Run_All_Workflow.py --skip-training
```

### Option B: Complete End-to-End Reanalysis (Full Training + 2D/3D SHAP + Controlled SM)
```bash
python 00_Run_All_Workflow.py
```
*(Verifies dependencies at startup, inhibits Windows/WSL2 host sleep while running, and saves the complete console transcript to `output_collection.txt`.)*

---

## Script-to-Manuscript Mapping

| Script | Description | Manuscript Deliverable |
| :--- | :--- | :--- |
| `tti_common.py` | Shared NetCDF4/xarray loader, VRAM PyTorch trainers, and pure-GPU 2D/3D Monte Carlo Shapley & 2-block inertia estimators (Štrumbelj & Kononenko, 2014) | Methods (Sections 2.3, 2.8) |
| `01_Kfold_MLP_SPAT.py` | 10-fold spatial block CV & 2D SHAP (`MLP_SPAT`) | Table 2, Table 4, Figs. 3, 4, S1 |
| `02_Kfold_LSTM_SPAT.py` | 10-fold spatial block CV, 3D trajectory SHAP & lag split (`LSTM_SPAT`) | Table 2, Table 4, Figs. 3, 4, S2 |
| `03_Rolling_MLP_TIME.py` | 30-year rolling-origin temporal extrapolation & 2D SHAP (`MLP_TIME`) | Table 3, Table 4, Fig. 2 |
| `04_Rolling_LSTM_TIME.py` | 30-year rolling-origin temporal extrapolation, 3D SHAP & inertia split (`LSTM_TIME`) | Table 3, Table 4, Fig. 2 |
| `05_Spatial_Ensemble_MLP_SPAT.py` | 10-fold full-basin Fisher-*z* ensemble reconstruction | Figs. 5, 6, S3 |
| `06_Generate_Manuscript_Tables.py` | Exports formatted performance & GFI tables (`.csv`) | Tables 2, 3, 4 |
| `07_Plot_Fig1_Spatial_Block_Map.py` | Maps Fold 1 (`10 × 10` blocks; 80% / 10% / 10% split) | Figure 1 |
| `08_Plot_Figs2_3_4_SHAP_Analysis.py` | Plots temporal/spatial GFI stability & 10-fold stratified beeswarms | Figures 2, 3, 4 |
| `09_Plot_Figs5_6_S3_Spatial_Ensemble.py` | Plots area-weighted TTI trajectory, spatial maps & SD maps | Figures 5, 6, S3 |
| `10_Plot_FigsS1_S2_Confusion_Matrices.py` | Plots aggregated global confusion matrices | Figures S1, S2 |
| `11_Controlled_SPAT_Comparison_SM.py` | Controlled common-period (1985–2020) & `MLP_SEQ7` recurrence comparison | Supplementary Table S1 |

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
