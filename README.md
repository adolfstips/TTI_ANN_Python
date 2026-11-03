# A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index (Python Implementation)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)

**Authors:** Adolf Stips*, Ove Pärn, Luca Polimene, Diego Macias-Moy*, Svetla Miladinova, Elisa Garcia-Gorriz, Olaf Duteil  
**Journal:** *Environmental Modelling & Software* (2026)  
**Contact:** `adolf.stips@ext.ec.europa.eu` | `adolf.stips@gmail.com`  
**Companion R Repository:** [https://github.com/adolfstips/TTI_ANN_FIG](https://github.com/adolfstips/TTI_ANN_FIG)

---

## Overview

This repository provides a complete, standalone **Python / PyTorch** translation of the R computational workflow for the manuscript **"A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index"**.

Every script (`00` through `10`), neural network architecture (`MLP` and `LSTM`), spatial/temporal partitioning scheme, zero-leakage standardization step, and Monte Carlo SHAP (`nsim = 50`) algorithm is a strict 1-to-1 mathematical counterpart to the reference R implementation.

---

## Repository Structure

```text
TTI_ANN_Python/
├── README.md                                  # Workflow documentation and instructions
├── LICENSE                                    # GNU General Public License v3.0 (GPL-3.0)
├── environment.yml                            # Conda environment specification (CUDA + PyTorch)
├── requirements.txt                           # Pip package dependencies
├── tti_common.py                              # Shared NetCDF loader, PyTorch models, training & GPU-SHAP engine
├── GES_Baltic_5year_ALLE_Combined_4D.nc.zip   # Place zipped or unzipped NetCDF input dataset in root
│
├── 00_Run_All_Workflow.py                     # Master script to execute the full pipeline
├── 01_Kfold_MLP_SPAT.py                       # 10-Fold Spatial Block CV (MLP_SPAT)
├── 02_Kfold_LSTM_SPAT.py                      # 10-Fold Spatial Block CV (LSTM_SPAT)
├── 03_Rolling_MLP_TIME.py                     # Rolling-Origin Temporal Forecast (MLP_TIME)
├── 04_Rolling_LSTM_TIME.py                    # Rolling-Origin Temporal Forecast (LSTM_TIME)
├── 05_Spatial_Ensemble_MLP_SPAT.py            # 10-Fold Spatial Bootstrap Ensemble Reconstruction
├── 06_Generate_Manuscript_Tables.py           # Generates Manuscript Tables 2, 3, and 4 (.docx & .csv)
├── 07_Plot_Fig1_Spatial_Block_Map.py          # Generates Figure 1
├── 08_Plot_Figs2_3_4_SHAP_Analysis.py         # Generates Figures 2, 3, and 4
├── 09_Plot_Figs5_6_S3_Spatial_Ensemble.py     # Generates Figures 5, 6, and Supplementary Fig S3
├── 10_Plot_FigsS1_S2_Confusion_Matrices.py    # Generates Supplementary Figures S1 and S2
│
├── Output/                                    # Generated .pkl model summaries, SHAP & ensemble outputs
├── Tables/                                    # Generated Manuscript Tables (.docx and .csv)
└── Figures/                                   # Generated Publication Figures (.png, 500 DPI)
```

---

## Quick Start & Installation

### 1. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate tti_ann
```
*(Alternatively, install into an existing environment via `pip install -r requirements.txt`).*

### 2. Place Input Dataset
Copy `GES_Baltic_5year_ALLE_Combined_4D.nc` (or `GES_Baltic_5year_ALLE_Combined_4D.nc.zip`) into the root folder of this repository. The workflow automatically unzips `.nc.zip` if the `.nc` file is not yet extracted.

### 3. Execute the Complete Pipeline
```bash
python 00_Run_All_Workflow.py
```
* Set `RUN_TRAINING = True` inside `00_Run_All_Workflow.py` (or pass `--skip-training` on the command line) to toggle between full neural network training and fast table/figure regeneration from `./Output/*.pkl`.
