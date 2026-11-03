# ==============================================================================
# Script: 00_Run_All_Workflow.py
# Title: Master End-to-End Execution Pipeline (Raw Data -> Models -> Tables/Figures)
# Description: Sequentially executes the complete Python computational workflow
#              for Stips et al. (2026).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import os
import sys
import time
import gc
import runpy
import torch
from tti_common import ensure_nc_file

# Set RUN_TRAINING = True to train all neural networks from scratch (GPU recommended).
# Set RUN_TRAINING = False (or pass --skip-training) to regenerate Tables & Figures from ./Output/*.pkl.
RUN_TRAINING = True


def run_step(script_name: str):
    if not os.path.exists(script_name):
        raise FileNotFoundError(f"Script not found: {script_name}")

    print("\n" + "=" * 70)
    print(f"EXECUTING: {script_name} | Started at: {time.strftime('%H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()
    runpy.run_path(script_name, run_name="__main__")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60.0
    print(f"COMPLETED: {script_name} in {elapsed:.2f} minutes.")


def main():
    global RUN_TRAINING
    if "--skip-training" in sys.argv:
        RUN_TRAINING = False

    for d in ["./Output", "./Tables", "./Figures"]:
        os.makedirs(d, exist_ok=True)

    ensure_nc_file("./GES_Baltic_5year_ALLE_Combined_4D.nc")
    pipeline_start = time.time()

    if RUN_TRAINING:
        training_scripts = [
            "01_Kfold_MLP_SPAT.py",
            "02_Kfold_LSTM_SPAT.py",
            "03_Rolling_MLP_TIME.py",
            "04_Rolling_LSTM_TIME.py",
            "05_Spatial_Ensemble_MLP_SPAT.py",
        ]
        for scr in training_scripts:
            run_step(scr)
    else:
        print("\nSkipping Phase 1 (Model Training). Using pre-computed .pkl files in ./Output/")

    postproc_scripts = [
        "06_Generate_Manuscript_Tables.py",
        "07_Plot_Fig1_Spatial_Block_Map.py",
        "08_Plot_Figs2_3_4_SHAP_Analysis.py",
        "09_Plot_Figs5_6_S3_Spatial_Ensemble.py",
        "10_Plot_FigsS1_S2_Confusion_Matrices.py",
    ]
    for scr in postproc_scripts:
        run_step(scr)

    total_mins = (time.time() - pipeline_start) / 60.0
    print("\n" + "=" * 70)
    print(f"ALL WORKFLOW STEPS COMPLETED SUCCESSFULLY IN {total_mins:.2f} MINUTES!")
    print("  - Intermediate PKL Data : ./Output/")
    print("  - Manuscript Tables     : ./Tables/")
    print("  - Publication Figures   : ./Figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
