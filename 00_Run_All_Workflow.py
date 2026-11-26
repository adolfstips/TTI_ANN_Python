# ==============================================================================
# Script: 00_Run_All_Workflow.py
# Title: Master Reproducible Pipeline for TTI Deep Learning Emulation (Python)
# Description:
#   1. Verifies required Python packages up front (with optional interactive install).
#   2. Inhibits Windows host sleep automatically on both native Windows and WSL2
#      via Win32 SetThreadExecutionState, restoring normal sleep upon exit.
#   3. Streams all live console output simultaneously to ./output_collection.txt.
#   4. Executes Scripts 01-05, 11, and 06-10 (pass --skip-training to regenerate
#      tables and figures in ~15 seconds without retraining).
# Reference: Stips et al. (2026) - Environmental Modelling & Software
# License: GPL-3.0
# ==============================================================================

import gc
import os
import sys
import time
import runpy
import shutil
import ctypes
import platform
import subprocess
import importlib.util
from datetime import datetime


def check_python_requirements(auto_install: bool = False, need_netcdf: bool = True) -> None:
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
    ]
    missing = [pip_name for mod_name, pip_name in required if importlib.util.find_spec(mod_name) is None]

    # Require at least one NetCDF backend (netCDF4 or xarray)
    if need_netcdf and importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("xarray") is None:
        missing.append("netCDF4")

    if missing:
        print("\n" + "=" * 70)
        print("MISSING PYTHON PACKAGES DETECTED: " + ", ".join(missing))
        print("=" * 70)
        do_install = auto_install
        if not do_install and sys.stdin is not None and sys.stdin.isatty():
            ans = input("Install missing Python packages now via pip? [Y/n]: ").strip().lower()
            do_install = (ans == "" or ans.startswith("y"))
        if do_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        else:
            raise ModuleNotFoundError(
                f"Please install missing packages first: pip install {' '.join(missing)}"
            )


class WindowsSleepInhibitor:
    """
    Holds Windows ES_CONTINUOUS | ES_SYSTEM_REQUIRED (0x80000001) solely while
    active, working seamlessly on both native Windows Python and WSL2 Ubuntu.
    """
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        self.mode = None
        self.proc = None

    def __enter__(self):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                self.mode = "win32"
            except Exception:
                self.mode = None
        else:
            rel = platform.uname().release.lower()
            ps_exe = shutil.which("powershell.exe")
            if ("microsoft" in rel or "wsl" in rel) and ps_exe:
                ps_cmd = (
                    "Add-Type -MemberDefinition '[DllImport(\"kernel32.dll\")] "
                    "public static extern uint SetThreadExecutionState(uint esFlags);' "
                    "-Name 'Win32Sleep' -Namespace 'Win32'; "
                    "try { [Win32.Win32Sleep]::SetThreadExecutionState(2147483649) | Out-Null; "
                    "[Console]::In.ReadLine() | Out-Null } "
                    "finally { [Win32.Win32Sleep]::SetThreadExecutionState(2147483648) | Out-Null }"
                )
                try:
                    self.proc = subprocess.Popen(
                        [ps_exe, "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self.mode = "wsl2_powershell"
                except Exception:
                    self.proc = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "win32":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except Exception:
                pass
        elif self.mode == "wsl2_powershell" and self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(b"\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        return False


class TeeLogger:
    """Mirrors live terminal output into output_collection.txt."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def run_step(script_name: str, torch_mod) -> None:
    print("\n" + "=" * 70)
    print(f"EXECUTING: {script_name} | Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    t0 = time.perf_counter()
    runpy.run_path(script_name, run_name="__main__")
    dt_min = (time.perf_counter() - t0) / 60.0
    print(f"COMPLETED: {script_name} in {dt_min:.2f} minutes.")
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def main() -> None:
    skip_training = "--skip-training" in sys.argv[1:]
    auto_install = "--install-packages" in sys.argv[1:]

    need_nc = (not skip_training) or os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc") or os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc.zip")
    check_python_requirements(auto_install=auto_install, need_netcdf=need_nc)

    import torch
    from tti_common import ensure_netcdf_unzipped

    log_path = "./output_collection.txt"
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    with WindowsSleepInhibitor() as sleep_guard, open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = TeeLogger(orig_stdout, log_f)
        sys.stderr = TeeLogger(orig_stderr, log_f)
        try:
            t_total_start = time.perf_counter()
            print("=" * 70)
            print("TTI_ANN_Python MASTER WORKFLOW LOG")
            print(f"Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Python     : {sys.version.split()[0]} ({platform.platform()})")
            if torch.cuda.is_available():
                print(f"GPU Device : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
            else:
                print("GPU Device : CPU")
            if sleep_guard.mode:
                print(f"Sleep Guard: Active ({sleep_guard.mode} -> ES_SYSTEM_REQUIRED)")
            print("=" * 70)

            if not skip_training or os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc") or os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc.zip"):
                if os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc") or os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc.zip"):
                    ensure_netcdf_unzipped()

            training_scripts = [
                "01_Kfold_MLP_SPAT.py",
                "02_Kfold_LSTM_SPAT.py",
                "03_Rolling_MLP_TIME.py",
                "04_Rolling_LSTM_TIME.py",
                "05_Spatial_Ensemble_MLP_SPAT.py",
                "11_Controlled_SPAT_Comparison_SM.py",
            ]

            post_scripts = [
                "06_Generate_Manuscript_Tables.py",
                "07_Plot_Fig1_Spatial_Block_Map.py",
                "08_Plot_Figs2_3_4_SHAP_Analysis.py",
                "09_Plot_Figs5_6_S3_Spatial_Ensemble.py",
                "10_Plot_FigsS1_S2_Confusion_Matrices.py",
            ]

            if not skip_training:
                for scr in training_scripts:
                    run_step(scr, torch)
            else:
                print("\nSkipping training scripts (01-05, 11); regenerating tables and figures...")

            for scr in post_scripts:
                if scr == "07_Plot_Fig1_Spatial_Block_Map.py" and not os.path.exists("./GES_Baltic_5year_ALLE_Combined_4D.nc"):
                    print(f"Skipping {scr} (NetCDF not present in current folder).")
                    continue
                run_step(scr, torch)

            tot_min = (time.perf_counter() - t_total_start) / 60.0
            print("\n" + "=" * 70)
            print(f"ALL PYTHON PIPELINE STEPS COMPLETED SUCCESSFULLY in {tot_min:.2f} minutes.")
            print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    print(f"Sleep lock released and full console transcript saved to: {log_path}")


if __name__ == "__main__":
    main()
