@echo off
REM ============================================================
REM  People Counter - GPU launcher (NVIDIA CUDA)
REM  Runs from source so it can use your GPU (e.g. RTX 3060).
REM  Requires Python 3.10-3.12 installed and on PATH.
REM  First run installs everything into a local .venv (a few GB,
REM  one-time). Later runs start instantly and use the GPU.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

REM Skip installation if a CUDA-enabled PyTorch is already present.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo [setup] Installing dependencies with CUDA PyTorch (one-time, large download)...
    python -m pip install --upgrade pip
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
    pip install -r requirements.txt
)

echo.
echo [run] Starting People Counter (GPU)...
python main.py

endlocal
