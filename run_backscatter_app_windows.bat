@echo off
setlocal
cd /d "%~dp0"
set "MPLCONFIGDIR=%CD%\.matplotlib_cache"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYEXE=py -3"
) else (
    set "PYEXE=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    %PYEXE% -m venv .venv
    if errorlevel 1 goto :python_error
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Tkinter is missing. Re-run the official Python installer and enable Tcl/Tk and IDLE.
    pause
    exit /b 1
)

python -c "import matplotlib.backends.backend_pdf, matplotlib.backends.backend_svg, matplotlib.backends.backend_tkagg" >nul 2>nul
if errorlevel 1 (
    echo Matplotlib installation looks incomplete. Repairing it now...
    python -m pip install --upgrade --force-reinstall --no-cache-dir "matplotlib>=3.7"
)

python backscatter_app.py
if errorlevel 1 pause
exit /b %ERRORLEVEL%

:python_error
echo.
echo Python 3 could not create a virtual environment. Install Python 3 from python.org and try again.
pause
exit /b 1
