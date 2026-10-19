@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (set "PYEXE=py -3") else (set "PYEXE=python")
if not exist ".venv\Scripts\python.exe" %PYEXE% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "Backscatter Analysis.spec" del /q "Backscatter Analysis.spec"
pyinstaller --noconfirm --clean --windowed --name "Backscatter Analysis" --hidden-import openpyxl --collect-all sklearn --collect-all scipy --collect-all matplotlib backscatter_app.py
echo.
echo Built Windows application in: dist\Backscatter Analysis\
pause
