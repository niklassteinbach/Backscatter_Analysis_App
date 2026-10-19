#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

rm -rf build dist "Backscatter Analysis.spec"
pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "Backscatter Analysis" \
  --hidden-import openpyxl \
  --collect-all sklearn \
  --collect-all scipy \
  --collect-all matplotlib \
  backscatter_app.py

echo
echo "Built: dist/Backscatter Analysis.app"
open dist
