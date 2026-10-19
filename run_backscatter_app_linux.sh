#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export MPLCONFIGDIR="$PWD/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3 and rerun this script."
  exit 1
fi

if ! python3 - <<'PY'
import tkinter
PY
then
  echo
  echo "Tkinter is not installed for your system Python."
  echo "Install your distribution's Tk package, for example:"
  echo "  Ubuntu/Debian: sudo apt install python3-tk"
  echo "  Fedora:        sudo dnf install python3-tkinter"
  echo "  Arch:          sudo pacman -S tk"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating Python environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi

if ! python - <<'PY'
import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg
import matplotlib.backends.backend_tkagg
PY
then
  echo "Matplotlib installation looks incomplete. Repairing it now..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir "matplotlib>=3.7"
fi

python backscatter_app.py
