#!/bin/bash
set -e
cd "$(dirname "$0")"

# Keep the Matplotlib cache in a known writable directory. This also prevents
# repeated long font-cache builds on some macOS installations.
export MPLCONFIGDIR="$PWD/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

if [ ! -d ".venv" ]; then
  echo "Creating Python environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi

# A normal Matplotlib install always contains backend_svg. Some macOS installs
# can end up incomplete after an interrupted/partial pip install. Repair it
# automatically instead of allowing the analysis to fail after all samples ran.
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
