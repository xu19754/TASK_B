#!/usr/bin/env bash
set -euo pipefail
export DISPLAY="${DISPLAY:-:20}"
taskb_unload_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$taskb_unload_root"
taskb_unload_python="$taskb_unload_root/.venv-unload/bin/python"
if [[ ! -x "$taskb_unload_python" ]]; then
  python -m venv --system-site-packages .venv-unload
  "$taskb_unload_python" -m pip install -r training/basket_unload/requirements.txt
fi
mkdir -p runs/basket_unload
if ! pgrep -f '^.*python.*-m training.basket_unload.dashboard' >/dev/null; then
  "$taskb_unload_python" -m training.basket_unload.dashboard > runs/basket_unload/dashboard.log 2>&1 &
fi
exec "$taskb_unload_python" -u -m training.basket_unload.train "$@"
