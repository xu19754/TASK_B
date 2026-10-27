#!/bin/zsh
set -eu
taskb_root="$(cd "$(dirname "$0")/.." && pwd)"
taskb_python="$taskb_root/.venv-basket/bin/python"
taskb_entry="$taskb_root/scripts/basket_viewer.py"
if [[ "${1:-}" == "--arm" ]]; then
  taskb_entry="$taskb_root/scripts/basket_arm_viewer.py"
  shift
fi
if [[ ! -x "$taskb_python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$taskb_root/.venv-basket" --python 3.12
    uv pip install --python "$taskb_python" -r "$taskb_root/scripts/requirements-basket-viewer.txt"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    "$HOME/.local/bin/uv" venv "$taskb_root/.venv-basket" --python 3.12
    "$HOME/.local/bin/uv" pip install --python "$taskb_python" -r "$taskb_root/scripts/requirements-basket-viewer.txt"
  else
    python3 -m venv "$taskb_root/.venv-basket"
    "$taskb_python" -m pip install -r "$taskb_root/scripts/requirements-basket-viewer.txt"
  fi
fi
exec "$taskb_python" "$taskb_entry" "$@"
