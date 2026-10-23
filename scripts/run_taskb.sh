#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TASKB_PYTHON:-python}"
# Kit loads CUDA libraries lazily; retain PyTorch's bundled linalg search path.
TORCH_LIB="$($PYTHON -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$PYTHON" -u "$ROOT/scripts/play_taskb.py" "$@"
