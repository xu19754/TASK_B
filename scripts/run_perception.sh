#!/usr/bin/env bash
set -euo pipefail

PERCEPTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERCEPTION_PYTHON="${TASKB_PYTHON:-python}"
PERCEPTION_TORCH_LIB="$("$PERCEPTION_PYTHON" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export LD_LIBRARY_PATH="${PERCEPTION_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$PERCEPTION_PYTHON" -u "$PERCEPTION_ROOT/training/perception/play.py" "$@"
