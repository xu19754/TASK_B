#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TASKB_PYTHON:-python}"
MODE="${1:-}"
case "$MODE" in
  recognition) SCRIPT="test_object_recognition.py" ;;
  traversal) SCRIPT="test_traversal_video.py" ;;
  *) echo "Usage: bash scripts/run_feature_test.sh {recognition|traversal} [arguments]" >&2; exit 2 ;;
esac
shift
# Help should work even before activating the Isaac environment.
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
  exec "$PYTHON" -u "$ROOT/scripts/$SCRIPT" "$@"
fi
TORCH_LIB="$("$PYTHON" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$PYTHON" -u "$ROOT/scripts/$SCRIPT" "$@"
