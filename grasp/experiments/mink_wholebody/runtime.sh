#!/usr/bin/env bash
GRASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASKB_PYTHON="${TASKB_PYTHON:-/opt/conda/envs/isaaclab/bin/python}"
export PYTHONPATH="$(dirname "$GRASP_ROOT"):$GRASP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TORCH_LIB="$("$TASKB_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMNI_KIT_ALLOW_ROOT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
