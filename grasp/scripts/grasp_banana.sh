#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_mink_banana.sh" --mode grasp --headless --device cpu --enable_cameras --record-video --trials 1 --config "$ROOT/experiments/mink_banana/configs/centered_offset_13.json" "$@"
