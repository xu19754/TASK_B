#!/bin/zsh
set -eu
taskb_root="$(cd "$(dirname "$0")" && pwd)"
export TASKB_BASKET_REVISION=v3
exec "$taskb_root/.venv-basket/bin/python" "$taskb_root/scripts/basket_random_trials.py" \
  --view "$taskb_root/reports/basket_v3/random_three_2026-10-26/mixed_three_seed_20261026"
