#!/bin/zsh
set -eu
taskb_root="$(cd "$(dirname "$0")" && pwd)"
export TASKB_BASKET_REVISION=v3
exec "$taskb_root/scripts/run_basket_viewer.sh" --arm --edge "$@"
