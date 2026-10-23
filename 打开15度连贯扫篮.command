#!/bin/zsh
set -eu
taskb_root="$(cd "$(dirname "$0")" && pwd)"
exec "$taskb_root/scripts/run_basket_viewer.sh" --arm --sweep "$@"
