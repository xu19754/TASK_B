#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/experiments/mink_wholebody/runtime.sh"
args=()
while (($#)); do
    case "$1" in
        --config|--output)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            args+=("$1" "$(realpath -m -- "$2")"); shift 2 ;;
        --config=*|--output=*)
            option="${1%%=*}"; value="${1#*=}"
            args+=("$option" "$(realpath -m -- "$value")"); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
cd "$ROOT"
exec /usr/bin/python3 -u "$ROOT/experiments/mink_banana/supervise.py" "${args[@]}"
