#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if (($# != 1)); then echo "usage: $0 OUTPUT_DIRECTORY" >&2; exit 2; fi
OUTPUT="$(realpath -m -- "$1")"
MANIFEST="${OUTPUT}.manifest.json"
if [[ -e "$OUTPUT" || -e "$MANIFEST" ]]; then echo "Output or manifest already exists; refusing to overwrite" >&2; exit 2; fi
mkdir -p "$(dirname "$OUTPUT")"
source "$ROOT/experiments/mink_wholebody/runtime.sh"
"$TASKB_PYTHON" - "$ROOT" "$MANIFEST" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1]).resolve(); repo=root.parent
files=[p for d in ('experiments','scripts') for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
files.append(root/'verify_nine.py')
tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','-z','assets','models','taskb_student'], text=True).split('\0')
files += [repo/p for p in tracked if p and (repo/p).is_file()]
manifest={str(p.relative_to(root)) if p.is_relative_to(root) else '../'+str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
Path(sys.argv[2]).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
exec "$TASKB_PYTHON" -u "$ROOT/verify_nine.py" --output "$OUTPUT" --expected "$MANIFEST"
