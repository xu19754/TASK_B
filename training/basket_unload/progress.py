"""Write the state consumed by the desktop monitor."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs/basket_unload"


def update(stage, detail, run_dir=RUN_DIR, **fields):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    row = dict(stage=stage, detail=detail, timestamp=stamp, **fields)
    temporary = run_dir / "status.tmp"
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(run_dir / "status.json")
    with (run_dir / "events.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} | {stage} | {detail}\n")


def save_image(array, filename, run_dir=RUN_DIR):
    from PIL import Image
    path = Path(run_dir) / filename
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)
