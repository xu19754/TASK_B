"""Desktop monitor for the live training process and its rendered observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/basket_unload")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    window = tk.Tk()
    window.option_add("*Font", ("Noto Sans CJK SC", 11))
    window.title("Task B | MuJoCo 整机视觉卸料训练")
    window.geometry("1280x900+20+20")
    style = ttk.Style()
    style.configure("Title.TLabel", font=("Noto Sans CJK SC", 19, "bold"))
    style.configure("TLabel", font=("Noto Sans CJK SC", 11))
    ttk.Label(window, text="MuJoCo · B2 + Z1 · 腕部 RGB-D · 尾窗卸料", style="Title.TLabel").pack(pady=12)
    status = tk.StringVar(value="阶段 1：建立独立训练环境与 GPU 依赖")
    ttk.Label(window, textvariable=status, wraplength=1200).pack(pady=5)
    detail = tk.StringVar(value="仅显示真实进度；尚未开始强化学习训练。")
    ttk.Label(window, textvariable=detail, wraplength=1200).pack(pady=5)
    frames = ttk.Frame(window)
    frames.pack(fill="both", expand=True, padx=12)
    labels = []
    for title in ("整机动力学实时画面", "腕部 RGB", "腕部光轴深度"):
        pane = ttk.LabelFrame(frames, text=title)
        pane.pack(side="left", fill="both", expand=True, padx=3)
        label = ttk.Label(pane, text="等待渲染数据")
        label.pack(fill="both", expand=True)
        labels.append(label)
    log = tk.Text(window, height=13, font=("Noto Sans Mono CJK SC", 10), bg="#101b26", fg="#dbe7ed", wrap="word")
    log.pack(fill="x", padx=12, pady=10)
    image_stamps = [None] * 3

    def refresh():
        try:
            row = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
            status.set(row.get("stage", "运行中"))
            detail.set(row.get("detail", ""))
        except (OSError, ValueError):
            pass
        for i, filename in enumerate(("latest_world.png", "latest_rgb.png", "latest_depth.png")):
            path = args.run_dir / filename
            try:
                stamp = path.stat().st_mtime_ns
                if image_stamps[i] != stamp:
                    with Image.open(path) as source:
                        picture = source.convert("RGB")
                        if i and picture.width < 240:
                            picture = picture.resize(
                                (240, round(picture.height * 240 / picture.width)), Image.Resampling.NEAREST)
                        picture.thumbnail((810 if i == 0 else 240, 480))
                    photo = ImageTk.PhotoImage(picture)
                    labels[i].configure(image=photo, text="")
                    labels[i].image = photo
                    image_stamps[i] = stamp
            except (OSError, ValueError):
                pass
        try:
            lines = (args.run_dir / "events.log").read_text(encoding="utf-8").splitlines()[-16:]
            log.delete("1.0", "end")
            log.insert("end", "\n".join(lines))
        except OSError:
            pass
        window.after(500, refresh)

    refresh()
    window.mainloop()


if __name__ == "__main__":
    main()
