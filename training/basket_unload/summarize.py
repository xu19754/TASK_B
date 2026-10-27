"""Summarize completed held-out evaluations without querying simulation truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def _rate(successes, episodes):
    z = 1.959963984540054
    fraction = successes / episodes
    denominator = 1 + z * z / episodes
    center = (fraction + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / episodes
                         + z * z / (4 * episodes * episodes)) / denominator
    return {"success": successes, "episodes": episodes, "success_rate": fraction,
            "wilson95": [max(0., center - half), min(1., center + half)]}


def summarize(paths):
    results, paired_records = [], []
    for path in map(Path, paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("episode_records", [])
        if (report.get("status") == "in_progress"
                or report.get("evaluation_kind") != "held_out_policy_evaluation"
                or report.get("environment", {}).get("stage") != "unload"
                or not rows or len(rows) != report.get("episodes")):
            raise ValueError(f"Requires a completed held-out unload evaluation: {path}")
        count = len(rows)
        if sorted(row["episode_id"] for row in rows) != list(range(count)):
            raise ValueError(f"Assigned episode coverage is incomplete or repeated: {path}")
        if len({row["seed"] for row in rows}) != count:
            raise ValueError(f"Repeated seeds cannot be counted as independent trials: {path}")
        objects = report["environment"]["num_objects"]
        by_combo, failures = defaultdict(list), Counter()
        for row in rows:
            key = "-".join(map(str, row["combination"][:objects]))
            by_combo[key].append(row)
            failures.update(name for name, value in row["failure_flags"].items() if value)
        groups = {key: _rate(sum(bool(row["success"]) for row in group), len(group))
                  for key, group in sorted(by_combo.items())}
        duration = [row["length"] * report["environment"]["control_dt"] for row in rows]
        results.append({
            "evaluation": str(path), "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": report["checkpoint"], "checkpoint_global_step": report["checkpoint_global_step"],
            "image_mode": report["image_mode"], "reset_bank": report["reset_bank"],
            "overall": _rate(sum(bool(row["success"]) for row in rows), count),
            "by_combination": groups, "failure_flags": dict(failures),
            "mean_duration_sim_s": sum(duration) / count,
            "maximum_duration_sim_s": max(duration),
            "minimum_clearance_m": min(row["min_clearance_m"] for row in rows),
            "maximum_speed_ratio": max(row["max_speed_ratio"] for row in rows),
        })
        paired_records.append({row["seed"]: row for row in rows})
    comparisons = []
    if len(paired_records) > 1:
        baseline = paired_records[0]
        for index, records in enumerate(paired_records[1:], 1):
            if (results[0]["reset_bank"] != results[index]["reset_bank"]
                    or baseline.keys() != records.keys() or any(
                    baseline[seed]["combination"] != row["combination"]
                    or baseline[seed]["bank_row"] != row["bank_row"]
                    for seed, row in records.items())):
                raise ValueError("Paired evaluations require identical assigned seed, class and bank rows")
            comparisons.append({
                "baseline": results[0]["evaluation"], "candidate": results[index]["evaluation"],
                "episodes": len(records),
                "both_success": sum(bool(row["success"]) and bool(baseline[seed]["success"])
                                    for seed, row in records.items()),
                "candidate_only_success_seeds": [seed for seed, row in sorted(records.items())
                                                  if row["success"] and not baseline[seed]["success"]],
                "baseline_only_success_seeds": [seed for seed, row in sorted(records.items())
                                                 if baseline[seed]["success"] and not row["success"]],
            })
    return {"scope": "Completed MuJoCo held-out unload trials; not formal Isaac score. Wilson intervals assume independent sampled seeds. Failure flags may overlap.",
            "evaluations": results, "paired_comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {Path(path).resolve() for path in args.evaluations}:
        parser.error("Output must not overwrite an input evaluation")
    summary = summarize(args.evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
