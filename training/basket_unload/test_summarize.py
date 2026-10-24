"""Independent-trial bookkeeping, never a physical success test."""
import json

import pytest

from .summarize import summarize


def _report(successes):
    return {"evaluation_kind": "held_out_policy_evaluation", "episodes": 2,
            "environment": {"stage": "unload", "num_objects": 1, "control_dt": .02},
            "checkpoint": "frozen.pt", "checkpoint_global_step": 100,
            "image_mode": "normal", "reset_bank": "independent.npz",
            "episode_records": [
                {"episode_id": i, "seed": 100 + i, "bank_row": i, "combination": [0, 0, 0],
                 "success": success, "failure_flags": {"failure_timeout": not success},
                 "length": 50, "min_clearance_m": .01, "max_speed_ratio": .5}
                for i, success in enumerate(successes)]}


def test_completed_seed_pairing_and_known_wilson_interval(tmp_path):
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report([False, True])))
    candidate.write_text(json.dumps(_report([True, True])))
    summary = summarize([baseline, candidate])
    rate = summary["evaluations"][0]["overall"]
    assert rate["success_rate"] == .5
    assert rate["wilson95"] == pytest.approx([.09453120573423074, .9054687942657693])
    pair = summary["paired_comparisons"][0]
    assert pair["candidate_only_success_seeds"] == [100]
    assert pair["baseline_only_success_seeds"] == []
    assert pair["both_success"] == 1


@pytest.mark.parametrize("fault", ["partial", "repeated_seed", "missing_episode", "different_seed"])
def test_unfinished_or_unpaired_trials_cannot_support_comparison(tmp_path, fault):
    baseline, candidate = _report([False, True]), _report([True, True])
    if fault == "partial":
        candidate["status"] = "in_progress"
    elif fault == "repeated_seed":
        candidate["episode_records"][1]["seed"] = 100
    elif fault == "missing_episode":
        candidate["episode_records"].pop()
    else:
        candidate["episode_records"][1]["seed"] = 103
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, report in zip(paths, [baseline, candidate]):
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        summarize(paths)
