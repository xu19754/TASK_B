"""Isaac integration test: one full scan, classified object crops and a JSON report."""

import json
import time
import traceback

from feature_test_support import (
    CONTROL_DT, create_environment, launch, make_run_directory, publish_summary, step_environment,
)


def run(args, app):
    import torch
    from demo.solution import AlgSolution

    directory = make_run_directory(args.output_dir)
    print(f"[RecognitionTest] seed={args.seed}; output={directory}", flush=True)
    env, solution = None, None
    step, score = 0, 0.0
    reason, error, exit_code = "simulation_closed", None, 2
    started = time.monotonic()
    records = []
    try:
        env = create_environment(args)
        obs, _ = env.reset(seed=args.seed)
        solution = AlgSolution(scan_output_dir=None)
        solution.reset()
        started = time.monotonic()
        with (directory / "scan_log.jsonl").open("w", encoding="utf-8") as log:
            while app.is_running():
                if step * CONTROL_DT >= args.max_seconds:
                    reason = "simulation_time_limit"
                    break
                if args.max_walltime_s is not None and time.monotonic() - started >= args.max_walltime_s:
                    reason = "wall_time_limit"
                    break
                with torch.inference_mode():
                    response = solution.predicts(obs, score)
                    finder = solution.object_finder
                    if step % 25 == 0 or finder.done:
                        log.write(json.dumps({"step": step, "sim_time": step * CONTROL_DT,
                                              "status": finder.status, "actual_turn_radians": finder.total_yaw,
                                              "confirmed": sum(item.confirmed for item in finder.objects)}) + "\n")
                    if finder.done:
                        reason = finder.status
                        break
                    obs, reward, terminated, truncated, _ = step_environment(env, response)
                    score += float(reward[0]) / CONTROL_DT
                step += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    reason = "environment_termination_or_timeout"
                    break
    except KeyboardInterrupt:
        reason, exit_code = "interrupted", 130
    except Exception as exc:
        reason, error, exit_code = "error", str(exc), 1
        traceback.print_exc()
    finally:
        try:
            if solution is not None:
                finder = solution.object_finder
                finder.save_results(directory)
                records = finder.save_object_images(directory / "items", confirmed_only=True)
                if not finder.ready:
                    finder.save_object_images(directory / "candidates")
            success = bool(solution is not None and solution.object_finder.ready
                           and reason == "complete" and len(records) == 18)
            if success:
                exit_code = 0
            summary = {"test": "object_recognition", "seed": args.seed, "success": success,
                       "reason": reason, "error": error, "sim_seconds": step * CONTROL_DT,
                       "wall_seconds": time.monotonic() - started, "saved_confirmed_images": len(records),
                       "expected_items": 18, "image_directory": "items", "report": "objects.json",
                       "limitations": "Recognition is heuristic; labels require visual inspection, not ground-truth accuracy scores."}
            publish_summary(directory, summary)
            print(f"[RecognitionTest] {reason}; saved {len(records)}/18 confirmed item images", flush=True)
        finally:
            if env is not None:
                env.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(launch("recognition", run))
