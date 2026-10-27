"""Isaac integration test: visit 18 poses, retain all objects and record the motion."""

import json
import time
import traceback

from feature_test_support import (
    CONTROL_DT, NoPathError, SimulationVideo, create_environment, launch, make_run_directory,
    prepare_route, publish_summary, step_environment, write_json,
)


def run(args, app):
    import torch
    from demo.solution import AlgSolution

    directory = make_run_directory(args.output_dir)
    print(f"[TraversalTest] seed={args.seed}; output={directory}", flush=True)
    env, solution, video = None, None, None
    step, score, visits = 0, 0.0, []
    reason, error, exit_code = "simulation_closed", None, 2
    started = time.monotonic()
    target_started, arrival_time = 0.0, None
    try:
        route = prepare_route(args.source)
        write_json(directory / "route.json", route)
        env = create_environment(args, video=True)
        obs, _ = env.reset(seed=args.seed)
        solution = AlgSolution(scan_before_start=False)
        solution.reset()
        items = route["items"]
        solution.set_traversal([item["target_pose"] for item in items], [item["position"] for item in items],
                               obstacle_radii=[item["obstacle_radius"] for item in items], **route["navigation"])
        nav = solution.traversal
        video = SimulationVideo(directory / "traversal.mp4", fps=args.fps)
        started = time.monotonic()
        with (directory / "trajectory.jsonl").open("w", encoding="utf-8") as log:
            while app.is_running():
                sim_time = step * CONTROL_DT
                if sim_time >= args.max_seconds:
                    reason = "simulation_time_limit"
                    break
                if args.max_walltime_s is not None and time.monotonic() - started >= args.max_walltime_s:
                    reason = "wall_time_limit"
                    break
                with torch.inference_mode():
                    response = solution.predicts(obs, score)
                    log.write(json.dumps({"step": step, "sim_time": sim_time,
                                          "estimated_position": solution.localization.position,
                                          "estimated_yaw": solution.localization.yaw,
                                          "command": solution._previous_command.tolist(),
                                          "target_index": nav.target_index, "awaiting_visit": nav.awaiting_pickup,
                                          "active_obstacle_count": len(nav.active_obstacle_indices)}) + "\n")
                    if nav.blocked_reason:
                        reason, error = "path_blocked", nav.blocked_reason
                        if video.due(sim_time):
                            video.append(env.unwrapped.render(), sim_time,
                                         f"BLOCKED | visited {len(visits)}/18 | sim {sim_time:.2f}s")
                        break
                    if nav.awaiting_pickup:
                        arrival_time = sim_time if arrival_time is None else arrival_time
                        if sim_time - arrival_time + 1e-9 >= args.dwell_seconds:
                            index = nav.confirm_visit()  # No physical pickup: keep this obstacle on the map.
                            visits.append({"target_index": index, "object_id": items[index]["object_id"],
                                           "sim_time": sim_time, "estimated_position": solution.localization.position,
                                           "estimated_yaw": solution.localization.yaw})
                            write_json(directory / "visits.json", visits)
                            print(f"[TraversalTest] visited {len(visits)}/18: object {items[index]['object_id']}", flush=True)
                            arrival_time, target_started = None, sim_time
                    if video.due(sim_time) or nav.completed:
                        caption = (f"Visit-only test | visited {len(visits)}/18 | target {min(nav.target_index + 1, 18)}/18"
                                   f" | sim {sim_time:.2f}s | all item obstacles retained")
                        # Retain a final completion frame, at most one frame interval after arrival.
                        frame_time = max(sim_time, video.next_time) if nav.completed else sim_time
                        video.append(env.unwrapped.render(), frame_time, caption)
                    if nav.completed:
                        reason = "completed_18_visits"
                        break
                    if sim_time - target_started >= args.per_target_timeout:
                        reason = "target_timeout"
                        break
                    obs, reward, terminated, truncated, _ = step_environment(env, response)
                    score += float(reward[0]) / CONTROL_DT
                step += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    # Isaac may auto-reset on termination: don't record reset frames as successful travel.
                    reason = "environment_termination_or_timeout"
                    break
    except KeyboardInterrupt:
        reason, exit_code = "interrupted", 130
    except NoPathError as exc:
        reason, error, exit_code = "path_blocked", str(exc), 2
    except Exception as exc:
        reason, error, exit_code = "error", str(exc), 1
        traceback.print_exc()
    finally:
        try:
            if video is not None:
                try:
                    video.close()  # Flush the MP4 even on interruption or an unreachable target.
                except Exception as exc:
                    reason, error, exit_code = "video_finalize_error", str(exc), 1
            success = reason == "completed_18_visits" and len(visits) == 18
            if success:
                exit_code = 0
            write_json(directory / "visits.json", visits)
            publish_summary(directory, {
                "test": "traversal_video", "seed": args.seed, "success": success, "reason": reason,
                "error": error, "visited_count": len(visits), "expected_items": 18,
                "sim_seconds": step * CONTROL_DT, "wall_seconds": time.monotonic() - started,
                "video": "traversal.mp4" if video is not None and video.frames else None,
                "video_frames": video.frames if video is not None else 0, "video_fps": args.fps,
                "source": args.source["source"], "obstacles_removed": False,
                "limitations": "Visits are judged using command-integrated localization; inspect the rendered video for physical drift. No grasping is tested.",
            })
            print(f"[TraversalTest] {reason}; visited {len(visits)}/18", flush=True)
        finally:
            if env is not None:
                env.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(launch("traversal", run))
