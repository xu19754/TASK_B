"""Isaac integration test: visit 18 poses, optionally wait for pickup, record motion."""

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
    target_started = 0.0
    pickup_enabled = getattr(args, "pickup", False)
    try:
        route = prepare_route(args.source)
        route["mode"] = "external_pickup_confirmation" if pickup_enabled else "visit_only_keep_all_obstacles"
        route["pickup_enabled"] = pickup_enabled
        route["dwell_seconds"] = args.dwell_seconds
        write_json(directory / "route.json", route)
        env = create_environment(args, video=True)
        obs, _ = env.reset(seed=args.seed)
        solution = AlgSolution(scan_before_start=False)
        solution.reset()
        items = route["items"]
        solution.set_traversal([item["target_pose"] for item in items], [item["position"] for item in items],
                               obstacle_radii=[item["obstacle_radius"] for item in items],
                               pickup_enabled=pickup_enabled, dwell_seconds=args.dwell_seconds, **route["navigation"])
        nav = solution.traversal
        if pickup_enabled:
            print("[TraversalTest] pickup ON: waiting for external confirm_pickup(); the default arm controller does not implement grasping.", flush=True)
        else:
            print(f"[TraversalTest] pickup OFF: dwell {args.dwell_seconds:g}s at each pose; keep all obstacles.", flush=True)
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
                                          "target_index": nav.target_index,
                                          "awaiting_visit": nav.awaiting_pickup and not pickup_enabled,
                                          "awaiting_pickup": nav.awaiting_pickup and pickup_enabled,
                                          "navigation": nav.diagnostics(),
                                          "active_obstacle_count": len(nav.active_obstacle_indices)}) + "\n")
                    if nav.blocked_reason:
                        reason, error = "path_blocked", nav.blocked_reason
                        if video.due(sim_time):
                            video.append(env.unwrapped.render(), sim_time,
                                         f"BLOCKED | visited {len(visits)}/18 | sim {sim_time:.2f}s")
                        break
                    # Traversal owns the dwell timer. In pickup mode only an
                    # actual external pickup confirmation advances the index.
                    while len(visits) < nav.target_index:
                        index = len(visits)
                        visits.append({"target_index": index, "object_id": items[index]["object_id"],
                                       "sim_time": sim_time, "estimated_position": solution.localization.position,
                                       "estimated_yaw": solution.localization.yaw,
                                       "completion": "pickup_confirmed" if pickup_enabled else "dwell_completed"})
                        write_json(directory / "visits.json", visits)
                        print(f"[TraversalTest] visited {len(visits)}/18: object {items[index]['object_id']}", flush=True)
                        target_started = sim_time
                    if video.due(sim_time) or nav.completed:
                        mode = "External pickup" if pickup_enabled else f"Visit-only {args.dwell_seconds:g}s dwell"
                        caption = (f"{mode} | visited {len(visits)}/18 | target {min(nav.target_index + 1, 18)}/18"
                                   f" | {nav.status} | sim {sim_time:.2f}s")
                        # Retain a final completion frame, at most one frame interval after arrival.
                        frame_time = max(sim_time, video.next_time) if nav.completed else sim_time
                        video.append(env.unwrapped.render(), frame_time, caption)
                    if nav.completed:
                        reason = "completed_18_visits"
                        break
                    # Arrival ends the navigation timeout; allow a full dwell
                    # even when arriving just before the navigation deadline.
                    if sim_time - target_started >= args.per_target_timeout and nav.status != "dwelling":
                        reason = "pickup_timeout" if nav.status == "awaiting_pickup" else "target_timeout"
                        error = f"Target {nav.target_index}: {nav.status}; position error {nav.diagnostics()['position_error']:.3f} m"
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
                "source": args.source["source"], "pickup_enabled": pickup_enabled,
                "dwell_seconds": args.dwell_seconds,
                "obstacles_removed": bool(solution is not None and solution.traversal is not None
                                          and len(solution.traversal.active_obstacle_indices) < 18),
                "final_navigation": solution.traversal.diagnostics() if solution is not None and solution.traversal is not None else None,
                "limitations": "Visits use command-integrated localization; inspect video for physical drift. Pickup mode requires an external grasp controller and success confirmation; the default arm controller only holds ARM_Q0.",
            })
            print(f"[TraversalTest] {reason}; visited {len(visits)}/18", flush=True)
        finally:
            if env is not None:
                env.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(launch("traversal", run))
