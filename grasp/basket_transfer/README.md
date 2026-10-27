# Experimental grasp and basket release (known target)

Status: one successful sugar-box simulation, 0 degree starting yaw, 0.55 m initial robot/object distance, using the team v3 basket. This is a single observation after multiple failed attempts; it is not a reliability estimate or the formal Task B policy. `evidence/attempts.json` retains the outcomes, including failures. Full raw traces, configs, logs and videos are in `/root/grasp-validation/basket-physical-01` through `basket-physical-10` on the server. Source hashes are retained for each attempt; earlier controller snapshots are next to those outputs. The verified video SHA-256 is recorded in `evidence/attempt_10_independent.json`.

`controller_jointplan.py`, `bench_jointplan.py`, `experimental_joint_plan.json`, and `config_sugar_p00.json` reproduce the experimental setup. The plan is specific to this robot, basket, object, grasp and starting state; it was built from prior simulation. The independent evaluator uses simulator ground truth to gate transfer and evaluate retention. It is a diagnostic, not a permitted competition policy or a perception interface. The team asset configuration already disables robot self-collision: no full arm/basket/body collision audit has been performed. No change was made to gravity, object collision, mass, friction or actuator limits; no object was attached or reset during the run.

From repo root, using the Isaac Lab Python environment, the execution shape is:

```bash
source grasp/experiments/mink_wholebody/runtime.sh
"$TASKB_PYTHON" -u grasp/basket_transfer/bench_jointplan.py --mode grasp --headless --device cpu --enable_cameras --record-video --trials 1 --config "$PWD/grasp/basket_transfer/config_sugar_p00.json" --output /absolute/new/output
```

Do not repeat a successful condition without the owner's permission. The benchmark does not provide general-purpose collision-free planning, visual detection, navigation handover, generalized object handling, loaded locomotion or real-robot validation. The basket evaluator requires the object to be within basket bounds, with gripper contacts absent and low speed for three seconds. Verify an existing record without rerunning physics:

```bash
source grasp/experiments/mink_wholebody/runtime.sh
"$TASKB_PYTHON" grasp/basket_transfer/verify_record.py /root/grasp-validation/basket-physical-10/trial_01 --output /absolute/new/verification.json
```

Initial task grasp uses a known target and independent evaluation. During the successful trial, native jaw forces remained above 26 N through carry; after release, recorded retention was 6.04 s; sampled minimum object-bottom clearance over basket rim during the turn was about 8.8 mm; TCP azimuth changed 178.9 degrees. These measurements are specific to this run. Robot self-collision and general trajectory safety remain unverified.

Successful trial video: [`evidence/attempt_10_sugar_basket.mp4`](evidence/attempt_10_sugar_basket.mp4). The video documents one known-target simulation; see the independent checks and limitations above.
